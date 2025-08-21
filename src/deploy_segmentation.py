#!/usr/bin/env python3
# Fast TensorRT build for RTMDet(-Ins) segmentation (FP16, opt-level 5).
# Usage:  python deploy_segmentation.py <model.onnx> <out.engine>

import os, sys, json, pathlib
from typing import Optional, Dict
import numpy as np
import tensorrt as trt

# ---------- helpers ----------
def find_sidecar_pipeline(onnx_path: str) -> Optional[str]:
    p = pathlib.Path(onnx_path)
    cand = p.with_name("pipeline.json")
    return str(cand) if cand.exists() else None

def load_pipeline(pipeline_path: Optional[str]) -> Optional[Dict]:
    if not pipeline_path or not os.path.exists(pipeline_path):
        print("No pipeline.json found; using defaults.")
        return None
    try:
        with open(pipeline_path, "r") as f:
            cfg = json.load(f)
        print(f"Loaded pipeline: {pipeline_path}")
        return cfg
    except Exception as e:
        print(f"[warn] Failed to read pipeline.json ({e}); using defaults.")
        return None

def extract_preprocess(cfg: Optional[Dict]) -> Dict:
    # Defaults (OpenMMLab-ish BGR stats)
    params = dict(
        input_size=(640, 640),                 # (H, W)
        mean=[103.53, 116.28, 123.675],        # BGR
        std=[57.375, 57.12, 58.395],
        to_rgb=False,
    )
    if not cfg:
        return params
    try:
        pipeline = cfg.get("pipeline", {})
        for task in pipeline.get("tasks", []):
            if task.get("name") == "Preprocess" and task.get("module") == "Transform":
                for t in task.get("transforms", []):
                    tt = t.get("type", "")
                    if tt == "Resize" and isinstance(t.get("size"), list) and len(t["size"]) >= 2:
                        params["input_size"] = (int(t["size"][0]), int(t["size"][1]))
                    elif tt == "Normalize":
                        if "mean" in t: params["mean"] = t["mean"]
                        if "std"  in t: params["std"]  = t["std"]
                        params["to_rgb"] = t.get("to_rgb", params["to_rgb"])
    except Exception:
        pass
    return params

def ones_like_rank(t: trt.ITensor):
    """Broadcastable scalar dims (1,1,...,1) with same rank as t."""
    rank = len(t.shape)
    return (1,) * max(1, rank)

def guess_mask_outputs(network: trt.INetworkDefinition):
    """Prefer outputs named like 'mask*'; else choose the largest spatial output."""
    outs = [(i, network.get_output(i)) for i in range(network.num_outputs)]
    idxs = [i for i, t in outs if "mask" in (t.name or "").lower()]
    if idxs:
        return idxs
    def spatial_score(t: trt.ITensor):
        s = t.shape
        if len(s) < 3: return -1
        H = s[-2] if isinstance(s[-2], int) and s[-2] > 0 else 1
        W = s[-1] if isinstance(s[-1], int) and s[-1] > 0 else 1
        return H * W
    scored = sorted(((i, spatial_score(t)) for i, t in outs), key=lambda x: x[1], reverse=True)
    return [i for i, sc in scored if sc > 1][:1]

# ---------- graph surgery ----------
def insert_preprocess(network: trt.INetworkDefinition, H: int, W: int,
                      MEAN: np.ndarray, STD: np.ndarray, to_rgb: bool) -> str:
    """
    Replace original input with NHWC uint8 'raw_input' and do cast/normalize in-engine:
      raw(NHWC u8) -> float32 -> (optional BGR->RGB) -> NHWC->NCHW -> normalize
    """
    orig_in = network.get_input(0)
    raw = network.add_input("raw_input", trt.DataType.UINT8, (1, H, W, 3))

    cast = network.add_identity(raw); cast.set_output_type(0, trt.DataType.FLOAT)
    x = cast.get_output(0)

    if to_rgb:
        # NHWC channel reorder
        sl_r = network.add_slice(x, (0, 0, 0, 2), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        sl_g = network.add_slice(x, (0, 0, 0, 1), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        sl_b = network.add_slice(x, (0, 0, 0, 0), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        cat = network.add_concatenation([sl_r, sl_g, sl_b]); cat.axis = 3
        x = cat.get_output(0)

    # NHWC -> NCHW
    shuf = network.add_shuffle(x); shuf.first_transpose = (0, 3, 1, 2)
    x = shuf.get_output(0)

    # (x - mean) / std (channel-wise)
    scale = (1.0 / STD).astype(np.float32)
    shift = (-MEAN * scale).astype(np.float32)
    sc = network.add_scale(x, trt.ScaleMode.CHANNEL, shift=shift, scale=scale)
    x = sc.get_output(0)

    # reconnect to consumers of the old input
    for li in range(network.num_layers):
        L = network.get_layer(li)
        for ii in range(L.num_inputs):
            if L.get_input(ii) is orig_in:
                L.set_input(ii, x)
    network.remove_tensor(orig_in)
    return "raw_input"

def attach_gpu_binary_mask_fp16(network: trt.INetworkDefinition, mask_tensor: trt.ITensor, thr: float = 0.5):
    """
    Portable GPU-side binarization (works on older TRT builds):
      bool_m = (mask > thr)    # ElementWise GREATER with broadcast scalar
      half_m = cast(bool_m -> FP16)
      scaled = half_m * 255    # FP16 0/255
      out    = scaled (FP16)
    Also: rename original tensor to '<name>__raw', new FP16 binarized keeps original name.
    """
    old_name = mask_tensor.name or "masks"

    # Avoid name collision later
    try:
        mask_tensor.name = old_name + "__raw"
    except Exception:
        pass

    # We can unmark the original as network output (it stays in graph with new name)
    network.unmark_output(mask_tensor)

    # Broadcastable scalar threshold
    one_dims = ones_like_rank(mask_tensor)
    thr_c = network.add_constant(one_dims, np.array([thr], dtype=np.float32)).get_output(0)

    # Compare via ElementWise GREATER (older TRT lacks add_comparison) -> BOOL
    gt = network.add_elementwise(mask_tensor, thr_c, trt.ElementWiseOperation.GREATER)
    bool_m = gt.get_output(0)

    # Cast BOOL -> FP16, then scale to 0/255 (still FP16)
    cast_f = network.add_identity(bool_m); cast_f.set_output_type(0, trt.DataType.HALF)
    half_m = cast_f.get_output(0)
    s255 = network.add_constant(one_dims, np.array([255.0], dtype=np.float16)).get_output(0)
    mul = network.add_elementwise(half_m, s255, trt.ElementWiseOperation.PROD)
    scaled = mul.get_output(0)

    # Keep original name on new (binarized) output
    scaled.name = old_name
    network.mark_output(scaled)

# ---------- build ----------
def build_engine(onnx_path: str, engine_path: str):
    # Optional sidecar pipeline
    pipeline_path = find_sidecar_pipeline(onnx_path)
    cfg = load_pipeline(pipeline_path)
    pp = extract_preprocess(cfg)
    H, W = int(pp["input_size"][0]), int(pp["input_size"][1])
    MEAN = np.array(pp["mean"], dtype=np.float32)
    STD  = np.array(pp["std"],  dtype=np.float32)
    print(f"Preprocess: size={W}x{H}, mean={MEAN.tolist()}, std={STD.tolist()}, to_rgb={pp['to_rgb']}")

    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ONNX parse failed:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit(1)

        # In-engine preprocess
        raw_name = insert_preprocess(network, H, W, MEAN, STD, to_rgb=pp["to_rgb"])

        # Add GPU-side binarizer for mask-like outputs (FP16 0/255)
        mask_idxs = guess_mask_outputs(network)
        if mask_idxs:
            for idx in mask_idxs:
                attach_gpu_binary_mask_fp16(network, network.get_output(idx), thr=0.5)
        else:
            print("[warn] Could not auto-detect mask output; leaving outputs unchanged.")

        # Builder config: FP16 + max tactics + generous workspace
        config = builder.create_builder_config()
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        try:
            config.builder_optimization_level = 5  # 0..5 (5=max)
            print("builder_optimization_level=5")
        except Exception:
            pass
        # 2 GB workspace (often a good speed default)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 * 1024 * 1024)

        # Fixed-shape profile (fastest for a single resolution)
        profile = builder.create_optimization_profile()
        profile.set_shape(raw_name, min=(1, H, W, 3), opt=(1, H, W, 3), max=(1, H, W, 3))
        config.add_optimization_profile(profile)

        print("Building TensorRT engine (FP16, opt-level 5)...")
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("Build failed.")
            sys.exit(2)

        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(plan)

        print(f"✅ Saved engine: {engine_path}")
        print("\n=== Engine Summary ===")
        print(f"Input : {raw_name} [1,{H},{W},3] uint8 -> in-engine cast/normalize")
        print("Output: masks thresholded on-GPU and exported as FP16 (0/255)")
        print("FP16  : enabled" if builder.platform_has_fast_fp16 else "FP16 : not available")
        try:
            print(f"Opt   : level {config.builder_optimization_level}")
        except Exception:
            pass

def main():
    if len(sys.argv) != 3:
        print("Usage: python deploy_segmentation.py <model.onnx> <out.engine>")
        sys.exit(64)
    build_engine(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
