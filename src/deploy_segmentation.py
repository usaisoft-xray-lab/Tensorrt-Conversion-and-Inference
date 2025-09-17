#!/usr/bin/env python3
# Simple fix - just add optimization flags to your working deployment
# Keeps FP16 mask output (same as original) but with better optimization

import os, sys, json, pathlib
from typing import Optional, Dict
import numpy as np
import tensorrt as trt

# ---------- helpers (exactly as original) ----------
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
    params = dict(
        input_size=(640, 640),
        mean=[103.53, 116.28, 123.675],
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
    rank = len(t.shape)
    return (1,) * max(1, rank)

def guess_mask_outputs(network: trt.INetworkDefinition):
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

# ---------- Keep original preprocessing ----------
def insert_preprocess(network: trt.INetworkDefinition, H: int, W: int,
                      MEAN: np.ndarray, STD: np.ndarray, to_rgb: bool) -> str:
    orig_in = network.get_input(0)
    raw = network.add_input("raw_input", trt.DataType.UINT8, (1, H, W, 3))

    cast = network.add_identity(raw)
    cast.set_output_type(0, trt.DataType.FLOAT)
    x = cast.get_output(0)

    if to_rgb:
        sl_r = network.add_slice(x, (0, 0, 0, 2), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        sl_g = network.add_slice(x, (0, 0, 0, 1), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        sl_b = network.add_slice(x, (0, 0, 0, 0), (1, H, W, 1), (1, 1, 1, 1)).get_output(0)
        cat = network.add_concatenation([sl_r, sl_g, sl_b])
        cat.axis = 3
        x = cat.get_output(0)

    shuf = network.add_shuffle(x)
    shuf.first_transpose = (0, 3, 1, 2)
    x = shuf.get_output(0)

    scale = (1.0 / STD).astype(np.float32)
    shift = (-MEAN * scale).astype(np.float32)
    sc = network.add_scale(x, trt.ScaleMode.CHANNEL, shift=shift, scale=scale)
    x = sc.get_output(0)

    for li in range(network.num_layers):
        L = network.get_layer(li)
        for ii in range(L.num_inputs):
            if L.get_input(ii) is orig_in:
                L.set_input(ii, x)
    network.remove_tensor(orig_in)
    return "raw_input"

# ---------- Keep original FP16 mask output ----------
def attach_gpu_binary_mask_fp16(network: trt.INetworkDefinition, mask_tensor: trt.ITensor, thr: float = 0.5):
    """Keep exactly as original - FP16 output"""
    old_name = mask_tensor.name or "masks"
    try:
        mask_tensor.name = old_name + "__raw"
    except Exception:
        pass

    network.unmark_output(mask_tensor)
    one_dims = ones_like_rank(mask_tensor)
    thr_c = network.add_constant(one_dims, np.array([thr], dtype=np.float32)).get_output(0)

    gt = network.add_elementwise(mask_tensor, thr_c, trt.ElementWiseOperation.GREATER)
    bool_m = gt.get_output(0)

    cast_f = network.add_identity(bool_m)
    cast_f.set_output_type(0, trt.DataType.HALF)
    half_m = cast_f.get_output(0)
    
    s255 = network.add_constant(one_dims, np.array([255.0], dtype=np.float16)).get_output(0)
    mul = network.add_elementwise(half_m, s255, trt.ElementWiseOperation.PROD)
    scaled = mul.get_output(0)

    scaled.name = old_name
    network.mark_output(scaled)

# ---------- Build with optimization flags ----------
def build_engine(onnx_path: str, engine_path: str, workspace_gb: int = 4):
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

        raw_name = insert_preprocess(network, H, W, MEAN, STD, to_rgb=pp["to_rgb"])

        mask_idxs = guess_mask_outputs(network)
        if mask_idxs:
            for idx in mask_idxs:
                attach_gpu_binary_mask_fp16(network, network.get_output(idx), thr=0.5)
        else:
            print("[warn] Could not auto-detect mask output")

        # OPTIMIZATION: Better builder config
        config = builder.create_builder_config()
        
        # Enable FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16: Enabled")
        
        # OPTIMIZATION: Max optimization level
        try:
            config.builder_optimization_level = 5
            print("Optimization level: 5 (maximum)")
        except Exception:
            print("Could not set optimization level")
        
        # OPTIMIZATION: Larger workspace
        workspace_bytes = workspace_gb * 1024 * 1024 * 1024
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        print(f"Workspace: {workspace_gb} GB")
        
        # OPTIMIZATION: Additional flags for better performance
        try:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            print("Additional optimization flags: Enabled")
        except:
            pass

        profile = builder.create_optimization_profile()
        profile.set_shape(raw_name, min=(1, H, W, 3), opt=(1, H, W, 3), max=(1, H, W, 3))
        config.add_optimization_profile(profile)

        print("Building optimized TensorRT engine...")
        print("This may take a few minutes for maximum optimization...")
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("Build failed.")
            sys.exit(2)

        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(plan)

        print(f"✅ Saved engine: {engine_path}")
        print("\n=== Optimization Summary ===")
        print(f"Input : {raw_name} [1,{H},{W},3] uint8")
        print(f"Output: Binary masks as FP16 (0/255)")
        print(f"FP16  : Enabled")
        print(f"Opt   : Level 5")
        print(f"Memory: {workspace_gb} GB workspace")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model")
    parser.add_argument("engine_output")
    parser.add_argument("--workspace", type=int, default=4,
                       help="Workspace size in GB (default: 4)")
    args = parser.parse_args()
    
    build_engine(args.onnx_model, args.engine_output, 
                workspace_gb=args.workspace)

if __name__ == "__main__":
    main()
