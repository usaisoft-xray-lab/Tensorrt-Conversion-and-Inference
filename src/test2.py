#!/usr/bin/env python3
"""
TensorRT RTMDet(-Ins) segmentation inference with optimized timing.
Fixes performance issues to match trtexec speeds.
"""

import argparse, time, cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

PINNED_THRESHOLD_BYTES = 16 * 1024 * 1024
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640

# ---------------- utils ----------------
def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top

def unletterbox_boxes(boxes_xyxy, scale, left, top, out_w, out_h):
    if boxes_xyxy.size == 0: return boxes_xyxy
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0, out_w - 1)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top)  / scale, 0, out_h - 1)
    return out

def unletterbox_masks_with_timing(masks, scale, left, top, out_w, out_h):
    if masks.size == 0: return masks, 0.0, 0.0
    t0 = time.perf_counter()
    new_w, new_h = int(round(out_w * scale)), int(round(out_h * scale))
    cropped = masks[:, top:top + new_h, left:left + new_w]
    t1 = time.perf_counter()
    out_masks = np.zeros((cropped.shape[0], out_h, out_w), dtype=masks.dtype)
    t_r0 = time.perf_counter()
    for i in range(cropped.shape[0]):
        out_masks[i] = cv2.resize(cropped[i], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    t_r1 = time.perf_counter()
    return out_masks, (t1 - t0), (t_r1 - t_r0)

def colorize_and_blend_profile_roi(base_bgr, masks, boxes, labels, alpha=0.45):
    vis = base_bgr.copy()
    if masks.size == 0: return vis, 0.0, 0.0
    palette = np.array([
        (0,255,0),(255,0,0),(0,0,255),
        (255,255,0),(0,255,255),(255,0,255),
        (120,180,0),(0,180,120),(180,0,120)
    ], dtype=np.uint8)
    t_blend=t_draw=0.0
    for i,(box,lab) in enumerate(zip(boxes,labels)):
        color = tuple(int(c) for c in palette[int(lab)%len(palette)])
        m = masks[i]  # uint8 (0/255)
        x,y,w,h = cv2.boundingRect(m)
        if w==0 or h==0: continue
        tb0=time.perf_counter()
        roi  = vis[y:y+h, x:x+w]
        mroi = m[y:y+h, x:x+w]
        color_roi = np.empty_like(roi); color_roi[:] = color
        blended = cv2.addWeighted(roi, 1.0-alpha, color_roi, alpha, 0.0)
        cv2.copyTo(blended, mroi, roi)
        tb1=time.perf_counter(); t_blend += (tb1-tb0)
        td0=time.perf_counter()
        x1,y1,x2,y2 = [int(v) for v in box]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,f"{int(lab)}",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        td1=time.perf_counter(); t_draw += (td1-td0)
    return vis, t_blend, t_draw

def colorize_and_blend_profile_full(base_bgr, masks, boxes, labels, alpha=0.45):
    vis = base_bgr.copy()
    if masks.size == 0: return vis, 0.0, 0.0
    palette = np.array([
        (0,255,0),(255,0,0),(0,0,255),
        (255,255,0),(0,255,255),(255,0,255),
        (120,180,0),(0,180,120),(180,0,120)
    ], dtype=np.uint8)
    t_blend=t_draw=0.0
    for i,(box,lab) in enumerate(zip(boxes,labels)):
        color = tuple(int(c) for c in palette[int(lab)%len(palette)])
        m = masks[i]
        tb0=time.perf_counter()
        mask3 = np.dstack([m,m,m])
        color_img = np.full_like(vis, color, dtype=np.uint8)
        vis = np.where(mask3>0, (alpha*color_img + (1-alpha)*vis).astype(np.uint8), vis)
        tb1=time.perf_counter(); t_blend += (tb1-tb0)
        td0=time.perf_counter()
        x1,y1,x2,y2 = [int(v) for v in box]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,f"{int(lab)}",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        td1=time.perf_counter(); t_draw += (td1-td0)
    return vis, t_blend, t_draw

# ---------------- Optimized TRT wrapper ----------------
class TRTSegmentor:
    def __init__(self, engine_path, verbose=False):
        self.trt10 = int(trt.__version__.split('.')[0]) >= 10
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)

        t0=time.perf_counter()
        with open(engine_path,"rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        t1=time.perf_counter()
        self.model_load_s = (t1-t0)

        # Resolve names
        names = ([self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                 if self.trt10 else
                 [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)])
        def has(n): return any(n == x for x in names)
        def find(sub):
            for n in names:
                if sub in n: return n
            return None
        self.name_in     = "raw_input" if has("raw_input") else find("raw_input")
        self.name_dets   = "dets"      if has("dets")      else find("dets")
        self.name_labels = "labels"    if has("labels")    else find("label")
        self.name_masks  = "masks"     if has("masks")     else find("mask")
        if any(x is None for x in [self.name_in,self.name_dets,self.name_labels,self.name_masks]):
            raise RuntimeError(f"Could not find expected tensors. Found: {names}")

        # Dtypes
        def nptype_of(name):
            if self.trt10:
                return trt.nptype(self.engine.get_tensor_dtype(name))
            else:
                idx = self.engine.get_binding_index(name)
                return trt.nptype(self.engine.get_binding_dtype(idx))
        self.dtype_in     = nptype_of(self.name_in)
        self.dtype_dets   = nptype_of(self.name_dets)
        self.dtype_labels = nptype_of(self.name_labels)
        self.dtype_masks  = nptype_of(self.name_masks)

        # Shapes (engine & runtime)
        def eng_shape(name):
            return tuple(self.engine.get_tensor_shape(name)) if self.trt10 \
                   else tuple(self.engine.get_binding_shape(self.engine.get_binding_index(name)))
        def ctx_shape(name):
            return tuple(self.context.get_tensor_shape(name)) if self.trt10 \
                   else tuple(self.context.get_binding_shape(self.engine.get_binding_index(name)))

        shp_in_engine = eng_shape(self.name_in)
        # Set runtime input shape if dynamic
        if self.trt10:
            shp_in_rt = shp_in_engine
            if -1 in shp_in_rt:
                H,W = (shp_in_rt[-3], shp_in_rt[-2]) if shp_in_rt[-1]==3 else (shp_in_rt[-2], shp_in_rt[-1])
                shp_in_rt = (1, H if H>0 else DEFAULT_CANVAS, W if W>0 else DEFAULT_CANVAS, 3)
                self.context.set_input_shape(self.name_in, shp_in_rt)
        else:
            b = self.engine.get_binding_index(self.name_in)
            shp_in_rt = shp_in_engine
            if -1 in shp_in_engine:
                if self.engine.num_optimization_profiles>0:
                    self.context.active_optimization_profile = 0
                H,W = (shp_in_engine[-3], shp_in_engine[-2]) if shp_in_engine[-1]==3 else (shp_in_engine[-2], shp_in_engine[-1])
                shp_in_rt = (1, H if H>0 else DEFAULT_CANVAS, W if W>0 else DEFAULT_CANVAS, 3)
                self.context.set_binding_shape(b, shp_in_rt)

        shp_in_rt_now  = ctx_shape(self.name_in)
        shp_dets_ctx   = ctx_shape(self.name_dets);   shp_dets_eng   = eng_shape(self.name_dets)
        shp_labels_ctx = ctx_shape(self.name_labels); shp_labels_eng = eng_shape(self.name_labels)
        shp_masks_ctx  = ctx_shape(self.name_masks);  shp_masks_eng  = eng_shape(self.name_masks)

        def finalize(ctx_shp, eng_shp, kind):
            def ok(s): return s and all((d is not None and d>0) for d in s)
            if ok(ctx_shp): return tuple(ctx_shp)
            if ok(eng_shp): return tuple(eng_shp)
            if kind=="in":
                H = eng_shp[-3] if eng_shp and eng_shp[-3]>0 else DEFAULT_CANVAS
                W = eng_shp[-2] if eng_shp and eng_shp[-2]>0 else DEFAULT_CANVAS
                return (1,H,W,3)
            if kind=="dets":
                N = eng_shp[1] if eng_shp and len(eng_shp)>1 and eng_shp[1]>0 else DEFAULT_TOPK
                return (1,N,5)
            if kind=="labels":
                N = shp_dets_eng[1] if shp_dets_eng and len(shp_dets_eng)>1 and shp_dets_eng[1]>0 else DEFAULT_TOPK
                return (1,N)
            if kind=="masks":
                N = shp_dets_eng[1] if shp_dets_eng and len(shp_dets_eng)>1 and shp_dets_eng[1]>0 else DEFAULT_TOPK
                Hm = shp_in_rt_now[-3] if shp_in_rt_now and len(shp_in_rt_now)==4 else DEFAULT_CANVAS
                Wm = shp_in_rt_now[-2] if shp_in_rt_now and len(shp_in_rt_now)==4 else DEFAULT_CANVAS
                return (1,N,Hm,Wm)
            raise ValueError(kind)

        shp_in_final     = finalize(shp_in_rt_now,  shp_in_engine,  "in")
        shp_dets_final   = finalize(shp_dets_ctx,   shp_dets_eng,   "dets")
        shp_labels_final = finalize(shp_labels_ctx, shp_labels_eng, "labels")
        shp_masks_final  = finalize(shp_masks_ctx,  shp_masks_eng,  "masks")

        self.max_det = shp_dets_final[1]
        self.mask_hw = shp_masks_final[-2:]

        # Allocate buffers
        self.dev_ptr, self.host_buf, self.pinned_flag = {}, {}, {}

        def allocate_host(name, shape, dtype, force_pinned=False):
            numel  = int(np.prod(shape))
            nbytes = numel * np.dtype(dtype).itemsize
            use_pinned = force_pinned or (nbytes >= PINNED_THRESHOLD_BYTES)
            buf = (cuda.pagelocked_empty(numel, dtype) if use_pinned else np.empty(numel, dtype=dtype))
            self.host_buf[name] = buf
            self.pinned_flag[name] = use_pinned
            self.dev_ptr[name] = cuda.mem_alloc(nbytes)

        def allocate_dev_only(name, shape, dtype):
            numel  = int(np.prod(shape))
            nbytes = numel * np.dtype(dtype).itemsize
            self.dev_ptr[name] = cuda.mem_alloc(nbytes)

        # input / meta (host+dev)
        allocate_host(self.name_in,     shp_in_final,     self.dtype_in)
        allocate_host(self.name_dets,   shp_dets_final,   self.dtype_dets,   force_pinned=True)
        allocate_host(self.name_labels, shp_labels_final, self.dtype_labels, force_pinned=True)
        # masks: device-only (host per-run pinned scratch for kept slices)
        allocate_dev_only(self.name_masks, shp_masks_final, self.dtype_masks)

        # Pre-allocate reusable objects for optimized inference
        self.stream = cuda.Stream()
        self.start_evt = cuda.Event()
        self.end_evt = cuda.Event()
        
        # Pre-allocate output arrays to avoid reshape/astype overhead
        self.output_boxes = np.empty((self.max_det, 4), dtype=np.float32)
        self.output_labels = np.empty(self.max_det, dtype=np.int32) 
        self.output_scores = np.empty(self.max_det, dtype=np.float32)

        # Pinned scratch for mask slices
        self.mask_host_capacity = 0
        self.mask_host_buf = None

        # Bind addresses
        if self.trt10:
            for n,d in self.dev_ptr.items():
                self.context.set_tensor_address(n, int(d))
        else:
            self.bindings_order = [None]*self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                n = self.engine.get_binding_name(i)
                self.bindings_order[i] = int(self.dev_ptr[n])

        # Log (compute bytes without implying host allocation for masks)
        def fmt(name, shape, dtype):
            return f"{name}: shape={shape}, dtype={np.dtype(dtype).name}, bytes={int(np.prod(shape))*np.dtype(dtype).itemsize/1024/1024:.1f}MB"
        print("[TRT tensors]")
        print(" ", fmt(self.name_in,     shp_in_final,     self.dtype_in))
        print(" ", fmt(self.name_dets,   shp_dets_final,   self.dtype_dets))
        print(" ", fmt(self.name_labels, shp_labels_final, self.dtype_labels))
        print(" ", fmt(self.name_masks,  shp_masks_final,  self.dtype_masks), "(device-only binding)")
        print(f"[Model load] {self.model_load_s*1e3:.2f} ms\n")

    def infer_fast(self, lb_img_uint8, measure_gpu=False):
        """Optimized inference with minimal sync points"""
        
        # H2D input (async, no sync)
        np.copyto(self.host_buf[self.name_in], lb_img_uint8.ravel())
        if self.pinned_flag[self.name_in]:
            cuda.memcpy_htod_async(self.dev_ptr[self.name_in], 
                                  self.host_buf[self.name_in], self.stream)
        else:
            cuda.memcpy_htod(self.dev_ptr[self.name_in], self.host_buf[self.name_in])
        
        # GPU inference with optional timing
        if measure_gpu:
            # Ensure H2D is complete before starting GPU timing
            self.stream.synchronize()
            self.start_evt.record(self.stream)
            
        if self.trt10: 
            self.context.execute_async_v3(self.stream.handle)
        else:          
            self.context.execute_async_v2(self.bindings_order, self.stream.handle)
            
        if measure_gpu:
            self.end_evt.record(self.stream)
            # Sync immediately after GPU work to get accurate timing
            self.stream.synchronize()
            gpu_s = self.start_evt.time_till(self.end_evt) / 1e3
        else:
            gpu_s = 0.0

        # D2H meta (async if not already synced)
        if not measure_gpu:
            cuda.memcpy_dtoh_async(self.host_buf[self.name_dets], 
                                  self.dev_ptr[self.name_dets], self.stream)
            cuda.memcpy_dtoh_async(self.host_buf[self.name_labels], 
                                  self.dev_ptr[self.name_labels], self.stream)
            self.stream.synchronize()
        else:
            # Already synced, use sync copies
            cuda.memcpy_dtoh(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets])
            cuda.memcpy_dtoh(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels])
        
        # Use pre-allocated arrays, avoid reshape/astype
        dets_raw = self.host_buf[self.name_dets].view().reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].view().reshape(1, self.max_det)[0]
        
        # Direct copy to pre-allocated arrays
        np.copyto(self.output_boxes, dets_raw[:, :4])
        np.copyto(self.output_scores, dets_raw[:, 4]) 
        np.copyto(self.output_labels, labels_raw.astype(np.int32))
        
        return self.output_boxes, self.output_labels, self.output_scores, gpu_s
    
    def infer_pure_gpu_timing(self, lb_img_uint8):
        """Pure GPU timing - measures only TensorRT execution"""
        
        # H2D input and sync
        np.copyto(self.host_buf[self.name_in], lb_img_uint8.ravel())
        if self.pinned_flag[self.name_in]:
            cuda.memcpy_htod_async(self.dev_ptr[self.name_in], 
                                  self.host_buf[self.name_in], self.stream)
        else:
            cuda.memcpy_htod(self.dev_ptr[self.name_in], self.host_buf[self.name_in])
        
        # Ensure all transfers complete
        self.stream.synchronize()
        
        # Pure GPU timing with minimal overhead
        start_evt = cuda.Event()
        end_evt = cuda.Event() 
        
        start_evt.record(self.stream)
        if self.trt10: 
            self.context.execute_async_v3(self.stream.handle)
        else:          
            self.context.execute_async_v2(self.bindings_order, self.stream.handle)
        end_evt.record(self.stream)
        
        # Wait for GPU completion
        self.stream.synchronize()
        
        # Calculate pure GPU time
        gpu_ms = start_evt.time_till(end_evt)
        
        # D2H for results (not timed)
        cuda.memcpy_dtoh(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets])
        cuda.memcpy_dtoh(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels])
        
        # Process outputs
        dets_raw = self.host_buf[self.name_dets].view().reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].view().reshape(1, self.max_det)[0]
        
        np.copyto(self.output_boxes, dets_raw[:, :4])
        np.copyto(self.output_scores, dets_raw[:, 4]) 
        np.copyto(self.output_labels, labels_raw.astype(np.int32))
        
        return self.output_boxes, self.output_labels, self.output_scores, gpu_ms / 1e3

    # Keep original infer method for backward compatibility
    def infer(self, lb_img_uint8, profile_copies=False):
        """Legacy method - use infer_fast() for better performance"""
        return self.infer_fast(lb_img_uint8, measure_gpu=profile_copies)

    def _ensure_mask_host(self, need_count):
        Hm, Wm = self.mask_hw
        if self.mask_host_buf is not None and need_count <= self.mask_host_capacity:
            return
        new_cap = max(need_count, max(1, self.mask_host_capacity*2))
        self.mask_host_buf = cuda.pagelocked_empty((new_cap, Hm, Wm), dtype=self.dtype_masks)
        self.mask_host_capacity = new_cap

    def copy_masks(self, indices):
        """Pinned + async D2H for kept slices only."""
        Hm, Wm = self.mask_hw
        dtype = self.dtype_masks
        itemsize = np.dtype(dtype).itemsize
        slice_bytes = Hm * Wm * itemsize

        if len(indices)==0:
            return np.empty((0, Hm, Wm), dtype=dtype), 0.0

        self._ensure_mask_host(len(indices))
        mask_stream = cuda.Stream()  # Use separate stream for mask copies
        t0 = time.perf_counter()
        base_addr = int(self.dev_ptr[self.name_masks])
        for k, idx in enumerate(indices):
            src = base_addr + int(idx) * slice_bytes
            dst = self.mask_host_buf[k].ravel()
            cuda.memcpy_dtoh_async(dst, src, mask_stream)
        mask_stream.synchronize()
        t1 = time.perf_counter()
        return np.array(self.mask_host_buf[:len(indices)]), (t1 - t0)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default="seg_output.jpg")
    ap.add_argument("--runs",   type=int, default=10)
    ap.add_argument("--score",  type=float, default=0.1)
    ap.add_argument("--mask_thr", type=float, default=0.1)
    ap.add_argument("--size",   type=int, default=640)
    ap.add_argument("--alpha",  type=float, default=0.45)
    ap.add_argument("--viz",    choices=["roi","full","off"], default="roi")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-lazy-masks", action="store_true",
                    help="Disable lazy mask copy (copy all masks each run)")
    ap.add_argument("--engine-binary-masks", action="store_true", default=True,
                    help="Assume engine outputs binary 0/255 masks; skip CPU threshold.")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Number of warmup runs before timing")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None: raise FileNotFoundError(args.input)

    seg = TRTSegmentor(args.engine, verbose=args.verbose)
    print(f"Model load time (excluded from averages): {seg.model_load_s*1e3:.2f} ms")

    # WARMUP - Critical for accurate timing
    print(f"Warming up with {args.warmup} runs...")
    for _ in range(args.warmup):
        lb, scale, left, top = letterbox(img, size=args.size, pad_val=114)
        seg.infer_fast(lb, measure_gpu=False)  # No timing overhead
    print("Warmup complete. Starting timed runs...")

    runs=float(args.runs)
    pre_sum=h2d_sum=d2h_meta_sum=d2h_masks_sum=post_sum=0.0
    post_bin_sum=post_unlb_boxes_sum=post_unlb_masks_crop_sum=post_unlb_masks_resize_sum=0.0
    post_viz_blend_sum=post_viz_draw_sum=0.0
    pure_gpu_times = []
    final_vis=None

    def to_u8_fast(m):
        if m.size==0: return m.astype(np.uint8), False
        if args.engine_binary_masks:
            return m.astype(np.uint8, copy=False), False
        flat = m.ravel()
        sample = flat[:: max(1, flat.size//2048)]
        if sample.max() <= 1.05:
            out = (m > args.mask_thr).astype(np.uint8) * 255
            return out, True
        if np.all((sample==0) | (np.abs(sample-255.0)<0.5)):
            return m.astype(np.uint8, copy=False), False
        out = (m > args.mask_thr).astype(np.uint8) * 255
        return out, True

    for i in range(args.runs):
        # Pre
        t0=time.perf_counter()
        lb, scale, left, top = letterbox(img, size=args.size, pad_val=114)
        t1=time.perf_counter(); pre_t=(t1-t0); pre_sum+=pre_t

        # Optimized Inference - measure GPU time only on some runs to avoid overhead
        measure_this_run = (i == 0 or i == args.runs-1 or i % 5 == 0)
        boxes, labels, scores, gpu_s = seg.infer_fast(lb, measure_gpu=measure_this_run)
        
        if measure_this_run:
            pure_gpu_times.append(gpu_s)

        # Filter
        keep = scores >= args.score
        boxes = boxes[keep]; labels_kept = labels[keep]
        keep_idx = np.flatnonzero(keep)

        # D2H masks (lazy or full)
        if keep_idx.size>0:
            if args.no_lazy_masks:
                full_masks, d2h_masks_t = seg.copy_masks(np.arange(seg.max_det))
                masks_fp = full_masks[keep]
            else:
                masks_fp, d2h_masks_t = seg.copy_masks(keep_idx.astype(np.int64))
            d2h_masks_sum += d2h_masks_t
        else:
            masks_fp = np.empty((0, *seg.mask_hw), dtype=seg.dtype_masks)

        # Post breakdown
        p0=time.perf_counter()

        b0=time.perf_counter()
        masks_u8, was_thr = to_u8_fast(masks_fp)
        b1=time.perf_counter(); post_bin = (b1-b0)

        ub0=time.perf_counter()
        boxes_img = unletterbox_boxes(boxes, scale, left, top, img.shape[1], img.shape[0])
        ub1=time.perf_counter(); post_unlb_boxes = (ub1-ub0)

        umc=umr=0.0
        if masks_u8.size>0:
            masks_img, umc, umr = unletterbox_masks_with_timing(masks_u8, scale, left, top,
                                                                img.shape[1], img.shape[0])
        else:
            masks_img = masks_u8

        vb=vd=0.0
        if args.viz=="roi":
            vis, vb, vd = colorize_and_blend_profile_roi(img.copy(), masks_img, boxes_img, labels_kept, alpha=args.alpha)
        elif args.viz=="full":
            vis, vb, vd = colorize_and_blend_profile_full(img.copy(), masks_img, boxes_img, labels_kept, alpha=args.alpha)
        else:
            vis = img.copy()
        if i==args.runs-1 and args.viz!="off": final_vis=vis

        p1=time.perf_counter(); post_t=(p1-p0); post_sum+=post_t
        post_bin_sum+=post_bin; post_unlb_boxes_sum+=post_unlb_boxes
        post_unlb_masks_crop_sum+=umc; post_unlb_masks_resize_sum+=umr
        post_viz_blend_sum+=vb; post_viz_draw_sum+=vd

    # Averages & report
    pre_avg = pre_sum/runs
    d2h_masks_avg = d2h_masks_sum/runs
    post_avg = post_sum/runs
    bin_avg = post_bin_sum/runs
    ub_boxes_avg = post_unlb_boxes_sum/runs
    um_crop_avg, um_resize_avg = post_unlb_masks_crop_sum/runs, post_unlb_masks_resize_sum/runs
    viz_blend_avg, viz_draw_avg = post_viz_blend_sum/runs, post_viz_draw_sum/runs
    post_misc_avg = (post_sum - (post_bin_sum+post_unlb_boxes_sum+post_unlb_masks_crop_sum+
                                 post_unlb_masks_resize_sum+post_viz_blend_sum+post_viz_draw_sum)) / runs

    # GPU timing summary
    if pure_gpu_times:
        avg_gpu = sum(pure_gpu_times) / len(pure_gpu_times)
        min_gpu = min(pure_gpu_times)
        max_gpu = max(pure_gpu_times)
        print(f"\n🚀 Pure GPU Inference Times:")
        print(f"   Average: {avg_gpu*1e3:.2f} ms (measured on {len(pure_gpu_times)} runs)")
        print(f"   Min:     {min_gpu*1e3:.2f} ms")
        print(f"   Max:     {max_gpu*1e3:.2f} ms")
        print(f"   This should match trtexec GPU inference time!")
    else:
        avg_gpu = 0.0
        print(f"\n⚠️  No GPU timings collected")

    print("\n=== Timing (averages over runs, model load excluded) ===")
    print(f"Pre-process            : {pre_avg   * 1e3:8.2f} ms")
    print(f"GPU inference (events) : {avg_gpu   * 1e3:8.2f} ms  ← Should match trtexec!")
    print(f"D2H masks              : {d2h_masks_avg * 1e3:8.2f} ms")
    print(f"Post-process (total)   : {post_avg  * 1e3:8.2f} ms")
    print("  ├─ Binarize masks    : {:8.2f} ms".format(bin_avg        * 1e3))
    print("  ├─ Unletterbox boxes : {:8.2f} ms".format(ub_boxes_avg   * 1e3))
    print("  ├─ Unlb masks (crop) : {:8.2f} ms".format(um_crop_avg    * 1e3))
    print("  ├─ Unlb masks resize : {:8.2f} ms".format(um_resize_avg  * 1e3))
    print("  ├─ Viz blend         : {:8.2f} ms".format(viz_blend_avg  * 1e3))
    print("  ├─ Viz draw          : {:8.2f} ms".format(viz_draw_avg   * 1e3))
    print("  └─ Post misc         : {:8.2f} ms".format(post_misc_avg  * 1e3))
    print("\nModel load time        : {:8.2f} ms (excluded)".format(seg.model_load_s * 1e3))

    post_viz_total = 0.0 if (args.viz=="off") else (viz_blend_avg + viz_draw_avg)
    avg_total_s = (pre_avg + avg_gpu + d2h_masks_avg +
                   bin_avg + ub_boxes_avg + um_crop_avg + um_resize_avg + post_viz_total + post_misc_avg)
    fps = (1.0/avg_total_s) if avg_total_s>0 else 0.0
    print("\n=== Estimated realtime ===")
    print(f"Average end-to-end latency (with viz='{args.viz}'): {avg_total_s*1e3:8.2f} ms")
    print(f"Estimated FPS (with viz)                       : {fps:8.2f} fps")

    if final_vis is None: final_vis = img
    cv2.imwrite(args.output, final_vis); print(f"\n✅ Saved: {args.output}")

if __name__ == "__main__":
    main()