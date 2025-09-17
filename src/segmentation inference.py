#!/usr/bin/env python3
"""
Optimized TensorRT RTMDet segmentation inference with full timing breakdown
Key optimizations:
- GPU-side thresholding of float masks to uint8 (cuts D2H bandwidth)
- GPU-side ROI-only resize of masks (fast path, --gpu-roi)
- CPU ROI fallback for uint8 engines
- Pinned buffers and minimal copies
- Detailed per-step timing that sums to TOTAL
"""

import argparse, time, cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from pycuda.compiler import SourceModule
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
PINNED_THRESHOLD_BYTES = 16 * 1024 * 1024  # Use pinned memory for large buffers
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640

# ---------------- Preprocessing utilities ----------------
def letterbox(img, size=640, pad_val=114):
    """Resize and pad image to square"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top

def unletterbox_boxes(boxes_xyxy, scale, left, top, out_w, out_h):
    """Convert boxes from letterboxed to original coordinates"""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    out = boxes_xyxy.astype(np.float32, copy=False)
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0, out_w - 1)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / scale, 0, out_h - 1)
    return out

def unletterbox_masks_roi(masks_u8, boxes_img, scale, left, top, out_w, out_h, max_workers=4):
    """
    CPU fallback: resize only bbox ROI instead of full 640x640.
    Returns (out_masks, prep_time, resize_time)
    """
    if masks_u8.size == 0:
        return masks_u8, 0.0, 0.0

    t0 = time.perf_counter()
    N, Hm, Wm = masks_u8.shape
    out_masks = np.zeros((N, out_h, out_w), dtype=np.uint8)

    boxes_img = boxes_img.astype(np.float32, copy=False)

    # Map original image box -> letterboxed (640x640) coords
    x1 = np.clip(left + boxes_img[:, 0] * scale, 0, Wm).astype(np.int32)
    y1 = np.clip(top  + boxes_img[:, 1] * scale, 0, Hm).astype(np.int32)
    x2 = np.clip(left + boxes_img[:, 2] * scale, 0, Wm).astype(np.int32)
    y2 = np.clip(top  + boxes_img[:, 3] * scale, 0, Hm).astype(np.int32)

    # Destination region in original image
    dx1 = np.clip(boxes_img[:, 0], 0, out_w).astype(np.int32)
    dy1 = np.clip(boxes_img[:, 1], 0, out_h).astype(np.int32)
    dx2 = np.clip(boxes_img[:, 2], 0, out_w).astype(np.int32)
    dy2 = np.clip(boxes_img[:, 3], 0, out_h).astype(np.int32)

    t1 = time.perf_counter()
    prep_time = t1 - t0

    def place(i):
        if x2[i] <= x1[i] or y2[i] <= y1[i] or dx2[i] <= dx1[i] or dy2[i] <= dy1[i]:
            return
        src = masks_u8[i, y1[i]:y2[i], x1[i]:x2[i]]
        if src.size == 0:
            return
        w_dst = int(dx2[i] - dx1[i])
        h_dst = int(dy2[i] - dy1[i])
        dst_roi = cv2.resize(src, (w_dst, h_dst), interpolation=cv2.INTER_NEAREST)
        out_masks[i, dy1[i]:dy2[i], dx1[i]:dx2[i]] = dst_roi

    t2 = time.perf_counter()
    if N > 3:
        with ThreadPoolExecutor(max_workers=min(max_workers, N)) as ex:
            list(ex.map(place, range(N)))
    else:
        for i in range(N):
            place(i)
    t3 = time.perf_counter()
    resize_time = t3 - t2

    return out_masks, prep_time, resize_time

def colorize_blend_optimized(base_bgr, masks, boxes, labels, alpha=0.45):
    """Faster visualization using OpenCV operations (ROI only)"""
    vis = base_bgr.copy()
    if masks.size == 0:
        return vis, 0.0, 0.0
    palette = np.array([
        (0,255,0), (255,0,0), (0,0,255),
        (255,255,0), (0,255,255), (255,0,255),
        (120,180,0), (0,180,120), (180,0,120)
    ], dtype=np.uint8)
    t_blend = t_draw = 0.0
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        color = tuple(int(c) for c in palette[int(lab) % len(palette)])
        m = masks[i]
        x, y, w, h = cv2.boundingRect(m)
        if w == 0 or h == 0:  # skip empty
            continue
        tb0 = time.perf_counter()
        roi = vis[y:y+h, x:x+w]
        mroi = m[y:y+h, x:x+w]
        color_roi = np.full_like(roi, color, dtype=np.uint8)
        blended = cv2.addWeighted(roi, 1.0-alpha, color_roi, alpha, 0.0)
        cv2.copyTo(blended, mroi, roi)
        tb1 = time.perf_counter()
        t_blend += (tb1 - tb0)

        td0 = time.perf_counter()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{int(lab)}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        td1 = time.perf_counter()
        t_draw += (td1 - td0)
    return vis, t_blend, t_draw

# ---------------- Optimized TRT Wrapper ----------------
class OptimizedTRTSegmentor:
    def __init__(self, engine_path, verbose=False):
        self.trt10 = int(trt.__version__.split('.')[0]) >= 10
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)

        t0 = time.perf_counter()
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        t1 = time.perf_counter()
        self.model_load_s = (t1 - t0)

        self._setup_io()
        self.stream = cuda.Stream()
        # Events used for pure GPU timing of inference only
        self.start_evt = cuda.Event()
        self.end_evt = cuda.Event()

        self.output_boxes = np.empty((self.max_det, 4), dtype=np.float32)
        self.output_labels = np.empty(self.max_det, dtype=np.int32)
        self.output_scores = np.empty(self.max_det, dtype=np.float32)

        # Host buffer for masks (uint8 or float32) - legacy path
        self.mask_host_capacity = 0
        self.mask_host_buf = None  # dtype=self.dtype_masks

        # --- GPU kernel to threshold float32->uint8 on device (full-mask pack path)
        self._thresh_mod = SourceModule(r"""
        extern "C" __global__
        void thresh_u8(const float* __restrict__ src,
                       unsigned char* __restrict__ dst,
                       int n, int src_offset, int dst_offset, float thr) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                float v = src[src_offset + i];
                dst[dst_offset + i] = (unsigned char)((v > thr) ? 255 : 0);
            }
        }
        """)
        self._k_thresh = self._thresh_mod.get_function("thresh_u8")

        # uint8 device + host staging for full-mask packed copies
        self.dev_masks_u8 = None
        self.u8_capacity = 0
        self.mask_host_buf_u8 = None

        # --- Kernel: threshold + ROI resize (nearest) per mask (fast path)
        self._roi_mod = SourceModule(r"""
        extern "C" __global__
        void resize_thresh_roi(
            const float* __restrict__ src,  // base pointer to all masks (float32)
            int Hm, int Wm,
            int src_offset_elems,           // starting element offset for this mask
            int sx, int sy, int sw, int sh, // ROI in letterboxed space
            unsigned char* __restrict__ dst,// base output buffer for all ROIs (u8)
            int dst_offset_elems,           // offset into dst in elements (bytes)
            int dw, int dh,                 // ROI size in original image space
            float thr
        ){
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= dw || y >= dh) return;

            // Map output (x,y) to source (ix,iy) in letterboxed ROI space (nearest)
            float fx = ((x + 0.5f) * (float)sw / (float)dw) - 0.5f;
            float fy = ((y + 0.5f) * (float)sh / (float)dh) - 0.5f;
            int ix = sx + (int)roundf(fx);
            int iy = sy + (int)roundf(fy);
            ix = max(0, min(Wm - 1, ix));
            iy = max(0, min(Hm - 1, iy));

            float v = src[src_offset_elems + iy * Wm + ix];
            dst[dst_offset_elems + y * dw + x] = (unsigned char)(v > thr ? 255 : 0);
        }
        """)
        self._k_roi = self._roi_mod.get_function("resize_thresh_roi")

        # staging for packed ROI outputs (device + host)
        self.dev_rois_u8 = None
        self.host_rois_u8 = None
        self.roi_bytes_capacity = 0

        self._warmup_cv2()

        print(f"[TRT Engine Loaded] {self.model_load_s*1e3:.2f} ms")
        print(f"[Max detections] {self.max_det}")
        print(f"[Mask size] {self.mask_hw}")

    def _setup_io(self):
        if self.trt10:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        def find_tensor(keywords):
            for name in names:
                if any(kw in name.lower() for kw in keywords):
                    return name
            return None
        self.name_in = find_tensor(['raw_input', 'input'])
        self.name_dets = find_tensor(['det', 'boxes'])
        self.name_labels = find_tensor(['label', 'class'])
        self.name_masks = find_tensor(['mask', 'seg'])
        if any(x is None for x in [self.name_in, self.name_dets, self.name_labels, self.name_masks]):
            raise RuntimeError(f"Could not find required tensors in: {names}")
        if self.trt10:
            self.dtype_in = trt.nptype(self.engine.get_tensor_dtype(self.name_in))
            self.dtype_dets = trt.nptype(self.engine.get_tensor_dtype(self.name_dets))
            self.dtype_labels = trt.nptype(self.engine.get_tensor_dtype(self.name_labels))
            self.dtype_masks = trt.nptype(self.engine.get_tensor_dtype(self.name_masks))
        else:
            def get_dtype(name):
                idx = self.engine.get_binding_index(name)
                return trt.nptype(self.engine.get_binding_dtype(idx))
            self.dtype_in = get_dtype(self.name_in)
            self.dtype_dets = get_dtype(self.name_dets)
            self.dtype_labels = get_dtype(self.name_labels)
            self.dtype_masks = get_dtype(self.name_masks)
        self._allocate_buffers()

    def _allocate_buffers(self):
        self.input_shape = (1, 640, 640, 3)
        self.max_det = 100
        self.mask_hw = (640, 640)
        self.dev_ptr, self.host_buf, self.pinned_flag = {}, {}, {}

        def allocate(name, shape, dtype, force_pinned=False):
            numel = int(np.prod(shape))
            nbytes = numel * np.dtype(dtype).itemsize
            use_pinned = force_pinned or (nbytes >= PINNED_THRESHOLD_BYTES)
            if use_pinned:
                buf = cuda.pagelocked_empty(numel, dtype)
            else:
                buf = np.empty(numel, dtype=dtype)
            self.host_buf[name] = buf
            self.pinned_flag[name] = use_pinned
            self.dev_ptr[name] = cuda.mem_alloc(nbytes)

        allocate(self.name_in, self.input_shape, self.dtype_in)
        allocate(self.name_dets, (1, self.max_det, 5), self.dtype_dets, force_pinned=True)
        allocate(self.name_labels, (1, self.max_det), self.dtype_labels, force_pinned=True)

        # Masks: device-only (copy on demand)
        mask_shape = (1, self.max_det, *self.mask_hw)
        mask_bytes = int(np.prod(mask_shape)) * np.dtype(self.dtype_masks).itemsize
        self.dev_ptr[self.name_masks] = cuda.mem_alloc(mask_bytes)

        if self.trt10:
            for name, ptr in self.dev_ptr.items():
                self.context.set_tensor_address(name, int(ptr))
        else:
            self.bindings = [None] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.bindings[i] = int(self.dev_ptr[name])

    def _warmup_cv2(self):
        dummy = np.zeros((100, 100), dtype=np.uint8)
        _ = cv2.resize(dummy, (200, 200), interpolation=cv2.INTER_NEAREST)

    def _ensure_u8_capacity(self, needed_masks):
        Hm, Wm = self.mask_hw
        if needed_masks <= self.u8_capacity:
            return
        self.u8_capacity = max(needed_masks * 2, 32)
        bytes_u8 = self.u8_capacity * Hm * Wm  # 1 byte per pixel
        if self.dev_masks_u8 is not None:
            self.dev_masks_u8.free()
        self.dev_masks_u8 = cuda.mem_alloc(bytes_u8)
        self.mask_host_buf_u8 = cuda.pagelocked_empty(
            (self.u8_capacity, Hm, Wm), dtype=np.uint8
        )

    def _ensure_roi_bytes(self, needed_bytes):
        if needed_bytes <= self.roi_bytes_capacity:
            return
        self.roi_bytes_capacity = max(needed_bytes * 2, 1 << 20)  # min 1MB
        if self.dev_rois_u8 is not None:
            self.dev_rois_u8.free()
        self.dev_rois_u8 = cuda.mem_alloc(self.roi_bytes_capacity)
        self.host_rois_u8 = cuda.pagelocked_empty(self.roi_bytes_capacity, dtype=np.uint8)

    # ---------- Inference I/O helpers with timing ----------
    def copy_input_h2d(self):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(self.dev_ptr[self.name_in], self.host_buf[self.name_in], self.stream)
        self.stream.synchronize()
        return time.perf_counter() - t0

    def run_inference(self):
        """
        Returns: (gpu_wall_s, gpu_pure_s)
        gpu_wall_s: wall-clock for context.execute (synchronized)
        gpu_pure_s: CUDA event elapsed time (seconds)
        """
        t0 = time.perf_counter()
        self.start_evt.record(self.stream)
        if self.trt10:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.end_evt.record(self.stream)
        self.stream.synchronize()
        wall = time.perf_counter() - t0
        pure_ms = self.start_evt.time_till(self.end_evt)
        return wall, pure_ms / 1e3

    def copy_dets_labels_d2h(self):
        t0 = time.perf_counter()
        cuda.memcpy_dtoh_async(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets], self.stream)
        cuda.memcpy_dtoh_async(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels], self.stream)
        self.stream.synchronize()
        return time.perf_counter() - t0

    # ---------- Legacy mask copy (full or packed) with timing ----------
    def copy_masks_optimized(self, indices, thr=0.5):
        """
        Legacy copy:
        - If engine outputs uint8: fast D2H copy. Returns timings dict.
        - If engine outputs float32: GPU thresh -> u8 (packed), then D2H. Returns timings dict.
        Returns (masks, timings) where timings has keys:
          - d2h_s
          - gpu_thresh_s (0 for uint8 engines)
          - total_s
        """
        timings = {"d2h_s": 0.0, "gpu_thresh_s": 0.0, "total_s": 0.0}
        if len(indices) == 0:
            return np.empty((0, *self.mask_hw), dtype=self.dtype_masks), timings

        t_total0 = time.perf_counter()
        Hm, Wm = self.mask_hw
        n_masks = len(indices)
        n_pix = Hm * Wm
        itemsize = np.dtype(self.dtype_masks).itemsize
        slice_bytes = Hm * Wm * itemsize
        base_addr = int(self.dev_ptr[self.name_masks])

        if self.dtype_masks == np.uint8:
            if self.mask_host_buf is None or self.mask_host_capacity < n_masks:
                self.mask_host_capacity = max(n_masks * 2, 32)
                self.mask_host_buf = cuda.pagelocked_empty(
                    (self.mask_host_capacity, Hm, Wm), dtype=self.dtype_masks
                )
            t0 = time.perf_counter()
            if n_masks > 1 and np.all(np.diff(indices) == 1):
                src = int(base_addr + int(indices[0]) * slice_bytes)
                cuda.memcpy_dtoh_async(self.mask_host_buf[:n_masks].ravel(), src, stream=self.stream)
            else:
                for k, idx in enumerate(indices):
                    src = int(base_addr + int(idx) * slice_bytes)
                    dst = self.mask_host_buf[k].ravel()
                    cuda.memcpy_dtoh_async(dst, src, stream=self.stream)
            self.stream.synchronize()
            timings["d2h_s"] = time.perf_counter() - t0
            timings["total_s"] = time.perf_counter() - t_total0
            return self.mask_host_buf[:n_masks].copy(), timings

        # float32 masks: GPU thresh -> u8 (packed), then D2H
        self._ensure_u8_capacity(n_masks)
        threads = 256
        t_gpu0 = time.perf_counter()
        if n_masks > 1 and np.all(np.diff(indices) == 1):
            src_offset = int(indices[0]) * n_pix
            dst_offset = 0
            n_total = n_masks * n_pix
            grid = ((n_total + threads - 1) // threads, 1, 1)
            self._k_thresh(
                self.dev_ptr[self.name_masks],
                self.dev_masks_u8,
                np.int32(n_total),
                np.int32(src_offset),
                np.int32(dst_offset),
                np.float32(thr),
                block=(threads, 1, 1), grid=grid, stream=self.stream
            )
        else:
            for k, idx in enumerate(indices):
                src_offset = int(idx) * n_pix
                dst_offset = k * n_pix
                grid = ((n_pix + threads - 1) // threads, 1, 1)
                self._k_thresh(
                    self.dev_ptr[self.name_masks],
                    self.dev_masks_u8,
                    np.int32(n_pix),
                    np.int32(src_offset),
                    np.int32(dst_offset),
                    np.float32(thr),
                    block=(threads, 1, 1), grid=grid, stream=self.stream
                )
        self.stream.synchronize()
        timings["gpu_thresh_s"] = time.perf_counter() - t_gpu0

        t_d2h0 = time.perf_counter()
        cuda.memcpy_dtoh_async(self.mask_host_buf_u8[:n_masks].ravel(), int(self.dev_masks_u8), stream=self.stream)
        self.stream.synchronize()
        timings["d2h_s"] = time.perf_counter() - t_d2h0

        timings["total_s"] = time.perf_counter() - t_total0
        return self.mask_host_buf_u8[:n_masks].copy(), timings

    # ---------- Fast path: ROI kernel + packed D2H + paste (all timed) ----------
    def copy_and_resize_masks_roi_gpu(self, indices, boxes_img, scale, left, top, out_w, out_h,
                                      thr=0.5):
        """
        Float32 masks fast path:
          - Prep ROIs (CPU)
          - Launch per-ROI kernel (threshold + ROI resize)
          - Single packed D2H copy
          - CPU paste of ROIs into full-size masks
        Returns (masks_full, timings) with keys:
          prep_cpu_s, kernel_s, d2h_s, paste_cpu_s, total_s
        """
        timings = {"prep_cpu_s": 0.0, "kernel_s": 0.0, "d2h_s": 0.0, "paste_cpu_s": 0.0, "total_s": 0.0}
        if len(indices) == 0:
            return np.empty((0, out_h, out_w), dtype=np.uint8), timings
        if self.dtype_masks != np.float32:
            raise RuntimeError("GPU ROI path expects float32 mask output from engine.")

        t_total0 = time.perf_counter()
        Hm, Wm = self.mask_hw
        n_pix = Hm * Wm
        N = len(indices)
        boxes_img = boxes_img.astype(np.float32, copy=False)

        # Prep (CPU): compute letterbox ROIs and destination ROIs
        t_p0 = time.perf_counter()
        sx = np.clip(left + boxes_img[:,0] * scale, 0, Wm).astype(np.int32)
        sy = np.clip(top  + boxes_img[:,1] * scale, 0, Hm).astype(np.int32)
        ex = np.clip(left + boxes_img[:,2] * scale, 0, Wm).astype(np.int32)
        ey = np.clip(top  + boxes_img[:,3] * scale, 0, Hm).astype(np.int32)
        sw = np.maximum(ex - sx, 0).astype(np.int32)
        sh = np.maximum(ey - sy, 0).astype(np.int32)

        dx = np.clip(boxes_img[:,0], 0, out_w).astype(np.int32)
        dy = np.clip(boxes_img[:,1], 0, out_h).astype(np.int32)
        ex2 = np.clip(boxes_img[:,2], 0, out_w).astype(np.int32)
        ey2 = np.clip(boxes_img[:,3], 0, out_h).astype(np.int32)
        dw = np.maximum(ex2 - dx, 0).astype(np.int32)
        dh = np.maximum(ey2 - dy, 0).astype(np.int32)

        sizes = (dw * dh).astype(np.int64)
        valid = (sw > 0) & (sh > 0) & (dw > 0) & (dh > 0)
        sizes[~valid] = 0
        offsets = np.zeros(N, dtype=np.int64)
        if N > 0:
            np.cumsum(sizes[:-1], out=offsets[1:])
        total_elems = int(offsets[-1] + sizes[-1]) if N > 0 else 0  # uint8 elems
        timings["prep_cpu_s"] = time.perf_counter() - t_p0

        # Ensure staging buffers
        self._ensure_roi_bytes(total_elems if total_elems > 0 else 1)

        # Kernel launches (GPU)
        t_k0 = time.perf_counter()
        masks_full = np.zeros((N, out_h, out_w), dtype=np.uint8)
        block = (16, 16, 1)
        for i, idx in enumerate(indices):
            if sizes[i] == 0:
                continue
            src_offset = int(idx) * n_pix
            dst_offset = int(offsets[i])  # in elems (uint8)
            grid = ( (int(dw[i]) + block[0]-1)//block[0],
                     (int(dh[i]) + block[1]-1)//block[1], 1 )
            self._k_roi(
                self.dev_ptr[self.name_masks],
                np.int32(Hm), np.int32(Wm),
                np.int32(src_offset),
                np.int32(int(sx[i])), np.int32(int(sy[i])),
                np.int32(int(sw[i])), np.int32(int(sh[i])),
                self.dev_rois_u8,
                np.int32(dst_offset),
                np.int32(int(dw[i])), np.int32(int(dh[i])),
                np.float32(thr),
                block=block, grid=grid, stream=self.stream
            )
        self.stream.synchronize()
        timings["kernel_s"] = time.perf_counter() - t_k0

        # Packed D2H
        t_d0 = time.perf_counter()
        if total_elems > 0:
            cuda.memcpy_dtoh_async(self.host_rois_u8[:total_elems], int(self.dev_rois_u8), stream=self.stream)
        self.stream.synchronize()
        timings["d2h_s"] = time.perf_counter() - t_d0

        # Paste (CPU)
        t_paste0 = time.perf_counter()
        base = self.host_rois_u8  # uint8 1D buffer
        for i in range(N):
            if sizes[i] == 0:
                continue
            count = int(sizes[i])
            off = int(offsets[i])
            roi = np.frombuffer(base, dtype=np.uint8, count=count, offset=off).reshape(int(dh[i]), int(dw[i]))
            masks_full[i, dy[i]:dy[i]+dh[i], dx[i]:dx[i]+dw[i]] = roi
        timings["paste_cpu_s"] = time.perf_counter() - t_paste0

        timings["total_s"] = time.perf_counter() - t_total0
        return masks_full, timings

    # ---------- Core inference ----------
    def infer_optimized(self, img_uint8):
        """
        Runs the whole I/O+inference (no masks) with timing on:
          - H2D input
          - GPU inference (wall) + pure GPU (events)
          - D2H dets/labels
        Returns: (boxes, labels, scores, timing_dict)
        """
        # copy input into pinned host
        np.copyto(self.host_buf[self.name_in], img_uint8.ravel())

        t_h2d = self.copy_input_h2d()
        t_gpu_wall, t_gpu_pure = self.run_inference()
        t_d2h = self.copy_dets_labels_d2h()

        # Process outputs (host)
        dets_raw = self.host_buf[self.name_dets].reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].reshape(1, self.max_det)[0]
        np.copyto(self.output_boxes, dets_raw[:, :4])
        np.copyto(self.output_scores, dets_raw[:, 4])
        np.copyto(self.output_labels, labels_raw.astype(np.int32, copy=False))

        timing = {
            "h2d_input_s": t_h2d,
            "gpu_infer_wall_s": t_gpu_wall,
            "gpu_infer_pure_s": t_gpu_pure,
            "d2h_dets_labels_s": t_d2h
        }
        return self.output_boxes, self.output_labels, self.output_scores, timing

# ---------------- Main inference loop ----------------
def main():
    parser = argparse.ArgumentParser(description="Optimized RTMDet segmentation inference (full timing)")
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="seg_output.jpg", help="Output image path")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs")
    parser.add_argument("--score", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--mask-thr", type=float, default=0.4, help="Mask threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask blend alpha")
    parser.add_argument("--viz", choices=["on", "off"], default="on", help="Visualization")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--engine-binary-masks", action="store_true", default=True,
                        help="Engine outputs binary masks (skip CPU threshold)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for CPU ROI resize")
    parser.add_argument("--gpu-roi", action="store_true", default=True,
                        help="GPU threshold + ROI resize + packed D2H copy (fastest)")

    args = parser.parse_args()

    # Optional: reduce OpenCV thread jitter
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.input}")

    print(f"Image shape: {img.shape}")
    print(f"Score threshold: {args.score}")
    print(f"Mask threshold: {args.mask_thr}")
    print(f"Binary masks: {args.engine_binary_masks}")

    seg = OptimizedTRTSegmentor(args.engine, verbose=args.verbose)

    print(f"\nWarming up with {args.warmup} runs...")
    for _ in range(args.warmup):
        lb, scale, left, top = letterbox(img, size=args.size, pad_val=114)
        seg.infer_optimized(lb)
    print("Warmup complete.\n")

    # Timing accumulators
    pre_times = []
    h2d_input_times = []
    gpu_infer_wall_times = []
    gpu_infer_pure_times = []
    d2h_dets_times = []
    filter_times = []
    mask_bin_cpu_times = []     # <--- NEW: explicit CPU binarize row
    mask_prep_cpu_times = []
    mask_gpu_kernel_times = []
    d2h_mask_times = []
    mask_resize_cpu_times = []
    mask_paste_cpu_times = []
    viz_blend_times = []
    viz_draw_times = []
    total_times = []

    final_vis = None

    for run_idx in range(args.runs):
        t_start = time.perf_counter()

        # 1) Preprocess
        t0 = time.perf_counter()
        lb, scale, left, top = letterbox(img, size=args.size, pad_val=114)
        t1 = time.perf_counter()
        pre_times.append(t1 - t0)

        # 2) Inference (H2D input, GPU exec, D2H dets/labels)
        boxes, labels, scores, t_inf = seg.infer_optimized(lb)
        h2d_input_times.append(t_inf["h2d_input_s"])
        gpu_infer_wall_times.append(t_inf["gpu_infer_wall_s"])
        gpu_infer_pure_times.append(t_inf["gpu_infer_pure_s"])
        d2h_dets_times.append(t_inf["d2h_dets_labels_s"])

        # 3) Filter
        t_f0 = time.perf_counter()
        keep = scores >= args.score
        boxes_kept = boxes[keep]
        labels_kept = labels[keep]
        keep_idx = np.flatnonzero(keep)
        t_f1 = time.perf_counter()
        filter_times.append(t_f1 - t_f0)

        # 4) Unletterbox boxes (CPU)
        boxes_img = unletterbox_boxes(boxes_kept, scale, left, top, img.shape[1], img.shape[0])

        # 5) Masks
        if keep_idx.size > 0:
            if args.gpu_roi and seg.dtype_masks == np.float32:
                # GPU fast path: threshold + ROI resize on device, packed D2H, paste
                masks_img, t_mask = seg.copy_and_resize_masks_roi_gpu(
                    keep_idx, boxes_img, scale, left, top,
                    img.shape[1], img.shape[0],
                    thr=args.mask_thr
                )
                mask_bin_cpu_times.append(0.0)  # binarize is fused on GPU
                mask_prep_cpu_times.append(t_mask["prep_cpu_s"])
                mask_gpu_kernel_times.append(t_mask["kernel_s"])
                d2h_mask_times.append(t_mask["d2h_s"])
                mask_paste_cpu_times.append(t_mask["paste_cpu_s"])
                mask_resize_cpu_times.append(0.0)  # resize done on GPU
            else:
                # Legacy copy (full u8 or packed u8), then CPU ROI resize
                masks_fp, t_pack = seg.copy_masks_optimized(keep_idx, thr=args.mask_thr)
                # If engine emits float32, t_pack['gpu_thresh_s'] > 0; attribute to GPU kernels
                mask_gpu_kernel_times.append(t_pack.get("gpu_thresh_s", 0.0))
                d2h_mask_times.append(t_pack["d2h_s"])
                mask_paste_cpu_times.append(0.0)

                # Binarize (or cast) – CPU (explicit row)
                if args.engine_binary_masks:
                    t_b0 = time.perf_counter()
                    masks_u8 = masks_fp.astype(np.uint8, copy=False)
                    bin_time = time.perf_counter() - t_b0  # ~0, but measured
                else:
                    t_b0 = time.perf_counter()
                    masks_u8 = ((masks_fp > args.mask_thr) * 255).astype(np.uint8) if masks_fp.size > 0 else masks_fp.astype(np.uint8)
                    bin_time = time.perf_counter() - t_b0
                mask_bin_cpu_times.append(bin_time)

                # CPU ROI prep/resize (prep_time = bbox math; resize_time = cv2.resize)
                masks_img, prep_time_cpu, resize_time_cpu = unletterbox_masks_roi(
                    masks_u8, boxes_img, scale, left, top, img.shape[1], img.shape[0],
                    max_workers=args.workers
                )
                mask_prep_cpu_times.append(prep_time_cpu)
                mask_resize_cpu_times.append(resize_time_cpu)
        else:
            masks_img = np.empty((0, img.shape[0], img.shape[1]), dtype=np.uint8)
            mask_bin_cpu_times.append(0.0)
            mask_prep_cpu_times.append(0.0)
            mask_gpu_kernel_times.append(0.0)
            d2h_mask_times.append(0.0)
            mask_resize_cpu_times.append(0.0)
            mask_paste_cpu_times.append(0.0)

        # 6) Visualization
        if args.viz == "on":
            vis, blend_time, draw_time = colorize_blend_optimized(
                img.copy(), masks_img, boxes_img, labels_kept, alpha=args.alpha
            )
            viz_blend_times.append(blend_time)
            viz_draw_times.append(draw_time)
            if run_idx == args.runs - 1:
                final_vis = vis
        else:
            viz_blend_times.append(0.0)
            viz_draw_times.append(0.0)
            if run_idx == args.runs - 1:
                final_vis = img.copy()

        total_times.append(time.perf_counter() - t_start)

    # Stats helpers
    def stats(times):
        if not times:
            return 0.0, 0.0, 0.0
        return np.mean(times), np.min(times), np.max(times)

    # Compute stats
    pre_avg, pre_min, pre_max = stats(pre_times)
    h2d_avg, h2d_min, h2d_max = stats(h2d_input_times)
    gpuw_avg, gpuw_min, gpuw_max = stats(gpu_infer_wall_times)
    gpup_avg, gpup_min, gpup_max = stats(gpu_infer_pure_times)  # pure GPU (FYI)
    d2hl_avg, d2hl_min, d2hl_max = stats(d2h_dets_times)
    filt_avg, filt_min, filt_max = stats(filter_times)
    mbin_avg, mbin_min, mbin_max = stats(mask_bin_cpu_times)    # NEW row stats
    mp_prep_avg, mp_prep_min, mp_prep_max = stats(mask_prep_cpu_times)
    mgpu_avg, mgpu_min, mgpu_max = stats(mask_gpu_kernel_times)
    d2hm_avg, d2hm_min, d2hm_max = stats(d2h_mask_times)
    mres_avg, mres_min, mres_max = stats(mask_resize_cpu_times)
    mpaste_avg, mpaste_min, mpaste_max = stats(mask_paste_cpu_times)
    vbl_avg, vbl_min, vbl_max = stats(viz_blend_times)
    vdr_avg, vdr_min, vdr_max = stats(viz_draw_times)
    total_avg, total_min, total_max = stats(total_times)

    # Accounted vs total (sanity)
    accounted_per_run = []
    for i in range(len(total_times)):
        accounted = (
            pre_times[i]
            + h2d_input_times[i]
            + gpu_infer_wall_times[i]
            + d2h_dets_times[i]
            + filter_times[i]
            + mask_bin_cpu_times[i]     # include explicit binarize row
            + mask_prep_cpu_times[i]
            + mask_gpu_kernel_times[i]
            + d2h_mask_times[i]
            + mask_resize_cpu_times[i]
            + mask_paste_cpu_times[i]
            + viz_blend_times[i]
            + viz_draw_times[i]
        )
        accounted_per_run.append(accounted)
    acc_avg, acc_min, acc_max = stats(accounted_per_run)
    unacc = [total_times[i] - accounted_per_run[i] for i in range(len(total_times))]
    unacc_avg, unacc_min, unacc_max = stats(unacc)

    # Print results
    print("\n" + "="*64)
    print(" OPTIMIZED PERFORMANCE RESULTS (All rows sum to TOTAL ≈)")
    print("="*64)
    print(f"Runs: {len(total_times)}, Warmup excluded from stats")
    print(f"Image: {img.shape[1]}x{img.shape[0]}, Engine input: {args.size}x{args.size}")
    print("-"*64)
    print(f"{'Step':<32} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
    print("-"*64)
    print(f"{'Preprocessing':<32} {pre_avg*1e3:>12.2f} {pre_min*1e3:>12.2f} {pre_max*1e3:>12.2f}")
    print(f"{'H2D Input':<32} {h2d_avg*1e3:>12.2f} {h2d_min*1e3:>12.2f} {h2d_max*1e3:>12.2f}")
    print(f"{'GPU Inference (wall)':<32} {gpuw_avg*1e3:>12.2f} {gpuw_min*1e3:>12.2f} {gpuw_max*1e3:>12.2f}")
    print(f"{'D2H Dets/Labels':<32} {d2hl_avg*1e3:>12.2f} {d2hl_min*1e3:>12.2f} {d2hl_max*1e3:>12.2f}")
    print(f"{'Filter & Selection':<32} {filt_avg*1e3:>12.2f} {filt_min*1e3:>12.2f} {filt_max*1e3:>12.2f}")
    print(f"{'Mask Binarize (CPU)':<32} {mbin_avg*1e3:>12.2f} {mbin_min*1e3:>12.2f} {mbin_max*1e3:>12.2f}")
    print(f"{'Mask Prep (CPU)':<32} {mp_prep_avg*1e3:>12.2f} {mp_prep_min*1e3:>12.2f} {mp_prep_max*1e3:>12.2f}")
    print(f"{'Mask GPU Kernels (thresh+ROI)':<32} {mgpu_avg*1e3:>12.2f} {mgpu_min*1e3:>12.2f} {mgpu_max*1e3:>12.2f}")
    print(f"{'D2H Masks':<32} {d2hm_avg*1e3:>12.2f} {d2hm_min*1e3:>12.2f} {d2hm_max*1e3:>12.2f}")
    print(f"{'Mask ROI Resize (CPU)':<32} {mres_avg*1e3:>12.2f} {mres_min*1e3:>12.2f} {mres_max*1e3:>12.2f}")
    print(f"{'Mask Paste (CPU)':<32} {mpaste_avg*1e3:>12.2f} {mpaste_min*1e3:>12.2f} {mpaste_max*1e3:>12.2f}")
    if args.viz == "on":
        print(f"{'Viz Blend':<32} {vbl_avg*1e3:>12.2f} {vbl_min*1e3:>12.2f} {vbl_max*1e3:>12.2f}")
        print(f"{'Viz Draw':<32} {vdr_avg*1e3:>12.2f} {vdr_min*1e3:>12.2f} {vdr_max*1e3:>12.2f}")
    print("-"*64)
    print(f"{'TOTAL':<32} {total_avg*1e3:>12.2f} {total_min*1e3:>12.2f} {total_max*1e3:>12.2f}")
    print(f"{'Accounted (sum of rows)':<32} {acc_avg*1e3:>12.2f} {acc_min*1e3:>12.2f} {acc_max*1e3:>12.2f}")
    print(f"{'Unaccounted (jitter)':<32} {unacc_avg*1e3:>12.2f} {unacc_min*1e3:>12.2f} {unacc_max*1e3:>12.2f}")
    print("="*64)
    print(f"Pure GPU Inference (events) — Avg: {gpup_avg*1e3:.2f} ms, Min: {gpup_min*1e3:.2f} ms, Max: {gpup_max*1e3:.2f} ms")

    fps = 1.0 / total_avg if total_avg > 0 else 0.0
    print(f"\n🚀 Estimated FPS (TOTAL): {fps:.2f}")

    if final_vis is not None and args.viz == "on":
        cv2.imwrite(args.output, final_vis)
        print(f"\n✅ Output saved: {args.output}")

if __name__ == "__main__":
    main()
