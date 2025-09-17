#!/usr/bin/env python3
"""
Real-time TensorRT RTMDet segmentation inference for video
Custom post-processing for baguette factory:
- Class 0 (baguette): filled GREEN masks (no boxes)
- Class 1 (tray): outline only (no fill, no boxes)
- ROI-only inference (center-based ROI)
- Only one COMPLETE tray inside ROI is considered; only baguettes inside that tray (>=50% overlap) are drawn.
- Preserves your optimized TRT I/O, GPU ROI mask path, FPS overlay, real-time simulation, etc.
"""

import argparse, time, cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from pycuda.compiler import SourceModule
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# =========================================
# Configuration constants
# =========================================
PINNED_THRESHOLD_BYTES = 16 * 1024 * 1024  # Use pinned memory for large buffers
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640
TARGET_FPS = 60.0
FRAME_TIME_TARGET = 1.0 / TARGET_FPS  # 16.67ms for 60 fps

# =========================================
# Utility & geometry helpers
# =========================================
def clamp_roi(cx, cy, w, h, img_w, img_h):
    """Return (x1,y1,x2,y2) ROI rect clamped to frame. (cx,cy) is center."""
    x1 = int(round(cx - w / 2)); y1 = int(round(cy - h / 2))
    x2 = x1 + int(w);            y2 = y1 + int(h)
    x1 = max(0, min(img_w - 1, x1)); y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2));     y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI after clamping.")
    return x1, y1, x2, y2

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

def add_fps_overlay(frame, fps, frame_count, inference_time_ms):
    """Add FPS and timing information overlay to frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Inference: {inference_time_ms:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status_color = (0, 255, 0) if fps >= TARGET_FPS * 0.9 else (0, 255, 255) if fps >= TARGET_FPS * 0.7 else (0, 0, 255)
    cv2.putText(frame, f"Target: {TARGET_FPS} FPS", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    return frame

# =========================================
# Custom selection & visualization
# =========================================
def tray_is_complete(mask_u8, margin_px=2):
    """True if the tray mask is fully inside ROI (doesn't touch any ROI border)."""
    if mask_u8.size == 0:
        return False
    h, w = mask_u8.shape
    if (mask_u8[:margin_px, :].any() or mask_u8[-margin_px:, :].any() or
        mask_u8[:, :margin_px].any() or mask_u8[:, -margin_px:].any()):
        return False
    return mask_u8.any()

def pick_complete_tray(tray_masks):
    """Return index of the largest-area 'complete' tray mask; -1 if none."""
    best_idx, best_area = -1, 0
    for i, m in enumerate(tray_masks):
        if tray_is_complete(m, margin_px=2):
            area = int((m > 0).sum())
            if area > best_area:
                best_idx, best_area = i, area
    return best_idx

def visualize_baguettes_and_tray(full_frame_bgr, roi_rect, baguette_masks_roi, tray_mask_roi, alpha=0.45, show_roi=False):
    """
    Draw tray contour (white) and filled green baguette masks (alpha blended) inside ROI on full_frame_bgr.
    """
    rx1, ry1, rx2, ry2 = roi_rect
    vis = full_frame_bgr

    # Tray outline only (white)
    if tray_mask_roi is not None and tray_mask_roi.any():
        contours, _ = cv2.findContours(tray_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt + np.array([[rx1, ry1]], dtype=np.int32)
            cv2.polylines(vis, [cnt], isClosed=True, color=(255, 255, 255), thickness=2)

    # Filled green baguette masks
    if len(baguette_masks_roi) > 0:
        roi_view = vis[ry1:ry2, rx1:rx2]
        base = roi_view.copy()
        green = np.zeros_like(roi_view)
        green[:, :, 1] = 255  # pure green
        # Build union mask for one copyTo
        union = np.zeros((roi_view.shape[0], roi_view.shape[1]), dtype=np.uint8)
        for m in baguette_masks_roi:
            if m.shape[:2] != union.shape:
                m = cv2.resize(m, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST)
            union = cv2.bitwise_or(union, (m > 0).astype(np.uint8))
        blended = cv2.addWeighted(base, 1.0 - alpha, green, alpha, 0.0)
        cv2.copyTo(blended, union, roi_view)

    if show_roi:
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

    return vis

# =========================================
# Optimized TRT Wrapper (unchanged)
# =========================================
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
        np.copyto(self.host_buf[self.name_in], img_uint8.ravel())
        t_h2d = self.copy_input_h2d()
        t_gpu_wall, t_gpu_pure = self.run_inference()
        t_d2h = self.copy_dets_labels_d2h()

        dets_raw = self.host_buf[self.name_dets].reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].reshape(1, self.max_det)[0]
        self.output_boxes[...] = dets_raw[:, :4]
        self.output_scores[...] = dets_raw[:, 4]
        self.output_labels[...] = labels_raw.astype(np.int32, copy=False)

        timing = {
            "h2d_input_s": t_h2d,
            "gpu_infer_wall_s": t_gpu_wall,
            "gpu_infer_pure_s": t_gpu_pure,
            "d2h_dets_labels_s": t_d2h
        }
        return self.output_boxes, self.output_labels, self.output_scores, timing

# =========================================
# Video processing (modified for ROI + custom viz)
# =========================================
def process_video_realtime(seg, video_path, output_path, args):
    """Process video with real-time 60fps simulation + custom post-processing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {original_fps:.1f} fps, {total_frames} frames")
    print(f"Simulating {TARGET_FPS} fps camera feed...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height))
    if not out.isOpened():
        raise ValueError(f"Cannot open output video: {output_path}")

    fps_queue = deque(maxlen=30)
    frame_count = 0
    start_time = time.perf_counter()
    last_display_time = start_time

    # Warmup on ROI to match real run
    print("Warming up model...")
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        rx1, ry1, rx2, ry2 = clamp_roi(args.roi_x, args.roi_y, args.roi_width, args.roi_height, width, height)
        roi = frame[ry1:ry2, rx1:rx2].copy()
        lb, scale, left, top = letterbox(roi, size=args.size, pad_val=114)
        seg.infer_optimized(lb)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Starting real-time processing...\n")

    try:
        while True:
            frame_start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            frame_count += 1

            # === ROI crop (inference ONLY inside ROI) ===
            rx1, ry1, rx2, ry2 = clamp_roi(args.roi_x, args.roi_y, args.roi_width, args.roi_height, width, height)
            roi = frame[ry1:ry2, rx1:rx2].copy()
            roi_h, roi_w = roi.shape[:2]

            # 1) Preprocess on ROI
            lb, scale, left, top = letterbox(roi, size=args.size, pad_val=114)

            # 2) Inference on ROI
            inference_start = time.perf_counter()
            boxes, labels, scores, t_inf = seg.infer_optimized(lb)
            inference_time = time.perf_counter() - inference_start

            # 3) Filter detections
            keep = scores >= args.score
            boxes_kept = boxes[keep]
            labels_kept = labels[keep]
            keep_idx = np.flatnonzero(keep)

            # 4) Map boxes back to ROI coords
            boxes_roi = unletterbox_boxes(boxes_kept, scale, left, top, roi_w, roi_h)

            # 5) Masks in ROI coordinates
            if keep_idx.size > 0:
                if args.gpu_roi and seg.dtype_masks == np.float32:
                    masks_roi, _ = seg.copy_and_resize_masks_roi_gpu(
                        keep_idx, boxes_roi, scale, left, top, roi_w, roi_h, thr=args.mask_thr
                    )
                else:
                    masks_fp, _ = seg.copy_masks_optimized(keep_idx, thr=args.mask_thr)
                    if args.engine_binary_masks:
                        masks_u8 = masks_fp.astype(np.uint8, copy=False)
                    else:
                        masks_u8 = ((masks_fp > args.mask_thr) * 255).astype(np.uint8) if masks_fp.size > 0 else masks_fp.astype(np.uint8)
                    masks_roi, _, _ = unletterbox_masks_roi(
                        masks_u8, boxes_roi, scale, left, top, roi_w, roi_h, max_workers=args.workers
                    )
            else:
                masks_roi = np.empty((0, roi_h, roi_w), dtype=np.uint8)

            # 6) Split by class
            baguette_masks = []
            tray_masks = []
            for m, lab in zip(masks_roi, labels_kept):
                if lab == 0:       # baguette
                    baguette_masks.append(m)
                elif lab == 1:     # tray
                    tray_masks.append(m)

            # 7) Choose one COMPLETE tray inside ROI
            chosen_tray_idx = pick_complete_tray(tray_masks)
            tray_mask = tray_masks[chosen_tray_idx] if chosen_tray_idx != -1 else None

            # 8) Keep only baguettes that significantly overlap chosen tray (>=50%)
            filtered_baguettes = []
            if tray_mask is not None:
                tray_bin = (tray_mask > 0).astype(np.uint8)
                for bm in baguette_masks:
                    if bm is None or bm.size == 0:
                        continue
                    b_bin = (bm > 0).astype(np.uint8)
                    inter = cv2.bitwise_and(b_bin, tray_bin)
                    b_area = int(b_bin.sum())
                    if b_area > 0 and int(inter.sum()) >= 0.5 * b_area:
                        filtered_baguettes.append((b_bin * 255).astype(np.uint8))

            # 9) Custom Visualization (on full frame)
            vis = frame.copy()
            vis = visualize_baguettes_and_tray(
                vis, (rx1, ry1, rx2, ry2),
                filtered_baguettes,
                tray_mask if tray_mask is not None else None,
                alpha=args.alpha,
                show_roi=args.show_roi_box
            )

            # FPS calc & overlay
            frame_end_time = time.perf_counter()
            frame_time = frame_end_time - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_queue.append(current_fps)
            avg_fps = np.mean(fps_queue)
            vis = add_fps_overlay(vis, avg_fps, frame_count, inference_time * 1000)

            # Write & optionally display
            out.write(vis)
            if args.display:
                cv2.imshow('RTMDet TRT (Custom Postproc)', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Real-time simulation
            if not args.no_sleep:
                target_frame_time = 1.0 / TARGET_FPS
                actual_frame_time = time.perf_counter() - frame_start_time
                if actual_frame_time < target_frame_time:
                    time.sleep(target_frame_time - actual_frame_time)

            # Periodic log
            current_time = time.perf_counter()
            if current_time - last_display_time >= 2.0:
                elapsed = current_time - start_time
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                eta = (elapsed / frame_count * total_frames - elapsed) if frame_count > 0 and total_frames > 0 else 0
                print(f"Frame {frame_count:5d} | FPS: {avg_fps:5.1f} | "
                      f"Inference: {inference_time*1000:5.1f}ms | "
                      f"Progress: {progress:5.1f}% | ETA: {eta:5.1f}s", end='\r')
                last_display_time = current_time

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        cap.release()
        out.release()
        if args.display:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    final_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n\nProcessing complete!")
    print(f"Processed {frame_count} frames in {total_time:.1f} seconds")
    print(f"Average FPS: {final_fps:.1f}")
    print(f"Output saved: {output_path}")

# =========================================
# Main / CLI
# =========================================
def main():
    parser = argparse.ArgumentParser(description="Real-time RTMDet segmentation inference for video (custom postproc)")
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="realtime_output.mp4", help="Output video path")
    parser.add_argument("--score", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--mask-thr", type=float, default=0.4, help="Mask threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--alpha", type=float, default=0.45, help="Baguette mask blend alpha")

    # Engine / perf flags (kept from your original)
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--engine-binary-masks", action="store_true", default=True,
                        help="Engine outputs binary masks (skip CPU threshold)")
    parser.add_argument("--workers", type=int, default=4, help="Workers for CPU ROI resize")
    parser.add_argument("--gpu-roi", action="store_true", default=True,
                        help="GPU threshold + ROI resize + packed D2H copy (fastest)")

    # Display / control
    parser.add_argument("--display", action="store_true", default=True,
                        help="Display real-time video window")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="Disable real-time display window")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop video when it ends")
    parser.add_argument("--no-sleep", action="store_true", default=False,
                        help="Disable frame rate limiting (run as fast as possible)")

    # NEW: ROI (center-based)
    parser.add_argument("--roi-x", type=float, required=True, help="ROI center x (pixels)")
    parser.add_argument("--roi-y", type=float, required=True, help="ROI center y (pixels)")
    parser.add_argument("--roi-width", type=int, required=True, help="ROI width (pixels)")
    parser.add_argument("--roi-height", type=int, required=True, help="ROI height (pixels)")
    parser.add_argument("--show-roi-box", action="store_true", help="Draw ROI rectangle (debug)")

    args = parser.parse_args()

    # Optional: reduce OpenCV thread jitter
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    print(f"🚀 RTMDet Real-time Video Segmentation (Custom Postproc)")
    print(f"{'='*50}")
    print(f"🎯 Target FPS: {TARGET_FPS}")
    print(f"📊 Score threshold: {args.score}")
    print(f"🎭 Mask threshold: {args.mask_thr}")
    print(f"⚡ GPU ROI optimization: {args.gpu_roi}")
    print(f"🖥️  Real-time display: {'✅ Enabled' if args.display else '❌ Disabled'}")
    print(f"🟨 ROI: center=({args.roi_x},{args.roi_y}), size=({args.roi_width}x{args.roi_height})")
    print(f"{'='*50}")

    seg = OptimizedTRTSegmentor(args.engine, verbose=args.verbose)
    process_video_realtime(seg, args.input, args.output, args)

if __name__ == "__main__":
    main()
