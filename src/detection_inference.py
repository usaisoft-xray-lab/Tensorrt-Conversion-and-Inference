#!/usr/bin/env python3
# trt_image_infer_timed.py
"""
Timed single‑image inference with a TensorRT RTMDet engine.

Usage:
  python trt_image_infer_timed.py --engine large.engine --input test.jpg \
                                  --output out.jpg --runs 10 --score 0.4
"""

import argparse, time, cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit   # 1 CUDA context for whole script

# -----------------------------------------------------------------------------#
# Utility: letter‑box with padding value 114 (YOLO / RTMDet convention)
# -----------------------------------------------------------------------------#
def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top            # BGR uint8, scale & pad

# -----------------------------------------------------------------------------#
# TensorRT wrapper (extracts pure GPU time with CUDA events)
# -----------------------------------------------------------------------------#
class TRTDetector:
    def __init__(self, engine_path, max_det=1000):
        logger  = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Binding names taken from original script
        self.iname, self.on_dets, self.on_labels = "raw_input", "dets", "labels"
        self.hbuf, self.dbuf = {}, {}

        # Allocate static buffers once (raw_input (1,640,640,3) uint8)
        self._alloc(self.iname , (1,640,640,3), np.uint8)
        self._alloc(self.on_dets , (1,max_det,5) , np.float32)
        self._alloc(self.on_labels, (1,max_det)   , np.int32)

        # Cache bindings array for execute_async_v2
        self.bindings = [int(self.dbuf[self.engine.get_binding_name(i)])
                         for i in range(self.engine.num_bindings)]

    def _alloc(self, name, shape, dtype):
        host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
        dev  = cuda.mem_alloc(host.nbytes)
        self.hbuf[name], self.dbuf[name] = host, dev

    def infer(self, img_arr):
        """Returns (boxes_xyxy, labels, scores, gpu_time_seconds)."""
        stream = cuda.Stream()

        # ---------------- host→device copy ----------------
        np.copyto(self.hbuf[self.iname], img_arr.ravel())
        cuda.memcpy_htod_async(self.dbuf[self.iname], self.hbuf[self.iname], stream)

        # ---------------- inference (timed with CUDA events) ----------------
        start_event, end_event = cuda.Event(), cuda.Event()
        start_event.record(stream)
        self.context.execute_async_v2(self.bindings, stream.handle)
        end_event.record(stream)

        # ---------------- device→host copy ----------------
        cuda.memcpy_dtoh_async(self.hbuf[self.on_dets]  , self.dbuf[self.on_dets]  , stream)
        cuda.memcpy_dtoh_async(self.hbuf[self.on_labels], self.dbuf[self.on_labels], stream)
        stream.synchronize()                       # wait for everything

        gpu_ms = start_event.time_till(end_event)  # kernel + copies inside stream
        dets   = self.hbuf[self.on_dets ].reshape(1,-1,5)
        labels = self.hbuf[self.on_labels].reshape(1,-1)

        return dets[0,:,:4], labels[0], dets[0,:,4], gpu_ms / 1_000  # sec

# -----------------------------------------------------------------------------#
def draw_boxes(img, boxes, labels, scores, scale, left, top, colours=None):
    if colours is None:
        colours = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
    h, w = img.shape[:2]
    for (x1,y1,x2,y2), lab, conf in zip(boxes, labels, scores):
        if conf < 0:   # keep all, filtering already done
            continue
        # undo padding + scale
        x1 = int(max(0, (x1 - left) / scale)); y1 = int(max(0, (y1 - top) / scale))
        x2 = int(min(w-1, (x2 - left) / scale)); y2 = int(min(h-1, (y2 - top) / scale))
        colour = colours[int(lab) % len(colours)]
        cv2.rectangle(img, (x1,y1), (x2,y2), colour, 2)
        cv2.putText(img, f"{int(lab)}:{conf:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT .engine path")
    ap.add_argument("--input" , required=True, help="image file for testing")
    ap.add_argument("--output", default="output.jpg", help="visualised result")
    ap.add_argument("--runs", type=int, default=10, help="# inference passes")
    ap.add_argument("--score", type=float, default=0.4, help="confidence threshold")
    args = ap.parse_args()

    # -------- static objects --------
    raw_img = cv2.imread(args.input)
    if raw_img is None:
        raise FileNotFoundError(args.input)
    detector = TRTDetector(args.engine)

    # -------- accumulators --------
    pre_sum = inf_sum = post_sum = 0.0

    # -------- repeated inference --------
    for i in range(args.runs):
        # --- PRE‑PROCESS -----------------------------------------------------
        t0 = time.perf_counter()
        lb_img, scale, left, top = letterbox(raw_img)
        t1 = time.perf_counter()
        pre_sum += (t1 - t0)

        # --- GPU INFERENCE ---------------------------------------------------
        boxes, labels, scores, gpu_t = detector.infer(lb_img)
        inf_sum += gpu_t

        # keep detections above score threshold
        keep = scores >= args.score
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        # --- POST‑PROCESS (rescale + draw) ----------------------------------
        t2 = time.perf_counter()
        if i == args.runs - 1:           # draw only on final pass (saves time)
            vis_img = raw_img.copy()
            draw_boxes(vis_img, boxes, labels, scores,
                       scale, left, top)
        post_sum += (time.perf_counter() - t2)

    # -------- averages & reporting --------
    runs = float(args.runs)
    print(f"\nAverage times over {args.runs} runs")
    print(f"  Pre‑process : {pre_sum  / runs * 1e3:7.2f}  ms")
    print(f"  GPU infer   : {inf_sum  / runs * 1e3:7.2f}  ms")
    print(f"  Post‑process: {post_sum / runs * 1e3:7.2f}  ms")

    # save visualised frame from the last iteration
    cv2.imwrite(args.output, vis_img)
    print(f"\n✅  Result saved to {args.output}")

if __name__ == "__main__":
    main()