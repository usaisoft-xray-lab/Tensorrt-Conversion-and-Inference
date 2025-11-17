# TensorRT Inference Fastest (TRT Infer Fastest)

## 🚀 Project Overview

TRT Infer Fastest is a high-performance machine learning inference project utilizing NVIDIA TensorRT for accelerated object detection and instance segmentation. This project provides optimized inference pipelines for converting ONNX models to TensorRT engines and performing fast, efficient inference on GPU.

## 🌟 Key Features

- 🚄 Ultra-fast GPU inference using TensorRT
- 🔍 Support for object detection and instance segmentation
- 📊 Performance benchmarking and timing measurements
- 🔧 Flexible preprocessing and engine building
- 🖼️ Visualization of inference results


 ## 📊 Benchmark

All benchmarks were conducted on an **NVIDIA RTX 4090 GPU** using the provided conversion and inference scripts.  
The results below are measured **end-to-end latency**, including preprocessing, inference, and post-processing.  

| Model Type         | Scenario                  | Latency (ms) | FPS    |
|--------------------|--------------------------|--------------|--------|
| Segmentation (Medium Model) | 2–3 masks detected       | 7.22 ms      | 138.48 |
| Segmentation (Medium Model) | >10 masks detected       | 12.67 ms     | 78.92  |    



## 📋 Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- TensorRT 8.6 (install from nvidia then run the pip wheel for the correct version)
- OpenCV
- NumPy
- CUDA Toolkit 11.8

## 🛠️ Installation


1. Clone the repository:
```bash
git clone https://github.com/your-username/trt_infer_fastest.git
cd trt_infer_fastest
```

2. Install dependencies:
```bash
pip install -r requirements.txt   
```   
use mmdeploy2 conda env on the workstation

## 🔬 Project Structure

```
trt_infer_fastest/
├── experiments/           # Output images and experiment results
├── models/                # Pre-trained models and TensorRT engines
├── newonnx/               # Additional ONNX models
├── src/                   # Source code
│   ├── deploy_segmentation.py    # Segmentation model deployment
│   ├── deploydetection.py        # Detection model deployment
│   ├── segmentation inference.py # Segmentation inference script
│   ├── trtdetection.py           # Detection inference script
│   └── tests/             # Test scripts
└── testimages/            # Sample input images
```

## 🚦 Usage

### Object Detection

Run object detection inference:
```bash
python src/trtdetection.py --engine models/small_model_det/model.engine --input testimages/demo.jpg --output output.jpg
```

Parameters:
- `--engine`: Path to TensorRT engine file
- `--input`: Input image path
- `--output`: Output visualization path
- `--runs`: Number of inference runs (default: 10)
- `--score`: Confidence threshold (default: 0.4)

### Instance Segmentation

Run segmentation inference:
```bash
python src/segmentation\ inference.py --engine models/small_model_seg/small-ins.engine --input testimages/apple.jpg --output seg_output.jpg
```

Parameters:
- `--engine`: Path to TensorRT segmentation engine
- `--input`: Input image path
- `--output`: Output visualization path
- `--runs`: Number of inference runs (default: 10)
- `--score`: Confidence threshold (default: 0.1)
- `--size`: Input image resize (default: 640)
- `--viz`: Visualization mode (roi/full/off, default: roi)

### Engine Building

Convert ONNX to TensorRT engine:

For Detection:
```bash
python src/deploydetection.py path/to/model.onnx path/to/output.engine
```

For Segmentation:
```bash
python src/deploy_segmentation.py path/to/model.onnx path/to/output.engine
```

## 📈 Performance Metrics

The scripts automatically measure and report:
- Pre-processing time
- Inference time
- Post-processing time
- Total pipeline performance

## 🔍 Troubleshooting

- Ensure CUDA and TensorRT are correctly installed
- Check GPU compatibility
- Verify model and engine file paths
- Confirm input image formats

---

**Note**: Performance and results may vary based on hardware and specific model configurations.

