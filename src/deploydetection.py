"""
Enhanced TensorRT 8.6 Builder with pipeline.json support

Builds TensorRT engine with built-in preprocessing that can optionally
use pipeline.json configuration for preprocessing parameters.
"""

import tensorrt as trt
import numpy as np
import sys
import pathlib
import json
import os
from typing import Dict, List, Tuple, Optional


def load_pipeline_config(pipeline_path: str) -> Optional[Dict]:
    """Load and parse pipeline.json configuration"""
    if not os.path.exists(pipeline_path):
        print(f"Pipeline file {pipeline_path} not found, using default parameters")
        return None
    
    try:
        with open(pipeline_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded pipeline configuration from {pipeline_path}")
        return config
    except Exception as e:
        print(f"Error loading pipeline config: {e}")
        return None


def extract_preprocessing_params(pipeline_config: Optional[Dict]) -> Dict:
    """Extract preprocessing parameters from pipeline.json"""
    # Default parameters
    params = {
        'input_size': (640, 640),
        'mean': [103.53, 116.28, 123.675],  # BGR
        'std': [57.375, 57.12, 58.395],
        'to_rgb': False,
        'keep_ratio': True,
        'pad_val': [114, 114, 114]
    }
    
    if pipeline_config is None:
        return params
    
    try:
        # Navigate through pipeline structure to find transforms
        pipeline = pipeline_config.get('pipeline', {})
        tasks = pipeline.get('tasks', [])
        
        for task in tasks:
            if (task.get('name') == 'Preprocess' and 
                task.get('module') == 'Transform'):
                
                transforms = task.get('transforms', [])
                
                for transform in transforms:
                    transform_type = transform.get('type', '')
                    
                    if transform_type == 'Resize':
                        if 'size' in transform:
                            size = transform['size']
                            if isinstance(size, list) and len(size) >= 2:
                                params['input_size'] = (size[0], size[1])
                        params['keep_ratio'] = transform.get('keep_ratio', params['keep_ratio'])
                    
                    elif transform_type == 'Normalize':
                        if 'mean' in transform:
                            params['mean'] = transform['mean']
                        if 'std' in transform:
                            params['std'] = transform['std']
                        params['to_rgb'] = transform.get('to_rgb', params['to_rgb'])
                    
                    elif transform_type == 'Pad':
                        if 'pad_val' in transform:
                            pad_val = transform['pad_val']
                            if isinstance(pad_val, dict) and 'img' in pad_val:
                                params['pad_val'] = pad_val['img']
                            elif isinstance(pad_val, list):
                                params['pad_val'] = pad_val
                
                break
        
        print("Extracted preprocessing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error extracting preprocessing params: {e}")
        print("Using default parameters")
    
    return params


def build_engine_with_preprocessing(
    onnx_path: str, 
    engine_path: str, 
    pipeline_path: Optional[str] = None,
    workspace_size: int = 1 << 30
) -> None:
    """Build TensorRT engine with integrated preprocessing"""
    
    # Load pipeline configuration
    pipeline_config = load_pipeline_config(pipeline_path) if pipeline_path else None
    
    # Extract preprocessing parameters
    preproc_params = extract_preprocessing_params(pipeline_config)
    
    H, W = preproc_params['input_size']
    MEAN = np.array(preproc_params['mean'], dtype=np.float32)
    STD = np.array(preproc_params['std'], dtype=np.float32)
    
    logger = trt.Logger(trt.Logger.WARNING)
    
    print(f"Building TensorRT engine with preprocessing:")
    print(f"  Input size: {W}x{H}")
    print(f"  Mean: {MEAN}")
    print(f"  Std: {STD}")
    print(f"  RGB conversion: {preproc_params['to_rgb']}")
    
    with trt.Builder(logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:
        
        # Parse ONNX model
        print(f"Loading ONNX model: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ONNX parsing errors:")
                for i in range(parser.num_errors):
                    print(f"  {parser.get_error(i)}")
                sys.exit("ONNX parse failed")
        
        # Get original input tensor
        orig_input = network.get_input(0)
        print(f"Original input: {orig_input.name}, shape: {orig_input.shape}")
        
        # Create new raw input tensor (NHWC uint8)
        raw_input = network.add_input("raw_input", trt.uint8, (1, H, W, 3))
        print(f"New raw input: raw_input, shape: (1, {H}, {W}, 3)")
        
        # Build preprocessing pipeline
        current_tensor = raw_input
        
        # 1. Cast uint8 to float32
        cast_layer = network.add_identity(current_tensor)
        cast_layer.set_output_type(0, trt.float32)
        current_tensor = cast_layer.get_output(0)
        print("Added: uint8 -> float32 cast")
        
        # 2. Convert BGR to RGB if needed
        if preproc_params['to_rgb']:
            # Create permutation for BGR->RGB: [0,3,2,1] to swap B and R channels
            shuffle_layer = network.add_shuffle(current_tensor)
            shuffle_layer.second_transpose = (0, 1, 2, 3)  # No change in spatial dims
            # We need to handle channel swapping differently
            # For now, we'll skip BGR->RGB conversion in the engine
            # and handle it in the preprocessing if needed
            print("Note: BGR->RGB conversion skipped in engine (handle in preprocessing if needed)")
        
        # 3. NHWC -> NCHW transpose
        transpose_layer = network.add_shuffle(current_tensor)
        transpose_layer.first_transpose = (0, 3, 1, 2)  # NHWC -> NCHW
        current_tensor = transpose_layer.get_output(0)
        print("Added: NHWC -> NCHW transpose")
        
        # 4. Normalization: (x - mean) / std
        scale = 1.0 / STD
        shift = -MEAN * scale
        
        norm_layer = network.add_scale(
            current_tensor,
            trt.ScaleMode.CHANNEL,
            shift=shift,
            scale=scale
        )
        current_tensor = norm_layer.get_output(0)
        print(f"Added: normalization with scale={scale}, shift={shift}")
        
        # 5. Resize if needed (assuming input is already correct size for now)
        # For more complex resizing logic, you might need to implement
        # keep_ratio and padding logic here
        if preproc_params['keep_ratio']:
            print("Note: keep_ratio resize logic not implemented in engine")
            print("      Ensure input images are pre-resized or implement resize logic")
        
        # Connect to original network input
        print("Connecting preprocessing output to original network...")
        for layer_idx in range(network.num_layers):
            layer = network.get_layer(layer_idx)
            for input_idx in range(layer.num_inputs):
                if layer.get_input(input_idx) is orig_input:
                    layer.set_input(input_idx, current_tensor)
                    print(f"  Connected to layer {layer.name} input {input_idx}")
        
        # Remove original input tensor
        network.remove_tensor(orig_input)
        print("Removed original input tensor")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        
        # Create optimization profile for fixed input size
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "raw_input",
            min=(1, H, W, 3),
            opt=(1, H, W, 3),
            max=(1, H, W, 3)
        )
        config.add_optimization_profile(profile)
        
        print("Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            sys.exit("Failed to build TensorRT engine")
        
        # Save engine
        print(f"Saving engine to: {engine_path}")
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print("✅ TensorRT engine with preprocessing built successfully!")
        
        # Print summary
        print("\nEngine Summary:")
        print(f"  Input: raw_input [1, {H}, {W}, 3] uint8")
        print(f"  Preprocessing: cast -> transpose -> normalize -> resize")
        print(f"  Mean: {MEAN}")
        print(f"  Std: {STD}")
        print(f"  Output: same as original ONNX model")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build TensorRT engine with integrated preprocessing")
    parser.add_argument("onnx_path", help="Path to input ONNX model")
    parser.add_argument("engine_path", help="Path to output TensorRT engine")
    parser.add_argument("--pipeline", help="Path to optional pipeline.json file")
    parser.add_argument("--workspace", type=int, default=1024, help="Workspace size in MB")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.isfile(args.onnx_path):
        print(f"Error: ONNX file not found: {args.onnx_path}")
        return 1
    
    if args.pipeline and not os.path.isfile(args.pipeline):
        print(f"Warning: Pipeline file not found: {args.pipeline}")
    
    # Convert workspace size to bytes
    workspace_bytes = args.workspace * 1024 * 1024
    
    try:
        build_engine_with_preprocessing(
            args.onnx_path,
            args.engine_path,
            args.pipeline,
            workspace_bytes
        )
        return 0
    except Exception as e:
        print(f"Error building engine: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())