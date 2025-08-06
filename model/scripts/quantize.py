"""
This script performs dynamic quantization on a trained ONNX model.

Quantization is a process that reduces the precision of the model's weights
(e.g., from 32-bit floating point to 8-bit integer), which can significantly
decrease the model's size and speed up inference time with minimal impact on
accuracy. This script uses the ONNX Runtime quantization tools to apply dynamic
quantization, where the scaling factors for activations are calculated on-the-fly.
"""

import os
import sys
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def quantize_model(model_input: str, model_output: str, weight_type: QuantType):
    """
    Performs dynamic quantization on an ONNX model.
    """
    print(f"Quantizing model: {model_input}")
    print(f"Saving quantized model to: {model_output}")
    print(f"Using weight type: {weight_type}")
    
    quantize_dynamic(
        model_input,
        model_output, 
        weight_type=weight_type
    )
    
    print("Quantization complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize an ONNX model.")
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained ONNX model (.onnx file)."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports",
        help="Directory to save the quantized .onnx file. Defaults to 'exports'."
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="int8",
        choices=["int8", "uint8"],
        help="Define the quantization weights type. Can be 'int8' or 'uint8'. Defaults to 'int8'."
    )

    args = parser.parse_args()

    if args.type == "int8":
        quantization_type = QuantType.QInt8
    elif args.type == "uint8":
        quantization_type = QuantType.QUInt8
    else:
        raise ValueError(f"Unsupported quantization type: {args.type}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.basename(args.model_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_model_name = f"{file_name}.quant{file_ext}"
    
    output_model_path = os.path.join(args.output_dir, output_model_name)

    quantize_model(args.model_path, output_model_path, quantization_type)
