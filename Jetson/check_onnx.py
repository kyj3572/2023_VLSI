import os
import argparse
import sys
import onnx

parser = argparse.ArgumentParser()
parser.add_argument(
	"--model", type=str, default="~/working/train/onnx/x4_270_480.onnx"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    
    model = onnx.load(opt.model)
    if (onnx.checker.check_model(model, full_check=True) == None):
        print("No problem\n")