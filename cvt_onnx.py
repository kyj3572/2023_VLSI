import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.onnx
import onnx
from model import abpn
from model import rlfn
from model import innopeak
from onnx import shape_inference

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight", type=str, default='run/abpn...'
)
parser.add_argument(
    "--height", type=int, default=270
)
parser.add_argument(
    "--width", type=int, default=480
)
parser.add_argument(
    "--model", type=str, choices=['abpn', 'rlfn', 'innopeak'], default='abpn'
)
parser.add_argument(
    "--save", type=str, default='onnx/x4_270_480.onnx'
)
parser.add_argument(
    "--scale", type=int, default=4
)
parser.add_argument(
    "--qat", action="store_true"
)

def Convert_ONNX(model:nn.Module, input_shape:tuple, name:str, device:torch.device):
    if not name.endswith('.onnx'):
        name = name + '.onnx'
    if os.path.exists(name):
        os.remove(name)
        print(f"[WARN] REMOVE FILE = {name}")

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(input_shape, requires_grad=True).to(device).detach()

    # Export the model
    torch.onnx.export(  model,                     # model being run
                        dummy_input,               # model input (or a tuple for multiple inputs)
                        name,                      # where to save the model
                        export_params=True,        # store the trained parameter weights inside the model file
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        opset_version = 11
    )

    # Add shape to onnx model
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(name)), name)

    print('# Model has been converted to ONNX')
    print(f'# CREATE NEW ONNX FILE = {name}')

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}\n")

    ins_dir = args.weight

    h, w = args.height, args.width

    if not args.qat:
        model = args.model
        if model == 'rlfn':
            model = rlfn.RLFN()
        elif model == 'abpn':
            model = abpn.ABPN()
        elif model == 'innopeak':
            model = innopeak.InnoPeak(upscale=args.scale)
        
        model.load_state_dict(torch.load(ins_dir))
    else:
        model = torch.jit.load(args.weight)

    model.to(device)

    exp_dir = args.save

    Convert_ONNX(model, (1, 3, h, w), exp_dir, device)    # (B, C, H, W)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
