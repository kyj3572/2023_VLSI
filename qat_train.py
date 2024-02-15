import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch_tensorrt
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

from pytorch_quantization import quant_modules

from libs.model import *
from libs.data_utils import *
from libs.common import train_one_epoch

# Constants ...
DATA_PRELOAD = True
FP32_WEIGHT = "./weights/1000_dts.pth"
DATA_ROOT = "../data/"
EPOCHS = 200
BATCH_SIZE = 16
TEST_IMAGE_PATH = "ms3_01.png"

imgTensor = npToTensor(openImage(TEST_IMAGE_PATH)).unsqueeze(0).float().cuda()


# (1) Adding quantized modules
quant_modules.initialize()

# (2) Post training quantization
# For efficient inference, we want to select a fixed range for each quantizer
quant_desc_input = QuantDescriptor(calib_method="max")
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_input)

model = edgeSR()
model.load_state_dict(torch.load(FP32_WEIGHT))
model.cuda()

traindataset = trainDataset(DATA_ROOT, preload=DATA_PRELOAD)
valdataset = valDataset(DATA_ROOT, preload=DATA_PRELOAD)
trainloader = DataLoader(
    traindataset, BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()-1, pin_memory=True
)
valloader = DataLoader(
    valdataset, BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()-1, pin_memory=True
)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=[0.9, 0.999], eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(1, EPOCHS+1):
    train_one_epoch(
        model, trainloader, valloader, criterion, optimizer, lr_scheduler
    )

    if epoch % 20 == 0:
    # Export to TensorRT
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        dummy_input = torch.randint(0, 255, (1, 3, 224, 320)).float().cuda()
        with torch.no_grad():
            jit_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(jit_model, f"./weights/edgeSR_max_qat_{epoch}.jit.pt")
            
        qat_model = torch.jit.load(f"./weights/edgeSR_max_qat_{epoch}.jit.pt").eval()
        compile_spec = {
            "inputs" : [torch_tensorrt.Input([1, 3, 224, 320])],
            "enabled_precisions" : torch.int8,
            "truncate_long_and_double" : True
        }
        trt_mod = torch_tensorrt.compile(qat_model, **compile_spec)
        sr_trt = trt_mod(imgTensor).squeeze(0).cpu().numpy().astype(np.uint8)

        srObj = np.transpose(sr_trt, [1, 2, 0])
        srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output_trt_max_{epoch}.jpg", srObj)
