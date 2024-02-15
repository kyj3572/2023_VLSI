import torch
import numpy as np
import cv2
import torchvision
from libs.data_utils import *

qat_model = torch.jit.load("./weights/edgeSR_max_qat_200.jit.pt")
qat_model.cuda()
TEST_IMAGE_PATH = "./results/ms3_01.png"

imgTensor = npToTensor(openImage(TEST_IMAGE_PATH)).unsqueeze(0).float().cuda()
srTensor = qat_model(imgTensor).squeeze(0).cpu().detach().numpy().astype(np.uint8)
srObj = np.transpose(srTensor, [1,2,0])
srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
while True:
    cv2.imshow("test", srObj)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()