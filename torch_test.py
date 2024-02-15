import os
import cv2
import torch
import argparse
import numpy as np
from utils.data_utils import *
from model import abpn, rlfn, innopeak

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="media/ori/270_480_01.png"
)
parser.add_argument(
	"--weight", type=str, default="run/abpn..."
)
parser.add_argument(
	"--model", type=str, choices=['abpn', 'rlfn', 'innopeak'], default='abpn'
)
parser.add_argument(
	"--scale", type=int, default=4
)
parser.add_argument(
	"--norm", action="store_true"
)
parser.add_argument(
	"--qat", action="store_true"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not opt.qat:
        model = opt.model
        if model == 'abpn':
            model = abpn.ABPN()
        elif model == 'rlfn':
            model = rlfn.RLFN()
        elif model == 'innopeak':
            model = innopeak.InnoPeak(upscale=opt.scale)
        
        if device == 'cuda':
            model.load_state_dict(torch.load(opt.weight)).cuda()
        else:
            model.load_state_dict(torch.load(opt.weight, map_location=device))
    else:
        model = torch.jit.load(opt.weight)
        model.to(device)        

    model.eval()

    imgToTensor = torch.from_numpy(preprocess(openImage(opt.image), opt.norm))
    with torch.no_grad():
        srObj = model(imgToTensor).detach().numpy()
    srObj = postprocess(srObj, opt.norm)
    
    cv2.imshow("SuperResolution", srObj)
    cv2.waitKey()
    cv2.destroyAllWindows()

    folder = "media/pth_test/" + opt.weight[9:22]
    if not os.path.exists(folder):
        os.makedirs(folder)
    idx = [opt.weight.find('_')+1, opt.weight.find('.')]
    save = folder + opt.weight[idx[0]:idx[1]] + ".png"
    print(save)
    cv2.imwrite(save, srObj)