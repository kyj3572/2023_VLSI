import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--model", type=str, default="./model/x4_270_480.trt"
)
parser.add_argument(
    "--height", type=int, default=270
)
parser.add_argument(
    "--width", type=int, default=480
)
parser.add_argument(
	"--save", type=str
)
parser.add_argument(
	"--scale", type=int, default=4
)
parser.add_argument(
	"--norm", action="store_true"
)

if __name__ == "__main__":
	opt = parser.parse_args()
		
	trt_model = edgeSR_TRT_Engine(
		engine_path=opt.model, scale=opt.scale, lr_size=(opt.height, opt.width)
	)
	lrOrig = openImage(opt.image)
	
	# SuperResolution
	lrObj = preprocess(lrOrig, opt.norm)
	srObj = postprocess(trt_model(lrObj), opt.norm)

	biObj = bicubicResize(openImage(opt.image))
	canvas = horizontalFusion(biObj, srObj)

	BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"
	cv2.namedWindow(BICUBIC_SR_WINDOW)
	cv2.moveWindow(BICUBIC_SR_WINDOW, 570, 250)
	cv2.imshow(BICUBIC_SR_WINDOW, canvas)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

	'''
	if not opt.save:
		save = "./media/result/trt_" + opt.image[-14:]
		cv2.imwrite(save, srObj)
	else:
		cv2.imwrite(opt.save, srObj)
	'''
