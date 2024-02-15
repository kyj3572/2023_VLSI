import argparse
import numpy as np
import cv2
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--img_dir", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--save_dir", type=str, default="./media/ref/x4_270_480_bicubic_01.png"
)
parser.add_argument(
	"--scale", type=int, default=4
)

if __name__ == "__main__":
	args = parser.parse_args()
	
	lrOrig = cv2.imread(args.img_dir)
	bicubic = bicubicResize(lrOrig, args.scale)
	
	cv2.imwrite(args.save_dir, bicubic)
