import os
import cv2
import argparse
import numpy as np
from utils.data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    save = "./media/ref/x4_" + opt.image[-14:]

    if not os.path.exists("./media/ref"):
        os.makedirs("./media/ref")

    biObj = bicubicResize(openImage(opt.image))
    
    cv2.imshow("Bicubic", biObj)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(save, biObj)