import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, default="./media/ori/270_480.mp4"
)
parser.add_argument(
    "--model", type=str, default="./model/x4_270_480.trt"
)
parser.add_argument(
    "--framerate", type=int, default=30
)
parser.add_argument(
    "--height", type=int, default=270
)
parser.add_argument(
    "--width", type=int, default=480
)
parser.add_argument(
    "--scale", type=int, default=4
)
parser.add_argument(
    "--norm", action="store_true"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    try:
        cap = cv2.VideoCapture(opt.video)
    except:
        raise ValueError(f"Failed to open video file")
    
    model_path = opt.model
    size = opt.height, opt.width
    upscale = opt.scale

    # load model
    trt_model = edgeSR_TRT_Engine(
        engine_path=model_path, scale=upscale, lr_size=size
    )
    
    frameRate = opt.framerate

    LR_WINDOW = "LR_WINDOW"
    BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

    cv2.namedWindow(LR_WINDOW)
    cv2.namedWindow(BICUBIC_SR_WINDOW)
    cv2.moveWindow(LR_WINDOW, 80, 250)
    cv2.moveWindow(BICUBIC_SR_WINDOW, 570, 250)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bicubic = bicubicResize(frame, upscale)
        input_np = preprocess(frame, opt.norm)
        sr_np = postprocess(trt_model(input_np), opt.norm)
        key = cv2.waitKey(frameRate)
        if key == 27:
            break
        
        # Left(BICUBIC) + Right(SuperResolution) ...
        canvas = horizontalFusion(bicubic, sr_np)

        cv2.imshow(BICUBIC_SR_WINDOW, canvas)        
        cv2.imshow(LR_WINDOW, frame)
        
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
