import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, default="ms03_vid.mp4"
)
parser.add_argument(
    "--model", type=str, default="x4_224_320.trt"
)
parser.add_argument(
    "--framerate", type=int, default=30
)

def preprocess(x:np.ndarray):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)          # ndarray는 채널 순서가 BGR 순서로 저장됨 -> RGB로 바꿔서 SR
    x = np.transpose(x, [2, 0, 1])                  # OpenCV는 기본적으로 [H,W,C] -> [C,H,W]
    #x = x * 1/255                                   # normalize (0~1)
    x = np.ascontiguousarray(x, dtype=np.float32)   # 메모리에 연속적으로 저장되지 않는 배열을 연속적으로 저장되는 배열로 변환 -> 데이터 빠르게 불러올 수 있음
    return x

def postprocess(x:np.ndarray):
    #x = x * 255                                     # denormalize
    #x = np.clip(x, min, max)                        # 픽셀값 범위 0~255로 제한
    x = x.astype(np.uint8)                          # 이미지로 나타내기 위해 uint8로 변환
    x = np.transpose(x, [1, 2, 0])                  # [C,H,W] -> [H,W,C]

    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)          # 다시 ndarray 채널 순서인 BGR로 바꿔줌
    return x    

def bicubicResize(x:np.ndarray, scale:int=4):
    h, w, _ = x.shape
    x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    return x

def horizontalFusion(bi:np.ndarray, sr:np.ndarray):
    assert bi.shape == sr.shape
    h, w, c = bi.shape
    canvas = np.zeros_like(bi).astype(np.uint8)
    canvas[:, 0:w//2, :] = bi[:, 0+200:w//2+200, :]
    canvas[:, w//2:w, :] = sr[:, 0+200:w//2+200, :]
    return canvas

if __name__ == "__main__":
    opt = parser.parse_args()
    try:
        cap = cv2.VideoCapture(opt.video)
    except:
        raise ValueError(f"Failed to open video file")
    
    model_path = os.path.join("./model", opt.model)
    size = opt.model[3:10]
    h, w = map(int, size.split("_"))
    size = (h, w)
    
    # load model
    trt_model = edgeSR_TRT_Engine(
        engine_path=model_path, scale=4, lr_size=size
    )
    
    frameRate = opt.framerate

    LR_WINDOW = "LR_WINDOW"
    BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

    cv2.namedWindow(LR_WINDOW)
    cv2.namedWindow(BICUBIC_SR_WINDOW)
    cv2.moveWindow(LR_WINDOW, 30, 20)
    cv2.moveWindow(BICUBIC_SR_WINDOW, 800, 300)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bicubic = bicubicResize(frame)
        input_np = preprocess(frame)
        sr_np = postprocess(trt_model(input_np))
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