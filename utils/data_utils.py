import cv2
import torch
import numpy as np

def preprocess(x:np.ndarray, norm:bool):
    x = np.transpose(x, [2, 0, 1])    
    x = np.expand_dims(x, axis=0)
    if norm == True:
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    else:
        x = np.ascontiguousarray(x, dtype=np.float32)    
    return x

def postprocess(x:np.ndarray, norm:bool):
    if norm == True:
        x = np.ascontiguousarray(x * 255.0, dtype=np.uint8).squeeze(0)
    else:
        x = np.ascontiguousarray(x, dtype=np.uint8).squeeze(0)
    x = np.transpose(x, [1, 2, 0])      # [C, H, W] -> [H, W, C]
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def bicubicResize(x:np.ndarray, scale:int=4):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    h, w, _ = x.shape
    x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return x

def horizontalFusion(bi:np.ndarray, sr:np.ndarray):
    assert bi.shape == sr.shape, "Check image shape"
    h, w, c = bi.shape
    canvas = np.zeros_like(bi).astype(np.uint8)
    canvas[:, 0:w//2, :] = bi[:, w//4:(w//2 + w//4), :]
    canvas[:, w//2:w, :] = sr[:, w//4:(w//2 + w//4), :]
    return canvas

def openImage(filepath):
    try:
        imgObj = cv2.imread(filepath, cv2.IMREAD_COLOR)
        imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2RGB)
        return imgObj
    except:
        raise ValueError()
    
def npToTensor(x:np.ndarray):
    x = np.transpose(x, [2, 0, 1])
    tensor = torch.from_numpy(x)
    return tensor