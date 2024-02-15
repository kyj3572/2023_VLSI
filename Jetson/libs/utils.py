import os
import time
# import torch
import cv2
import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

def preprocess(x:np.ndarray, norm:bool):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, [2, 0, 1])    
    if norm == True:
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    else:
        x = np.ascontiguousarray(x, dtype=np.float32)    
    return x

def postprocess(x:np.ndarray, norm:bool):
    if norm == True:
        x = np.ascontiguousarray(x * 255.0, dtype=np.uint8)
    else:
        x = np.ascontiguousarray(x, dtype=np.uint8)
    x = np.transpose(x, [1, 2, 0])      # [C, H, W] -> [H, W, C]
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def bicubicResize(x:np.ndarray, scale:int=4):
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
        return imgObj
    except:
        raise ValueError()


class edgeSR_TRT_Engine(object):
    # engine_path : "./model/x4_180_320.trt" 
    # scale : scaling factor
    # lr_size : shape of low resolution image (h, w)
    def __init__(self, engine_path, scale:int=4, lr_size=(180, 320)):
        self.lr_size = lr_size
        self.scale = scale
        self.hr_size = (lr_size[0] * scale, lr_size[1] * scale)
        
        # 로깅 레벨 설정 및 Runtime 설정
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        # ".trt"를 binary 형식으로 열기
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        # TensorRT engine 파일 역직렬화
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        # execution_context 객체 생성 -> 입력과 출력 바인딩 관리
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        # CUDA stream 생성 -> 비동기적인 연산
        self.stream = cuda.Stream()
        for binding in engine:
            # binding의 shape, data type 가져오기
            # engine.get_binding_shape(binding)의 반환값 : (3, 180, 320) -> trt.volume의 반환값 : 3 * 180 * 320 = 172,800
            # engine.get_binding_dtype(binding)의 반환값 : trt.float32 -> trt.nptype의 반환값 : np.float32
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Host memory (입출력 데이터) & Device memory (GPU 메모리 할당) 생성
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # bindings 배열에 device 메모리의 주소를 int형으로 변환하여 추가
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                # 입력 바인딩인 경우,
                # 입력 데이터를 저장하기 위한 딕셔너리를 생성하고, inputs 배열에 추가
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                # 출력 바인딩인 경우,
                # 출력 데이터를 저장하기 위한 딕셔너리를 생성하고, outputs 배열에 추가
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def __call__(self, lr:np.ndarray):
        # 입력 이미지를 1차원으로 변환 후 inputs[0]['host']에 할당
        # ex. (3, 10, 10) -> (300, )
        self.inputs[0]['host'] = np.ravel(lr)
        # inputs에 있는 각 입력 데이터에 대해 cuda.memcpy_htod_async 함수를 사용하여
        # host memory에 있는 데이터를 device memory로 비동기적으로 복사하는 것으로, 입력 데이터를 GPU로 전달
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # 모델을 비동기적으로 실행
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        # outputs에 있는 각 입력 데이터에 대해 cuda.memcpy_dtoh_async 함수를 사용하여
        # device memory에 있는 데이터를 host memory로 비동기적으로 복사하는 것으로, 출력 데이터를 CPU로 전달
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # 현재 진행중인 CUDA stream 작업(비동기적)이 완료될 때까지 대기
        self.stream.synchronize()
        
        # outputs에서 'host' 데이터를 가져와서 data list 생성
        data = [out['host'] for out in self.outputs]
        # 첫 번째 데이터를 이미지 형태로 재구성
        # ex. (300, ) -> (3, 10, 10)
        data = data[0]
        sr = np.reshape(data, (3, self.hr_size[0], self.hr_size[1]))
        
        return sr
    
        
