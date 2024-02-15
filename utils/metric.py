import torch
import torch.nn as nn

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = []
        self.val = 0.0  # current value
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
    
    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):     
            # Tensor에 들어있는 값을 꺼내온다. 
            # 이때 Tensor 안에는 하나의 원소만 있어야 한다.
            value = value.item()
        value = float(value)                    
        
        self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
    def minmax(self):
        min_value = min(self.data)      
        max_value = max(self.data)
        return min_value, max_value
    
    # avg, min, max를 Dictionary 형태로 저장하여 반환
    def extract(self):
        d = {}      # Dictionary 선언
        # d['data'] = self.data
        d['avg'] = self.avg
        
        min_value, max_value = self.minmax()
        d['min'] = min_value
        d['max'] = max_value
        return d    # {'avg': 5.1, 'min': 3.2, ...} 형태