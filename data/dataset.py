import torch
import os
import cv2
import random
import numpy as np
import albumentations as A          
from tqdm import tqdm              
from typing import List, Tuple
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

# Original Color image -> HR : [256 x 256], LR : [64 x 64] & Normalization
def to_tensor(arr: np.ndarray,
              normalization:bool=False) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input type must be numpy.ndarray. Your input : {type(arr)}")

    if len(arr.shape) != 3 and len(arr.shape) != 4:
        raise ValueError(f"Input shape must be (H, W, C) or (B, H, W, C). Your shape : {arr.shape}")

    if not normalization:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32) / 255.0

    # Change axis
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))      # (H, W, C) -> (C, H, W)
    else:
        arr = np.transpose(arr, (0, 3, 1, 2))   # (B, H, W, C) -> (B, C, H, W)

    # numpy -> torch.Tensor
    return torch.from_numpy(arr)

def read_img(path:str) -> np.ndarray:
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
        return img
    except:
        return None
    
# List[str] : str 타입의 원소를 가지는 리스트
def check_imgs(img_paths:List[str]):
    valid, error = 0, 0
    isblack = 0

    valid_img_path =[]
    
    bar = tqdm(img_paths, ncols=120)  
    
    for img_path in bar:
        obj = read_img(img_path)    
        if obj is None:             
            error += 1
        elif isinstance(obj, np.ndarray):   
            h, w, c = obj.shape
            if c == 1:     
                isblack += 1
            elif c == 3:   
                valid += 1
                valid_img_path.append(img_path)
        del obj
    
    print(f"Chekced imgs ... valid={valid}, io_error={error}, is_black={isblack}")
    
    return valid_img_path

# float=0.8, int=1 -> Default 값
def split_data(img_paths:List[str], r:float=0.8, seed:int=1):
    total_number = len(img_paths)               
    train_number = int(total_number * r)        
    valid_number = total_number - train_number
    
    random.seed(seed)
    random.shuffle(img_paths)
    
    train_img_paths = img_paths[:train_number]
    valid_img_paths = img_paths[train_number:]
    
    print(f"# TRAIN_DATA = {train_number}, VALID_DATA = {valid_number}")
    
    return train_img_paths, valid_img_paths

# Dataset preprocessing & Preload 
class SR_Dataset(Dataset):
    def __init__(self, 
                 data_paths:list, 
                 preload:bool=False,
                 normalization:bool=False, 
                 train:bool=True, 
                 hr_size:int=256, 
                 lr_size:int=64, 
                 workers:int=8):
        super().__init__()      
        self.length_of_data = len(data_paths)       
        self.workers = workers                     
        self.preload = preload                     
        self.norm = normalization
        self.data_list = self._preload(data_paths, preload)
        """
            preload = True : pixel values in data_list
            preload = False : image paths in data_list      
        """

        # Data augmentation
        if isinstance(train, bool) and train == True:
            self.aug_methods = [
                A.AdvancedBlur(),
                A.OneOf([A.GaussNoise(), A.ISONoise()]),
                A.OneOf([A.Downscale(scale_min=0.25, scale_max=0.7, interpolation=cv2.INTER_AREA),
                         A.Downscale(scale_min=0.25, scale_max=0.7, interpolation=cv2.INTER_CUBIC),
                         A.Downscale(scale_min=0.25, scale_max=0.7, interpolation=cv2.INTER_LINEAR)]),
                A.ImageCompression(quality_lower=80, quality_upper=100)
            ]
            random.shuffle(self.aug_methods)

            # Original -> High resolution [256 x 256]
            self.o2hr = A.Compose([
                A.RandomCrop(height=hr_size, width=hr_size),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(limit=45, interpolation=cv2.INTER_CUBIC)
            ])
            # High resolution -> Low resolution [64 x 64]
            self.hr2lr = A.Compose([
                *self.aug_methods,                                                            
                A.Resize(height=lr_size, width=lr_size)
            ])
            
        elif isinstance(train, bool) and train == False:
            self.o2hr = A.CenterCrop(height=hr_size, width=hr_size)
            self.hr2lr = A.Resize(height=lr_size, width=lr_size)
        else:
            raise RuntimeError(train)
    
    # Load image values on RAM in binary.
    def _preload(self, data_paths:list, preload:bool=False):
        if not preload:
            return data_paths
        else:   
            with Pool(processes=self.workers) as pool:
                preloaded_data = []
                for img_data in tqdm(pool.imap_unordered(read_img, data_paths, chunksize=16), 
                                     total=self.length_of_data):
                    preloaded_data.append(img_data)
            
            pool.close()    
            pool.join()     

            print("\033[F\033[J ==> IMAGE PRELOAD COMPLETE")
            return preloaded_data

    def __len__(self) -> int:
        return self.length_of_data
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:    # preload = True
            orig_obj = self.data_list[idx]
        else:               # preload = False
            orig_obj = read_img(self.data_list[idx])
        
        # Return type of o2hr & hr2lr is Dictionary -> key : "image"
        hr_obj = self.o2hr(image=orig_obj)["image"]
        lr_obj = self.hr2lr(image=hr_obj)["image"]
        
        hr_tensor = to_tensor(hr_obj, self.norm)
        lr_tensor = to_tensor(lr_obj, self.norm)
        
        return lr_tensor, hr_tensor


def get_dataloader(root:str,
                   check:bool=False, 
                   split_ratio:float=0.8, 
                   preload:bool=False,
                   normalization:bool=False,
                   batch_size:int=32, 
                   workers:int=8,
                   hr_size:int=256, 
                   lr_size:int=64):
    
    total_img_paths = [os.path.join(root, x) for x in os.listdir(root)]

    # Check data
    if check:
        imgs_list = check_imgs(total_img_paths)
    else:
        imgs_list = total_img_paths

    # Split data
    train_imgs_list, valid_imgs_list = split_data(imgs_list, split_ratio)

    # build Dataset
    train_dataset = SR_Dataset(train_imgs_list, preload, normalization, True, hr_size, lr_size, workers)
    valid_dataset = SR_Dataset(valid_imgs_list, preload, normalization, False, hr_size, lr_size, workers)
    
    # build DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=workers, persistent_workers=True)
    
    return train_loader, valid_loader