import os
import json
import shutil   
import torch
import torch.nn as nn
import torch.quantization as qt
import matplotlib.pyplot as plt
import pytorch_quantization

from time import time
from pytz import timezone
from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from data.dataset import get_dataloader
from model import get_model
from utils.cawr import CosineAnnealingWarmUpRestarts
from utils.metric import AverageMeter
from utils import innopeak_loss
from torchmetrics.functional.image import peak_signal_noise_ratio, \
        structural_similarity_index_measure
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

class Trainer:
    def __init__(self, args):
        # backup args
        self.args = args
        self.start_time = None
        # prepare data pipeline
        self.train_loader, self.valid_loader = get_dataloader(
            root=args.data_path,
            check=args.file_check,
            preload=args.preload,
            normalization=args.normalization,
            batch_size=args.batch_size,
            workers=args.workers,
            hr_size=args.hr_size,
            lr_size=args.lr_size
        )
        # define model
        self.qat = args.qat
        self.model = get_model(args)
        
        if args.load is not None:
            try:
                self.model.load_state_dict(torch.load(args.load))
            except:
                print(f"Failed to load weights")
        
        if not self.qat:
            self.model_name = args.model
        else:
            self.model_name = args.model + '_QAT'
            quant_modules.initialize()
            quant_desc_input = QuantDescriptor(calib_method="max")
            quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
            quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_input)
        
        self.criterion = self.get_criterion(args)
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.device = torch.device(args.device)
        self.step = args.step
        self.gamma = args.gamma
        self.epochs = args.epochs
        
        self.exp_dir = None
        self.save_dir = None
        self.save_interval = args.save_interval
        
        # metrics buffer area
        self.train_loss_data = []
        self.valid_psnr_data = []
        self.valid_ssim_data = []

    def get_criterion(self, args):
        if args.loss == 'l1':
            return nn.L1Loss()
        elif args.loss == 'l2':
            return nn.MSELoss()
        elif args.loss == 'inno_loss':
            return innopeak_loss.InnoPeak_loss()

    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            optimizer = Adam(self.model.parameters(), lr=args.lr)
            # scheduler = StepLR(optimizer, args.step, args.gamma)
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer, T_0=args.step, T_mult=1, eta_max=1e-3, T_up=5, gamma=args.gamma
            )
            return optimizer, scheduler
        
        elif args.optimizer == 'sgd':
            optimizer = SGD(self.model.parameters(), momentum=args.momentum, lr=args.lr)
            scheduler = StepLR(optimizer, args.step, args.gamma)
            return optimizer, scheduler
    
    def train(self):
        self._prepare()
        self.start_time = time()
        for e in range(1, self.epochs+1):
            print(f"# Train Super Resolution Model [{e}/{self.epochs}]")
            self._train()
            self._valid()
            # print("\033[F\033[J",end="")
            if e % self.save_interval == 0:
                self._save_model(e)
                
        # after train finished
        self._finish()
    
    def _prepare(self):
        self.model.to(self.device)

        if not os.path.exists("./run"):
            os.makedirs("./run")
        
        exp_name = self.args.model + "-" + datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H-%M")
        exp_dir = os.path.join("./run", exp_name)
        
        # remove duplicate dir
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        
        os.makedirs(exp_dir)
        self.exp_dir = exp_dir
        os.makedirs(os.path.join(exp_dir, "weights"))
        self.save_dir = os.path.join(exp_dir, "weights")
        
        args_dict = vars(self.args)
        with open(os.path.join(exp_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)
        
    def _train(self):
        self.model.train()
        train_bar = tqdm(self.train_loader, ncols=120)
        train_loss_meter = AverageMeter()
        for lr_tensor, hr_tensor in train_bar:
            lr_tensor, hr_tensor = lr_tensor.to(self.device), hr_tensor.to(self.device)
            self.optimizer.zero_grad()
            
            # forward
            sr_tensor = self.model(lr_tensor)

            # calculate loss
            loss = self.criterion(sr_tensor, hr_tensor)
            
            # backpropagation
            loss.backward()
            self.optimizer.step()
            train_loss_meter.update(loss)
            train_bar.set_description(
                f"# TRAIN : loss(avg)={train_loss_meter.avg:.5f}"
            )
        # print("\033[F\033[J",end="")
        self.train_loss_data.append(train_loss_meter.extract())
        del train_loss_meter
        self.scheduler.step()
        
    def _valid(self):
        self.model.eval()   
        valid_bar = tqdm(self.valid_loader, ncols=120)
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        with torch.no_grad():
            for lr_tensor, hr_tensor in valid_bar:
                lr_tensor, hr_tensor = lr_tensor.to(self.device), hr_tensor.to(self.device)
                sr_tensor = self.model(lr_tensor)
                
                psnr_value = peak_signal_noise_ratio(sr_tensor, hr_tensor)
                ssim_value = structural_similarity_index_measure(sr_tensor, hr_tensor)
                
                psnr_meter.update(psnr_value)
                ssim_meter.update(ssim_value)
                
                valid_bar.set_description(
                    f"# VALID : PSNR={psnr_meter.avg:.3f}, SSIM={ssim_meter.avg:.5f}"
                )
        self.valid_psnr_data.append(psnr_meter.extract())
        self.valid_ssim_data.append(ssim_meter.extract())
        # print("\033[F\033[J",end="")
        del psnr_meter
        del ssim_meter        
    
    def _save_model(self, e:int):
        if self.qat == True:
            model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_{e}.jit.pth")
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
            dummy_input = torch.randint(0, 255, (1, 3, 270, 480)).float().to(self.device)
            with torch.no_grad():
                jit_model = torch.jit.trace(self.model, dummy_input)
                torch.jit.save(jit_model, model_weight_path)
        else:
            model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_{e}.pth")
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            # Save as Dictionary -> torch.load(), load_state_dict()
            torch.save(self.model.state_dict(), model_weight_path)      
        
    def _finish(self):
        # save final model weights
        if self.qat == True:
            model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_final.jit.pth")
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
            dummy_input = torch.randint(0, 255, (1, 3, 270, 480)).float().to(self.device)
            with torch.no_grad():
                jit_model = torch.jit.trace(self.model, dummy_input)
                torch.jit.save(jit_model, model_weight_path)
        else:
            model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_final.pth")
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            torch.save(self.model.state_dict(), model_weight_path) 
        
        # find min/max psnr/ssim value
        psnr_avg = 0.0
        psnr_max = 0.0
        ssim_avg = 0.0
        ssim_max = 0.0
        
        for i, (psnr_data, ssim_data) in enumerate(zip(self.valid_psnr_data, self.valid_ssim_data)):
            current_psnr_avg = psnr_data['avg']
            current_psnr_max = psnr_data['max']
            current_ssim_avg = ssim_data['avg']
            current_ssim_max = ssim_data['max']
            
            if current_psnr_avg > psnr_avg:
                psnr_avg = current_psnr_avg
                psnr_avg_indices = i + 1
            
            if current_psnr_max > psnr_max:
                psnr_max = current_psnr_max
                psnr_max_indices = i + 1
            
            if current_ssim_avg > ssim_avg:
                ssim_avg = current_ssim_avg
                ssim_avg_indices = i + 1
            
            if current_ssim_max > ssim_max:
                ssim_max = current_ssim_max
                ssim_max_indices = i + 1
                 
        # Draw training_loss graph
        indices = []
        buffer_min = []
        # buffer_max = []
        buffer_avg = []
        loss_avg_min = 300.0

        for i, train_loss in enumerate(self.train_loss_data):
            indices.append(i)
            buffer_min.append(train_loss['min'])
            # buffer_max.append(train_loss['max'])
            buffer_avg.append(train_loss['avg'])
            if train_loss['avg'] < loss_avg_min:
                loss_avg_min = train_loss['avg']
                loss_avg_min_indices = i + 1
       
        print(f"# TRAIN RESULTs")
        print(f"#\t Min of Average Loss = {loss_avg_min:.5f} ({loss_avg_min_indices})")
        print(f"#\t PSNR(avg/max) = {psnr_avg:.3f} / {psnr_max:.3f} ({psnr_avg_indices} / {psnr_max_indices})")
        print(f"#\t SSIM(avg/max) = {ssim_avg:.5f} / {ssim_max:.5f} ({ssim_avg_indices} / {ssim_max_indices})")
        
        plt.figure(figsize=(10, 6))
        plt.plot(indices, buffer_min, 'r', label='Loss(min)')
        plt.plot(indices, buffer_avg, 'b', label='Loss(avg)')
        # plt.plot(indices, buffer_max, 'g', label='Loss(max)')
        plt.legend(loc='upper right')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.xticks(range(min(indices), max(indices)+1, 1), fontsize=5)
        
        for i in range(self.step, self.epochs, self.step):
            plt.axvline(x=i, color='olivedrab', linestyle='--')
        plt.savefig(os.path.join(self.exp_dir, 'training_loss.png'))
        
        # Extract average value from list
        valid_psnr_avg_data = [item['avg'] for item in self.valid_psnr_data]
        valid_ssim_avg_data = [item['avg'] for item in self.valid_ssim_data]

        # Draw PSNR & SSIM graph
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # PSNR
        ax1.plot(range(min(indices), max(indices)+1, 1), valid_psnr_avg_data, 'r', label='PSNR')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR', color='r')

        # SSIM
        ax2 = ax1.twinx()
        ax2.plot(range(min(indices), max(indices)+1, 1), valid_ssim_avg_data, 'b', label='SSIM')
        ax2.set_ylabel('SSIM', color='b', rotation=-90, labelpad=15)

        ax1.set_xticks(range(min(indices), max(indices)+1, 1))
        ax1.set_xticklabels(range(min(indices), max(indices)+1, 1), fontsize=5)

        plt.title('Average of PSNR & SSIM')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        for i in range(self.step, self.epochs, self.step):
            plt.axvline(x=i, color='olivedrab', linestyle='--')
        plt.savefig(os.path.join(self.exp_dir, 'psnr_ssim_average.png'))

        elapsed_time = time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"# elapsed_time = {int(hours)}hr {int(minutes)}m {seconds:.1f}s")

        with open(os.path.join(self.exp_dir, 'summary.txt'), 'w') as f:
            f.write(f"# Elapsed_time = {int(hours)}hr {int(minutes)}m {seconds:.1f}s\n")
            f.write(f"# min of Average Loss = {loss_avg_min:.5f} ({loss_avg_min_indices})\n")
            f.write(f"# PSNR(avg/max) = {psnr_avg:.3f} / {psnr_max:.3f} ({psnr_avg_indices} / {psnr_max_indices})\n")
            f.write(f"# SSIM(avg/max) = {ssim_avg:.5f} / {ssim_max:.5f} ({ssim_avg_indices} / {ssim_max_indices})\n")
