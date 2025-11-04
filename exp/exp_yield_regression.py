import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import warnings

# --- (关键) 导入我们的新数据加载器 ---
from data_provider import data_provider_yield 
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.metrics import metric # TSLib 自带的 metric

warnings.filterwarnings('ignore')

class Exp_Yield_Regression(Exp_Basic):
    """
    用于产量预测 (Seq2One 回归) 的自定义实验 Pipeline
    """
    def __init__(self, args):
        super(Exp_Yield_Regression, self).__init__(args)

    def _build_model(self):
        print(f"正在构建模型: {self.args.model}")
        
        # (我们不需要动态导入, Exp_Basic 会帮我们处理)
        # model = self.model_dict[self.args.model].Model(self.args).float()
        
        # (我们必须在 exp_basic.py 中注册模型)
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # flag 是 'train', 'val', 'test'
        # data_provider_yield 是我们在 data_loader_yield.py 中创建的函数
        dataset, data_loader = data_provider_yield(self.args, flag)
        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 回归任务使用 MSE (均方误差)
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion() # nn.MSELoss()

        best_val_loss = float('inf')
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            total_loss = 0
            
            for i, (x_dynamic, x_static, y_target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 移动数据到 GPU
                x_dynamic = x_dynamic.to(self.device)
                x_static = x_static.to(self.device)
                y_target = y_target.to(self.device)
                
                # --- 核心修改 (处理 MoE 的辅助损失) ---
                if 'MoE' in self.args.model:
                    # MoE 模型返回 (prediction, load_loss)
                    predictions, load_loss = self.model(x_dynamic, x_static)
                    main_loss = criterion(predictions, y_target)
                    loss = main_loss + load_loss
                else:
                    # 基线模型只返回 prediction
                    predictions = self.model(x_dynamic, x_static)
                    loss = criterion(predictions, y_target)
                
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f"Epoch: {epoch+1} | Batch: {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

            train_loss = total_loss / len(train_loader)
            val_loss = self.vali(vali_loader, criterion) # 运行验证
            
            print(f"Epoch: {epoch+1} | 耗时: {time.time() - epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                print("  -> Val loss 降低, 保存模型 checkpoint.")
                
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (x_dynamic, x_static, y_target) in enumerate(vali_loader):
                x_dynamic = x_dynamic.to(self.device)
                x_static = x_static.to(self.device)
                y_target = y_target.to(self.device)
                
                if 'MoE' in self.args.model:
                    predictions, load_loss = self.model(x_dynamic, x_static)
                    loss = criterion(predictions, y_target) + load_loss
                else:
                    predictions = self.model(x_dynamic, x_static)
                    loss = criterion(predictions, y_target)
                    
                total_loss += loss.item()
                
        self.model.train() # 恢复训练模式
        return total_loss / len(vali_loader)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (x_dynamic, x_static, y_target) in enumerate(test_loader):
                x_dynamic = x_dynamic.to(self.device)
                x_static = x_static.to(self.device)
                y_target = y_target.to(self.device)
                
                if 'MoE' in self.args.model:
                    predictions, _ = self.model(x_dynamic, x_static)
                else:
                    predictions = self.model(x_dynamic, x_static)
                
                preds.append(predictions.detach().cpu().numpy())
                trues.append(y_target.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 计算回归指标
        # (注意: TSLib 的 metric 函数计算 MAE 和 MSE)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # (我们还需要 R2)
        try:
            from sklearn.metrics import r2_score
            r2 = r2_score(trues, preds)
        except:
            r2 = -999.0 # R2 计算失败
        
        print(f"--- 测试结果 (RMSE | MAE | R2) ---")
        print(f"  {rmse:.4f} | {mae:.4f} | {r2:.4f}")
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([rmse, mae, r2]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        
        return
