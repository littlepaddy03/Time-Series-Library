from data_provider.data_loader_yield import data_provider_yield
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

class Exp_Yield_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Yield_Regression, self).__init__(args)
        self.writer = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == 'train':
            return data_provider_yield(self.args, flag)
        else:
            # This is handled by the train flag call now
            return None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_static, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.model == 'MoE_Regression':
                    outputs, _, _ = self.model(batch_x, batch_x_static)
                else:
                    outputs = self.model(batch_x, batch_x_static)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        r2 = r2_score(trues, preds)

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        train_data, train_loader, vali_data, vali_loader, test_data, test_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self.writer = SummaryWriter(log_dir=os.path.join('runs', setting))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 专门为 MoE 准备一个计数器
            if self.args.model == 'MoE_Regression':
                expert_counts = torch.zeros(self.args.n_experts, dtype=torch.long).to(self.device)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_static, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.model == 'MoE_Regression':
                    outputs, aux_loss, expert_indices = self.model(batch_x, batch_x_static)
                    loss = criterion(outputs, batch_y) + aux_loss

                    # 更新专家计数 (如果模型返回了 expert_indices)
                    if expert_indices is not None:
                        expert_counts.index_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=torch.long))
                else:
                    outputs = self.model(batch_x, batch_x_static)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                self.writer.add_scalar('Loss/train_step', loss.item(), epoch * train_steps + i)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val', vali_loss, epoch)
            self.writer.add_scalar('R2/val', vali_r2, epoch)

            # 记录 MoE 专家使用情况
            if self.args.model == 'MoE_Regression':
                print(f"Expert utilization for epoch {epoch + 1}: {expert_counts.cpu().numpy()}")
                for i in range(self.args.n_experts):
                    self.writer.add_scalar(f'Expert_Usage/expert_{i}', expert_counts[i], epoch)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            print(f"Vali R2: {vali_r2:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Perform final test
        print("------ Final Test ------")
        self.test(test_loader)

        return self.model

    def test(self, test_loader):
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_static, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.model == 'MoE_Regression':
                    outputs, _, _ = self.model(batch_x, batch_x_static)
                else:
                    outputs = self.model(batch_x, batch_x_static)

                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        r2 = r2_score(trues, preds)
        print(f'Test R2: {r2:.4f}')

        if self.writer:
            self.writer.add_scalar('R2/test', r2, 0) # Log test R2 at a single step
            self.writer.close()

        return
