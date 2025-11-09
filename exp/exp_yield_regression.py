from data_provider.data_loader_yield import data_provider_yield
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
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
            return None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(self, batch_data):
        batch_x, batch_x_static, batch_y = batch_data
        batch_x = batch_x.float().to(self.device)
        batch_x_static = batch_x_static.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if 'MoE_Regression' in self.args.model:
            outputs, aux_loss, affinities = self.model(batch_x, batch_x_static)
        else:
            outputs, aux_loss, affinities = self.model(batch_x, batch_x_static), None, None

        return outputs, batch_y, aux_loss, affinities, batch_x_static

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds, trues = [], []
        static_features, expert_affinities = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                outputs, batch_y, _, affinities, batch_x_static = self._process_one_batch(batch_data)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())
                if affinities is not None:
                    static_features.append(batch_x_static.cpu())
                    expert_affinities.append(affinities.cpu())

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        r2 = r2_score(trues.numpy(), preds.numpy())
        mae, mse, rmse, _, _ = metric(preds.numpy(), trues.numpy())

        if len(expert_affinities) > 0:
            static_features = np.concatenate(static_features, axis=0)
            expert_affinities = np.concatenate(expert_affinities, axis=0)
        else:
            static_features, expert_affinities = None, None

        self.model.train()
        return total_loss, r2, mae, mse, rmse, (static_features, expert_affinities)

    def train(self, setting):
        train_data, train_loader, _, vali_loader, _, test_loader = self._get_data(flag='train')
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

            # Placeholders for affinity analysis data
            epoch_static_features, epoch_expert_affinities = [], []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                outputs, batch_y, aux_loss, affinities, batch_x_static = self._process_one_batch(batch_data)

                loss = criterion(outputs, batch_y) + (aux_loss if aux_loss is not None else 0)
                train_loss.append(loss.item())

                if affinities is not None:
                    epoch_static_features.append(batch_x_static.cpu().numpy())
                    epoch_expert_affinities.append(affinities.cpu().detach().numpy())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                self.writer.add_scalar('Loss/train_step', loss.item(), epoch * train_steps + i)

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss, vali_r2, vali_mae, vali_mse, vali_rmse, _ = self.vali(vali_loader, criterion)

            # --- Logging ---
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val', vali_loss, epoch)
            self.writer.add_scalar('R2/val', vali_r2, epoch)
            self.writer.add_scalar('MAE/val', vali_mae, epoch)
            self.writer.add_scalar('MSE/val', vali_mse, epoch)
            self.writer.add_scalar('RMSE/val', vali_rmse, epoch)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            print(f"Vali R2: {vali_r2:.4f}, MAE: {vali_mae:.4f}, MSE: {vali_mse:.4f}, RMSE: {vali_rmse:.4f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # --- Save training affinities ---
        if len(epoch_expert_affinities) > 0:
            train_static_features = np.concatenate(epoch_static_features, axis=0)
            train_expert_affinities = np.concatenate(epoch_expert_affinities, axis=0)
            affinity_folder_path = os.path.join(path, 'affinity_analysis')
            os.makedirs(affinity_folder_path, exist_ok=True)
            np.savez_compressed(os.path.join(affinity_folder_path, 'train_affinities.npz'),
                                static_features=train_static_features,
                                expert_affinities=train_expert_affinities)
            print("Saved training affinity data.")

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        print("------ Final Test ------")
        self.test(setting, test_loader)

        return self.model

    def test(self, setting, test_loader):
        preds, trues = [], []
        static_features, expert_affinities = [], []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                outputs, batch_y, _, affinities, batch_x_static = self._process_one_batch(batch_data)

                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())
                if affinities is not None:
                    static_features.append(batch_x_static.cpu())
                    expert_affinities.append(affinities.cpu())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        r2 = r2_score(trues.numpy(), preds.numpy())
        mae, mse, rmse, _, _ = metric(preds.numpy(), trues.numpy())

        print(f'Test R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

        if self.writer:
            self.writer.add_scalar('R2/test', r2, 0)
            self.writer.add_scalar('MAE/test', mae, 0)
            self.writer.add_scalar('MSE/test', mse, 0)
            self.writer.add_scalar('RMSE/test', rmse, 0)
            self.writer.close()

        results_folder_path = os.path.join('./results/', setting)
        os.makedirs(results_folder_path, exist_ok=True)
        np.save(os.path.join(results_folder_path, 'metrics.npy'), np.array([r2, mae, mse, rmse]))
        np.save(os.path.join(results_folder_path, 'pred.npy'), preds.numpy())
        np.save(os.path.join(results_folder_path, 'true.npy'), trues.numpy())

        # --- Save test affinities ---
        if len(expert_affinities) > 0:
            test_static_features = np.concatenate(static_features, axis=0)
            test_expert_affinities = np.concatenate(expert_affinities, axis=0)
            np.savez_compressed(os.path.join(results_folder_path, 'test_affinities.npz'),
                                static_features=test_static_features,
                                expert_affinities=test_expert_affinities)
            print("Saved test affinity data.")

        return
