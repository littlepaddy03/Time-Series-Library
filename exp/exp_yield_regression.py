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

# Import new regression models
from models import TimeXer_Regression, TimeMixer_Regression

warnings.filterwarnings('ignore')

class Exp_Yield_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Yield_Regression, self).__init__(args)
        self.writer = None

    def _build_model(self):
        # We override this method to inject our custom regression models
        if self.args.model == 'TimeXer':
            model_class = TimeXer_Regression.Model
        elif self.args.model == 'TimeMixer':
            model_class = TimeMixer_Regression.Model
        else:
            model_class = self.model_dict[self.args.model].Model

        model = model_class(self.args).float()

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
        batch_x, batch_x_static, batch_y, batch_x_static_unnormalized = batch_data
        batch_x = batch_x.float().to(self.device)
        batch_x_static = batch_x_static.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # Handle models that may return a single tensor or a tuple
        model_output = self.model(batch_x, batch_x_static)
        if isinstance(model_output, tuple):
            outputs, aux_loss, affinities = model_output
        else:
            outputs, aux_loss, affinities = model_output, None, None

        return outputs, batch_y, aux_loss, affinities, batch_x_static_unnormalized

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds, trues = [], []
        unnormalized_static_features_list, expert_affinities_list = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                outputs, batch_y, _, affinities, unnormalized_static_features = self._process_one_batch(batch_data)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())
                if affinities is not None:
                    unnormalized_static_features_list.append(unnormalized_static_features.cpu())
                    expert_affinities_list.append(affinities.cpu())

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        r2 = r2_score(trues.numpy(), preds.numpy())
        mae, mse, rmse, _, _ = metric(preds.numpy(), trues.numpy())

        affinity_data = None
        if len(expert_affinities_list) > 0:
            final_static_features = np.concatenate(unnormalized_static_features_list, axis=0)
            final_expert_affinities = np.concatenate(expert_affinities_list, axis=0)
            affinity_data = (final_static_features, final_expert_affinities)

        self.model.train()
        return total_loss, r2, mae, mse, rmse, affinity_data

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
        import json
        CROP_MAP = {1.0: 'Maize', 2.0: 'Rice', 3.0: 'Soybean', 4.0: 'Wheat'}

        preds, trues = [], []
        unnormalized_static_features_list, expert_affinities_list = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                outputs, batch_y, _, affinities, unnormalized_static_features = self._process_one_batch(batch_data)

                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())
                unnormalized_static_features_list.append(unnormalized_static_features.cpu())
                if affinities is not None:
                    expert_affinities_list.append(affinities.cpu())

        preds = torch.cat(preds, 0).numpy()
        trues = torch.cat(trues, 0).numpy()
        static_features_all = np.concatenate(unnormalized_static_features_list, axis=0)
        
        # --- Overall Metrics ---
        r2 = r2_score(trues, preds)
        mae, mse, rmse, _, _ = metric(preds, trues)
        print(f'Test R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

        metrics_data = {
            'overall': {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse},
            'per_crop': {}
        }

        # --- Per-Crop Metrics ---
        print("\n------ Per-Crop Test Metrics ------")
        print(f"{'Crop':<10} | {'R2':<8} | {'MAE':<8} | {'MSE':<8} | {'RMSE':<8}")
        print("-" * 50)

        crop_ids = static_features_all[:, 3]
        for crop_id, crop_name in CROP_MAP.items():
            mask = crop_ids == crop_id
            if np.sum(mask) == 0:
                continue

            preds_crop, trues_crop = preds[mask], trues[mask]

            r2_crop = r2_score(trues_crop, preds_crop)
            mae_crop, mse_crop, rmse_crop, _, _ = metric(preds_crop, trues_crop)

            print(f"{crop_name:<10} | {r2_crop:<8.4f} | {mae_crop:<8.4f} | {mse_crop:<8.4f} | {rmse_crop:<8.4f}")

            metrics_data['per_crop'][crop_name] = {
                'r2': r2_crop, 'mae': mae_crop, 'mse': mse_crop, 'rmse': rmse_crop
            }
        print("-" * 50)

        # --- Logging to Tensorboard ---
        if self.writer:
            self.writer.add_scalar('R2/test', r2, 0)
            self.writer.add_scalar('MAE/test', mae, 0)
            self.writer.add_scalar('MSE/test', mse, 0)
            self.writer.add_scalar('RMSE/test', rmse, 0)
            self.writer.close()

        # --- Saving Results ---
        results_folder_path = os.path.join('./results/', setting)
        os.makedirs(results_folder_path, exist_ok=True)

        with open(os.path.join(results_folder_path, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)

        np.save(os.path.join(results_folder_path, 'pred.npy'), preds)
        np.save(os.path.join(results_folder_path, 'true.npy'), trues)

        # --- Save test affinities ---
        if len(expert_affinities_list) > 0:
            test_static_features = static_features_all
            test_expert_affinities = np.concatenate(expert_affinities_list, axis=0)
            np.savez_compressed(os.path.join(results_folder_path, 'test_affinities.npz'),
                                static_features=test_static_features,
                                expert_affinities=test_expert_affinities)
            print("Saved test affinity data.")

        return
