from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # The loss is computed inside the model for pre-training
        return None

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_t = []
            train_loss_f = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # The model's forward for pretrain returns: loss, x_t_pred, x_f_pred, loss_t, loss_f
                outputs = self.model(batch_x, None, None, None)
                loss = outputs[0]
                loss_t = outputs[3]
                loss_f = outputs[4]

                train_loss.append(loss.item())
                train_loss_t.append(loss_t.item())
                train_loss_f.append(loss_f.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_t: {3:.7f} | loss_f: {4:.7f}".format(
                        i + 1, epoch + 1, loss.item(), loss_t.item(), loss_f.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("---------------------------- Epoch {} ------------------------".format(epoch + 1))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_t = np.average(train_loss_t)
            train_loss_f = np.average(train_loss_f)
            
            vali_data, vali_loader = self._get_data(flag='val')
            vali_loss, vali_loss_t, vali_loss_f = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            print("    Train Loss(t): {0:.7f}, Train Loss(f): {1:.7f} | Vali Loss(t): {2:.7f}, Vali Loss(f): {3:.7f}".format(
                    train_loss_t, train_loss_f, vali_loss_t, vali_loss_f))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader):
        total_loss = []
        total_loss_t = []
        total_loss_f = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                loss = outputs[0]
                loss_t = outputs[3]
                loss_f = outputs[4]

                total_loss.append(loss.item())
                total_loss_t.append(loss_t.item())
                total_loss_f.append(loss_f.item())

        total_loss = np.average(total_loss)
        total_loss_t = np.average(total_loss_t)
        total_loss_f = np.average(total_loss_f)
        
        self.model.train()
        return total_loss, total_loss_t, total_loss_f

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        test_loss, test_loss_t, test_loss_f = self.vali(test_data, test_loader)

        print_str = f"Test Loss: {test_loss:.7f} | Test Loss(t): {test_loss_t:.7f} | Test Loss(f): {test_loss_f:.7f}"
        print(print_str)
        
        with open(self.args.logs_path, 'a') as f:
            print(f"Test Loss: {test_loss:.7f} | Test Loss(t): {test_loss_t:.7f} | Test Loss(f): {test_loss_f:.7f}", file=f)
            print(f"Pretrain finished! Pre-trained model saved in:", file=f)
            print(f"./checkpoints/{setting}/checkpoint.pth", file=f)

        return
