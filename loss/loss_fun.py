

from torch import nn


class Loss:
    def __init__(self, loss_type='mse'):
        self.loss_fun = nn.MSELoss()
        if loss_type == 'mse':
            self.loss_fun = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fun = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fun = nn.SmoothL1Loss()
        elif loss_type == 'cross_entropy':
            self.loss_fun = nn.CrossEntropyLoss()

    def get_loss_fun(self):
        return self.loss_fun
