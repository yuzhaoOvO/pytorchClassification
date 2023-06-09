

from torch import optim


class Optimizer:
    def __init__(self, net, opt_type='Adam'):
        super(Optimizer, self).__init__()
        self.opt = optim.Adam(net.parameters())
        if opt_type == 'SGD':
            self.opt = optim.SGD(net.parameters(), lr=0.01)
        elif opt_type == 'ASGD':
            self.opt = optim.ASGD(net.parameters())
        elif opt_type == 'Adam':
            self.opt = optim.Adam(net.parameters(), lr=0.001)
        elif opt_type == 'AdamW':
            self.opt = optim.AdamW(net.parameters())
        elif opt_type == 'Adamax':
            self.opt = optim.Adamax(net.parameters())
        elif opt_type == 'Adagrad':
            self.opt = optim.Adagrad(net.parameters())
        elif opt_type == 'Adadelta':
            self.opt = optim.Adadelta(net.parameters())
        elif opt_type == 'SparseAdam':
            self.opt = optim.SparseAdam(net.parameters())
        elif opt_type == 'LBFGS':
            self.opt = optim.LBFGS(net.parameters())
        elif opt_type == 'Rprop':
            self.opt = optim.Rprop(net.parameters())
        elif opt_type == 'RMSprop':
            self.opt = optim.RMSprop(net.parameters())

    def get_optimizer(self):
        return self.opt
