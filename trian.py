import argparse
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
from loss.loss_fun import Loss
from net.net import ClassifierNet
from optimizer.optim import Optimizer
from dataset.dataset import *
from utils import utils

parse = argparse.ArgumentParser(description='train_demo of argparse')
parse.add_argument('--weights_path', default=r'w\test\weight\best.pth')


class Train:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(config['model_dir']):  # 是否有该模型文件夹
            os.makedirs(config['model_dir'])
        self.summary_writer = SummaryWriter(config['log_dir'])  # 加载日志文件夹
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否使用cpu进行训练
        self.net = ClassifierNet(self.config['net_type'], len(self.config['class_names']),
                                 self.config['pretrained']).to(self.device)  # 网络类型 在config里设置
        self.loss_fun = Loss(self.config['loss_type']).get_loss_fun()  # 损失函数
        self.optimizer = Optimizer(self.net, self.config['optimizer']).get_optimizer()  # 优化器
        self.dataset = ClassDataset(self.config['data_dir'], config)  # 初始化
        self.train_dataset, self.test_dataset = random_split(self.dataset,
                                                             [int(len(self.dataset) * config['train_rate']),
                                                              len(self.dataset) - int(
                                                                  len(self.dataset) * config['train_rate'])])
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train(self, weights_path):
        print(f'device:{self.device}  训练集:{len(self.train_dataset)}  测试集:{len(self.test_dataset)}')
        if weights_path is not None:  # 加载已有模型
            if os.path.exists(weights_path):
                self.net.load_state_dict(torch.load(weights_path))
                print('successfully loading model weights!')
            else:
                print('no loading model weights')
        temp_acc = 0
        for epoch in range(1, self.config['epochs'] + 1):  # 训练
            self.net.train()
            with tqdm.tqdm(self.train_data_loader) as t1:
                for i, (image_data, image_label) in enumerate(self.train_data_loader):
                    image_data, image_label = image_data.to(self.device), image_label.to(self.device)

                    out = self.net(image_data)
                    if self.config['loss_type'] == 'cross_entropy':
                        train_loss = self.loss_fun(out, image_label)
                    else:
                        train_loss = self.loss_fun(out, utils.label_one_hot(image_label).type(torch.FloatTensor).to(
                            self.device))
                    t1.set_description(f'Train-Epoch {epoch} 轮 {i} 批次 : ')
                    t1.set_postfix(train_loss=train_loss.item())
                    time.sleep(0.1)
                    t1.update(1)
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()
                    if i % 10 == 0:
                        torch.save(self.net.state_dict(), os.path.join(self.config['model_dir'], 'last.pth'))
                self.summary_writer.add_scalar('train_loss', train_loss.item(), epoch)
            self.net.eval()
            acc, temp = 0, 0

        self.net.eval()  # 验证
        acc, temp = 0, 0
        with torch.no_grad():
            with tqdm.tqdm(self.test_data_loader) as t2:
                for j, (image_data, image_label) in enumerate(self.test_data_loader):
                    image_data, image_label = image_data.to(self.device), image_label.to(self.device)

                    out = self.net(image_data)
                    if self.config['loss_type'] == 'cross_entropy':
                        test_loss = self.loss_fun(out, image_label)
                    else:
                        test_loss = self.loss_fun(out, utils.label_one_hot(image_label).type(torch.FloatTensor).to(
                            self.device))
                    out = torch.argmax(out, dim=1)
                    test_acc = torch.mean(torch.eq(out, image_label).float()).item()
                    acc += test_acc
                    temp += 1
                    t2.set_description(f'Test-Epoch {epoch} 轮 {j} 批次 : ')
                    t2.set_postfix(test_loss=test_loss.item(), test_acc=test_acc)
                    time.sleep(0.1)
                    t2.update(1)
                print(f'Test-Epoch {epoch} 轮准确率为 : {acc / temp}')
                if (acc / temp) > temp_acc:
                    temp_acc = acc / temp
                    torch.save(self.net.state_dict(), os.path.join(self.config['model_dir'], 'best.pth'))
                else:
                    torch.save(self.net.state_dict(), os.path.join(self.config['model_dir'], 'last.pth'))
                self.summary_writer.add_scalar('test_loss', test_loss.item(), epoch)
                self.summary_writer.add_scalar('test_acc', acc / temp, epoch)


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util('config/config.yaml')
    Train(config).train(args.weights_path)

    #  FNN
