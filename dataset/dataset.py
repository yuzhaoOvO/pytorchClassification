import argparse
import os

from PIL import Image
from torch.utils.data import *
from model.utils import utils
from torchvision import transforms


parse = argparse.ArgumentParser(description='dataset')


class ClassDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        print("预处理开始")
        self.transform = transforms.Compose([     # 预处理：随机旋转，转换为tensor
            # transforms.RandomRotation(60),
            # 后期加入
            # #transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.transforms.RandomCrop(64),
            transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor()
        ])
        print("预处理结束")
        self.dataset = []
        class_dirs = os.listdir(data_dir)
        for class_dir in class_dirs:
            image_names = os.listdir(os.path.join(data_dir, class_dir))
            for image_name in image_names:
                self.dataset.append(
                    [os.path.join(data_dir, class_dir, image_name),
                     int(config['class_names'].index(class_dir))])
        print()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        image_path, image_label = data
        image = Image.open(image_path)
        image = utils.keep_shape_resize(image, self.config['image_size'])
        return self.transform(image), image_label


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util('E:\Sitting_posture_recognition\pytorchTest\config\config.yaml')
    dataset = ClassDataset(os.path.join("E:\Sitting_posture_recognition\pytorchTest",config['data_dir']), config)

    print(dataset[1410])