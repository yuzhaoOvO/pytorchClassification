from net.net import ClassifierNet
from dataset.dataset import *
from utils.utils import *
from PIL import Image
import tqdm
import random


"""
    使用pth进行预测
"""

def init_model(net_type, class_num, model_path):
    model = ClassifierNet(net_type, class_num)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def init_img(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(img_path)
    image = utils.keep_shape_resize(image, size=224)
    transform = transforms.Compose([  # 预处理：随机旋转，转换为tensor
        transforms.ToTensor()
    ])
    img = transform(image)
    img = img.view(1, 3, 224, 224)
    # print(img.shape)
    return img


def pre(model, img):
    output = model(img)
    _, prediction = torch.max(output, 1)
    # 将预测结果从tensor转为array，并抽取结果
    prediction = prediction.numpy()[0]
    return prediction


if __name__ == '__main__':
    net_type = "resnet50"
    class_num = 6
    model_path = "w/test/weight/" + "best.pth"

    dataset = []
    data_dir = "data\\train"
    class_dirs = os.listdir(data_dir)
    for class_dir in class_dirs:
        image_names = os.listdir(os.path.join(data_dir, class_dir))
        for image_name in image_names:
            dataset.append(
                [os.path.join(data_dir, class_dir, image_name),
                 int(['back', 'front', 'head', 'left', 'normal', 'right'].index(class_dir))])

    model = init_model(net_type, class_num, model_path)
    dataset = random.shuffle(dataset)
    acc, temp = 0, 0
    with tqdm.tqdm(dataset) as t1:
        for i in dataset:
            img = init_img(i[0])
            prediction = pre(model, img)
            prediction = int(prediction)
            if prediction == i[1]:
                acc += 1
            temp += 1
            t1.update(1)
    print(f'Test准确率为 : {acc / temp}')

    # print("预测为：",prediction)
