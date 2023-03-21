import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import onnxruntime
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
from utils import utils
import time

parse = argparse.ArgumentParser(description='onnx model infer!')
parse.add_argument('--demo', type=str,default='image', help='推理类型支持：image/video/camera')
parse.add_argument('--config_path', type=str,default=r'..\config\config.yaml', help='配置文件存放地址')
parse.add_argument('--onnx_path', type=str, default=r'.\resnet50ddd.onnx', help='onnx包存放路径')
parse.add_argument('--image_path', type=str, default=r'..\data\infer_data', help='图片存放路径')
parse.add_argument('--device', type=str, default='cpu', help='默认设备cpu')
parse.add_argument('--batchsize',type=int, default=1,help='batchsize')
parse.add_argument('--label_file', default=r'..\data\infer_data\mc_image_label_new.csv',help="""label file""")


def to_numpy(tensor):
    return tensor.requires_grad if tensor.requires_grad else tensor.cpu().numpy()

def read_file(image_name, path):
    label_name = pd.read_csv(path, sep=",", encoding="utf-8")
    temp = label_name.loc[label_name['Filename'] == image_name]
    if temp['FO'].values[0] == 1:
        num = 0
    elif temp['FS'].values[0] == 1:
        num = 1
    elif temp['OB'].values[0] == 1:
        num = 2
    elif temp['OS'].values[0] == 1:
        num = 3
    else:
        num = 4
    return num


def image_process(image_path, label_file,config):

    imagelist = []
    labellist = []
    images_count = 0
    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        image_name = str(image_file)[-12:]
        transform = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_file)
        image_data = utils.keep_shape_resize(image, config['image_size'])
        image_data = transform(image_data)
        image_data = torch.unsqueeze(image_data, dim=0)
        imagelist.append(image_data)
        images_count = images_count + 1
        # read image label from label_file
        label = read_file(image_name, label_file)
        labellist.append(label)

    return np.array(imagelist), np.array(labellist),images_count

args = parse.parse_args()
config = utils.load_config_util(args.config_path)
imagelist,labellist,images_count = image_process(args.image_path, args.label_file,config)

def onnx_infer_image(args, config):
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    imagelist, labellist, images_count = image_process(args.image_path, args.label_file, config)
    pre_label = []
    if args.device == 'cpu':
        infer_time = []
        for i in range(len(os.listdir(args.image_path))):
            file = os.listdir(args.image_path)[i]
            image_file = os.path.join(args.image_path, file)
            image = Image.open(image_file)

            ort_input = {ort_session.get_inputs()[0].name: to_numpy(imagelist[i])}

            start = time.time()
            ort_out = ort_session.run(None, ort_input)#执行推理

            end = time.time()
            infer_time.append(round(end-start,4))

            out = np.argmax(ort_out[0], axis=1)
            result = config['class_names'][int(out)]
            result = short_to_string(result)
            pre_label.append(out[0])
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(r"C:\Windows\Fonts\HarmonyOS_Sans_SC_Regular.TTF", 35)
            draw.text((10, 10), result, font=font, fill='red')
            #image.show()
            image.save(r'..\data\infer_results\pics\%s'%file)

    elif args.device == 'cuda':
        pass
    else:
        exit(0)
    return labellist, pre_label,infer_time


def short_to_string(st):
    shorts = {
        'FO': "Obstacle",
        "FS": "Displaced joint",
        "OB": "Surface damage",
        "OS": "Lateral cuts",
        "RB": "Cracks breaks"
    }
    return shorts.get(st, None)

def conf_matrix(true_classes,predicted_classes):

    classes = set(true_classes)
    number_of_classes = len(classes)
    cm = pd.DataFrame(
        np.zeros((number_of_classes, number_of_classes), dtype=int),
        index=classes,
        columns=classes)
    for true_label, prediction in zip(true_classes, predicted_classes):
        cm.loc[true_label, prediction] += 1
    return cm.values

if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util(args.config_path)

    labellist, pre_label, infer_time = onnx_infer_image(args, config)
    #统计：准确率Accuracy、召回率、精确率
    #混淆矩阵
    cm = conf_matrix(labellist, pre_label)
    print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    print(FP)
    print(FN)
    print(TP)
    print(TN)



    accuracy = sum(TP) / 1000
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    print("准确率：", accuracy)
    print("召回率：", recall)
    print("精确率：", precision)
    print("e2e最大推理时延：", max(infer_time[1:]))
    print("e2e平均推理时延：", round(np.mean(infer_time[1:]), 4))
