'''
 _*_coding:utf-8 _*_
 @Time     :2022/5/25  8:00
 @Author   : huangwenlan
 @File     : pytorch_to_onnx.py
 @Software : PyCharm
 '''
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # 同层及上层模块调用

import numpy as np
import torch.onnx
import torch.cuda
import onnx, onnxruntime
from net.net import *
from utils import utils
import argparse

parse = argparse.ArgumentParser(description='pack onnx model')
parse.add_argument('--config_path', type=str, default=r'..\config\config.yaml', help='配置文件存放地址')
parse.add_argument('--weights_path', type=str, default=r'..\w\test\weight\best.pth', help='模型权重文件地址')


# resnet101_mc

def pack_onnx(model_path, config):
    model = ClassifierNet(config['net_type'], len(config['class_names']),
                          False)
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    batch_size = 1
    input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    torch.onnx.export(model,
                      input,
                      config['net_type'] + 'ddd.onnx',
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}
                      },
                      verbose=True
                      )
    print('onnx打包成功！')

    output = model(input)
    onnx_model = onnx.load(config['net_type'] + 'ddd.onnx')
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(config['net_type'] + 'ddd.onnx')# 'InferenceSession' +

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy

    ort_input = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_output = ort_session.run(None, ort_input)

    np.testing.assert_allclose(to_numpy(output), ort_output[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util(args.config_path)
    pack_onnx(args.weights_path, config)
