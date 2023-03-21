import argparse

import torch


from net.net import ClassifierNet
from utils import utils

parse = argparse.ArgumentParser(description='pack onnx model')
parse.add_argument('--config_path', type=str, default=r'..\config\config.yaml', help='配置文件存放地址')
parse.add_argument('--weights_path', type=str, default=r'..\w\test\best.pth', help='模型权重文件地址')

def save(model_path, config):
    # An instance of your model.
    model = ClassifierNet(config['net_type'], len(config['class_names']),
                          False)
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    model.load_state_dict(torch.load(model_path, map_location=map_location))

    # Switch the model to eval model
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save the TorchScript model
    traced_script_module.save(config['net_type'] +"_model.pt")
    print("saveModel/"+config['net_type'] +"_model.pt"+"--------------导出完成---------------------")


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util(args.config_path)
    save(args.weights_path, config)
