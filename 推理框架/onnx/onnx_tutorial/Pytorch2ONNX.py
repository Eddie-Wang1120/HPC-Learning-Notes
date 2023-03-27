import torch
import torchvision

import onnx

# input = torch.randn(1, 3, 224, 224)
# model = torchvision.models.resnet18(pretrained=True)
# torch.onnx.export(model, input, "alexnet.onnx")

model = onnx.load("res18.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))