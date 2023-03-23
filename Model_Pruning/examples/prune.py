import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from ResNet import ResNet,BasicBlock,Bottleneck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认 num_classes=10
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

model = ResNet18().to(device)
# model.load_state_dict(torch.load("CIFAR10_ResNet18.pth"))
module = model.conv1
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))

# prune weights
module = prune.random_unstructured(module, name="weight", amount=0.3)
# print(list(module.named_buffers()))
# print(list(module.weight))
# print(module._forward_pre_hooks)

module = prune.l1_unstructured(module, name="weight", amount=3)
# print(list(module.named_buffers()))

# recursion prune
module = prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
# print(module.weight)

prune.remove(module, "weight")
# print(list(module.named_parameters()))

new_model = ResNet50()
for name, module1 in new_model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module1, torch.nn.Conv2d):
        prune.l1_unstructured(module1, name="weight", amount=0.2)
    # prune 40% of connections in all linear layers
    elif isinstance(module1, torch.nn.Linear):
        prune.l1_unstructured(module1, name="weight", amount=0.4)

# print(dict(new_model.named_buffers()).keys())

# global pruning
global_model = ResNet50()
parameters_to_prune = (
    (global_model.conv1, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3
)

print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(global_model.conv1.weight == 0))
        / float(global_model.conv1.weight.nelement())
    )
)
