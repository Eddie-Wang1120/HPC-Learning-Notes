from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

model = resnet18(num_classes = 10, pretrained=False)

def evaluate(model, loader, device):
    # correct为总正确数量，total为总测试数量
    correct = 0
    total = len(loader.dataset)
    # 取测试数据
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # validation和test过程不需要反向传播
        model.eval()
        with torch.no_grad():
            out = model(x)       # 计算测试数据的输出logits
            # 计算出out在第一维度上最大值对应编号，得模型的预测值
            prediction = out.argmax(dim=1)
        # 预测正确的数量correct
        correct += torch.eq(prediction, y).float().sum().item()
    # 最终返回正确率
    return correct / total

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

quantization_config = torch.quantization.get_default_qconfig("fbgemm")
# Custom quantization configurations
# quantization_config = torch.quantization.default_qconfig
# quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
quantized_model = QuantizedResNet18(model)
quantized_model.qconfig = quantization_config

# Print quantization configurations
print(quantized_model.qconfig)

# https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
torch.quantization.prepare_qat(quantized_model, inplace=True)

def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=200):

    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_accuracy = evaluate(model, test_loader, device)
    print("Epoch: {:02d} Eval Acc: {:.3f}".format(-1, eval_accuracy))

    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_accuracy = evaluate(model, test_loader, device)

        # Set learning rate scheduler
        scheduler.step()

        print("Epoch: {:03d} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            epoch, train_loss, train_accuracy, eval_accuracy))

    return model

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

train_loader, test_loader = prepare_dataloader(
        num_workers=8, train_batch_size=128, eval_batch_size=256)
train_model(model, train_loader, test_loader, device, 0.1, 30)