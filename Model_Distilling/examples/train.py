import torch
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from torch import optim
from ResNet18 import ResNet18
from ResNet50 import ResNet50

BATCH_SIZE = 512                        # 超参数batch大小
EPOCH = 30                              # 总共训练轮数
save_path = "./CIFAR10_ResNet50.pth"    # 模型权重参数保存位置
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    # CIFAR10数据集类别
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                         # 创建GPU运算环境
print(device)

data_transform = {                      # 数据预处理
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
# 加载数据集，指定训练或测试数据，指定于处理方式
train_data = datasets.CIFAR10(root='./CIFAR10/', train=True, transform=data_transform["train"], download=True)
test_data = datasets.CIFAR10(root='./CIFAR10/', train=False, transform=data_transform["val"], download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_data, BATCH_SIZE, False, num_workers=0)

# # 展示图片
# x = 0
# for images, labels in train_data:
#     plt.subplot(3,3,x+1)
#     plt.tight_layout()
#     images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后
#     plt.title(str(classes[labels]))
#     plt.imshow(images)
#     plt.xticks([])
#     plt.yticks([])
#     x += 1
#     if x == 9:
#         break
# plt.show()

# 创建一个visdom，将训练测试情况可视化
# viz = visdom.Visdom()

# 测试函数，传入模型和数据读取接口
def evalute(model, loader):
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


net = ResNet50(len(classes))
net.to(device)                                      # 实例化网络模型并送入GPU
# net.load_state_dict(torch.load(save_path))          # 使用上次训练权重接着训练
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器
loss_function = CrossEntropyLoss()                  # 多分类问题使用交叉熵损失函数

best_acc, best_epoch = 0.0, 0                       # 最好准确度，出现的轮数
global_step = 0                                     # 全局的step步数，用于画图
for epoch in range(EPOCH):      
    running_loss = 0.0                              # 一次epoch的总损失
    net.train()                                     # 开始训练
    for step, (images, labels) in enumerate(train_dataloader, start=0):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()                 # 将一个epoch的损失累加
        # 打印输出当前训练的进度
        rate = (step + 1) / len(train_dataloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\repoch: {} train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch+1, int(rate * 100), a, b, loss), end="")
        # 记录test的loss
        # viz.line([loss.item()], [global_step], win='loss', update='append')
        # 每次记录之后将横轴x的值加一
        global_step += 1

    # 在每一个epoch结束，做一次test
    if epoch % 1 == 0:
        # 使用上面定义的evalute函数，测试正确率，传入测试模型net，测试数据集test_dataloader
        test_acc = evalute(net, test_dataloader)
        print("  epoch{} test acc:{}".format(epoch+1, test_acc))
        # 根据目前epoch计算所得的acc，看看是否需要保存当前状态（即当前的各项参数值）以及迭代周期epoch作为最好情况
        if test_acc > best_acc:
            # 保存最好数据
            best_acc = test_acc
            best_epoch = epoch
            # 保存最好的模型参数值状态
            torch.save(net.state_dict(), save_path)

            # 记录validation的val_acc
            # viz.line([test_acc], [global_step], win='test_acc', update='append')
print("Finish !")

