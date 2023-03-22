## 模型蒸馏原理
概念 -> 把一个大模型或者多个模型ensemble学到的知识迁移到另一个轻量级单模型上，方便部署。简单的说就是用小模型去学习大模型的预测结果，而不是直接学习训练集中的label。  
术语 ->  
- 教师模型：原始大模型
- 学生模型：新的小模型
- hard-label：训练集中标签
- soft-label：教师模型预测输出  
核心思想 -> **好模型的目标不是拟合训练数据，而是学习如何泛化到新的数据**  
理论上，学习soft-label的学生模型比学习hard-label的学生模型效果好  
## 如何蒸馏
1. 直接拿训练结果作为soft-label -> hard-label效果相似
2. 使用概率值 ->   
![](pictures/1.png)  
其中T用来更好的控制输出概率的平滑程度  
学生模型新的loss ->   
![](pictures/2.png)  
CE->交叉熵 y->真实label p->学生预测结果 a->蒸馏权重 T^2将梯度乘回  
## BERT蒸馏
蒸馏提升：  
- 精调阶段蒸馏->预训练阶段蒸馏
- 蒸馏最后一层知识->蒸馏隐层知识->蒸馏注意力矩阵  

### Distilled BiLSTM
将BERT-large蒸馏到单层BiLSTM中，参数量减少100倍，速度提升15倍，效果降低到ELMo  
- 教师模型：精调过的BERT-large
- 学生模型：BiLSTM+ReLU
- loss：CE(hrad-label)/MSE(logits)

### BERT-PKD
提出Patient Knowledge Distillation -> 从教师模型中间层提取知识，避免在蒸馏最后一层拟合过快  
![](https://pic3.zhimg.com/80/v2-190fb9b0d777e2c72ce9f7b74e8a5c3a_720w.webp)  
PT LOSS -> 归一化后的MSE  
- 教师模型：精调BERT-base
- 学生模型：  
           PKD-skip -> BERT-base[2,4,6,8,10]  
           PKD-last -> BERT-base[7,8,9,10,11]  

### DIstillBERT
预训练阶段进行蒸馏，尺寸缩小40%，速度提升60%，效果好于BERT-PKD  
- 教师模型：预训练BERT-base
- 学生模型：6层transformer  
不同点：新增cosine embedding loss，蒸馏最后一层hidden
- loss：MLM loss、CE（最后一层）、cosine loss

### TinyBERT
精调阶段和预训练阶段联合蒸馏  
![](https://pic3.zhimg.com/80/v2-803f1809b6db3ad30d52b6b38878b5ca_720w.webp)  
作者参考其他研究的结论，即注意力矩阵可以捕获到丰富的知识，提出了注意力矩阵的蒸馏，采用教师-学生注意力矩阵logits的MSE作为损失函数（这里不取attention prob是实验表明前者收敛更快）。另外，作者还对embedding进行了蒸馏，同样是采用MSE作为损失。  
整体loss：  
![](https://pic1.zhimg.com/80/v2-4c9703b675a0ed1ba291b86805cfdd2c_720w.webp)  
预训练阶段 -> 只对中间层蒸馏  
精调阶段 -> 先对中间层蒸馏20epoch，再对最后一层蒸馏3epoch  

### MobileBERT
专注于减少每层的维度（基于bottleneck机制）  
![](https://pic2.zhimg.com/80/v2-998207497278455a883bc5081381f1e1_720w.webp)  
其中a是标准的BERT，b是加入bottleneck的BERT-large，作为教师模型，c是加入bottleneck的学生模型。Bottleneck的原理是在transformer的输入输出各加入一个线性层，实现维度的缩放。对于教师模型，embedding的维度是512，进入transformer后扩大为1024，而学生模型则是从512缩小至128，使得参数量骤减。  

### MiniLM  
创新点在于蒸馏Valur-Value矩阵：  
![](https://pic4.zhimg.com/80/v2-a96a5c1332aa89c5ad5f6574074c167b_720w.webp)  
只蒸馏最后一层，只蒸馏这两个矩阵的KL散度。  
助教机制 -> 学生层数维度都小很多：先蒸到助教再给学生  

## BERT蒸馏技巧
### 剪层还是剪维度
预训练蒸馏：剪层+纬度缩减（数据充分）  
只想蒸馏精调BERT：剪层，用教师模型的层对学生模型初始化  
### loss选择
CE/KL/MSE  
### T和a如何设置？
a -> soft-label和hard-label的loss比例，建议soft-label更高  
T -> 控制预测分布的平滑程度，T越大越能学到teacher泛化信息  
### 是否逐层蒸馏？
不建议  
