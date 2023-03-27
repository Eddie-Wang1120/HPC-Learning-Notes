## 量化基础知识
### 2.1. 硬件背景
![](https://pic4.zhimg.com/80/v2-b94f85e47c487bff11948e26018cadbb_720w.webp "y = Wx + b 示意图")
此为乘累加示意图，公示如下：  

![](https://pic4.zhimg.com/80/v2-0f45a1d2786542062c213066ef7edd03_720w.webp)

神经网络使用FP32权重和激活进行训练 -> 处理单元（PE即图中Cn,m）和累加器（ACC即图中A）必须支持浮点运算，同时32bit写回内存。I/O占据大部分功耗。  

使用INT8 -> 减少数据传输量，降低MAC规模和能耗。

上图公式转化->  
![](https://pic2.zhimg.com/80/v2-8d3840df2ec529e600408200f8c1c041_720w.webp)

神经网络加速器结构变化为->  
![](https://pic1.zhimg.com/80/v2-8f42eb2c1a14c17d9f20f410acb1a088_720w.webp)
- 保证累加器位宽 -> 避免积累误差造成溢出损失
- 重新量化 -> 减少数据传输和下一层操作复杂性（下一层也要用INT8）

### 2.2. 
#### 2.2.1. 均匀仿射量化（非对称量化）
三个参数定义，分别为：
- 比例因子s（通常为浮点数）  ->  将浮点值映射到整数范围内
- 零点z （整数）      ->  将浮点值映射到整数范围内，确保真实零点(真实的0)的量化没有错误
- 比特宽度b  -> 在实际操作的时候一般是没有位宽这个选项的，因为大多硬件已经定好了支持8bit还是4bit，不能支持任意bit的选择  

1. 进行量化操作  
![](https://pic4.zhimg.com/80/v2-1d54a5024c2b90fd8ee9742a78248b33_720w.webp)  
其中clamp为四舍五入取整
2. 反量化
![](https://pic2.zhimg.com/80/v2-4303bdf27a7ed0bc2409461e35fe3c31_720w.webp)  

总定义为：  
![](https://pic2.zhimg.com/80/v2-d0677cd2ecc4cd0e9b53260824735ad5_720w.webp)  
理解为：
- 比例因子：按比例缩放至量化后表示范围内
- 量化和反量化操作：使浮点数靠近量化后相似的整数数值  

如何确定量化范围极限？  
反量化： x = s(xint - z) = s*xint - s*z 其中 xint >=0 && xint <= 2^(b-1)  
故 min -> -sz / max -> s(2^b-1 - z)  

超出量化范围的数值被截断 -> 截断误差
减少截断误差 -> 增大比例因子 -> 增大舍入误差  
trade-off between clipping error and rounding error  

#### 2.2.2. 对称均匀量化（对称量化）
是非对称量化的简化版本 -> 将零点z限制为真是的零  
- 减少对z计算额外开销
- 限制整数和浮点数的映射范围
- 可选择有符号整数和无符号整数
![](https://pic3.zhimg.com/80/v2-7a9be98f75e5c037c7e25c40e8ae113a_720w.webp)  
对称有符号、对称无符号、非对称  
![](https://pic4.zhimg.com/80/v2-6a2d09427984ce914582a867d44764db_720w.webp)  
无符号对称量化适用于单尾分布（RELU）  
有符号对称量化适用于零对称分布（TANH）

#### 2.2.3. 二次幂量化
特例：比例因子被限制为二次幂，即s=2^(-k)，k为整数，此时对s的缩放为位移，但会使trade-off变复杂  

#### 2.2.4. 量化粒度
- 为每个张量定义一组权重量化参数和激活量化参数 -> 按张量量化（per tensor / per layer）
- 对张量的各个部分定义一个单独的量化器 -> 按通道量化（per channel）
- 按组。。。

### 2.3. 量化模拟
![](https://pic1.zhimg.com/80/v2-9516cb38c03b0fd32d5515b8e1955214_720w.webp) 

#### 2.3.1. 批量归一化的折叠(Batch normalization folding)
将批量归一化融合到前一个或者后一个线性层中  
- 减少额外的缩放和偏移计算  
- 省去额外的数据搬移和输出层的量化  
![](https://pic1.zhimg.com/80/v2-1faeeeb386dde59fe366b04dfea0b5e8_720w.webp) 
演示了如何融合到前一个线性层中 -> 将BatchNorm(Wk;x)转换为y=Wx+b的形式  

#### 2.3.2. 激活函数融合
反量化在矩阵乘法或卷积运算（线性层）之后 -> 将线性层结果加载到非线性层计算 -> I/O浪费
解决办法：  
反量化之前进行非线性操作 -> relu（去掉负半轴）、sigmoid/swish（专门支持-泰勒或LUT表）

#### 2.3.3. 其它层量化
- Max pooling：输入输出范围一致（不改变最大最小值），无需激活量化
- Average pooling：平均后需再量化一次（平均后不一定为整数）
- Element-wise addtion：两个输入量化范围必须完全匹配  
                        -- 增加一个反量化步骤  
                        -- 绑定多个量化器
- Concatenation：需要反量化或fine-tuning

#### 2.4.1. 对称和非对称量化
非对称量化有更好的表达能力，但会导致更多的计算开销  
非对称权重与非对称激活相乘  
![](https://pic4.zhimg.com/80/v2-324f5e6a6ba64a633d6354106f816f27_720w.webp) 
第二项取决于输入数据x，故每次都要重新计算，因此 -->  
对激活使用非对称，对权重使用对称（避免第二项）

#### 2.4.2. 按张量还是按通道
- 按行 -> 对权重按通道量化 -> 对每个权重通道使用不同缩放系数  
- 按列 -> 对激活按通道量化 -> 无法将缩放系数从求和中简单的分解出来，每列输入的改变会影响所有行的结果，也就是每个累加器都会受到输入的影响，因此一旦输入数据变量所有的累加器都要去进行一个抵消操作

## 3. 训练后量化（Post-training quantization）PTQ
无需训练过程将预训练FP32转为定点网络 -> data-free or calibration  
                                  -> 几乎无需调整超参数  

### 3.1. 量化范围的设置（Quantization range setting）
即确定qmin和qmax，权衡截断误差和舍入误差  
- Min-Max  
![](https://pic3.zhimg.com/80/v2-d2135fb27e39e4a8ee2791e6cead8a4a_720w.webp)  
对异常值敏感，可能导致过多舍入误差
- MSE
![](https://pic2.zhimg.com/80/v2-de12d37f43b4908a71ea88049cabde01_720w.webp)  
缓解强离群值问题
- Cross entropy
![](https://pic3.zhimg.com/80/v2-6b94888a53c571f97d2c393794d185ee_720w.webp) 
避免MSE对关键类和较小相关类一视同仁的缺点，考虑量化的值不同等重要（softmax等）。
- BN based range setting
![](https://pic2.zhimg.com/80/v2-584f42d2964d9066223a767abc61d38d_720w.webp)  
用BN的均值和标准差找合适的量化参数  

Comparision  
![](https://pic2.zhimg.com/80/v2-45c1111bd61a5f1f7ba40b6668a5db85_720w.webp)   
![](https://pic2.zhimg.com/80/v2-3505cc452113cbba73b9938a97381a1d_720w.webp) 

### 3.2. 跨层均衡化
同一张量中元素有明显不同的大小（BN会增加这种效果 --> BN折叠使权重进一步改变） --> 导致量化误差（对per-channel问题不大，对per-tensor影响很大）  
克服方案：  
对于RELU、PRELU等，按比例缩放关系成立：  
![](https://pic2.zhimg.com/80/v2-7eac1b5236724ff2d6930644344ce495_720w.webp) 
对于两个连续的层 -->  
![](https://pic1.zhimg.com/80/v2-784105b7035ac657d4b536421474c4e4_720w.webp) 
S为一个对角矩阵，～W(2)=W(2)S，～W(1)=S^-1W(1)和～b(1)=S^-1b(1)  
![](https://pic3.zhimg.com/80/v2-578d3a4696f1718afe47bc3ed6231b06_720w.webp) 
可以找到一个缩放因子 si 使得重新缩放层中的量化噪声最小，通过设置 S 来实现最优的权重均衡  
![](https://pic3.zhimg.com/80/v2-a93dc4fa5e1a86ebd7862c1e6548ce52_720w.webp) 
其中ri是权重张量 j 的通道 i 的动态范围

``` python
def equalize(weight1, bias1, weight2):
 # 重排列
 weight2 = weight2.permute(1, 0, 2, 3)
 out_channel = weight1.shape[0]
 for i in range(out_channel):
    r1 = compute_range(weight1, i)  # 计算kernel数值范围
    r2 = compute_range(weight2, i)
    s =  r1 / sqrt(r1 * r2)
    weight1[i] = weight1[i] * (1. / s)
    weight2[i] = weight2[i] * s
    bias1[i] = bias1[i] * (1. / s)
  # 调整回之前的数据排布
 weight2 = weight2.permute(1, 0, 2, 3)
 return weight1, weight2
```

Abosrbing hgih biases 吸收高偏差  
某些情况下，CLE后高偏差对导致激活动态范围不同 --> 可能的话将高偏差吸收到下一层  
![](https://pic3.zhimg.com/80/v2-5bb063ba378621227a29d003b149a2ae_720w.webp) 
高偏差获得方式 -->   
![](https://pic2.zhimg.com/80/v2-7e8f2eb594d0b3025e4a8348b7300a95_720w.webp) 
其中minx是在一个小的校准集上获得的

### 3.3. 偏差校正（Bias Correction）
量化误差的偏差主要因素往往是截断误差（少数截断过大）
预期输出分布：  
![](https://pic4.zhimg.com/80/v2-9e669c62e062138cfdb52f73b9a7aa6f_720w.webp) 
偏差即E[∆Wx]，由于∆W是常数，我们有E[∆Wx] = ∆WE[x]。  
为了抵消这种偏移，我们可以从输出中减去它  
![](https://pic2.zhimg.com/80/v2-8b1b2deb1cb48124bd6cdaa528274485_720w.webp) 
校正项和偏差项具有相同形状，无需额外开销。  
计算偏差校正项的方法：经验偏差校正和分析偏差校正
- Empirical bias correction  
需要校准数据集，通过比较量化模型和全精度模型的激活来计算  
![](https://pic4.zhimg.com/80/v2-9b1065d52adfb0ba1d9a9d024588628f_720w.webp)   
- Analytic bias correction
无需数据，使用前一层的BN统计量计算预期的输入分布E[x]  
![](https://pic3.zhimg.com/80/v2-6bf4ece01c0e9f425536014d85a54e06_720w.webp) 

### 3.4. 自适应取整（AdaRound）
权重通常由FP32映射到最近的量化网格点 --> 四舍五入（MSE最小）  
四舍五入并不是最佳选择 
![](https://pic3.zhimg.com/80/v2-7632bd00b97ec2258a1b3dc601dee402_720w.webp) 
AdaRound --> 为PTQ寻找好的权重舍入选择的方法
![](https://pic1.zhimg.com/80/v2-f4978c09d2ec4ab9cac3073e221abfc4_720w.webp) 
其中ˆx是该层的输入，前面所有的层都被量化，fa是激活函数。(35)的目标可以使用随机梯度下降法进行有效和高效的优化。这种优化权重舍入的方法被称为AdaRound。

### 3.5. 标准的PTQ流程
![](https://pic4.zhimg.com/80/v2-93b0c85105b2005a70f6b35a11f5e65b_720w.webp) 

### 3.6. 实验
![](https://pic3.zhimg.com/80/v2-65a19fc5d2cae01aed4937e9f5591d66_720w.webp)

### 3.7. 调试
![](https://pic1.zhimg.com/80/v2-8b0f4cdb9b2911b06076c44983510390_720w.webp)

## 4. 训练时量化（Quantization-aware training QAT）
- PTQ --> 非常有效、计算速度快、无需带标签数据 / 对激活低比特量化（<4bit）有局限性
- QAT --> 对量化噪声源进行建模，效果更优 / 训练成本、时间更长，需要带标签数据集，超参搜索

### 4.1. 反向传播模拟 --> 针对量化模块
四舍五入后梯度处处为零或未定义，基于梯度的训练无法正常进行 --> 直通估计器 --> 将舍入算子的梯度近似为1 -->   
![](https://pic2.zhimg.com/80/v2-3a73f6f8eaa0e3925f689e099408eaf5_720w.webp)
可使用该近似值计算量化操作梯度  
--> 对称量化  
![](https://pic3.zhimg.com/80/v2-f57b6baf85832a039d1fe9ebb98842da_720w.webp)    
前向传播相同，反向传播有效跳过  
![](https://pic4.zhimg.com/80/v2-5669a384a6ecc35f061c56fc5f443297_720w.webp)  
对s的梯度：  
![](https://pic3.zhimg.com/80/v2-101c6be5694eb04a40f7ee772c787762_720w.webp)  
对z的梯度： 
![](https://pic3.zhimg.com/80/v2-4be8358daf902bc416de9e668bb0d63a_720w.webp)  

### 4.2. 批量归一化折叠和QAT
在QAT中建立BN折叠模型的一个简单而有效的方法是将BN缩放和偏移量静态地折叠到线性层的权重和偏置中。这相当于权重的重新参数化，并有效地从网络中完全删除了批量规范化操作。当从一个收敛的预训练模型开始时，静态折叠是非常有效的

### 4.3. QAT的初始化
**Effect of range estimation 量化范围的影响**  
![](https://pic3.zhimg.com/80/v2-8d1f0004a6d417fa03201862dddba0fa_720w.webp)  
![](https://pic1.zhimg.com/80/v2-ab353647cc8dab1c57df42508adbabfc_720w.webp)  
更好的初始化可以导致更好的QAT结果，但是收益通常很小，并且在训练持续的时间越长，收益就会微弱甚至消失。  

**Effect of CLE CLE的影响**  
对于那些使用普通PTQ有严重问题的模型，我们可能需要更先进的PTQ技术，如CLE来初始化QAT。但是在大多数其它情况下，改进的PTQ初始化只对最终QAT性能有着微小的改善。  

### 4.4. 标准的QAT流程
![](https://pic4.zhimg.com/80/v2-3c9ccb9685a7e292670ccb90e8a526bb_720w.webp)  

### 4.5. 实验
![](https://pic2.zhimg.com/80/v2-bcbcdedb8fb82c519b58854b173ee7d5_720w.webp)  
深度分离可卷积？  

## 5. 总结和结论
 - 训练后量化（PTQ）技术将预先训练好的FP32网络转换为固定点网络，而不需要原始训练流程。这使得它们成为一种轻量级的、一键式的量化方法，具有低工程量和计算成本。我们描述了PTQ的一系列最新进展，并介绍了一个PTQ流程，该流程使各种模型和机器学习任务的量化精度可以接近浮点模型精度。特别的，使用所提出的流程，我们可以在所有网络的浮点精度1%之内实现权重和激活的8位量化。我们进一步表明，许多网络甚至可以量化到4位权重，而仅有很小的额外性能下降。此外我们还介绍了一个调试工作流程，以有效地识别和解决量化新网络时可能出现的问题。
 - 量化感知训练（QAT）通过模拟量化操作对训练期间的量化噪声进行建模。与 PTQ 相比，此训练过程可以找到更好的量化解决方案，同时实现更有效和更激进的激活量化。与 PTQ 类似，我们引入了使用该领域最新算法的标准训练流程。我们还特别关注 QAT 期间的批量归一化折叠，并表明简单的静态折叠优于其他计算成本更高的方法。我们证明，通过我们的QAT流程，我们可以实现权重的4位量化，对于某些模型甚至可以实现4位激活量化，而且与浮点运算相比，精度只有小幅下降。

## 6.混合精度量化
8bit以上量化基本无损，8bit以下还在刷SOTA。  
混合精度量化对模型进行进一步的高效压缩，实现Accuracy和(FLOPS && Parameters)之间的trade-off。  
定义：通过设置某种policy，对模型各层的weights和activations的量化位宽进行合理分配。  
![](https://pic1.zhimg.com/80/v2-b96290645d6540205c2a71316ae1f628_720w.webp)   
