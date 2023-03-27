### 混合精度训练
通常模型使用float32进行训练。  
是否可以全使用float16？ -> 不合适  
float16优点：  
- 降低显存占用
- 减少网络通信的开销
- 硬件对float16又优化，速度更快  
float16缺点：  
- 下溢：相比于float32大多数权重不再更新，模型难以收敛
- 舍入误差：权重和梯度相差太大，梯度更新时权重无变化  
--> 混合精度训练  
--> 模型权重、梯度使用float16，优化器参数使用float32，优化器还要保存一份float32权重。  
具体训练过程：  
- 使用float16权重进行前向传播；
- 反向传播得到float16的梯度；
- 通过优化器计算出float32精度的权重更新量；
- 更新float32权重；
- 将float32权重转换为float16；  
NV apex混合精度训练工具
- O0：纯float32精度训练，可作为参照的baseline； ​
- O1：根据黑白名单自动决定使用float16还是float32(推荐)；
- O2：绝大多数都使用float16，除了batch norm；
- O3：纯float16，训练不稳定； 

### 显存占用分析
#### 主要显存消耗
一个参数量为X（INT8）的模型，并使用Adam作为优化器，模型参数和梯度使用Float16->2X+2X。Adam会维护一个float32的模型副本，消耗4X。Adam还会维护两个状态变量v和r，由于v和r均是float32，所以显存占用为4X+4X。  
总的来说  
- 模型：2X+2X=4X  
- Adam：4X+4X+4X=12X  
则总显存消耗4X+12X=16X。GPT2 1.5B参数，显存消耗至少24GB。  
#### 剩余显存消耗
- 激活（Activations）：前向传播时z=h(x)，完成反向g(z)前要保存z。降低方法：Activation checkpointing or Activation recomputation --> 33%重计算，变为总激活的均分。  
- 临时缓存区：临时buffers消耗大量显存。例如在all-reduce时，需要一个平坦的buffer来融合所有的梯度，从而改善吞吐量。例如，跨设备的all-reduce操作会随着消息的增大而增加。虽然，梯度本文是fp16的张量，但是有些操作中可能需要融合的buffer为fp32。当模型尺寸很大时，临时的buffer也不小。例如，对于1.5B参数的模型，一个fp32的buffer需要6GB的显存。
- 显存碎片：即使在有足够显存的情况下，也可能会导致Out of Memory，这是由于显存碎片导致的。在进程发出显存请求时，如果没有连续的显存来满足请求，即使总的显存仍然足够，该请求也会失败。当训练非常大的模型时，可以观察到明显的显存碎片。极端情况下，可能会导致30%的显存碎片。

### 分布式训练系统架构
1. Parameter Server Architecture  
节点分为：  
- parameter server：存放模型参数
- worker：负责计算参数的梯度  
worker从ps获得参数，计算后将梯度返回ps，ps聚合worker梯度，更新参数，参数广播给worker

2. Ring-allreduce架构
各个设备全为worker，形成一个环，没有中心节点来聚合所有worker计算的梯度。在一个迭代过程中，每个worker完成自己的mini-batch训练，计算出梯度，并将梯度传递给环中的下一个worker，同时也接受上一个worker的梯度。对于一个包含N-1个worker的环，各个worker需要收到其它N-1个worker的梯度后就可以更新模型参数。  
算法的基本思想是取消Reducer，让数据在gpu形成的环内流动，整个ring-allreduce的过程分为两大步，第一步是scatter-reduce，第二步是allgather。  
scatter-reduce ->  
![](https://img-blog.csdn.net/20180918142847567?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p3cWpveQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
allgather ->  
![](https://img-blog.csdn.net/20180918143101598?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p3cWpveQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  


### 数据并行(DP)  
训练数据集太大，无法一次全部加载进内存，因此将数据集分为N个batch，分别装载到N个GPU上进行梯度求导，再将所有结点的求导结果进行加权平均，再sync update给所有节点。 
每张GPU上相同参数，不同数据。  
GPU1 GPU2上都有完整模型，但使用不同的子数据集进行训练，每次每个节点推导结果加权平均后同步到所有GPU节点上，再进行下一轮迭代。  
#### 数据并行中需要注意的问题：  
1. SyncBatchNorm  
global batch size被切分到不同的进程上，每个进程上只有部分输入数据，则计算平均值和方差未使用global，造成精度下降。  
SyncBatchNorm -> 额外同步通信（计算完local 后 同步到sum）  
2. 数据切分均匀  
- local batch size大小相同  
- 进程上分配到相同的batch数量  
#### 优化技巧：  
1. 通信融合（Fuse Allreduce）  
同步梯度通过进程间Allreduce实现，可能一个step中有很多Allreduce通信。  
通信耗时从通信延迟和数据传输时间两方面考虑。通信延迟相对固定，传输时间由数据量和带宽决定。为减少总通信消耗可以减少通信频率，因此使用通信融合。  
通信融合 -> 将N个梯度的Allreduce通信合并成一次Allreduce通信，可以减少N-1次通信延迟时间。  
![](https://pics6.baidu.com/feed/2e2eb9389b504fc28a9d8eb18f26561892ef6d4c.jpeg@f_auto?token=5d72c3ee2aba74914d906649d172c24b)  

![](https://pics1.baidu.com/feed/0823dd54564e9258fa7cee62f7796051cebf4ef2.jpeg@f_auto?token=e7ddb6efdeb541b997b42108be7a3971)  
2. 通信计算重叠  
通信和计算的异步流水实现 ->   
数据并行中的梯度同步Allreduce通信是在训练的反向过程中进行的，而Allreduce 后得到的同步梯度是在训练的更新过程中才被使用，在反向中并没有被使用。也就是说上一个梯度的通信和下一个梯度的计算间并没有依赖，通信和计算可以并行，让两者的耗时相互重叠掩盖，减少反向的耗时。  
![](https://pics2.baidu.com/feed/21a4462309f79052268e11e3670866c379cbd55b.jpeg@f_auto?token=ea597da756d0ba50f17907db3f22386b)  
![](https://pics5.baidu.com/feed/77094b36acaf2edd412003fee8ebb0e03b0193dd.jpeg@f_auto?token=fdb5f45d3b989e5a25af5c7287639311)  

#### ZeRO-DP
来源：《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》  
在标准的数据并行中，每个显卡(rank)都会保存独立的权重、梯度和优化器状态，如上图中的baseline所示。那么每个显卡是否有必要存储全部的这些信息呢？ZeRO-DP的答案是**不需要**。ZeRO-DP能够对模型状态(权重、梯度和优化器状态)进行划分(不像标准DP那样进行复制)，然后通过动态通信调度来最小化通信开销。ZeRO-DP能够在保持整体通信开销接近标准DP的同时，线性地降低模型的单显卡显存占用。  
ZeRO-DP可以分为三个阶段：Pos/Pg/Pp。三个阶段对应优化器状态划分、梯度划分和模型参数划分，并且三个阶段可以叠加使用(上图展示了三个阶段的叠加)。  
![](https://pic1.zhimg.com/80/v2-cee54b33e803d98fc0ddbbe341ec8ee8_720w.webp)  

ZeRO offload
- 将部分数据和计算资源转移到CPU上；
- 将优化器参数同样分散到多个CPU上；
- 将CPU上状态更行与Step N+ 1 的GPU上的计算重叠起来；
- 40TFOPs/GPU on V100 for 10B model (在CPU上计算优化器状态比较耗时，所以将通信时间进行重叠。)

### 模型并行(MP)
模型太大，无法将整个模型载入一个GPU，因此将模型分为N个部分，分别加载到不同的N个GPU节点上。模型1000层，GPU0->1-100层，GPU1->101-200层以此类推。前向传播时GPU0->GPU1->GPU2....，反向传播时GPU9->GPU8->GPU7....  
Megatorn-LM  
- 按行切分
- 按列切分  

### 流水线并行（PP）
基于模型并行，一个batch结束前开始下一个batch，以充分利用计算资源。将模型按层进行切分，将不同的层放入不同的GPU，训练的时候数据像流水一样在GPU上进行流动。  
切分方式：按层切分（流水线并行）、层内切分（模型并行）。  

### 混合并行（HP）
混合使用上述方法

补充知识来源列表：  
[[论文翻译] 分布式训练 Parameter sharding 之 ZeRO]([https://www.runoob.com](https://www.cnblogs.com/rossiXYZ/p/15785669.html))
