# Linear-regression
# 线性回归
线性回归输出是一个连续值，因此适用于回归问题。回归问题在实际中很常见，如预测房屋价格、气温、销售额等连续值的问题。

## 1.基本要素
我们以一个简单的房屋价格预测作为例子来解释线性回归的基本要素。这个应用的目标是预测一栋房子的售出价格（元）。我们知道这个价格取决于很多因素，如房屋状况、地段、市场行情等。为了简单起见，这里我们假设价格只取决于房屋状况的两个因素，即面积（平方米）和房龄（年）。接下来我们希望探索价格与这两个因素的具体关系。
### 1.1 模型定义
设房屋的面积为 x1，房龄为 x2​，售出价格为 y。我们需要建立基于输入 x1 ​	
和 x2来计算输出 y的表达式，也就是模型（model）。顾名思义，线性回归假设输出与各个输入之间是线性关系：
y' = x1w1 + x2w2 + b
其中 w1和 w2是权重（weight），b 是偏差（bias），且均为标量。它们是线性回归模型的参数，y' 是线性回归对真实价格 y 的预测或估计。我们通常允许它们之间有一定误差。
### 1.2 模型训练
接下来我们需要通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。这个过程叫作模型训练（model training）。下面我们介绍模型训练所涉及的3个要素。
#### （1）训练数据
我们通常收集一系列的真实数据，例如多栋房屋的真实售出价格和它们对应的面积和房龄。我们希望在这个数据上面寻找模型参数来使模型的预测价格与真实价格的误差最小。在机器学习术语里，该数据集被称为训练数据集（training data set）或训练集（training set），一栋房屋被称为一个样本（sample），其真实售出价格叫作标签（label），用来预测标签的两个因素叫作特征（feature）。特征用来表征样本的特点。
#### （2）损失函数
在模型训练中，我们需要衡量价格预测值与真实值之间的误差。通常我们会选取一个非负数作为误差，且数值越小表示误差越小。一个常用的选择是平方函数。
#### （3）优化算法
当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。本文使用的线性回归和平方误差刚好属于这个范畴。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。
在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。它的算法很简单：先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。
## 2.代码
### （1）导入模块
首先，导入本文中实验所需的包或模块

```python
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
```
### （2）生成数据集
 我们生成数据集，其中features是训练数据特征，labels是标签。
 

```python
'''
制作数据集
'''
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
#  numpy.random.normal(loc=0,scale=1e-2,size=shape) 正态分布，loc为均值，scale为标准差，size为大小
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 加入一些噪声
```

### （3）读取数据
PyTorch提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用Data代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

```python
'''
读取数据
'''
batch_size = 10
# 将训练数据的样本和标签进行组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```
### （4）定义模型
首先，导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。下面先来看看如何用nn.Module实现一个线性回归模型。

输入：
```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)
```
输出：

```python
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
```
我们可以查看模型所有的可学习参数，此函数将返回一个生成器并打印模型参数。
输入：

```python
'''
打印模型参数
'''
for name, parameters in net.named_parameters():
    print(name, ':', parameters, '->', parameters.size())
```
输出：

```python
linear.weight : Parameter containing:
tensor([[0.3901, 0.1191]], requires_grad=True) -> torch.Size([1, 2])
linear.bias : Parameter containing:
tensor([0.6805], requires_grad=True) -> torch.Size([1])
```

### （5）初始化模型参数
在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。PyTorch在init模块中提供了多种参数初始化方法。这里的init是initializer的缩写形式。我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。

```python
'''
初始化模型参数
'''
init.normal_(net.linear.weight, mean=0, std=0.01)  # 正态分布初始化，满足mean,std分布
init.constant_(net.linear.bias, val=0)  # 常数初始化，使值为常数val
```

### （6）定义损失函数
PyTorch在nn模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，PyTorch也将这些损失函数实现为nn.Module的子类。我们现在使用它提供的均方误差损失作为模型的损失函数。

```python
'''
定义损失函数
'''
loss = nn.MSELoss()

```
### （7）定义优化算法
同样，我们也无须自己实现小批量随机梯度下降算法。torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。下面我们创建一个用于优化net所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。
输入：
```python
'''
定义优化算法
'''
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

```
输出：

```python
SGD (
Parameter Group 0
    dampening: 0
    lr: 0.03
    momentum: 0
    nesterov: False
    weight_decay: 0
)
```
有时候我们不想让学习率固定成一个常数，那如何调整学习率呢？主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。

```python
'''
调整学习率
'''
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率为之前的0.1倍
```
### （8）训练模型
输入：
```python
'''
训练模型
'''
num_epochs = 10
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        # print(output)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    # print(l, l.item())  l返回的是tensor，l.item()返回的是该tensor的数值
```
输出：

```python
epoch 1, loss: 12.250201
epoch 2, loss: 2.193312
epoch 3, loss: 0.516244
epoch 4, loss: 0.053094
epoch 5, loss: 0.031211
epoch 6, loss: 0.016656
epoch 7, loss: 0.006196
epoch 8, loss: 0.001707
epoch 9, loss: 0.000423
epoch 10, loss: 0.000251
```
下面我们分别比较学到的模型参数和真实的模型参数。我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。学到的参数和真实的参数很接近。
输入：
```python
print(true_w, net.linear.weight)
print(true_b, net.linear.bias)
```
输出：

```python
[2, -3.4] Parameter containing:
tensor([[ 1.9965, -3.3900]], requires_grad=True)
4.2 Parameter containing:
tensor([4.1910], requires_grad=True)
```
