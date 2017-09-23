**文章结构**
- 神经网络的关键问题：过拟合
    + 什么是过拟合
    + 什么原因导致了过拟合
- 防止过拟合的方法
- Python实现
=================================
### 1.神经网络的关键问题：过拟合
简单来说，正则化（Regularization）的目的是防止过拟合（overfitting）。
#### 1.1 什么是过拟合？
先放图：
![过拟合说明]("http://img.blog.csdn.net/20170414112819079?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTEFXXzEzMDYyNQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center")
上图来自于吴恩达老师机器学习课程，第一张图是“欠拟合”（underfit），第三图是过拟合（overfit），第二张图是正好的状态。
有图可以看出来，过拟合就是训练的模型与数据集匹配的太完美，以至于“过了”。
过拟合的危害是：模型不够“通用”，也就是说如果将模型应用在新的数据上，得到的效果不好。
举例说明，如果一个模型在训练集上的准确率达到99.9%，而将模型应用在测试集时，准确率只有80%，很明显这是一个过拟合的状态。
#### 1.2 什么原因导致了过拟合？
**数据问题**
数据太少了，模型没有足够多的意外数据用来使模型更加“通用”。
**模型问题**
神经网络模型的复杂度太高了！
以至于模型的复杂程度高于问题的复杂程度！
或者说，复杂模型的拟合能力太强拥有了拟合噪声的能力。
**算法问题**
1. 模型权重太多
2. 模型的权重分配差异太大
3. 权重的取值太大
权重太多，导致，模型的复杂度太大。
而模型的权重分配差异约大，模型越不够平滑。
想像一个例子，模型a的所有参数都是1，模型b的参数是[1，9999]范围内随机取值，暂不考虑准确度问题，前一个模型一定比后一个平滑的多。
关于权重取值大小的问题，先看下图
[激活函数]("https://pic3.zhimg.com/50/v2-f3121f5af646c8a5a1e239594557098e_hd.png")
无论是图中的sigmoid函数还是Relu函数，当权重较小时，激活函数工作在线性区，此时神经元的拟合能力较弱。
综上所述，我们得出结论：
1. 模型权重数量不宜太多
1. 权重的取值范围应尽量一致
2. 权重越小拟合能力越差

### 2.防止过拟合的方法  
本篇文章重点介绍一下两种方法：
1. 限制网络结构（Dropout）：这个很好理解，减少网络的层数、神经元个数等均可以限制网络的拟合能力；
2. 限制权重 ：Weight-decay或正则化regularization，保持较小的权重有利于使模型更加平滑。
**Dropout**
[Dropout]("https://pic1.zhimg.com/50/v2-9dfdb3a81e5aaa377831f8deb36d5814_hd.png")
Dropout很好理解，随机“关闭”一些神经元。

**Regularization:**
$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost}$$
Regularization的数学解释如上面的公式所示，在cost函数的后面增加一个惩罚项！如果一个权重太大，将导致代价过大，因此在后向传播后，就会对该权重进行惩罚，使其保持一个较小的值。
ok，简单介绍了原理，下面是python实现。
### 3.Python实现
#### 3.1 regularization
确切的说，我们采用的是L2 regularization，L2代表什么呢，代表求权重的欧几里得范数，也叫做L2范数。   
公式如下：$$||x||_2 = \sqrt{\sum_{i=1}^{n}x_i^2}$$
关于矩阵范数的详细内容请看[深度学习中的线性代数2：矩阵的操作和性质]("https://zhuanlan.zhihu.com/p/28926180")   
根据采用了L2 regularization后的代价函数公式，我们可以写出计算代价的函数：   
```python
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost = (1/m)*(lambd/2)*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
```
为例使代码更加简单清晰，这里假设我们采用的是三层的神经网络，代价函数公式可以看做包含两部分：交叉熵部分和L2正则化部分！   
分别计算上述两部分后相加即可得到最红的代价。   
这里注意W是矩阵运算，因此求和和求平方利用numpy的运算：np.sum和np.square
另外需要注意的是，公式中多了一个参数$\lambda$，这个参数是人为设置的超参数（hyper parameter），用来人为调节大小进而优化模型性能。   
想象一下，既然L2正则化的本质是在原有代价的基础上增加了一个‘惩罚项’，那么如果$\lambda$越大，这个‘惩罚’的力度也就越大，越能够使我们的模型更加平滑，也就更能防止过拟合！
由于我们更新了代价函数的公式，因此代价函数对权重的求导也随着发生了改变。   
也就是说，权重的导数在原公式的基础上也增加了'惩罚项'：公式如下：$\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$
增加了L2正则化的反向传播python实现：  
```python
def backward_propagation_with_regularization(X, Y, cache, lambd):

    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    """ 
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y 
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3 
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```
最后关于如何调整超参数$\lambda$?   
前面说了，调整超参数$\lambda$可以使我们的模型更平滑，但是如果$\lambda$太大也会导致过平滑。
一般来说，我们可通过测试集调整$\lambda$，一般来说，当达到训练集和测试集的正确率接近时，我们认为$\lambda$调整到了合适的值。   
本质上L2正则化是控制权重使其保持较小的值，因此这种方法也叫做weight decay。   
#### 3.2 Dropout
Dropout很好理解，通过在神经网络迭代过程中随机关闭一些神经元达到防止过拟合的目的。
为什么Dropout能够防止过拟合，如果我们在迭代过程中随机关闭一些神经元，那么模型将不会对某一个或一些神经元特别‘敏感’，因为无论哪个神经元随机都有被关闭的风险。
这也就间接的导致权重的取值范围尽量一致。
首先理解一下，神经元如何被“关闭”？   
实践中，我们通过把神经元的输出置0来“关闭”神经元。
具体来说，执行下面4步：
1. 建立一个维度与本层神经元数目相同的矩阵$D^{[l]}$.
2. 根据概率（这里用变量keep_prob代表）将$D^{[l]}$中的元素设置为0或1。
3. 将本层激活函数的输出与$D^{[l]}$相乘作为新的输出。
4. 新的输出除以keep_prob，这一步是为了保证得到的代价与未进行Dropout前一致，想像一下，你有5个1，求和等于5，现在随机删除了1/5的数字，为了保证结果还是5，需要对剩下来的每一个数字都除以1/5。   
下面是前向传播的代码：
``` python
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # 4 steps
    D1 = np.random.rand(Z1.shape[0], Z1.shape[1])     # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                               # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1*D1                                        # Step 3: shut down some neurons of A1
    A1 = A1/keep_prob                                 # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.rand(Z2.shape[0], Z2.shape[1])                                       # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                                # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2*D2                                         # Step 3: shut down some neurons of A2
    A2 = A2/keep_prob                                  # Step 4: scale the value of neurons that haven't been shut down

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
```
这里的一个小技巧是用`D2 = D2 < keep_prob`来生成0或1，因为python中的False和True等于0和1.   
下面要写的代码是反向传播，注意，反向传播也要关闭对应的神经元，同样也需要除以keep_prob。
```python
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = D2*dA2                     # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)

    dA1 = D1*dA1                     # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```
ok，至此，我们的Dropout算法部署完毕！   
### 4.参考资料
[用简单易懂的语言描述「过拟合 overfitting」？]("https://www.zhihu.com/question/32246256")
[机器学习中使用「正则化来防止过拟合」到底是一个什么原理？为什么正则化项就可以防止过拟合？
]("https://www.zhihu.com/question/20700829")
[机器学习中用来防止过拟合的方法有哪些？]("https://www.zhihu.com/question/59201590")
[什么是 L1/L2 正则化 (Regularization)]("https://zhuanlan.zhihu.com/p/25707761")