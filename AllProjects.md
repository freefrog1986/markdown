#项目整理
##Python
###Python Basics With Numpy
**项目摘要**
利用numpy编写深度学习中的常用功能：sigmoid函数、sigmoid求梯度、矩阵归一化、softmax函数、L1、L2代价函数。理解numpy库与math库的区别，理解numpy中的“广播”，理解向量化的作用。学习常用函数`np.shape`、`np.reshape`、`np.dot`、`np.outer`、`np.multiply`等。
[完整项目报告地址：Python Basics With Numpy](https://freefrog1986.github.io/Neural-Networks-and-Deep-Learning/Neural%20Networks%20Basics/PythonBasicsWithNumpy.html)


##Deep Learning
###Tensorflow Tutorial
**项目摘要**
学习和使用Tensorfolow、PaddlePaddle、Torch、Caffe、Keras等深度学习框架库能够加速算法的开发和部署效率。此文是上手探索热门框架库Tensorfolow的基础教程，本文中主要练习了如何初始化变量、运行一个session、训练一个神经网络算法以及部署该算法。
[完整项目报告地址：Tensorflow Tutorial](https://freefrog1986.github.io/Deep-Learning-Specialization/Improving%20Deep%20Neural%20Networks/TensorflowTutorial.html)

###Optimization Methods
**项目摘要**
改善神经网络的基本方法是利用优化算法并选择适当的超参数，也就是‘调参数’的过程。本文主要讨论了随机梯度下降、mini-Batch梯度下降与梯度下降的异同，优化梯度下降路径的Momentum算法，结合RMSProp和Momentum的Adam算法，利用‘moon’测试集对比测试上述算法对神经网络性能的影响。
[完整项目报告地址：Optimization Methods](https://freefrog1986.github.io/Deep-Learning-Specialization/Improving%20Deep%20Neural%20Networks/Optimization.html)

###Initialization of Neural Network Weights
**项目摘要**
搭建神经网络的第一步也是非常关键的一步就是初始化网络权重，初始化的过程要考虑到矩阵维度、赋值方法、灵活性（对于优化算法的灵活调整）等。好的初始化方法应该能够加速梯度的收敛，并且增大使梯度收敛到最小值的概率。本文比较了赋0值法、随机初始化、He初始化等方法，进而总结经验。
[完整项目报告地址：Initialization of Neural Network Weights](https://freefrog1986.github.io/Deep-Learning-Specialization/Improving%20Deep%20Neural%20Networks/Initialization.html)

###Regularization of Deep Neural Network
**项目摘要**
神经网络，尤其是深度神经网络，相较于传统的机器学习算法有更强的性能提升和适应性，但是随之而来算法‘过拟合’问题非常严重，导致在训练集表现良好的模型在新的实际数据面前表现很差。解决‘过拟合’的方法称为'正则化（Regularization）'。本文通过对一个足球运动员数据集应用神经网络进行开球位置预测的例子，应用和比较利用典型的泛化的方法能够在多大程度上改善‘过拟合’的问题，对于本文来讲，典型的正则化方法采用了L2 Regularization和Drop out。最后通过对比上述方法的实际结果，得到结论：‘正则化’能够有效的改善神经网络过拟合问题。
[完整项目报告地址：Regularization of Deep Neural Network](https://freefrog1986.github.io/Deep-Learning-Specialization/Improving%20Deep%20Neural%20Networks/Regularization.html)

###Gradient Checking
**项目摘要**
后向传播是搭建神经网络算法的重要一步！在这一过程中，涉及到了大量的、繁琐的对各种函数求梯度的过程，设计并实现这一过程非常复杂和富有挑战性，为了尽量避免出现错误或产生bug，很有必要设计梯度检查器。梯度检查器是利用求导的数学定义对梯度的计算结果进行验证，本文讨论了梯度检查器的工作原理，为什么采用梯度检查器而不是在后向传播过程中利用求导的定义进行计算。最后动手设计了一维检查器，进而实现多维梯度检查器。
[完整项目报告地址：Gradient Checking](https://freefrog1986.github.io/Deep-Learning-Specialization/Improving%20Deep%20Neural%20Networks/GChecking.html)

###Bird recognition in the city(case study of Machine Learning strategy)
**项目摘要**
在搭建成功的机器学习项目过程中如何把握正确的工作方向意义重大！作为项目负责人，其中一项最重要的工作是保证团队的工作方向没有‘偏航’。这其中的关键方法就是应用正确的工作策略，其中包括：如何设置正确的测量标准、如何合理分配数据、关注数据的统计分布、设定正确的人类水平作为性能目标、加速工作效率等。当然，如何选择这些策略视实际任务情况而定。
本文接下来的内容主要是一个关于识别鸟类的案例实践，
该文的内容来源于MOOC网站Coursera上的课程course Structuring Machine Learning Projects。
How to set right direction in the process of building up a successful machine Learning Projects is significant. As a leader of project, the main task is to make sure your team aren't moving away from your goals. The key method is adopt appropriate strategies including setting metrics, structuring your data, considering dataset distribution, choosing optimal methods, defining right human-level performance, Speeding up your work etc. Making decicions of those methods based on actual conditions.
The following content actually are case study about recognition of birds in city.
This case study is origenally from a test in coursera. You can find it in course Structuring Machine Learning Projects.
[完整项目报告地址：Bird Case](https://freefrog1986.github.io/Deep-Learning-Specialization/Structuring%20Machine%20Learning%20Projects/BirdCase.html)

### Logistic Regression with a Neural Network mindset
**项目摘要**
该项目采用神经网络的模式解决逻辑回归问题，具体来说，我们将要搭建一个逻辑分类器用于识别图片中的猫。搭建一个神经网络模型的一般步骤是1.初始化权重。2.计算代价和梯度。3.应用优化算法。文章按照上述步骤逐步搭建好模型的各功能模块，最终组合在一起打造一个功能简单的神经网络模型，利用该模型对猫的图片进行学习，我们将看到算法逐步自我‘学习’和‘迭代优化’的过程，并能够达到训练集99%，测试集70%的准确率！
[完整项目报告地址：Logistic Regression with a Neural Network mindset](https://freefrog1986.github.io/Deep-Learning-Specialization/Neural%20Networks%20and%20Deep%20Learning/LogisticNN.html)

### Planar data classification with one hidden layer
**项目摘要**   
非线性分布的数据很难用传统的逻辑回归模型解决，然而采用神经网络模型能够给我们提供一个简单有效的解决方法。本项目通过搭建具有1层隐藏层的神经网络来解决‘花朵数据’的分类问题。具体来说，本项目介绍并采用非线性的tanh函数作为激活函数，采用交叉熵作为代价函数，部署前向和后向传播。最终组合所有‘零件’在‘花朵数据’上进行训练和测试，结果表明，传统的逻辑回归模型只有47%的准确率（还不如靠扔骰子瞎猜的50%！），而我们的1层神经网络模型能够达到至少90%的准确率！ 
[完整项目报告地址：Planar data classification with one hidden layer](https://freefrog1986.github.io/Deep-Learning-Specialization/Neural%20Networks%20and%20Deep%20Learning/oneLayer.html)

### Building your Deep Neural Network: Step by Step
**项目摘要**
一步步打造L层神经网络模型，具体来说，构建了初始化权重、线性函数前向传播、激活函数前向传播、计算代价、线性函数后向传播、激活函数后向传播、权重优化等功能函数，所有功能函数的设计都满足能够搭建一个L层的神经网络模型。
[完整项目报告地址：Building your Deep Neural Network: Step by Step](https://freefrog1986.github.io/Deep-Learning-Specialization/Neural%20Networks%20and%20Deep%20Learning/NNfun.html)

### Deep Neural Network for Image Classification: Application
**项目摘要**
此项目的目的是构建一个L层神经网络模型，用于识别猫的图片。此项目使用的功能函数全部来自于项目[“Building your Deep Neural Network: Step by Step”](https://freefrog1986.github.io/Deep-Learning-Specialization/Neural%20Networks%20and%20Deep%20Learning/NNfun.html)。作为比较，我们构建了2层和5层神经网络进行比对，实验结果显示，5层神经网络的准确率为80%大于2层神经网络的72%。
[完整项目报告地址：Deep Neural Network for Image Classification: Application](https://freefrog1986.github.io/Deep-Learning-Specialization/Neural%20Networks%20and%20Deep%20Learning/NNmodel.html)

## Machine Learning
### Boston Housing Prices prediction
**项目摘要**
在此项目中，我们通过在“波士顿郊区房屋”数据集进行训练和测试机器学习模型，进而评估该模型的性能和预测能力。通过该数据集得到的模型可以用于评估一个房屋的合理价格，对于在日常工作中使用此类信息的职业，例如房产经纪人，搭建并应用此模型的价值是显而易见的。
“波士顿郊区房屋”数据集来自于UCI Machine Learning Repository，该数据收集于1978年，共506条来自波士顿郊区的各类房屋的数据，数据集共包含14个相关特征。
本文的结构包含四部分：数据探索，开发模型，分析模型，评估模型。第一步，通过探索数据得到数据的重要特征和统计信息。第二步，我们将数据划分为训练集和测试集，并设定合适的性能指标。第三步，通过改变参数值和训练集的大小分析算法的性能，这将用于帮助我们选择泛化行较好的最优模型。第四步，通过在真实数据上应用我们的模型来评估表现。
本项目来自于[Udacity的机器学习纳米基石学位](https://cn.udacity.com/mlnd)。

In this project, a machine learning model has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts，to evaluate the performance and predictive power. A model trained on this data that is seen as a good fit could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.

The dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represents aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. 

The structure of this project consists of four parts: Data Exploration, Developing a Model, Analyzing Model Performance, Evaluating Model Performance. Exploring the data to obtain important features and descriptive statistics about the dataset is the first step. Next, we split the data into testing and training subsets, and determine a suitable performance metric for this problem. Then this paper analyzes performance graphs for a learning algorithm with varying parameters and training set sizes. This will help us to pick the optimal model that best generalizes for unseen data. Finally, evaluating model performance by applying this model on a new sample and compare the predicted selling price to your statistics.

This project comes from [Machine Learning Foundation Nanodegree of Udacity](https://cn.udacity.com/mlnd).

[完整项目报告地址：Boston Housing Prices prediction](https://freefrog1986.github.io/Boston-Housing-Prices-prediction/boston_housing/report.html)


(https://freefrog1986.github.io/Deep-Learning-Specialization/Convolutional%20Neural%20Networks/Auto%20drive.html)


https://freefrog1986.github.io/Boston-Housing-Prices-prediction/boston_housing/report.html/

https://freefrog1986.github.io/Applied-Data-Science-with-Python/Introduction%20to%20Data%20Science%20in%20Python/Week%252B3.html

https://github.com/freefrog1986/Applied-Data-Science-with-Python

https://github.com/freefrog1986/Applied-Data-Science-with-Python/tree/master/Introduction%20to%20Data%20Science%20in%20Python


## 人脸识别
Face verification: 人脸确认
Face recognition: 人脸识别
one-shot learning 问题: learning from one example to recognize the person again. 每个人的训练集大小只有1.
“similarity” function：degree of diffeerence betweeen images，两个图片的相似程度, 解决one-shot learning.
Siamese network: 两幅图片输入两个相同的神经网络，最后一层不要代入sigmoid函数，而是作为两幅图片的encoding，然后计算两个向量的距离norm，该网络的目标是当输入图片是同一个人时，使两者的距离D小，相反则距离值大。该网络是DeepFace的一部分。
### Triplet Loss   
Triplet Loss: 同时看anchor positive and negative image。定义max（(fa-fp）^2-(fa-fn)^2+alpha,0), 意思：只要`(fa-fp）^2-(fa-fn)^2+alpha`小于0即可，否则后向传播改善参数。
margin： alpha防止得到这样的解：cnn的输出永远是0.
choose the triplet A P N:不要随机，选难度大的，否则网很难进行“学习”。
### logistic classification

##神经风格转移生成绘画艺术品（Art generation with Neural Style Transfer）
### 项目背景
Leon A. Gatys于2015年8月26日“A Neural Algorithm of Artistic Style”。
本项目：
- 部署神经风格转移算法。
- 利用算法生成艺术风格图片。
用于实现风格转移的神经网络与传统神经网络的不同是，该网络不是通过优化cost函数得到参数值，而是得到像素值。
将一副法国卢浮宫的摄影照片转换为莫奈的《亚嘉杜的罂粟田》印象派风格。

### 实现过程
搭建VGG-19网络，通过事先训练的网络得到low level features (at the earlier layers) and high level features (at the deeper layers).
1. 创建“内容代价函数”: J_content(C, G)，该函数用于帮助我们得到内容图片与生成图片之间的相似度程度。
2. 创建“风格代价函数”: J_style(S, G)，该函数用于帮助我们得到风格图片与生成图片之间的相似度程度。通过不同的隐藏层组合得到风格costs。
3. 组合“内容代价函数”和“风格代价函数”得到“生成代价函数”
### 项目结果



### 参考资料 
[CSDN人脸识别](http://blog.csdn.net/neu_chenguangq/article/details/52983093)
[Siamese network](DeepFace closing the gap to human level performance 2014)
[FaceNet](FaceNet: A unified embedding for face recognition and clustering)
[Zeiler and Fergus, 2013, Visualizing and Understanding convolutional network]


Most of the algorithms you've studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, you'll optimize a cost function to get pixel values!

ImageNet database


