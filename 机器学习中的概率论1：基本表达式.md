<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>此专栏文章随时更新编辑，如果文章还没写完，请耐心等待，正常的频率是日更。
此文章主要是[斯坦福大学CS224n课程“深度学习与自然语言处理”]("http://web.stanford.edu/class/cs224n/"")的学习笔记。
首发于知乎专栏：[深度学习+自然语言处理（NLP）]("https://zhuanlan.zhihu.com/c_123183356")。
该专栏专注于梳理深度学习相关学科的基础知识。
首先是文章结构：
================================================================
1. 基本表达式（Elements of probability)
    1.1 条件概率和独立（Conditional probability and independence）
================================================================
以下是正文：
概率论是关于研究不确定性的科学。很多机器学习算法的推导都依赖于概率论的概念，所以打好概率论基础，对于机器学习非常重要。概率的数学理论非常复杂，本文中我们只涉及一些基本方法。

## 1. 基本表达式（Elements of probability)
首先定义一些基本表达式：
- **样本空间$\Omega$**: 就是一个随机试验的所有可能的输出结果集合。
- **事件空间$F$**: $F$中的元素$A \in F$（事件）是$\Omega$的子集合。
- **概率度量**：满足下面条件的函数$P:F \rightarrow \mathbb{R}$。
    + $P(A) \geq 0$, for all $A \in F$
    + $P(\Omega) = 1$
    + $P(\cup_iA_i)=\sum P(A_i)$，互不相容事件的并集的概率为各事件概率之和。
以上这三条性质，被称为概率公理（**Axioms of Probability**）
**举例说明**:这里以掷一个骰子的试验为例，所有的可能输出结果是$\Omega = (1,2,3,4,5,6)$，这就是一个样本空间。我们假设事件$F$是$F={\varnothing,\Omega}$，$F$中的元素($\varnothing,\Omega$)都是是$\Omega$的子集合。$P(\varnothing)=0,p(\Omega)=1$

**性质**
- $If A \subseteq B \Longrightarrow P(A) \leq P(B)$
- $P(A \cap B) \leq min(P(A),P(B))$
- 一致限（the union bound）：$P(A \cup B) \leq P(A) + P(B)$
- $P(\Omega \setminus A)=1−P(A)$
- （全概率公式 Law of Total Probability)如果$A_1,\dots,A_k$是互不相容事件，且$\cup_{i=1}^{k}A_i=\Omega$，那么$\sum_{i=1}^{k}P(A_k)=1$

### 1.1 条件概率和独立（Conditional probability and independence）
假设$B$是概率不为0的事件。那么在B发生的条件下任何事件A发生的概率为：
$$P(A|B) \triangleq \frac{P(A \cap B)}{P(B)}$$
当且仅当$P(A \cap B)=P(A)P(B)$时，我们说$A$和$B$是相互独立事件。也可以理解为$B$的发生对$A$发生的概率没有任何影响。

OK, 这篇文章仅讨论一些基本概念和定义，下一篇讨论关于随机变量的内容。

参考资料
[LaTeX 各种命令，符号]("http://blog.csdn.net/garfielder007/article/details/51646604")
[机器学习中常见的字母解析及MarkDown代码]("http://blog.csdn.net/sanqima/article/details/51275104?ref=myread")
[Markdown数学符号]("http://blog.csdn.net/zcf1002797280/article/details/51289555")
[概率的公理化定义如何理解？]("https://www.zhihu.com/question/50046323")
[系统学习机器学习之误差理论]("http://blog.csdn.net/app_12062011/article/details/50577717")
[等号上面有一个三角是什么运算符？]("https://www.zhihu.com/question/24302002")

欢迎关注[我的微博](https://weibo.com/goldend/profile?rightmod=1&wvr=6&mod=personinfo)
