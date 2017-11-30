# ANACONDA
## Anaconda介绍
Anaconda是免费的、易于安装的、专门为数据科学开发用于管理环境和packages的工具。
Anaconda可以在Windows, macOS或者Linux等平台使用。
使用Anaconda可以解决如下问题：
- 工作中既需要Python2又需要Python3的环境。
- 安装工具包很麻烦。
- 需要的工具包又依赖其他工具包。

注：pip是python的默认packages管理工具，Conda与pip的区别是，Conda专注于数据科学而pip是通用包管理工具。

Anaconda来源于conda，你可以使用conda来创建独立的环境包含不同版本的python和packages。还可以使用其来安装、升级、卸载工具包。

一句话总结：使用Anaconda使数据处理工作更美好！

## 安装Anaconda
安装比较简单，这里给出教程链接：https://www.continuum.io/downloads

## 管理你的packages   
**注意：以下的命令均在mac系统terminal下使用**
**查看当前环境的packages列表：**
```
$ conda list
```

**安装numpy、pandas、matplotlib包：**
```
$ conda install numpy、pandas、matplotlib
```

**安装指定版本的包：**
```
$ conda install numpy=1.10
```

**安装jupyter notebook**
```
$ conda install jupyter notebook
```

**升级指定安装包和升级全部安装包**
```
$ conda upgrade conda
```

**卸载安装包**
```
$ conda remove package_name
```

**搜索包**
```
conda search search_term
```


## 管理你的enviroment   
**创建环境**
创建环境的话命令如下`$ conda create -n env_name list of packages`其中`env_name`是你要创建的环境的名称，`list of packages`是你要随着环境一起安装的包。例如，如果我们想创建一个名字是`env_name`的环境，并安装`python3`的话，使用下面的命令：
```
$ conda create -n env_name pathon=3 
```

**进入你的环境**
```
$ source activate env_name
```

**离开环境**
```
$ source deactivate
```

**保存环境**
使用命令
```
$ conda env export > environment.yaml
```
其中`conda env export`是输出所有环境文件，`> environment.yaml`是保存到environment.yaml文件。

**读取环境**
如何从yaml文件创建环境：
```
$ conda env create -f environment.yaml
```

**列出所有的环境**
```
$ conda env list
```

**删除环境**
```
$ conda env remove -n env_name
```
其中env_name是环境名称。