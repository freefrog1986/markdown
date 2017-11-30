
# 安装Tensorflow on Mac
## 决定如何安装Tensorflow
- virtualenv
- "native" pip
- Docker
- installing from sources, 另外一个单独的指南[guide](https://www.tensorflow.org/install/install_sources).

推荐使用[virtualenv](https://virtualenv.pypa.io/en/stable)安装。

## 通过virtualenv安装   
1. 打开terminal 
2. 输入以下命令安装pip和virtualenv
```
$ sudo easy_install pip   
$ pip install --upgrade virtualenv
```
3. 输入以下命令之一创建virtualenv环境
```
$ virtualenv --system-site-packages targetDirectory # for Python 2.7
$ virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
```


### 参考资料   
[Install Tensorflow on mac](https://www.tensorflow.org/install/install_mac)