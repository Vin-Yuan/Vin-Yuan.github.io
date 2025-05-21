---
title: conda
mathjax: true
date: 2024-02-29 19:12:02
categories:
tags:
---



conda 是 Anaconda 的工具箱，它是 pip 和 vitualenv 的组合，也就是说他可以像pip来管理包，也可以像vitualenv来切换环境

#### installation

<https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart>

#### 在conda中使用pip

注意：当在conda的虚拟环境中使用Pip安装包时，需要使用`pip -V`查看Pip所使用的路径，如果conda没有安装Pip，会使用系统默认的pip命令，这种结果使得pip安装的包被安装到了系统库位置，从而在当前虚拟环境中的python下无法使用或找不到。
一般conda的虚拟环境中自带pip，如果你是用pip3安装，可能使用的是系统的pip，这一点要注意

```shell
(face_detector) ➜  ~ pip -V
pip 9.0.1 from /opt/anaconda3/envs/face_detector/lib/python3.5/site-packages (python 3.5)
(face_detector) ➜  ~ pip3 -V
pip 20.0.2 from /usr/local/lib/python3.7/site-packages/pip (python 3.7)
```

的开发环境会被默认安装在你conda目录下的envs文件目录下。可以指定一个其他的路径；去通过 conda create -h了解更
果我们没有指定安装python的版本，conda会安装我们最初安装conda时所装的那个版本的python。

```python
# 列举当前所有环境
conda info
conda env list
# 创建环境
conda create --name new_env_name python=2.7.9
conda create -n new_env_name python=2.7.9
# 克隆环境 (例如当前环境是base, 需要克隆一个copy_base，地址在~/path)
conda create -n copy_base --clone ~/path
# 激活环境
source activate snowflakes  #linux
activate new_env_name #windows
# 释放环境
source deactivate #linux
deactivate #windows
# 移除环境
conda remove --name new_env_name --all
conda remove -n new_env_name --all
# 保存环境\分享环境
conda env export > environment.yml
# 恢复环境
conda env create -f environment.yml
# 查看当前环境所有package
conda list
# 为指定环境安装某个包
conda install -n env_name package_name
# 查找包有哪些版本
conda search tensorflow-gpu 
# 将conda放入PATH
eval "$(/home/yuanwenwu/anaconda3/bin/conda shell.YOUR_SHELL_NAME hook)"
```

* 在python2.7环境中启动notebook 使kernel变为python3、python2共存

```python
# 进入python2虚拟环境，执行下面语句，然后启动jupyter notebook即可
python -m ipykernel install --user
```

* 问题1：python能找到的包，jupyter notebook找不到

是因为python执行路径不一致。
定位这个问题可以通过sys包

```python
import sys
print(sys.executable)
```

往往这个问题是只安装了python，但要使用ipython或jupyter notebook，由于虚拟环境没有，会去主系统找可用版本呢，从而导致启动路径不一致，解决方式是在虚拟环境conda install需要的包（ ipython or jupyter notebook )
参考：<https://blog.csdn.net/sunxinyu/article/details/78801534>

#### conda 更换源

#### conda 安装指定version包

```python
conda search tensorflow
conda install tensorflow-gpu==2.0.0
```
