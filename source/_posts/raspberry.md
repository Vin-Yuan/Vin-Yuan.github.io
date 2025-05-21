---
title: raspberry
mathjax: true
date: 2024-06-04 15:19:57
categories:
tags:
---


## 安装

64-bit镜像：
<https://downloads.raspberrypi.org/raspios_lite_arm64/images/raspios_lite_arm64-2021-05-28/>

## 获取树莓派ip

树莓派在接入路由器后是动态分配的，需要使用路由器的管理界面查看链接设备的MAC和ip地址
另外一种方式是扫描当前局域网的设备信息，比如可以使用IOS的Fing App去发现当前连接设备，一般树莓派的设备名称是Raspberry开头的

## 挂载移动硬盘

<https://shumeipai.nxez.com/2013/09/08/raspberry-pi-to-mount-the-removable-hard-disk.html>

```python
sudo mount -o uid=pi,gid=pi /dev/sda1 /mnt/1GB_USB_flash
```
<!-- more -->

## 安装samba

<https://www.cnblogs.com/xiaowuyi/p/4051238.html>

## DLNA

<https://shumeipai.nxez.com/2015/07/12/raspberry-pi-install-dlna-streaming-media-server.html>

## 开机执行指定脚本

例如脚本路径/home/pi/Documents/start.sh

```sh
#!/bin/sh

mount -o uid=pi,gid=pi /dev/sda1 /home/pi/Share
/etc/init.d/smbd restart
/etc/init.d/nmbd restart
```

然后修改文件：/etc/rc.local
sudo vi /etc/rc.local
在exit 0之前添加要执行脚本的命令即可，例如

```sh
# /etc/rc.local
……
sh /home/pi/Documents/start.sh
exit 0
```

## 格式化U盘

格式化硬盘（U盘）
1.树莓派成功识别硬盘
sudo fdisk -l | grep '^Disk'
2.查看硬盘格式
sudo blkid
3.格式化为ext4
sudo mkfs.ext4 /dev/sda
挂载硬盘（U盘）
1.建立挂载点
sudo mkdir /media/xxx        #xxx代表你的挂载点名称，可以自定义
2.设置目录的所有人和所有组
sudo chown pi:pi /media/xxx
3.挂载
sudo mount -t ext4 /dev/sda /media/xxx
4.确认挂载
cd /media/xxx
5.卸载
sudo umount /media/xxx
6.开机自动挂载
查看UUID：sudo blkid
记下UUID：将UUID复制下来（不要带引号）
备份文件：sudo cp /etc/fstab /etc/fstab.bakup
编辑文件：sudo nano /etc/fstab
最后行加上： UUID=@@@ （@@@表示刚刚复制下来的UUID）（注意此处有一个空格）/media/xxx ext4 defaults 0 0

## To Be Continue

安装<https://pigallery2.herokuapp.com/gallery/>
<https://github.com/bpatrik/pigallery2/blob/master/docker/README.md>
<https://github.com/bpatrik/pigallery2>

## 出现问题

docker 安装完成后测试hello-world出现问题（Unable to find image 'hello-world:latest' locally）
<https://blog.csdn.net/wireless911/article/details/88989620>
