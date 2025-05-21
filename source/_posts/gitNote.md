---
title: gitNote
mathjax: true
date: 2024-02-20 13:28:00
categories:
tags: git
---
## 新建（关联）远程分支

```python
### Git global setup
git config --global user.name "yuanwenwu3"
git config --global user.email "yuanwenwu3@jd.com"

### Create a new repository

git clone git@git.jd.com:yuanwenwu3/draw_tensorboard.git
cd draw_tensorboard
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master

### Push an existing folder
cd existing_folder
git init
git remote add origin git@git.jd.com:yuanwenwu3/draw_tensorboard.git
git add .
git commit -m "Initial commit"
git push -u origin master

### Push an existing Git repository
cd existing_repo
git remote rename origin old-origin
git remote add origin git@git.jd.com:yuanwenwu3/draw_tensorboard.git
git push -u origin --all
git push -u origin --tags

### 迁移项目
# 新建一个空的repository, 例如地址为url-destination
# 在当前项目主分支上
git remote add destination ${url-destination}
git push -u origin --all
```
<!-- more -->

## 已经commit的回退到add状态

```python
git reset --soft HEAD~1
```

## 最近一次修改了那些文件（列表形式展示)

```python
 git diff --name-only HEAD^ HEAD
 git log --name-only #修改的文件列表
 git log --stat #修改的文件列表, 及文件修改的统计
 git log --name-status #修改的文件列表, 显示状态
```

## git 查看日志精简

```python
git log --pretty=oneline --author="vinyuan"

15f3e2fa840fa49d0a576eb5e3f63a295a0ea522 Merged PR 1211: fix shopping bpr large scale dag file
c1a323928e399357c427ec507c3b6c7927e3f60f Merged PR 1208: shopping_bpr modify lookbackwindow
945c30534c71fee996757a3fb730cc9f7ee3bf67 Merged PR 1207: shopping bpr fix scope timezone problem
……
```

## Git 补充commit

https://blog.csdn.net/chilun8494/article/details/100645862

```git@ssh.dev.azure.com:v3/bingreco/recommendation/StcaNewsCF
# 第一次commit内容
$ echo 'Hello world' > README.md
$ git add .
$ git commit -m "Add README.md"
$ git log --oneline
c56f680 Add README.md
# 修改文件内容并合并到上一次的commit变更当中
$ echo 'Hello voidint' >> README.md
$ git add .
$ git commit --amend --no-edit
$ git log --oneline
eb6c8cb Add README.md // hash值发生了变化
```

## ssh配置多个git账户

情景：[【参考博客】](https://blog.csdn.net/onTheRoadToMine/article/details/79029331)
同一台机器有多个git账户的问题。在配置git权限时，需要配置ssh-keygen，会运行下面命令：

```python
 ssh-keygen -t rsa -C "vinyuan@microsoft.com"
```

命令会在`~/.ssh/`目录下生成两个文件: `id_rsa` 和 `id_rsa.pub`，然后将`id_rsa.pub`的内容复制到git或者gitLab中的`ssh setting`中即可。但如果有多个用户就会面临如下问题：每个用户都需要配置自己的ssh setting，生成一个私钥就会覆盖之前的 `id_rsa` 和`id_ras.pub`。如何避免？

答案是在 “~/.ssh/config” (如果没有可以建一个）中配置。
~/.ssh/config 的规则可以查看[【参考链接】](https://deepzz.com/post/how-to-setup-ssh-config.html)
常用的参数有如下几个：

```python
Host example                       # 关键词
    HostName example.com           # 主机地址
    User root                      # 用户名
    IdentityFile ~/.ssh/id_ecdsa # 认证文件
    Port 22                      # 指定端口
```

~/.ssh/config 的作用可使你通过别名（即关键词）登录目标服务器，例如`ssh example`而不必 `ssh -p xxx admin@xxx.xxx.xxx.xxx`
这个文件还可以用来管理多git用户。
以两个用户user1, user2为例，先后生成各自的公钥、密钥：

```python
**ssh-keygen -t rsa -C "user1@hotmail.com"**
 填写密钥存放位置：例如"~/.ssh/user1_id_rsa
 ssh-keygen -t rsa -C "user2@gmail.com"
 填写密钥存放位置：例如"~/.ssh/user2_id_rsa
 生成完成后会在~/.ssh/下出现四个文件
 ls ~/.ssh/
.
├── user1_id_rsa
├── user1_id_rsa.pub
├── user2_id_rsa
└── user2_id_rsa.pub
```

之后可以在~/.ssh/config 配置不同用户的ssh连接方式

```python
github user1@hotmail.com
host github.com  #别名，随便定 后面配置地址有用
    Hostname github.com #要连接的服务器
    User user1 #用户名
    IdentityFile ~/.ssh/user1_id_rsa  #密钥文件的地址，注意是私钥

github user2@gmail.com
host user2 #别名，随便定
    Hostname github.com
    User user2
    IdentityFile ~/.ssh/user2_id_rsa
```

使用ssh的ssh-add命令将密钥添加到 ssh-agent 的高速缓存中，这样在当前会话中就不需要再次输入密码了 。

```python
 ssh-agent bash
//A账户的私钥
ssh-add ~/.ssh/user1_id_rsa
//B账户的私钥
 ssh-add ~/.ssh/user2_id_rsa
```

配置完成后可以使用

```python
ssh -T git@user1
会返回 Welcome to GitLab, @user1!
```

来测试连接
每个git project 目录下面都有一个".git/config"文件，里面配置了远程仓库的地址,
这时候，我们需要修改跟密钥对应的地址，上面在配置ssh时，为每个Hostname配置了一个host的别名，这时候，我们就不能使用原来的Hostname来提交了，要用别名来代替Hostname。

>url = git@**github.com**:user2/Sample.git
改成
>url = git@**user2**:user2/Sample.git

可以看到host起到了别名的作用，并且在不同的项目不同的账户可以配置自己的私钥方式，通过配置不同的xxx_id_rsa，达到互不影响的效果。

经过这样配置后，再git push时就会经过不同的公钥，私钥验证而互不影响了

### 实际案例

#### 1. 生成密钥

```python
ssh-keygen -t rsa -C “yuanwenwu3@jd.com”
```

#### 2.堡垒机配置~/.ssh/config

```python
Host yuanwenwu3
    HostName git.jd.com
    User yuanwenwu3
    IdentityFile /home/rec/yuanwenwu3/.ssh/id_rsa
    IdentitiesOnly yes
```

#### 3. 堡垒机git项目下.git/config配置

注意下面`git@git.jd.com`被换成了`git@yuanwenwu3`，对应上面Host名称

```python
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[remote "origin"]
    url = git@yuanwenwu3:yuanwenwu3/dnn-convert-tfrecord.git
    fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
    remote = origin
    merge = refs/heads/master
[user]
    name = yuanwenwu3
    email = yuanwenwu3@jd.com