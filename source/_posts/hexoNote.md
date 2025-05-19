---
title: hexoNote
mathjax: true
date: 2024-02-20 12:44:32
categories:
tags: hexo
---

## 常用hexo 命令

hexo new xxx ==> hexo n xxx, #新建文章
hexo generate ==> hexo g,  #生成静态页面至public目录
hexo server ==> hexo s,  #开启预览访问端口
hexo deploy ==> hexo d,  #部署到GitHub
hexo help  #查看帮助
hexo version  #查看Hexo的版本

## 组合命令

hexo s -g # 生成本地预览
hexo d -g # 生成并上传

## 预览部分文字

Use < !-- more -- > in your article to break your article manually, which is recommended by Hexo.

## 遇到hexo generation 出错处理

检查Markedown (vscode markdown语法检查)

```shell
hexo clean
hexo --debug
hexo generate
```

## clone whole repository in a new machine

<https://iphysresearch.github.io/blog/post/programing/git/git_submodule/>
since there is submodule in main repository ( the theme/next)

```shell
git clone xxxx (main repository)
git submodule init # 初始化子模块
git submodule update # 更新子模块
npm install
hexo generate or hexo s
# 调试的时候遇到不名曲问题，可以安装pandoc, https://pandoc.org/installing.html
# 这会指示那一个文件的哪一行出错了
```

<https://hexo.io/zh-cn/docs/commands.html>

<!-- more -->
## 参考链接

整体配置：<https://learnku.com/articles/45697>

latex: (<https://www.jianshu.com/p/d95a4795f3a8?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation>)

hexo: <https://github.com/next-theme/hexo-theme-next>

hexo post settings: <https://theme-next.js.org/docs/theme-settings/posts>


## Q&A
hexo generate编译出错记录
markdown 文件有不合法的link, 比如:
```python
感叹号+[柏松分部 统计出现车辆数](https://newonlinecourses.science.psu.edu/stat414/sites/onlinecourses.science.psu.edu.stat414/files/lesson52/147882_traffic/index.jpg)
[1]. <https://newonlinecourses.science.psu.edu/stat414/node/241/>
大概率是大括号的问题
<200b>符号
```
