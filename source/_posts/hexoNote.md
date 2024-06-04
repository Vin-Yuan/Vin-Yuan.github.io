---
title: hexoNote
mathjax: true
date: 2024-02-20 12:44:32
categories:
tags:
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

<https://hexo.io/zh-cn/docs/commands.html>

## 参考链接
<!-- more -->
整体配置： <https://learnku.com/articles/45697>
latex:  <https://www.jianshu.com/p/d95a4795f3a8?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation>
hexo: <https://github.com/next-theme/hexo-theme-next>
hexo post settings:  <https://theme-next.js.org/docs/theme-settings/posts>