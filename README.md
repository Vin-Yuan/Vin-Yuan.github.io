# Vin-Yuan.github.io

access url: https://vin-yuan.github.io/

```shell
git clone xxxx (main repository)
git submodule init # 初始化子模块
git submodule update # 更新子模块
npm install
hexo generate or hexo s
#调试的时候遇到问题，可以安装pandoc, https://pandoc.org/installing.html
#这会指示那一个文件的哪一行出错了
```


## problem

### deploy failed
when use hexo deploy, it shows:
fatal: LF would be replaced by CRLF in 2018/05/10/First-Blog/index.html
FATAL Something's wrong. Maybe you can find the solution here: https://hexo.io/docs/troubleshooting.html
Error: Spawn failed
get solution from: https://hexo.io/docs/troubleshooting.html

remove the .deploy_git and then 
```
hexo clean && hexo generate && hexo deploy
```

### can't git add
when use git add .
fatal: CRLF would be replaced by LF in source/_posts/hexoNote.md
git config --global core.safecrlf false