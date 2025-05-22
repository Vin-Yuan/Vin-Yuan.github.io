# Vin-Yuan.github.io

access url: https://vin-yuan.github.io/

```shell
git clone xxxx (main repository)
git submodule init # 初始化子模块
git submodule update # 更新子模块
or
git clone --recurse-submodules (main repository)
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
```
git config --global core.safecrlf false
```

### remote page build failed
github pages build deployment failed
"Logging at level: debug Configuration file: /github/workspace./_config.yml Theme: next github-pages 232 | Error: The next theme cound not be found
```shell
# local run
npm install hexo-deployer-git --save
# then 
hexo deploy
```

### hexo deploy后再pull 提示fatal: refusing to merge unrelated histories
说明你本地分支（通常是 Hexo 的源码分支，比如 main）和远程部署分支（如 master 或 gh-pages）之间的 Git 提交历史不一致，没有共同祖先。这是因为 Hexo 部署时往往用的是 public/ 目录单独作为另一个分支的内容，并用 --force 推送过去
最干净的做法是：

1. 将 Hexo 的源码放在 main 分支；

2. 将部署生成后的 public/ 推送到 gh-pages 分支；

3. GitHub Pages 设置为从 gh-pages 分支部署；

4. 两者完全隔离，不需要 pull 也不会冲突。

具体做法：  
1. 修改 _config.yml：

```yaml
deploy:
  type: git
  repo: git@github.com:Vin-Yuan/Vin-Yuan.github.io.git
  branch: gh-pages
```
2. 清除旧的部署缓存：

```bash
rm -rf .deploy_git
```
3. 重新部署：

```bash
hexo clean
hexo generate
hexo deploy
```
4. 在 GitHub 项目的页面中：  

进入 Settings > Pages  
设置 Source 为 gh-pages 分支

### git pull 来自于gh-pages 
when git pull出现如下信息
 * branch            gh-pages   -> FETCH_HEAD
Already up to date.
这说明你本地分支（如 main 或其他）当前的默认“跟踪分支”是远程的 gh-pages，而不是 master。
```shell
git branch --set-upstream-to=origin/master main
# 或者
git fetch origin
git branch -u origin/master
```
