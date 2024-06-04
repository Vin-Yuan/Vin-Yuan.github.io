---
title: linux
mathjax: true
date: 2024-02-29 17:57:34
categories:
tags:
---

## 数组

```shell
# 创建数组(注意不需要逗号，以空格分隔)
array=(1 2 3 4)

#获取所有元素
echo ${array[@]}

#获取第一个元素
echo ${array[0]}

#获取数组元素个数
echo ${#array[@]}

#如果某个元素是字符串，还可以通过指定下标的方式获得该元素的长度，如下所示：
echo ${#array[2]}

#因为字符串获取长度如下
str="hello world"
echo ${#str}
```

<!-- more -->

### 将command结果存入数组(不受空格影响)
<https://stackoverflow.com/questions/11426529/reading-output-of-a-command-into-an-array-in-bash>
The other answers will break if output of command contains spaces (which is rather frequent) or glob characters like *, ?, [...].

To get the output of a command in an array, with one line per element, there are essentially 3 ways:

### 1. With Bash≥4 use mapfile—it's the most efficient

mapfile使用方式：<https://wangchujiang.com/linux-command/c/mapfile.html>
箭头紧贴括号是将内部命令转换为临时文件，参照<https://tldp.org/LDP/abs/html/process-sub.html>

```
mapfile -t my_array < <( my_command )
```

### 2. Otherwise, a loop reading the output (slower, but safe)

```
my_array=()
while IFS= read -r line; do
    my_array+=( "$line" )
done < <( my_command )
```

### 3. As suggested by Charles Duffy in the comments (thanks!), the following might perform better than the loop method in number 2

```
IFS=$'\n' read -r -d '' -a my_array < <( my_command && printf '\0' )
```

查看IFS值

```
echo "$IFS" | od -b
0000000 040 011 012 012
0000004
```

### set命令
<http://www.ruanyifeng.com/blog/2017/11/bash-set.html>

```shell
set -eu
set -o pipefail


folder="MSNClick"

#IFS=$'\n' read -r -d '' -a array < <( du -d 1 $folder/* | sort -n -t 1 -r && printf '\0' )
mapfile -t array < <( du -d 1 $folder/* | sort -n -t 1 -r )

#echo -n "$IFS" | od -b

function fun(){
    local IFS=$' '

    #echo -n "$IFS" | od -b

    for line in ${array[@]};do
        size=`echo $line |cut -f 1`
        model_dir=`echo $line |cut -f 2`
        debug_case_dir="$model_dir/debug_case"
        if [[ $size -gt 100000 ]];then
            echo "$model_dir, size=$(($size / 1024))M"
            if [ -d $debug_case_dir ]; then
                echo "rm $model_dir/*.bin"
                rm $model_dir/*.bin
            else
                echo "rm $model_dir/*.tsv"
                rm $model_dir/*.tsv
                echo "rm $model_dir/*.bin"
                rm $model_dir/*.bin
            fi
        fi
    done

    return 0
}

fun
```

## 参数判空

```shell
if [ ! -n "$end_dt" ];then
    echo "end_dt is None"
    exit 1
fi
```

## 参数数量校验

```shell
param_len=$#
if [ $param_len != 3 ];then
    echo "train need 3 parameters, now $param_len"
    exit 1;
fi
```

## 循环日期

```shell
t_end=`date -d "$end_dt" +%s`
cur_dt=$start_dt
while true
do
  t_cur=`date -d "$cur_dt" +%s`
  #当开始时间小于结束时间时，直接结束脚本
  if [ $t_cur -gt $t_end ]; then
    break
  fi
  echo -e "\n"
  echo "<<<<<<<<<<<<< $cur_dt <<<<<<<<<<<<<<<<"

  sh run.sh $cur_dt
  check_result "backtrace from rank $cur_dt"
  cur_dt=`date --date "$cur_dt 1 day" +%Y-%m-%d`
done
```

## 分割文件

### Split文件
<https://www.cnblogs.com/OliverQin/p/10240222.html>

```
# file contains 5000 lines
split -l 1000 tmp.log -d -a 2 tmp_part_
# will result like below
tmp_part_00
tmp_part_01
...
tmp_part_04
```

### Merg文件

```
cat part_dir/tmp_part_* >> tmp.log
```

## 统计用户存储占用情况(sort)

du -sh ./* | sort -rh
find ./data -size +200M | xargs ls -lSh

## 字符处理

### 统计字符串单词个数

```bash
# 方法1
echo 'one two three four five' | wc -w
# 方法2
echo 'one two three four five' | awk '{print NF}'
# 方法3 
s='one two three four five' 
set ${s}
echo $#
# 方法4 通过数组方式获取
='one two three four five' 
a=($s)
echo ${#a[@]}
# 方法5
echo 'one two three four five' | tr ' ' '\n' | wc -l
```

### 提取数字

如果某一行如下
file.txt

```
……
 {space}{space}{\t}numRows{space}{\t}122321
……
```

上述文件里有某一行需要提取numRows，但此行有空格，制表符,如果想提取数字保存变量，可以使用tr命令

```shell
numRows=`cat file.txt | grep "numRows" | tr -cd [0-9]`

# tr 命令
# tr -cd ['字符集合']
#   -d：delete；-c：complement；-cd：删除后边的参数以外的
#   ps: complement意味着补集
```

### 打印信息

echo -e 可以输出'\t

#### 打印一条线
<https://blog.csdn.net/u013670453/article/details/113462422>

```
printf '#%.0s' {1..100}
```

或者

```
printf '%100s\n' | tr ' ' =
```

### 选定某几行

[参考资料](http://www.cnblogs.com/xianghang123/archive/2011/08/03/2125977.html)
【一】从第3000行开始，显示1000行。即显示3000~3999行
`cat filename | tail -n +3000 | head -n 1000`
【二】显示1000行到3000行
`cat filename| head -n 3000 | tail -n +1000`
*注意两种方法的顺序
分解：

```
tail -n 1000：显示最后1000行
tail -n +1000：从1000行开始显示，显示1000行以后的
head -n 1000：显示前面1000行
```

【三】用sed命令
 `sed -n '5,10p' filename`
 这样你就可以只查看文件的第5行到第10行。
 如果不加n, 会将5,10行打印一遍，同时打印全部文件；加n是取消default print的执行
 -n, --quiet, --silent: suppress automatic printing of pattern space

### linux sort,uniq,cut,wc命令详解
<https://www.cnblogs.com/ggjucheng/archive/2013/01/13/2858385.html>)

#### cut
<https://www.cnblogs.com/f-ck-need-u/p/7521357.html>

#### sort
<https://www.cnblogs.com/ding2016/p/9668425.html>
sort可以对行进行排序，包括按数字，以及按照split后某一列排序等
常见一种情况，例如目录有如下文件：

```
logs/
├── 202005_0_detect.log
├── 202005_10_detect.log
├── 202005_11_detect.log
├── 202005_1_detect.log
├── 202005_2_detect.log
├── 202005_3_detect.log
├── 202005_4_detect.log
├── 202005_5_detect.log
├── 202005_6_detect.log
├── 202005_7_detect.log
├── 202005_8_detect.log
└── 202005_9_detect.log
```

想按照数字排序，如果直接使用ls会以字母序排序，得到的结果会如上所示，因为ls将数字按照字符串排序，先排第一位再排第二位，类似基数排序的思想。
如果想按照数字顺序就需要sort命令了，sort命令如下

```
sort --help
-n # sort by num 按数字排序
-k # split后选则第k列作为排序依据
-t # terminate 指定分隔符
……
```

形成的命令如下:

```
ls logs/*.log | sort -n -k 2 -t _

# another example
# 按照第四列，以数字方式排列，分割方式是Tab键， 取某一区间结果
sort -k 4 -n -t$'\t' -r userIdMap.tsv | sed -n '30000, 30500p' > userIdMapSort.tsv
```

##### 处理tab键分隔符

```
sort -n -k 4 -r -t$'\t' userIdMap.tsv
```

##### sort 排列日期字符串

```
#例如文件内容 data.txt：
2020-11-14_2020-12-01/train/2020-11-30_2020-12-01
2020-12-01_2020-12-20/train/2020-12-01_2020-12-02
2020-12-01_2020-12-20/train/2020-12-02_2020-12-03
2020-12-01_2020-12-20/train/2020-12-03_2020-12-04
2020-12-01_2020-12-20/train/2020-12-04_2020-12-05
2020-12-01_2020-12-20/train/2020-12-05_2020-12-06
2020-12-01_2020-12-20/train/2020-12-06_2020-12-07

sort -n -k 2 -k 3 -k 4 -k 5 -k 7 -k 8 -t '-' data.txt 
```

##### sort MAC OS换行符不起作用

原因就是Max OS X上的sed是BSD的版本，Linux上的是Gnu的版本，导致的不一致
可以brew install gnu-sed和替换调用sed与gsed。
如果不想在“ g”之前加上sed，可以brew install gnu-sed --with-default-names

##### sort 处理负数

sort -g 注意使用了-g就要去掉-n两者是冲突的

```
sort -g ……
# or --general-numeric-sort
# or --sort=general-numeric
```

### xarg

<http://www.ruanyifeng.com/blog/2019/08/xargs-tutorial.html>

```
ps -ef | grep /bin/bash | grep -v grep | awk '{print $2}' | xargs -n 1 kill -9
# -n参数指定每次将多少项，作为命令行参数。
$ echo {0..9} | xargs -n 2 echo
0 1
2 3
4 5
6 7
8 9
```

#### 从一系列日志文件中提取相同一行内容

例如，从多个spark日志中提取tracking URL

```
ls -t logs/extract_*.log | head -10 | xargs -L 1 grep "tracking URL" -m 1
```

几个知识点：

- xargs 的形式是：

```
xargs [-options] [command]
```

- xargs -L n : 如果输入是多行，需要以**几行**作为参数：

```
>>> cat tmp.txt
img-1.jpeg
img-1-backup.jpeg
img-2.jpeg
img-2-backup.jpeg

cat tmp.txt | xargs -L 2 cp
# 上面命令实现：cp img-d img-d-backup的功能
```

意味着xargs后面需要接command，如果直接接管道会出错

- grep 取第一个匹配结果: grrep 'xxx' -m 1

### 查找文件名

#### Find

find <指定目录> <指定条件> <指定动作>

```
find . -name 'my*'
find . -name 'my*' -ls
find . -type f -mmin -10 #搜索当前目录中，所有过去10分钟中更新过的普通文件。如果不加-type f参数，则搜索普通文件+特殊文件+目录
find ./ -name '*.csv' -maxdepth 1 #至查找当前目录，不递归
```

#### locate

### 检索关键字

指定文件类型

```
grep -rn --include='*.后缀名' "检索词"
例如：
grep -rn --include='*.sh' "kill command"
```

### 比较两个文件夹内容
<https://www.tecmint.com/compare-find-difference-between-two-directories-in-linux/>

```
$ diff [OPTION]… FILES
$ diff options dir1 dir2 

# 非递归，只对比当前文件内容
$ diff -q directory-1/ directory-2/

# 递归对比
$ diff -qr directory-1/ directory-2/ 

# 比较两文件指定段落内容
vimdiff <(sed -n '7088,7122p' /user/vinyuan/a.txt) <(sed -n '9022,9043p' /user/vinyuan/b.txt)
```

### 文件名称替换
<https://www.cnblogs.com/xiaomai333/p/9760304.html>

```bash
>> var=person.jpg
#想替换成person_large.jpg
>> var=${var%.*}_large.${var##*.}
echo $var
>> person_large.jpg
# ${var##*.}
# ${var#*.}
# ${var%.*}
# ${var%%.*}
```

其实`${}`并不是专门为提取文件名或目录名的，它的使用是变量的提取和替换等等操作，它可以提取非常多的内容，并不一定是上面五个例子中的'/'或'.'。也就是说，上面的使用方法只是它使用的一个特例。
**记忆方式：**
查看键盘布局，`#`在`$`的左边，`%`在`$`的右边

### 字符串删除指定部分

[Substring Removal][1]
两种方式都是删除**匹配到的字符串**，对于符号右边的通配符匹配，只要匹配到，**则删除匹配到的部分，留下其他部分**

#### 1.”#“号的使用

`${string#substring}`
Deletes shortest match of `$substring` from front of `$string`.
`${string##substring}`
Deletes longest match of `$substring` from front of `$string`

```
stringZ=abcABC123ABCabc
#       |----|          shortest
#       |----------|    longest
echo ${stringZ#a*C}      # 123ABCabc
# Strip out shortest match between 'a' and 'C'.

echo ${stringZ##a*C}     # abc
# Strip out longest match between 'a' and 'C'.

# You can parameterize the substrings.
X='a*C'
echo ${stringZ#$X}      # 123ABCabc
echo ${stringZ##$X}     # abc
                        # As above
```

### 数值计算与判断

循环是根据奇偶性做出不同处理，switch flag

```
# HOW TO FIND A NUMBER IS EVEN OR ODD IN SHELL SCRIPT
# WRITTEN BY SURAJ MAITY
# TUTORIALSINHAND.COM
clear 
echo "---- EVEN OR ODD IN SHELL SCRIPT -----"
echo -n "Enter a number:"
read n
echo -n "RESULT: "
if [ `expr $n % 2` == 0 ]
then
 echo "$n is even"
else
 echo "$n is Odd"
fi
```

#### 2."%"号的使用

`${string%substring}`
Deletes shortest match of `$substring` from back of `$string`.
`string%%substring}`
Deletes longest match of `$substring` from back of `$string`.

```

stringZ=abcABC123ABCabc
#                    ||     shortest
#        |------------|     longest

echo ${stringZ%b*c}      # abcABC123ABCa
# Strip out shortest match between 'b' and 'c', from back of $stringZ.

echo ${stringZ%%b*c}     # a
# Strip out longest match between 'b' and 'c', from back of $stringZ.
```

#### 3.删除中间匹配的字符串

这种需求可以使用替换来完成
替换

|pattern|explain|
|---|---|
|`${string/substring/replacement}`|使用`$replacement`, 来代替第一个匹配的`$substring`|
|`${string//substring/replacement}`|使用`$replacement`, 代替所有匹配的`$substring`|
|`${string/#substring/replacement}`| 如果`$string`的前缀匹配`$substring`, 那么就用`$replacement`来代替匹配到的`$substring`|
|`${string/%substring/replacement}` |如果`$string`的后缀匹配`$substring`, 那么就用`$replacement`来代替匹配到的`$substring`|

### sed
<https://www.cnblogs.com/zhangzongjian/p/10708222.html>

#### 先检索指定位置，然后替换（非全局替换）
<https://www.golinuxhub.com/2017/09/sed-perform-search-and-replace-only-on/>

```
>> cat /tmp/file
four five six
one
seve eight nine
one two three
one
ten eleven twelve
one

>> sed -e '/two/s/one/replaced/g' /tmp/file
four five six
one
seve eight nine
replaced two three
one
ten eleven twelve
one
```

sed -e ’command-1' -e 'command-2' -e 'command-3' file.txt
sed -e 用来执行一系列操作，例如多个替换：

```
sed -e '/seve/s/nine/replaced/g' -e '/two/s/one/replaced/g' /tmp/file
```

sed -i -e xxxx 在执行多个的同时**本地替换**

```shell
sed -i -e "/validation_data_path/s#recpro#yuanwenwu3#g" -e "/output_path/s#recpro#yuanwenwu3#g" -e "/summary_path/s#recpro#yuanwenwu3#g" ${conf_path}
sed -i -e "/conf_file/s#conf_file=.*#conf_file='${conf}'#g" -e "s#'node':.*,#'node':'${node}',#g" tfconf.py
```

#### 匹配并选择部分替换

```
sed -i "s#\(statPath=\).*#\1${statPath}#g" ${conf_dir}/stat.conf
sed -i "s#\(output=\).*#\1${output}#g;s#\(statPath=\).*#\1${statPath}#g" ${conf_dir}/train.conf
#sed -i "s#\(output=\).*#\1${output}#g;s#\(statPath=\).*#\1${statPath}#g;" ${conf_dir}/train_id2i
```

#### 带变量的替换

如果有**shell变量**，则需要使用**双引号**即可,如果使用单引号则不会替换shell变量，而将`${var}`视为一个字符串

```
var="hello"
sed -n "s/$[var}/word/g" xxx.dat
#sed -n 's/${var}/wor/g' xxx.dat 不会得到预期结果
```

#### 一般用法

参考: <http://bbs.linuxtone.org/thread-1731-1-1.html>

1. 当前行进行替换:`s/XXX/YYY/g`
XXX是需要替换的字符串,YYY是替换后的字符串。
2. 全局替换:`% s/XXX/YYY/g`.
3. 对指定部分进行替换用V进入visual模式,再进行:`s/XXX/YYY/g`. <font color="green">ps(命令模式，显示的`:'<,'>`的是区域选择的意思，不要删除，紧接着在后面用`s/xxx/yyy/g`即可)</font>

比如，要将目录/modules下面所有文件中的zhangsan都修改成lisi，这样做：
sed -i "s/zhangsan/lisi/g" `grep zhangsan -rl /modules`

解释一下：

-i 表示inplace edit，就地修改文件
-r 表示搜索子目录
-l 表示输出匹配的文件名

这个命令组合很强大，要注意备份文件。

1. 字典映射

```
sed 'y/1234567890/ABCDEFGHIJ/' test_sed
sed 'y/1234567890/ABCDEFGHIJ/' filename
```

ABCDEFGHIJ
BCDEFGHIJA
CDEFGHIJAB
DEFGHIJABC
注意变换关系是按两个list的位置对应变换
其中：test_sed的内容是：
1234567890
2345678901
3456789012
4567890123
2. 替换每行所有匹配
`sed 's/01/Ab/g' test_sed`
1234567890
23456789Ab
3456789Ab2
456789Ab23
注意：第一行的0，1没有分别替换为A,b

#### 删除：d命令

```
sed '2d' example          #删除example文件的第二行。
sed '2,$d' example        #删除example文件的第二行到末尾所有行。
sed '$d' example          #删除example文件的最后一行。
sed '/test/'d example     #匹配'test'并删除 
```

example-----删除example文件所有包含test的行。

替换：s命令
`$ sed 's/test/mytest/g' example`-----在整行范围内把test替换为mytest。如果没有g标记，则只有每行第一个匹配的test被替换成mytest。
`$ sed -n 's/^test/mytest/p' example`-----(-n)选项和p标志一起使用表示只打印那些发生替换的行。也就是说，如果某一行开头的test被替换成mytest，就打印它。
`$ sed 's/^192.168.0.1/&localhost/'example`-----&符号表示替换换字符串中被找到的部份。所有以192.168.0.1开头的行都会被替换成它自已加localhost，变成192.168.0.1localhost。
`$ sed -n 's/\(love\)able/\1rs/p' example`-----love被标记为1，所有loveable会被替换成lovers，而且替换的行会被打印出来。
`$ sed 's#10#100#g' example`-----不论什么字符，紧跟着s命令的都被认为是新的分隔符，所以，“#”在这里是分隔符，代替了默认的“/”分隔符。表示把所有10替换成100。

选定行的范围：逗号
`$ sed -n '/test/,/check/p' example`-----所有在模板test和check所确定的范围内的行都被打印。
`$ sed -n '5,/^test/p' example`-----打印从第五行开始到第一个包含以test开始的行之间的所有行。
`$ sed '/test/,/check/s/$/sed test/' example`-----对于模板test和west之间的行，每行的末尾用字符串sed test替换。

多点编辑：e命令  
`$ sed -e '1,5d' -e 's/test/check/'example`-----(-e)选项允许在同一行里执行多条命令。如例子所示，第一条命令删除1至5行，第二条命令用check替换test。命令的执行顺序对结果有影响。如果两个命令都是替换命令，那么第一个替换命令将影响第二个替换命令的结果。
`$ sed --expression='s/test/check/' --expression='/love/d' example`-----一个比-e更好的命令是--expression。它能给sed表达式赋值。

从文件读入：r命令  
`$ sed '/test/r file' example`-----file里的内容被读进来，显示在与test匹配的行后面，如果匹配多行，则file的内容将显示在所有匹配行的下面。

写入文件：w命令  
`$ sed -n '/test/w file' example`-----在example中所有包含test的行都被写入file里。

追加命令：a命令  
`$ sed '/^test/a\\--->this is a example' example`<-----'this is a example'被追加到以test开头的行后面，sed要求命令a后面有一个反斜杠。

插入：i命令
`$ sed '/test/i\\ new line -------------------------' example`
如果test被匹配，则把反斜杠后面的文本插入到匹配行的前面。
下一个：n命令  
`$ sed '/test/{ n; s/aa/bb/; }' example`-----如果test被匹配，则移动到匹配行的下一行，替换这一行的aa，变为bb，并打印该行，然后继续。

变形：y命令  
`$ sed '1,10y/abcde/ABCDE/' example`-----把1--10行内所有abcde转变为大写，注意，正则表达式元字符不能使用这个命令。

退出：q命令  
`$ sed '10q' example`-----打印完第10行后，退出sed。

保持和获取：h命令和G命令  
`$ sed -e '/test/h' -e`
example-----在sed处理文件的时候，每一行都被保存在一个叫模式空间的临时缓冲区中，除非行被删除或者输出被取消，否则所有被处理的行都将打印在屏幕上。接着模式空间被清空，并存入新的一行等待处理。在这个例子里，匹配test的行被找到后，将存入模式空间，h命令将其复制并存入一个称为保持缓存区的特殊缓冲区内。第二条语句的意思是，当到达最后一行后，G命令取出保持缓冲区的行，然后把它放回模式空间中，且追加到现在已经存在于模式空间中的行的末尾。在这个例子中就是追加到最后一行。简单来说，任何包含test的行都被复制并追加到该文件的末尾。

保持和互换：h命令和x命令  
`$ sed -e '/test/h' -e '/check/x' example` -----互换模式空间和保持缓冲区的内容。也就是把包含test与check的行互换。

7.脚本

Sed脚本是一个sed的命令清单，启动Sed时以-f选项引导脚本文件名。Sed对于脚本中输入的命令非常挑剔，在命令的末尾不能有任何空白或文本，如果在一行中有多个命令，要用分号分隔。以#开头的行为注释行，且不能跨行。

8.小技巧
在sed的命令行中引用shell变量时要使用双引号，而不是通常所用的单引号。下面是一个根据name变量的内容来删除named.conf文件中zone段的脚本：

```
name='zone\ "localhost"'
sed "/$name/,/};/d" named.conf
sed -i "s/oldstring/newstring/g" `grep oldstring -rl yourdir`
```

例如：替换/home下所有文件中的www.itbbs.cn为chinafar.com

```
sed -i "s/www.itbbs.cn/chinafar.com/g" `grep www.itbbs.cn -rl /home` 
```

## vim & vi

### 自己使用的简单配置template

```
# ~/.vimrc
hi CursorLine   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
hi CursorLine term=bold cterm=bold ctermbg=237
hi CursorColumn   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
hi CursorColumn term=bold cterm=bold ctermbg=237

set ts=4
set expandtab

set smartindent
set tabstop=4
set shiftwidth=4
set expandtab
set softtabstop=4

# https://www.dyxmq.cn/linux/vim-setting-mouse-place.html
# 设置vim打开上次编辑地方
au BufReadPost * if line("'\"") > 0 | if line("'\"") <= line("$") | exe("norm '\"") | else |exe "norm $"| endif | endif
```

### 多功能templete

```shell
"=========================================================================
" DesCRiption: 适合自己使用的vimrc文件，for Linux/Windows, GUI/Console
"
" Last Change: 2010年08月02日 15时13分
"
" Version:     1.80
"
"=========================================================================

set nocompatible            " 关闭 vi 兼容模式
syntax on                   " 自动语法高亮
"colorscheme molokai         " 设定配色方案
set number                  " 显示行号
set noswapfile                          " 不产生.swap, .swo文件
"set cursorline              " 突出显示当前行
set ruler                   " 打开状态栏标尺
set shiftwidth=4            " 设定 << 和 >> 命令移动时的宽度为 4
set softtabstop=4           " 使得按退格键时可以一次删掉 4 个空格
set tabstop=4               " 设定 tab 长度为 4
set nobackup                " 覆盖文件时不备份
"set autochdir               " 自动切换当前目录为当前文件所在的目录
filetype plugin indent on   " 开启插件
set backupcopy=yes          " 设置备份时的行为为覆盖
set ignorecase smartcase    " 搜索时忽略大小写，但在有一个或以上大写字母时仍保持对大小写敏感
"set nowrapscan              " 禁止在搜索到文件两端时重新搜索
set incsearch               " 输入搜索内容时就显示搜索结果
set hlsearch                " 搜索时高亮显示被找到的文本
set noerrorbells            " 关闭错误信息响铃
set novisualbell            " 关闭使用可视响铃代替呼叫
set t_vb=                   " 置空错误铃声的终端代码
" set showmatch               " 插入括号时，短暂地跳转到匹配的对应括号
" set matchtime=2             " 短暂跳转到匹配括号的时间
set magic                   " 设置魔术
set hidden                  " 允许在有未保存的修改时切换缓冲区，此时的修改由 vim 负责保存
set guioptions-=T           " 隐藏工具栏
set guioptions-=m           " 隐藏菜单栏
set smartindent             " 开启新行时使用智能自动缩进
set backspace=indent,eol,start
                            " 不设定在插入状态无法用退格键和 Delete 键删除回车符
set cmdheight=1             " 设定命令行的行数为 1
set laststatus=2            " 显示状态栏 (默认值为 1, 无法显示状态栏)
set statusline=\ %<%F[%1*%M%*%n%R%H]%=\ %y\ %0(%{&fileformat}\ %{&encoding}\ %c:%l/%L%)\
                            " 设置在状态行显示的信息
"set foldenable              " 开始折叠
"set foldmethod=syntax       " 设置语法折叠
"set foldcolumn=0            " 设置折叠区域的宽度
"setlocal foldlevel=0        " 设置折叠层数为
"set foldclose=all           " 设置为自动关闭折叠
" nnoremap <space> @=((foldclosed(line('.')) < 0) ? 'zc' : 'zo')<CR>
                            " 用空格键来开关折叠
au BufReadPost * if line("'\"") > 0 | if line("'\"") <= line("$") | exe("norm '\"") | else |exe "norm $"| endif | endif
hi CursorLine   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
hi CursorLine term=bold cterm=bold ctermbg=237
hi CursorColumn   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
hi CursorColumn term=bold cterm=bold ctermbg=237

" return OS type, eg: windows, or linux, mac, et.st..
function! MySys()
    if has("win16") || has("win32") || has("win64") || has("win95")
        return "windows"
    elseif has("unix")
        return "linux"
    endif
endfunction

" 用户目录变量$VIMFILES
if MySys() == "windows"
    let $VIMFILES = $VIM.'/vimfiles'
elseif MySys() == "linux"
    let $VIMFILES = $HOME.'/.vim'
endif

" 设定doc文档目录
let helptags=$VIMFILES.'/doc'

" 设置字体 以及中文支持
if has("win32")
    set guifont=Inconsolata:h12:cANSI
endif

" 配置多语言环境
if has("multi_byte")
    " UTF-8 编码
    set encoding=utf-8
    set termencoding=utf-8
    set formatoptions+=mM
    set fencs=utf-8,gbk

    if v:lang =~? '^\(zh\)\|\(ja\)\|\(ko\)'
        set ambiwidth=double
    endif

    if has("win32")
        source $VIMRUNTIME/delmenu.vim
        source $VIMRUNTIME/menu.vim
        language messages zh_CN.utf-8
    endif
else
    echoerr "Sorry, this version of (g)vim was not compiled with +multi_byte"
endif

" Buffers操作快捷方式!
nnoremap <C-RETURN> :bnext<CR>
nnoremap <C-S-RETURN> :bprevious<CR>

" Tab操作快捷方式!
nnoremap <C-TAB> :tabnext<CR>
nnoremap <C-S-TAB> :tabprev<CR>

"关于tab的快捷键
" map tn :tabnext<cr>
" map tp :tabprevious<cr>
" map td :tabnew .<cr>
" map te :tabedit
" map tc :tabclose<cr>

"窗口分割时,进行切换的按键热键需要连接两次,比如从下方窗口移动
"光标到上方窗口,需要<c-w><c-w>k,非常麻烦,现在重映射为<c-k>,切换的
"时候会变得非常方便.
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

"一些不错的映射转换语法（如果在一个文件中混合了不同语言时有用）
nnoremap <leader>1 :set filetype=xhtml<CR>
nnoremap <leader>2 :set filetype=css<CR>
nnoremap <leader>3 :set filetype=javascript<CR>
nnoremap <leader>4 :set filetype=php<CR>

" set fileformats=unix,dos,mac
" nmap <leader>fd :se fileformat=dos<CR>
" nmap <leader>fu :se fileformat=unix<CR>

" use Ctrl+[l|n|p|cc] to list|next|previous|jump to count the result
" map <C-x>l <ESC>:cl<CR>
" map <C-x>n <ESC>:cn<CR>
" map <C-x>p <ESC>:cp<CR>
" map <C-x>c <ESC>:cc<CR>


" 让 Tohtml 产生有 CSS 语法的 html
" syntax/2html.vim，可以用:runtime! syntax/2html.vim
let html_use_css=1

" Python 文件的一般设置，比如不要 tab 等
autocmd FileType python set tabstop=4 shiftwidth=4 expandtab
autocmd FileType python map <F12> :!python %<CR>

" 选中状态下 Ctrl+c 复制
vmap <C-c> "+y

" 打开javascript折叠
let b:javascript_fold=1
" 打开javascript对dom、html和css的支持
let javascript_enable_domhtmlcss=1
" 设置字典 ~/.vim/dict/文件的路径
autocmd filetype javascript set dictionary=$VIMFILES/dict/javascript.dict
autocmd filetype css set dictionary=$VIMFILES/dict/css.dict
autocmd filetype php set dictionary=$VIMFILES/dict/php.dict

"-----------------------------------------------------------------
" plugin - bufexplorer.vim Buffers切换
" \be 全屏方式查看全部打开的文件列表
" \bv 左右方式查看   \bs 上下方式查看
"-----------------------------------------------------------------


"-----------------------------------------------------------------
" plugin - taglist.vim  查看函数列表，需要ctags程序
" F4 打开隐藏taglist窗口
"-----------------------------------------------------------------
if MySys() == "windows"                " 设定windows系统中ctags程序的位置
    let Tlist_Ctags_Cmd = '"'.$VIMRUNTIME.'/ctags.exe"'
elseif MySys() == "linux"              " 设定windows系统中ctags程序的位置
    let Tlist_Ctags_Cmd = '/usr/bin/ctags'
endif
nnoremap <silent><F4> :TlistToggle<CR>
let Tlist_Show_One_File = 1            " 不同时显示多个文件的tag，只显示当前文件的
let Tlist_Exit_OnlyWindow = 1          " 如果taglist窗口是最后一个窗口，则退出vim
let Tlist_Use_Right_Window = 1         " 在右侧窗口中显示taglist窗口
let Tlist_File_Fold_Auto_Close=1       " 自动折叠当前非编辑文件的方法列表
let Tlist_Auto_Open = 0
let Tlist_Auto_Update = 1
let Tlist_Hightlight_Tag_On_BufEnter = 1
let Tlist_Enable_Fold_Column = 0
let Tlist_Process_File_Always = 1
let Tlist_Display_Prototype = 0
let Tlist_Compact_Format = 1


"-----------------------------------------------------------------
" plugin - mark.vim 给各种tags标记不同的颜色，便于观看调式的插件。
" \m  mark or unmark the word under (or before) the cursor
" \r  manually input a regular expression. 用于搜索.
" \n  clear this mark (i.e. the mark under the cursor), or clear all highlighted marks .
" \*  当前MarkWord的下一个     \#  当前MarkWord的上一个
" \/  所有MarkWords的下一个    \?  所有MarkWords的上一个
"-----------------------------------------------------------------


"-----------------------------------------------------------------
" plugin - NERD_tree.vim 以树状方式浏览系统中的文件和目录
" :ERDtree 打开NERD_tree         :NERDtreeClose    关闭NERD_tree
" o 打开关闭文件或者目录         t 在标签页中打开
" T 在后台标签页中打开           ! 执行此文件
" p 到上层目录                   P 到根目录
" K 到第一个节点                 J 到最后一个节点
" u 打开上层目录                 m 显示文件系统菜单（添加、删除、移动操作）
" r 递归刷新当前目录             R 递归刷新当前根目录
"-----------------------------------------------------------------
" F3 NERDTree 切换
map <F3> :NERDTreeToggle<CR>
imap <F3> <ESC>:NERDTreeToggle<CR>


"-----------------------------------------------------------------
" plugin - NERD_commenter.vim   注释代码用的，
" [count],cc 光标以下count行逐行添加注释(7,cc)
" [count],cu 光标以下count行逐行取消注释(7,cu)
" [count],cm 光标以下count行尝试添加块注释(7,cm)
" ,cA 在行尾插入 /* */,并且进入插入模式。 这个命令方便写注释。
" 注：count参数可选，无则默认为选中行或当前行
"-----------------------------------------------------------------
let NERDSpaceDelims=1       " 让注释符与语句之间留一个空格
let NERDCompactSexyComs=1   " 多行注释时样子更好看


"-----------------------------------------------------------------
" plugin - DoxygenToolkit.vim  由注释生成文档，并且能够快速生成函数标准注释
"-----------------------------------------------------------------
let g:DoxygenToolkit_authorName="Asins - asinsimple AT gmail DOT com"
let g:DoxygenToolkit_briefTag_funcName="yes"
map <leader>da :DoxAuthor<CR>
map <leader>df :Dox<CR>
map <leader>db :DoxBlock<CR>
map <leader>dc a /*  */<LEFT><LEFT><LEFT>


"-----------------------------------------------------------------
" plugin – ZenCoding.vim 很酷的插件，HTML代码生成
" 插件最新版：http://github.com/mattn/zencoding-vim
" 常用命令可看：http://nootn.com/blog/Tool/23/
"-----------------------------------------------------------------


"-----------------------------------------------------------------
" plugin – checksyntax.vim    JavaScript常见语法错误检查
" 默认快捷方式为 F5
"-----------------------------------------------------------------
let g:checksyntax_auto = 0 " 不自动检查


"-----------------------------------------------------------------
" plugin - NeoComplCache.vim    自动补全插件
"-----------------------------------------------------------------
let g:AutoComplPop_NotEnableAtStartup = 1
let g:NeoComplCache_EnableAtStartup = 1
let g:NeoComplCache_SmartCase = 1
let g:NeoComplCache_TagsAutoUpdate = 1
let g:NeoComplCache_EnableInfo = 1
let g:NeoComplCache_EnableCamelCaseCompletion = 1
let g:NeoComplCache_MinSyntaxLength = 3
let g:NeoComplCache_EnableSkipCompletion = 1
let g:NeoComplCache_SkipInputTime = '0.5'
let g:NeoComplCache_SnippetsDir = $VIMFILES.'/snippets'
" <TAB> completion.
inoremap <expr><TAB> pumvisible() ? "\<C-n>" : "\<TAB>"
" snippets expand key
imap <silent> <C-e> <Plug>(neocomplcache_snippets_expand)
"smap <silent> <C-e> <Plug>(neocomplcache_snippets_expand)


"-----------------------------------------------------------------
" plugin - matchit.vim   对%命令进行扩展使得能在嵌套标签和语句之间跳转
" % 正向匹配      g% 反向匹配
" [% 定位块首     ]% 定位块尾
"-----------------------------------------------------------------


"-----------------------------------------------------------------
" plugin - vcscommand.vim   对%命令进行扩展使得能在嵌套标签和语句之间跳转
" SVN/git管理工具
"-----------------------------------------------------------------


"-----------------------------------------------------------------
" plugin – a.vim
"-----------------------------------------------------------------
imap <F9> <esc>:tabN<cr>
imap <F10> <esc>:tabn<cr>
nmap <F9> :tabN<cr>
nmap <F10> :tabn<cr>
"set showtabline=1
hi Comment ctermfg=2



highlight Visual cterm=bold ctermbg=Blue ctermfg=NONE #visual 选中高亮的底色变成浅色
```

### Vimdiff

更改对比颜色

```
:colo desert
```

### vi语法高亮shell脚本

```
# ~/.vimrc
filetype plugin indent on
syntax on
```

### vim & vi 配置
<https://www.dazhuanlan.com/2019/10/02/5d94884599d6c/>

在vimrc中加入:

```
hi CursorLine term=bold cterm=bold ctermbg=Red
# 在vi中使用
set cursorline
or
set cul
```

### 查找

```
# 高亮显示搜索匹配的词
:set hlsearch    

# 每输入一个字符，搜索一次，表现是同时高亮多个正则匹配
# 在普通搜索前执行此命令
:set incsearch

# 忽略大小写
:set ignorecase   

```

查找当前光标所在单词:
\* 向下查找全部匹配的字符串（按住shift+\*键），同时左下角会显示等效的命令模式指令
\# 向上查找全部匹配的字符（按住shift+\#键），同时左下角会显示等效的命令模式指令
g\* 向下查找包含当前字符的匹配字符串（按住shift+\*键），同时左下角会显示等效的命令模式指令
g\# 向上查找包含当前字符的匹配字符串（按住shift+\#键），同时左下角会显示等效的命令模式指令
\% 括号匹配 包括 () [] { }.

### 全部替换

```
:[addr]s/源字符串/目标字符串/[option]
# 全局替换
:%s/源字符串/目标字符串/g

# 全部替换和添加确认环节
:%s/search_for_this/replace_with_this/    - search whole file and replace
:%s/search_for_this/replace_with_this/c   - confirm each replace
```

### cursor颜色

```
set cursorline
hi CursorLine   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
set cursorcolumn
hi CursorColumn   cterm=NONE ctermbg=lightgray ctermfg=white guibg=lightgray guifg=white
# 设置更浅的灰色
hi CursorLine term=bold cterm=bold ctermbg=237
```

### 空格替换tab
<https://blog.csdn.net/jiang1013nan/article/details/6298727>
在.vimrc中添加以下代码后，重启vim即可实现按TAB产生4个空格：

```
#~/.vimrc
set ts=4  (注：ts是tabstop的缩写，设TAB宽4个空格)
set expandtab
```

对于已保存的文件，可以使用下面的方法进行空格和TAB的替换：
TAB替换为空格：

```
:set ts=4
:set expandtab
:%retab!
```

空格替换为TAB：

```
:set ts=4
:set noexpandtab
:%retab!
```

## 后台运行

单纯的后天运行直接似使用
参考：[link][2]

```shell
command & #后台运行
ctrl + z #将一个正在前台执行的命令放到后台，并且暂停
jobs -l
fg %1 #前台显示
bg %1 #将任务后台运行，比如cltr+z后会挂起任务，并显示任务号，此时就可以用这个命令让其后台运行
```

### [nohup](https://www.cnblogs.com/jinxiao-pu/p/9131057.html)

nohup 命令运行由 Command参数和任何相关的 Arg参数指定的命令，忽略所有挂断（SIGHUP）信号。在注销后使用 nohup 命令运行后台中的程序。要运行后台中的 nohup 命令，添加 & （ 表示“and”的符号）到命令的尾部。

**nohup 是 no hang up 的缩写，就是不挂断的意思。**

### [nohup和 & 的区别](https://www.cnblogs.com/jinxiao-pu/p/9131057.html)

& ： 指在后台运行
nohup ： 不挂断的运行，注意并没有后台运行的功能，，就是指，用nohup运行命令可以使命令永久的执行下去，和用户终端没有关系，例如我们断开SSH连接都不会影响他的运行，注意了nohup没有后台运行的意思；&才是后台运行

### [获得nohup后台运行进程的PID]( https://www.jianshu.com/p/5a04e2452e3f)

nohup command > logfile.txt & echo $! > pidfile.txt
其实也可以在command.sh里面写入：

```shell
echo current pid = $$
```

|变量名|含义|
|---|---|
|$$|Shell本身的PID（ProcessID）|
|$!|Shell最后运行的后台Process的PID|
|$?|最后运行的命令的结束代码（返回值）|
|$-|使用Set命令设定的Flag一览|
|$*|所有参数列表。如"$*"用「"」括起来的情况、以"$1 $2 …$n"的形式输出所有参数。|
|$@|所有参数列表。如"$@"用「"」括起来的情况、以"$1" "$2" … "$n" 的形式输出所有参数。|
|$#|添加到Shell的参数个数|
|$0|Shell本身的文件名|
|\$1-\$n|添加到Shell的各参数值。$1是第1参数、$2是第2参数…。|

### [2>&1解析](https://www.cnblogs.com/zzyoucan/p/7764590.html)

```shell
cmmand >out.file 2>&1 &
```

- command>out.file是将command的输出重定向到out.file文件，即输出内容不打印到屏幕上，而是输出到out.file文件中。
- 2>&1 是将标准出错重定向到标准输出，这里的标准输出已经重定向到了out.file文件，即将标准出错也输出到out.file文件中。最后一个&， 是让该命令在后台执行。
- 试想2>1代表什么，2与>结合代表错误重定向，而1则代表错误重定向到一个文件1，而不代表标准输出；换成2>&1，&与1结合就代表标准输出了，就变成错误重定向到标准输出.

### [查看后台任务或进程](https://www.cnblogs.com/baby123/p/6477429.html)

- jobs -l
jobs命令只看当前终端生效的，关闭终端后，在另一个终端jobs已经无法看到后台跑得程序了，此时利用ps（进程查看命令）
- ps -ef

```shell
s -aux|grep chat.js
 a:显示所有程序 
 u:以用户为主的格式来显示 
 x:显示所有程序，不以终端机来区分
```

### wait 命令+获取上一个后台进程的进程号以及状态

`wait_test.sh`

```shell
#!/usr/bin/env bash

echo "current job pid: " $$

bash wait_error_job.sh 1 &
job_1=$!
bash wait_error_job.sh 2 &
job_2=$!

wait $job_1   
job_1_status=$?
wait $job_2
job_2_status=$?

echo "job 1 pid: " ${job_1} " status: " ${job_1_status}
echo "job 2 pid: " ${job_2} " status: " ${job_2_status}
```

`wait_error_job.sh`

```shell
#!/usr/bin/env bash

echo "I am job: " $1 "my pid is: " $$
sleep 5 
if [ $1 -eq 1 ];then
    echo "this is job 1, success"
else
    echo "this is job 2, failed"
    lls # 拼写错误的命令，用来报错
fi
```

### disown

场景：<http://www.ibm.com/developerworks/cn/linux/l-cn-nohup/>
我们已经知道，如果事先在命令前加上 nohup 或者 setsid 就可以避免 HUP 信号的影响。但是如果我们未加任何处理就已经提交了命令，该如何补救才能让它避免 HUP 信号的影响呢？
解决方法：
这时想加 nohup 或者 setsid 已经为时已晚，只能通过作业调度和 disown 来解决这个问题了。让我们来看一下 disown 的帮助信息：

```
disown [-ar] [-h] [jobspec ...]
 Without options, each jobspec is  removed  from  the  table  of
 active  jobs.   If  the -h option is given, each jobspec is not
 removed from the table, but is marked so  that  SIGHUP  is  not
 sent  to the job if the shell receives a SIGHUP.  If no jobspec
 is present, and neither the -a nor the -r option  is  supplied,
 the  current  job  is  used.  If no jobspec is supplied, the -a
 option means to remove or mark all jobs; the -r option  without
 a  jobspec  argument  restricts operation to running jobs.  The
 return value is 0 unless a jobspec does  not  specify  a  valid
 job.
```

可以看出，我们可以用如下方式来达成我们的目的。
灵活运用 CTRL-z
在我们的日常工作中，我们可以用 CTRL-z 来将当前进程挂起到后台暂停运行，执行一些别的操作，然后再用 fg 来将挂起的进程重新放回前台（也可用 bg 来将挂起的进程放在后台）继续运行。这样我们就可以在一个终端内灵活切换运行多个任务，这一点在调试代码时尤为有用。因为将代码编辑器挂起到后台再重新放回时，光标定位仍然停留在上次挂起时的位置，避免了重新定位的麻烦。
用disown -h jobspec来使某个作业忽略HUP信号。
用disown -ah 来使所有的作业都忽略HUP信号。
用disown -rh 来使正在运行的作业忽略HUP信号。
需要注意的是，当使用过 disown 之后，会将把目标作业从作业列表中移除，我们将不能再使用jobs来查看它，但是依然能够用ps -ef查找到它。
但是还有一个问题，这种方法的操作对象是作业，如果我们在运行命令时在结尾加了"&"来使它成为一个作业并在后台运行，那么就万事大吉了，我们可以通过jobs命令来得到所有作业的列表。但是如果并没有把当前命令作为作业来运行，如何才能得到它的作业号呢？答案就是用 CTRL-z（按住Ctrl键的同时按住z键）了！
CTRL-z 的用途就是将当前进程挂起（Suspend），然后我们就可以用jobs命令来查询它的作业号，再用bg jobspec来将它放入后台并继续运行。需要注意的是，如果挂起会影响当前进程的运行结果，请慎用此方法。
disown 示例1（如果提交命令时已经用“&”将命令放入后台运行，则可以直接使用“disown”）

```
[root@pvcent107 build]# cp -r testLargeFile largeFile &
[1] 4825
[root@pvcent107 build]# jobs
[1]+  Running                 cp -i -r testLargeFile largeFile &
[root@pvcent107 build]# disown -h %1
[root@pvcent107 build]# ps -ef |grep largeFile
root      4825   968  1 09:46 pts/4    00:00:00 cp -i -r testLargeFile largeFile
root      4853   968  0 09:46 pts/4    00:00:00 grep largeFile
[root@pvcent107 build]# logout
```

disown 示例2（如果提交命令时未使用“&”将命令放入后台运行，可使用 CTRL-z 和“bg”将其放入后台，再使用“disown”）

```
[root@pvcent107 build]# cp -r testLargeFile largeFile2

[1]+  Stopped                 cp -i -r testLargeFile largeFile2
[root@pvcent107 build]# bg %1
[1]+ cp -i -r testLargeFile largeFile2 &
[root@pvcent107 build]# jobs
[1]+  Running                 cp -i -r testLargeFile largeFile2 &
[root@pvcent107 build]# disown -h %1
[root@pvcent107 build]# ps -ef |grep largeFile2
root      5790  5577  1 10:04 pts/3    00:00:00 cp -i -r testLargeFile largeFile2
root      5824  5577  0 10:05 pts/3    00:00:00 grep largeFile2
[root@pvcent107 build]#
```

### screen

用screen -dmS session name来建立一个处于断开模式下的会话（并指定其会话名）。
用screen -list 来列出所有会话。
用screen -r session name来重新连接指定会话。
用快捷键CTRL-a d 来暂时断开当前会话。
screen 示例

```
[root@pvcent107 ~]# screen -dmS Urumchi
[root@pvcent107 ~]# screen -list
There is a screen on:
        12842.Urumchi   (Detached)
1 Socket in /tmp/screens/S-root.

[root@pvcent107 ~]# screen -r Urumchi
```

1.未使用 screen 时新进程的进程树

```
[root@pvcent107 ~]# ping www.google.com &
[1] 9499
[root@pvcent107 ~]# pstree -H 9499
init─┬─Xvnc
     ├─acpid
     ├─atd
     ├─2*[sendmail] 
     ├─sshd─┬─sshd───bash───pstree
     │       └─sshd───bash───ping
我们可以看出，未使用 screen 时我们所处的 bash 是 sshd 的子进程，当 ssh 断开连接时，HUP 信号自然会影响到它下面的所有子进程（包括我们新建立的 ping 进程）。
```

2.使用了 screen 后新进程的进程树

```
[root@pvcent107 ~]# screen -r Urumchi
[root@pvcent107 ~]# ping www.ibm.com &
[1] 9488
[root@pvcent107 ~]# pstree -H 9488
init─┬─Xvnc
     ├─acpid
     ├─atd
     ├─screen───bash───ping
     ├─2*[sendmail]
```

而使用了 screen 后就不同了，此时 bash 是 screen 的子进程，而 screen 是 init（PID为1）的子进程。那么当 ssh 断开连接时，HUP 信号自然不会影响到 screen 下面的子进程了。

## TOP 命令
<https://www.binarytides.com/linux-top-command/>
<https://www.zhihu.com/question/378345922/answer/1069674567>

## 批量下载图片

[参考代码](https://www.jb51.net/article/122242.htm)

```shell
#while 循环读取文件，文件放在 done 后面
while read line;
do
    #echo "read line" $i ":" $line | tee -a $log_file
    let "i=$i+1"
    arr=($line)
    rnd=$(rand 0 6)
    curl_cmd="curl -o ${img_dir}${arr[0]}.jpg ${host[$rnd]}${arr[1]}"
    echo $curl_cmd
    eval "$curl_cmd "
done < $sku_info

```

## 随机数

```shell
function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(cat /proc/sys/kernel/random/uuid | cksum | awk -F ' ' '{print $1}')
    echo $(($num%$max+$min))
}
#使用方法
>> rnd=$(rand 0 10) #0-10内的整数
```

### 随机筛选数据

```
shuf dataset.dat -n 20 -o dataset_sample.dat
```

## 判断文件是否存在

[参考文件](https://www.jb51.net/article/122242.htm)

### 1.文件夹不存在则创建

```
if [ ! -d "/data/" ];then
mkdir /data
else
echo "文件夹已经存在"
fi
```

### 2.文件存在则删除

```
if [ ! -f "/data/filename" ];then
echo "文件不存在"
else
rm -f /data/filename
fi
```

### 3.判断文件夹是否存在

```
if [ -d "/data/" ];then
echo "文件夹存在"
else
echo "文件夹不存在"
fi
```

### 4.判断文件是否存在

```
if [ -f "/data/filename" ];then
echo "文件存在"
else
echo "文件不存在"
fi
```

## 日期处理

```shell
# n 天后的日期 yyyy-mm-dd
n_days_later=`date --date "${cur_day} n day" +%Y-%m-%d`
# n 天前的日期 
n_days_ago=`date --date "${cur_day} n day ago" +%Y-%m-%d`


## 获取时间戳，用以比较大小
today_t=`date -d "${today}" +%s`
tomorrow_t=`date -d "${tomorrow}" +%s`


## 循环一段时间 yyyy-mm-dd
start_day=2019-09-08
end_day=2019-09-17
dt=${start_day} 
while [ "${dt}" != "${end_day}" ];do
    echo "-------------统计日期: ${dt}----------------"
    predt=$(date -I -d "${dt} - 1 day")
    dt=$(date -I -d "${dt} + 1 day")
done


```

## 条件语句(if,while）

```shell
## while 条件语句
while [ ${today_t} -le ${tomorrow} ];
do
    # do sth
done
## if 条件语句
if [ ${today_t} -le ${tomorrow_t} ];
then
    #do sth
    else
    #do otherthing
fi
```

### if 条件判断大括号，中括号，圆括号
<https://www.jianshu.com/p/3e1eaaa3fee8>

### 判断文件、文件夹是否存在
<https://www.cnblogs.com/emanlee/p/3583769.html>

```
if [ ! -d "$folder"]; then
  mkdir "$folder"
fi
```

### 判断变量是否为空

```
# 1.判断变量
if [ ! -n "$var" ];then ...
# 2.判断输入参数
if [ ! -n "$1" ];then ...
# 3.直接变量判断
if [ ! $var ];then ...
# 4.使用test判断
if test -z "$var" then ...
# 5.使用""判断
if [ "$var" == "" ] then ...
```

2.

3.

## 检索

### 查找文件

```
#find 搜索目录 -name 目标名字
#eg.
find / -name filename  # /代表是全盘搜索,也可以指定目录搜索
 
##find 搜索文件的命令格式：
##find [搜索范围] [匹配条件]
#选项：
#    -name  根据名字查找（精确查找）
#    -iname 根据文件名查找，但是不区分大小写
#    -size  根据文件大小查找, +,-:大于设置的大小,直接写大小是等于
#    -user  查找用户名的所有者的所有文件
#    -group 根据所属组查找相关文件
#    -type  根据文件类型查找(f文件,d目录,l软链接文件)
#    -inum  根据i节点查找
#    -amin  访问时间access
#    -cmin  文件属性change
#    -mmin  文件内容modify
#example    
find /etc -iname "*.sh"
find /etc -iname "*.service" 2> /dev/null
```

## 重定向与文件操作符
<https://www.zhihu.com/question/53295083/answer/135258024>

```
# 分开写
cmd 2>stderr.txt 1>stdout.txt
# 写入一个文件
cmd > output.txt 2>&1
cmd &> output.txt
cmd >& output.txt  # 两个表达式效果一样哒~
```
<https://zhuanlan.zhihu.com/p/58419951>
find /etc -iname "*.service" 2>&1 1>services.txt
这是因为 Bash 从左到右处理 find 的每个结果。这样想：当 Bash 到达 2>&1 时，stdout （1）仍然是指向终端的通道。如果 find 给 Bash 的结果包含一个错误，它将被弹出到 2，转移到 1，然后留在终端！

相比之下，在：
find /etc -iname "*.service" 1>services.txt 2>&1
1 从一开始就指向 services.txt，因此任何弹出到 2 的内容都会导向到 1 ，而 1 已经指向最终去的位置 services.txt，这就是它工作的原因。

在任何情况下，如上所述 &> 都是“标准输出和标准错误”的缩写，即 2>&1。

## shell函数返回字符串参数

[参考博客](https://roc-wong.github.io/blog/2017/03/shell-%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9A%84%E6%96%B9%E6%B3%95.html)

```shell
# function 关键字可以省略，return value只能是整数，代表返回状态
function function_name () {
    list of commands
    [ return value ]
}

# 赋值方式
function fun1(){
    res=`cmd xxxx`
    echo $res
}
res=`fun1`

# 传参赋值方式
function fun2(){
    tmp=`res`
    eval "$1=$res"
}
res=""
fun2 $res
echo $res
```

|variable|含义
:-:|-
\$0|脚本本身的名字；
\$#|传给脚本的参数个数；
\$@|传给脚本的所有参数的列表，即被扩展为”\$1” “\$2” “\$3”等；
\$*|以一个单字符串显示所有向脚本传递的参数，与位置变量不同，参数可超过9个即被扩展成”\$1c\$2c\$3”，其中c是IFS的第一个字符；
\$\$|是脚本运行的当前进程ID号；
\$?|是显示最后命令的退出状态，0表示没有错误，其他表示有错误；
\$n|获取参数的值，\$1表示第一个参数，当n>=10时，需要使用\${n}来获取参数。

## Shell 中 \$(())", \$(), ``, \${} 的区别

\$()和``类似，对当中的表达式计算得出结果,
\$(()) 计算数学表达式
参考：<https://www.cnblogs.com/chengd/p/7803664.html>
注意

```shell
#file.txt
#hello world
#this is a test file

# 下面这种for循环结果并不是按行显示，其实未按字符显示，主要原因是`cat xxx`命令的到的结果是"hello{space}world{\n}this{sapce}is……
for line in `cat file.txt`
do 
echo $line
done;

# 正确应该用如下方式
while read line
do
echo $line
done < file.txt

```

## 多个命令先后执行

[在命令行中同时输入多个语句][3]：
直接在linux命令行中可以依次执行多个命令，多个命令间可采用“;”、“&&”和"||"分割，三个分隔符作用不同：
（1）;分割符：前后命令间没有必然的联系，前一个执行结束后、再执行第二个，没有逻辑关联；
（2）&&分隔符：前后命令有逻辑关联，后面的命令是否执行取决于前面的命令是否执行成功，前者执行成功，才会执行后面的命令。
（3）||分隔符：前后命令有逻辑关联，与&&相反，前面的命令执行失败后才能执行后面的命令。

  [3]: <https://blog.csdn.net/cooperdoctor/article/details/84333686>"
