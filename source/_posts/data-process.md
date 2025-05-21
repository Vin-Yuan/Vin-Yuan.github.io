---
title: data_process
mathjax: true
date: 2024-06-04 14:52:38
categories:
tags:
---

# Byte, KB, MB的笔记

<http://myrepono.com/faq/4>

> A byte is a sequence of 8 bits (enough to represent one alphanumeric character) processed as a single unit of information. A single letter or character would use one byte of memory (8 bits), two characters would use two bytes (16 bits).

1024 bytes = 1 KB
1024 KB = 1 MB
1024 MB = 1 GB
1024 GB = 1 TB
1024 TB = 1 PB
KB = KilobyteM
B = MegabyteG
B = GigabyteT
B = TerabyteP
B = Petabyte

## dataframe 估算内存

```python

def memory_usage(df):
    types = df.dtypes
    s = df.memory_usage(deep=True)
    s = s/1024**2
    total_mem = s.sum()
    for column in df.columns:
        if s[column] < 0.01:
            print("{} = {} KB,  {}".format(column, s[column] * 1024, types[column]))
        else:
            print("{} = {:1.2f} MB,  {}".format(column, s[column], types[column]))
    print("totoal memory: {:.2f} MB".format(total_mem))

```
<!-- more -->

## 读文件

```python
df = pd.read_csv(live_data_path, names=['id', 'index_image', 'view_num', 'dt'], dtype={'id':str, 'index_image':str, 'view_num':str, 'dt':str}, parse_dates=['dt'], index_col=['id'])
```

## dataframe 列数据格式format

1.读文件时
frame = pandas.DataFrame({..some data..},dtype=[str,int,int])

2.读文件后
frame['some column'] = frame['some column'].astype(float)

3.遇到错误（NULL不能转换时）
df['count'] = df['count'].apply(pd.to_numeric, errors='coerce').fillna(0.0)

## pandas 取行，列的方式

pandas 有两种索引：

1. name（index,column名称)
2. int (整数，列或行的数组位置）

- df[...]: 用索引取
- ix: int 和 name 混用，现在已经deprecated
- loc: name
- iloc: int

参看[《pandas取dataframe特定行/列》](https://www.cnblogs.com/nxf-rabbit75/p/10105271.html)

```python
import numpy as np
from pandas import DataFrame
import pandas as pd
 
df=DataFrame(np.arange(12).reshape((3,4)),index=['one','two','thr'],columns=list('abcd'))
 
df['a']#取a列
df[['a','b']]#取a、b列
 
#ix可以用数字索引，也可以用index和column索引
df.ix[0]#取第0行
df.ix[0:1]#取第0行
df.ix['one':'two']#取one、two行
df.ix[0:2,0]#取第0、1行，第0列
df.ix[0:1,'a']#取第0行，a列
df.ix[0:2,'a':'c']#取第0、1行，abc列
df.ix['one':'two','a':'c']#取one、two行，abc列
df.ix[0:2,0:1]#取第0、1行，第0列
df.ix[0:2,0:2]#取第0、1行，第0、1列
 
#loc只能通过index和columns来取，不能用数字
df.loc['one','a']#one行，a列
df.loc['one':'two','a']#one到two行，a列
df.loc['one':'two','a':'c']#one到two行，a到c列
df.loc['one':'two',['a','c']]#one到two行，ac列
 
#iloc只能用数字索引，不能用索引名
df.iloc[0:2]#前2行
df.iloc[0]#第0行
df.iloc[0:2,0:2]#0、1行，0、1列
df.iloc[[0,2],[1,2,3]]#第0、2行，1、2、3列
 
#iat取某个单值,只能数字索引
df.iat[1,1]#第1行，1列
#at取某个单值,只能index和columns索引
df.at['one','a']#one行，a列
```

## 按行条件判断处理

```python
import pandas as pd
df = pd.DataFrame({'A': [50, 50], 
                   'B' : [150, 30],
                   'C': [11, 40]})    

def adjust_min(row):
    res = min(row['A'], row['B'], row['C'])
    if res == row['A']:
        row['min'] = 'A'
    elif res == row['B']:
        row['min'] = 'B'
    else:
        row['min'] = 'C'
    return row['min']


df['Min'] = df.apply(adjust_min, axis=1)
```

## group by

### count sort

<https://stackoverflow.com/questions/40454030/count-and-sort-with-pandas>

```python
df = pd.DataFrame({'STNAME':list('abscscbcdbcsscae'),
                   'CTYNAME':[4,5,6,5,6,2,3,4,5,6,4,5,4,3,6,5]})

print (df)
    CTYNAME STNAME
0         4      a
1         5      b
2         6      s
3         5      c
4         6      s
5         2      c
6         3      b
7         4      c
8         5      d
9         6      b
10        4      c
11        5      s
12        4      s
13        3      c
14        6      a
15        5      e

df = df[['STNAME','CTYNAME']].groupby(['STNAME'])['CTYNAME'] \
                             .count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) \
                             .head(5)

print (df)
  STNAME  count
2      c      5
5      s      4
1      b      3
0      a      2
3      d      1
```

### groupby 后保留列

<https://stackoverflow.com/questions/19202093/how-to-select-columns-from-groupby-object-in-pandas/26668184>

```python
df = pd.DataFrame({'a': [1, 1, 3],
                   'b': [4.0, 5.5, 6.0],
                   'c': [7L, 8L, 9L],
                   'name': ['hello', 'hello', 'foo']})
df.groupby(['a', 'name']).median()
# 会得到结果如下
            b    c
a name            
1 hello  4.75  7.5
3 foo    6.00  9.0
```

通过groupby后可能groupBy的字段无法通过df['name']访问，因为已经变成了index
如果保证其能访问，可以在groupby里设置参数
groupby(..., **as_index=False**)

## pandas 读取 excel文件

需要安装xlrd
<https://www.cnblogs.com/yfacesclub/p/11232736.html>

```python
pd.read_excel(path, sheet_name=0, header=0, names=None, index_col=None, 
              usecols=None, squeeze=False,dtype=None, engine=None, 
              converters=None, true_values=None, false_values=None, 
              skiprows=None, nrows=None, na_values=None, parse_dates=False, 
              date_parser=None, thousands=None, comment=None, skipfooter=0, 
              convert_float=True, **kwds)
```

## pandas print格式调整

<https://blog.csdn.net/weekdawn/article/details/81389865>
<https://www.cnblogs.com/yoyo1216/p/12367713.html>

```python
#pd.options.display.max_colwidth = 100
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',200)
pd.set_option('expand_frame_repr', False
```

## map, filter

```python
import glob
import os
import re

train_logs = glob.glob('/data/vinyuan/bpr-models/MSNClick/sgd-unified_event_data_0715_0807*/train.log')
train_logs= sorted(train_logs, key=lambda x: os.path.getmtime(x), reverse=True)

text='WeightedSample'

fix_params = ['UserNegCross=0.2', 'PosL2=0.1', 'NegL2=0.15']
fix_params_str = "\\n".join(fix_params)+'\\n' if len(fix_params) > 0 else ""
pattern=re.compile(r".*{}-(\d+\.?\d*)".format(text))
train_logs = train_logs[:4]

def filter_by_fix_param(x):
    value, model_name = x
    for param in fix_params:
        if param not in model_name:
            return -1, model_name
    return value, model_name
            
    
def fun(x):
    res = pattern.match(x)
    model_name = x.split('/')[5]
    if res is None:
        return (-1, model_name)
    else:
        value = float(res.group(1))
        if value <= 0:
            return (0, model_name)
        else:
            return (value, model_name)
        
train_logs = map(fun, train_logs)
train_logs = filter(filter_by_fix_param, train_logs)
train_logs = filter(lambda x: x[0] > 0, train_logs)
train_logs = sorted(train_logs, key=lambda x: x[0], reverse=True)
```

## (co1, co2, co3 -> new_col)

```python
def cal_diff(row):
    diff = set(row['a']) - set(row['b'])
    diff = list(diff)
    return diff
df['c'] = df.apply(lambda row: cal_diff(row), axis=1)
```

## Numpy

## 按行和列填值

```python
a = np.zeros([5,3])
a[[0,1],[2,2]] = 1
a
>>>
array([[0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
```

## 从数值范围创建递增数组

1. numpy.arange(start, stop, step, dtype)
2. np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
3. numpy.logspace

## DataFrame 统计空值

```python
data1 = {'itemId':[1,2,3,4], 'other':['a','b','c','d']}
data2 = {'itemId':[2,3], 'label':['B','C']}
df_data1 = pd.DataFrame(data1)
df_data2 = pd.DataFrame(data2)
df_merge = pd.merge(df_data1, df_data2, on=['itemId'], how='left')
print(df_data1)
print(df_data2)
print(df_merge)
a = df_tmp.where(df_merge['label'].isnull())['itemId'].count()
```
