---
title: spark
mathjax: true
date: 2024-06-04 14:51:08
categories:
tags:
---

## pyspark API
<https://spark.apache.org/docs/2.2.1/api/python/search.html?q=dataframe.write>

## 简明教程
<https://sparkbyexamples.com/pyspark/pyspark-orderby-and-sort-explained/>
<http://www.learnbymarketing.com/1100/pyspark-joins-by-example/>

<!-- more -->
## Spark调优
<https://tech.meituan.com/2016/04/29/spark-tuning-basic.html>
资源参数调优
了解完了Spark作业运行的基本原理之后，对资源相关的参数就容易理解了。所谓的Spark资源参数调优，其实主要就是对Spark运行过程中各个使用资源的地方，通过调节各种参数，来优化资源使用的效率，从而提升Spark作业的执行性能。以下参数就是Spark中主要的资源参数，每个参数都对应着作业运行原理中的某个部分，我们同时也给出了一个调优的参考值。

- num-executors
参数说明：该参数用于设置Spark作业总共要用多少个Executor进程来执行。Driver在向YARN集群管理器申请资源时，YARN集群管理器会尽可能按照你的设置来在集群的各个工作节点上，启动相应数量的Executor进程。这个参数非常之重要，如果不设置的话，默认只会给你启动少量的Executor进程，此时你的Spark作业的运行速度是非常慢的。
参数调优建议：每个Spark作业的运行一般设置50~100个左右的Executor进程比较合适，设置太少或太多的Executor进程都不好。设置的太少，无法充分利用集群资源；设置的太多的话，大部分队列可能无法给予充分的资源。
- executor-memory
参数说明：该参数用于设置每个Executor进程的内存。Executor内存的大小，很多时候直接决定了Spark作业的性能，而且跟常见的JVM OOM异常，也有直接的关联。
参数调优建议：每个Executor进程的内存设置4G~8G较为合适。但是这只是一个参考值，具体的设置还是得根据不同部门的资源队列来定。可以看看自己团队的资源队列的最大内存限制是多少，num-executors乘以executor-memory，是不能超过队列的最大内存量的。此外，如果你是跟团队里其他人共享这个资源队列，那么申请的内存量最好不要超过资源队列最大总内存的1/3~1/2，避免你自己的Spark作业占用了队列所有的资源，导致别的同学的作业无法运行。
- executor-cores
参数说明：该参数用于设置每个Executor进程的CPU core数量。这个参数决定了每个Executor进程并行执行task线程的能力。因为每个CPU core同一时间只能执行一个task线程，因此每个Executor进程的CPU core数量越多，越能够快速地执行完分配给自己的所有task线程。
参数调优建议：Executor的CPU core数量设置为2~4个较为合适。同样得根据不同部门的资源队列来定，可以看看自己的资源队列的最大CPU core限制是多少，再依据设置的Executor数量，来决定每个Executor进程可以分配到几个CPU core。同样建议，如果是跟他人共享这个队列，那么num-executors * executor-cores不要超过队列总CPU core的1/3~1/2左右比较合适，也是避免影响其他同学的作业运行。
- driver-memory
参数说明：该参数用于设置Driver进程的内存。
参数调优建议：Driver的内存通常来说不设置，或者设置1G左右应该就够了。唯一需要注意的一点是，如果需要使用collect算子将RDD的数据全部拉取到Driver上进行处理，那么必须确保Driver的内存足够大，否则会出现OOM内存溢出的问题。
- spark.default.parallelism
参数说明：该参数用于设置每个stage的默认task数量。这个参数极为重要，如果不设置可能会直接影响你的Spark作业性能。
参数调优建议：Spark作业的默认task数量为500~1000个较为合适。很多同学常犯的一个错误就是不去设置这个参数，那么此时就会导致Spark自己根据底层HDFS的block数量来设置task的数量，默认是一个HDFS block对应一个task。通常来说，Spark默认设置的数量是偏少的（比如就几十个task），如果task数量偏少的话，就会导致你前面设置好的Executor的参数都前功尽弃。试想一下，无论你的Executor进程有多少个，内存和CPU有多大，但是task只有1个或者10个，那么90%的Executor进程可能根本就没有task执行，也就是白白浪费了资源！因此Spark官网建议的设置原则是，设置该参数为num-executors * executor-cores的2~3倍较为合适，比如Executor的总CPU core数量为300个，那么设置1000个task是可以的，此时可以充分地利用Spark集群的资源。
- spark.storage.memoryFraction
参数说明：该参数用于设置RDD持久化数据在Executor内存中能占的比例，默认是0.6。也就是说，默认Executor 60%的内存，可以用来保存持久化的RDD数据。根据你选择的不同的持久化策略，如果内存不够时，可能数据就不会持久化，或者数据会写入磁盘。
参数调优建议：如果Spark作业中，有较多的RDD持久化操作，该参数的值可以适当提高一些，保证持久化的数据能够容纳在内存中。避免内存不够缓存所有的数据，导致数据只能写入磁盘中，降低了性能。但是如果Spark作业中的shuffle类操作比较多，而持久化操作比较少，那么这个参数的值适当降低一些比较合适。此外，如果发现作业由于频繁的gc导致运行缓慢（通过spark web ui可以观察到作业的gc耗时），意味着task执行用户代码的内存不够用，那么同样建议调低这个参数的值。
- spark.shuffle.memoryFraction
参数说明：该参数用于设置shuffle过程中一个task拉取到上个stage的task的输出后，进行聚合操作时能够使用的Executor内存的比例，默认是0.2。也就是说，Executor默认只有20%的内存用来进行该操作。shuffle操作在进行聚合时，如果发现使用的内存超出了这个20%的限制，那么多余的数据就会溢写到磁盘文件中去，此时就会极大地降低性能。
参数调优建议：如果Spark作业中的RDD持久化操作较少，shuffle操作较多时，建议降低持久化操作的内存占比，提高shuffle操作的内存占比比例，避免shuffle过程中数据过多时内存不够用，必须溢写到磁盘上，降低了性能。此外，如果发现作业由于频繁的gc导致运行缓慢，意味着task执行用户代码的内存不够用，那么同样建议调低这个参数的值。
资源参数的调优，没有一个固定的值，需要同学们根据自己的实际情况（包括Spark作业中的shuffle操作数量、RDD持久化操作数量以及spark web ui中显示的作业gc情况），同时参考本篇文章中给出的原理以及调优建议，合理地设置上述参数。

资源参数参考示例
以下是一份spark-submit命令的示例，大家可以参考一下，并根据自己的实际情况进行调节：

```python
/bin/spark-submit \
  --master yarn-cluster \
  --num-executors 100 \
  --executor-memory 6G \
  --executor-cores 4 \
  --driver-memory 1G \
  --conf spark.default.parallelism=1000 \
  --conf spark.storage.memoryFraction=0.5 \
  --conf spark.shuffle.memoryFraction=0.3 \

```

## spark 本地执行

```python
# 本机4个CPU核心上执行
./bin/pyspark --master local[4]
# 本机所有CPU核心上执行
./bin/pyspark --master local[*]
# 查看当前的运行模式
 sc.master
# 读取本地文件（路径前用file:///)
textFile=sc.textFile("file:///usr/local/spark/README.md")
```

# [spark tutorial](https://www.tutorialspoint.com/pyspark/pyspark_sparkcontext.htm)

spark-shell 不用创建sparkContext，默认已经启用了一个，如果再次生成会提示："ValueError: Cannot run multiple SparkContexts at once".

# SparkSession和SparkContext的关系

```
.
└── SparkSession
    └── SparkContext
        ├── RDD1
        ├── RDD2
        └── RDD3

```

SparkSession是Spark 2.0引入的新概念。SparkSession为用户提供了统一的切入点，来让用户学习spark的各项功能。
在spark的早期版本中，SparkContext是spark的主要切入点，由于RDD是主要的API，我们通过SparkContext来创建和操作RDD。对于每个其他的API，我们需要使用不同的context：

- Streaming使用StreamingContext
- sql使用SqlContext
- hive使用HiveContext

但是随着DataSet和DataFrame的API逐渐成为标准的API，就需要为他们建立接入点。所以在spark2.0中，引入SparkSession作为DataSet和DataFrame API的切入点。
SparkSession封装了SparkContext和SQLContext。为了向后兼容，SQLContext和HiveContext也被保存下来。
在大多数情况下，我们不需要显式初始化SparkContext; 而尽量通过SparkSession来访问它。
<https://www.jianshu.com/p/4705988b0c84>

```python
# Creating a SparkSession in Python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("Word Count")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()
    
# 上面的spark时SparkSession对象，使用type可以验证
>>> type(spark)
>>> pyspark.sql.session.SparkSession
```

# [Spark Cluster Components](https://spark.apache.org/docs/latest/cluster-overview.html)

![spark cluster components][1]

Spark applications run as independent sets of processes on a cluster, coordinated by the SparkContext object in your main program (called the **driver program**).
> driver program是协调集群进程的hub，所以当driver不在集群里时（client模式），网络带宽延迟会很影响通信与状态更新。

1. Specifically, to run on a cluster, the SparkContext can **connect** to several types of **cluster managers** (either Spark’s own standalone cluster manager, Mesos or YARN), which allocate resources across applications.
2. Once connected, Spark **acquires executors** on nodes in the cluster, which are **processes** that run computations and store data for your application.
3. Next, it **sends your application code** (defined by JAR or Python files passed to SparkContext) **to the executors**.
4. Finally, SparkContext **sends tasks** to the executors to run.

# Spark Config

## spark.sql.shuffle.partitions
<http://blog.madhukaraphatak.com/dynamic-spark-shuffle-partitions/>

# Spark Submit

```shell
/bin/spark-submit \
--class <main-class> \  # 程序入口，如果是java则是类，java
--master <master-url> \
--deploy-mode <deploy-mode> \ #分为cluster和client两种
--conf <conf-param> \  #一些配置参数
... # other options
<application-jar> \  # 可以是jar或python文件的路径，如果是url则需要所有node可见
[application-arguments] \ 参数
```

**注意**：当提交spark 任务时，如果jar包有多个，用逗号隔开，**逗号前后不能有空格!,有逗号会导致后续参数解析错误，可能提交的scalar程序提示class notFound!**
具体的例子参见[《Submitting Applications》](https://spark.apache.org/docs/latest/submitting-applications.html#submitting-applications)

python的例子：
对于 main.py 依赖的 util.py, module1.py, module2.py，需要先压缩成一个 .zip 文件，再通过 spark-submit 的 --py--files 选项上传到 yarn，mail.py 才能 import 这些子模块。[命令如下](https://www.jianshu.com/p/92be93cfbb97)：

```shell
$ spark-submit 
--master=yarn \
--deploy-mode=cluster \
--jars elasticsearch-hadoop-5.3.1.jar \
--py-files deps.zip \
main.py
```

## deploy-mode

具体参考[《standalone mode》](https://blog.csdn.net/qq_39131779/article/details/83539608)
总结起来就一句话：**culster/client的主要区别就是driver是否在cluster里**

### cluster 集群模式

```shell
/bin/spark-submit \
--master  spark://node01:7077 \
--class org.apache.spark.examples.SparkPi \
../lib/spark-examples-1.6.0-hadoop2.6.0.jar 
100
```

1. client模式提交任务后，会在客户端启动Driver进程。
2. Driver会向Master申请启动Application启动的资源。
3. 资源申请成功，Driver端将task发送到worker端执行。
4. worker将task执行结果返回到Driver端。(由代码设置)

总结：client模式适用于测试调试程序。**Driver进程是在客户端启动的**，这里的客户端就是指提交应用程序的当前节点。**在Driver端可以看到task执行的情况。生产环境下不能使用client模式**，是因为：假设要提交100个application到集群运行，Driver每次都会在client端启动，那么就会导致客户端100次网卡流量暴增的问题。（因为要监控task的运行情况，会占用很多端口，如上图的结果图）客户端网卡通信，都被task监控信息占用。
集群模式如果是用来本地文件，需要添加--files参数
例如：

```shell

#此处代码使用了本地文件'stat.conf'
#如果直接使用 deploy-mode=cluster会报错找不到stat.conf
#所以需要在submit时添加 "--files $con_file"

$start_day='2020-10-01'
$end_day='2020-10-02'
$conf_file='./stat.conf'

spark-submit --class com.jd.rec.FeatStat\
   --num-executors 500 \
   --executor-memory 45g \
   --driver-memory 10g \
   --executor-cores 6 \
   --master yarn \
   --deploy-mode cluster \
   --conf spark.sql.catalogImplementation=hive \
   --conf spark.sql.shuffle.partitions=10000 \
   --conf spark.shuffle.consolidateFiles=true\
   --jars protobuf-java-3.5.1.jar,proto-1.10.0.jar \
   --conf spark.executor.userClassPathFirst=true \
   --files $conf_file \ # 注意此处
   tfrecord-hadoop-trans-11.0.0.jar $conffile $start_day $end_day
```

### client

```shell
/bin/spark-submit \
--master spark://node01:7077 \
--deploy-mode cluster \
--class org.apache.spark.examples.SparkPi \
../lib/spark-examples-1.6.0-hadoop2.6.0.jar  
100
```

1. 客户端使用命令spark-submit --deploy-mode cluster 后会启动spark-submit进程
2. 此进程为Driver向Master申请资源，Driver进程默认需要1G内存和1Core，
2. Master会**随机选择一台**worker节点来启动Driver进程（这样通信会较近，利于信息、状态收集）
3. **Driver启动成功后，spark-submit关闭**，然后Driver向Master申请资源
4. Master接收到请求后，会在资源充足的worker节点上启动Executor进程
5. Driver分发Task到Executor中执行

总结：这种模式会将单节点的网卡流量激增问题分散到集群中。在客户端看不到task执行情况和结果，要去**webui**中看。cluster模式适用于生产环境，Master模式先启动Driver，再启动Application

**spark shell 模式是以client提交的**(第一种)，所以不能加入--deploy-mode cluster的(第二种) **client方式用于测试环境，用于方便查看结果**，因为 spark shell 模式以client方式提交，所以 spark shell 模式不支持--deploy-mode cluster提

# RDD and DataFrame

## Transform函数

### flatMap

```
df = spark.sql("select id_feat from table where dt = '2020-10-01'")
df_split = split(df['id_feat'], '\t')
attrs = ['item_c3','item_br','item_c2','item_sh','item_pw','item_sku']
#print("列名：", attrs)
for index, value in enumerate(attrs):
    df = df.withColumn(value, df_split.getItem(index))
rdd = df.rdd.flatMap(lambda x: fun(x)).reduceByKey(lambda x, y: x+y)
rdd = rdd.map(lambda x: "{0},{1},{2}".format(x[0][0], x[0][1], x[1]))
```

## DataFrame 速查表
<https://sparkbyexamples.com/pyspark/pyspark-structtype-and-structfield/>
<https://www.cnblogs.com/liaowuhen1314/p/12792202.html>

## DataFrame Split 列生成新column
<https://sparkbyexamples.com/spark/spark-split-dataframe-column-into-multiple-columns/>

[split 教程][2]

```
df = spark.sql("select * from xxx where dt = '2020-10-20'")
df[]
df_split = split(df['id_feat'], '\t')
attrs = ['item_c3','item_br','item_c2']
for index, value in enumerate(attrs):
    df = df.withColumn(value, df_split.getItem(index))
print(df.select('item_br').take(3))
```

## DataFrame 筛选
<https://blog.csdn.net/sinat_26917383/article/details/80500349>

### join

例子

```
a_rdd = sc.parallelize([('a', 13132), ('b', 121212),('c',56577)])
b_rdd = sc.parallelize([('a', 23232), ('b',333333)])
columns = ['sku', 'feat']
df_a = a_rdd.toDF(columns)
df_b = b_rdd.toDF(columns)

df_a.join(df_b, df_a.sku==df_b.sku, 'left').where(df_b.sku.isNull()).select(df_a.sku, df_b.feat).show()
```

### Where
<https://stackoverflow.com/questions/35870760/filtering-a-pyspark-dataframe-with-sql-like-in-clause>

```
df = sc.parallelize([(1, "foo"), (2, "x"), (3, "bar")]).toDF(("k", "v"))
df.registerTempTable("df")
sqlContext.sql("SELECT * FROM df WHERE v IN {0}".format(("foo", "bar"))).count()


from pyspark.sql.functions import col
df.where(col("v").isin({"foo", "bar"})).count()


from pyspark.sql.functions import col
df.where(col("v").isin(["foo", "bar"])).count()

# example-2
df = df.where((df['dt'] >= '2020-09-14') & (df['dt'] <= '2020-10-05')).groupby('dt').count()
```

## DataFrame 统计

### GroupBy

```
df = df.where((df['dt'] >= '2020-09-14') & (df['dt'] <= '2020-10-05')).groupby('dt').count()
```

## DataFrame 转 rdd

[参考资料](https://blog.csdn.net/helloxiaozhe/article/details/89414735)
DataFrame的表相关操作不能处理一些问题，例如需要对一些数据利用指定的函数进行计算时，就需要将DataFrame转换为RDD。DataFrame可以直接利用df.rdd获取对应的RDD对象，此RDD对象的每个元素使用Row对象来表示，**每列值会成为Row对象的一个域=>值映射**。例如:

```
>>> lists = [['a', 1], ['b', 2]]
>>> list_dataframe = sqlContext.createDataFrame(lists,['col1','col2'])
>>> list_dataframe.show()
+----+----+                                                                     
|col1|col2|
+----+----+
|   a|   1|
|   b|   2|
+----+----+

>>> rdd=list_dataframe.rdd
>>> rdd.collect()
[Row(col1=u'a', col2=1), Row(col1=u'b', col2=2)] 

>>> rdd.map(lambda x: [x[0], x[1:]]).collect()
[[u'a', (1,)], [u'b', (2,)]]
>>> rdd.map(lambda x: [x[0], x[1]]).collect()
[[u'a', 1], [u'b', 2]]
```

## DataFrame写Hive表

```python
# 2. 处理数据
import pandas as pd
df = pd.read_csv('/user/recsys/recpro/xxx.csv')
df = spark.sql("select * from tmpr.live_person_attributes_yuanwenwu3")
# 3. 写hive表
sc = spark.sparkContext
hiveContext = HiveContext(sc)
data_to_hive_df = hiveContext.createDataFrame(df)
data_to_hive_df.partitionBy('dt').write.format("parquet").mode("overwrite").saveAsTable("tmpr.output_table") 
```

## rdd 转 dataframe

```
df = sqlContext.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
```

schema：DataFrame各列类型信息，在提前知道RDD所有类型信息时设定。例如:

```
schema = StructType([StructField('col1', StringType()), StructField('col2', IntegerType())])
```

## 读写文件

[参考资料1](https://zhuanlan.zhihu.com/p/105893298)
[参考资料2](https://www.jianshu.com/p/d1f6678db183)

### 读文件

```
data = spark.read.csv(filepath, sep=',', header=True, inferSchema=True)
```

### rdd写文件

```
## 单独一个文件
data.repartition(1).write.csv(writepath,mode="overwrite")
data.coalesce(1).write.csv(writepath,mode="overwrite")
## 写成特殊格式，去掉括号
data.map(lambda (k,v): "{0} {1}".format(k,v)).coalesce(1).write.csv(writepath,mode="overwrite")
## 分块儿文件
data.repartition(1000).write.csv(writepath,mode="overwrite")
data.coalesce(1000).write.csv(writepath,mode="overwrite")
```

通常rdd直接write到文件，内容会是如下形式：

```
>>cat rdd_res.txt
……
(k1,v1)
(k2,v2)
……
```

rdd想**没有括号**输出到文本文件可以使用如下方式：

```
data.map(lambda (k,v): "{0} {1}".format(k,v)).coalesce(1).write.csv('path')
#或者转成DataFrame在写
rdd.toDF().write.csv("path")
```

### DataFrame写文件

```
# dataframe 写csv文件
df.write.format("csv").option("header", "false").mode("overwrite").save("hdfs://user/xxx/data.csv")
hadoop fs -getmerge "hdfs://user/xxx/data.csv" "./data.csv"
```

**option** 支持参数
path: csv文件的路径。支持通配符;
header: csv文件的header。默认值是false;
delimiter: 分隔符。默认值是',';
quote: 引号。默认值是"";
mode: 解析的模式。支持的选项有：

## dataframe 添加一列

如果数据很多，想分块处理，以打到如下目的：

### 添加 id 选择区间

```
sku_img_df = spark.sql(sql)
def flat(l):
    sku_img = l[0]
    index = l[1]
    return (sku_img[0], sku_img[1], index)
## 添加 'id' 生成新的dataframe
rdd = sku_img_df.rdd.zipWithIndex()
schema = sku_img_df.schema.add(StructField("id", LongType()))
rdd = rdd.map(lambda x: flat(x))
sku_img_df = spark.createDataFrame(rdd, schema)
## 分区间处理dataframe中的行
batch_size = 30000000
start_index = 0
while start_index < total_count:
    if start_index + batch_size < total_count:
        batch_data_num = batch_size
    else:
        batch_data_num = total_count - start_index
    extract_batch(sku_img_df, start_index, batch_data_num)
    start_index = start_index + batch_size
```

## 读取TXT文件

- spark.sparkContext.textFile(file_path)
读取到的是**RDD** (pyspark.rdd.RDD)，每一个元素直接是字符串

```python
>> rdd = spark.sparkContext.textFile(file_path)
>> rdd.take(3)

[u'first line',
 u'second line',
 u'third line']
```

- spark.read.text(file_path)
读取到的是**DataFrame** (pyspark.sql.dataframe.DataFrame)，每一个元素是 Row (pyspark.sql.types.Row)

```python
>> df = spark.read.text(file_path)
>> rdd = df.rdd
>> rdd.take(3)

[Row(value=u'first line'),
 Row(value=u'second line'),
 Row(value=u'third line')]
```

所以这里如果直接rdd.map(lambda line: line.split(" ")) 会报错，因为操作的元素是Row，Row对象没有split方法。

## DataFrame 写入hive表

### 一般步骤

参考：<https://blog.csdn.net/a2639491403/article/details/80044121>

#### 1.创建数据集的spark DattaFrame

```df_tmp = spark.createDataFrame(RDD,schema)```
这里schema是由StructFied函数定义的

#### 2.将数据集的DataFrames格式映射到零时表

```df_tmp.createOrReplaceTempView('tempTable')```

#### 3.用spark sql语句将零时表的数据导入hive的tmp_table表中

```sqlContext.sql('insert overwrite table des_table select *from tempTable')```

### 写入分区表

参考：<https://xinancsd.github.io/Python/pyspark_save_hive_table.html>

#### df.write.saveAsTable() 方法

需要注意的是hive表名**是不明感大小写**的，经历过如下现象：

```python
df = spark.sql("select tmpr.abc")
df.write.option("delimiter", '\t').saveAsTable('tmpr.ABC', format='hive', mode='append', partitionBy='dt')
会写到原地址，因为如果原地址是：hdfs://xxxx/tmpr.db/abc，在使用新表tmpr.ABC时会从新创建地址，恰好是这一地址，所以数据又写回原地址了

`mode=’overwrite’ `模式时，会创建新的表，若表名已存在则会被删除，整个表被重写。而 `mode=’append’` 模式会在直接在原有数据增加新数据，这一模式可以写入已存在的表。
当使用overwrite时，saveAsTable 会自动创建hive表，partitionBy指定分区字段，默认存储为 parquet 文件格式。对于从文件生成的DataFrame，字段类型也是自动转换的，有时会转换成不符合要求的类型
##### format:
```

hive （hive默认格式，数据文件纯文本无压缩存储）
parquet （spark默认采用格式）
orc
json
csv
text（若用saveAsTable只能保存只有一个列的df）
jdbc
libsvm

```python
df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day'])
需要自定义字段类型的，可以在创建DataFrame时指定类型：
```

from pyspark.sql.types import StringType, StructType, BooleanType, StructField

schema = StructType([
    StructField("vin", StringType(), True),
    StructField("cust_id", StringType(), True),
    StructField("is_maintain", BooleanType(), True),
    StructField("is_wash", BooleanType(), True),
    StructField("pt_day", StringType(), True),
  ]
)

data = pd.read_csv('/path/to/data.csv', header=0)
df = spark.createDataFrame(data, schema=schema)

# 写入hive表时就是指定的数据类型了

df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day']

```python
#### option("delimiter", '\t')
写入hive表需要指定分割符时可使用如上方式

#### df.partitionBy('dt')和df.saveAsTable(partitionBy=['dt'])的区别
https://blog.csdn.net/qq_33536353/article/details/106165924
对于两种写回分区表的方法：
第一种：这种会清理hdfs路径，生成新的dt，**以为着重名旧分区会被删除，切记**
```

df.write.mode("overwrite").format("orc").partitionBy("dt").saveAsTable("aicloud.cust_features")

```python
第二种：这种只会重写覆盖的分区，其他旧分区不会被删除

df.write.saveAsTable("aicloud.cust_features", format="orc", mode="overwrite", partitionBy="dt")

```

## Scala

### Scala匿名函数不可用return

Scala - return in anonymous function
<https://www.jianshu.com/p/2053634328d3>

  [1]: https://spark.apache.org/docs/latest/img/cluster-overview.png
  [2]: <https://sparkbyexamples.com/pyspark/pyspark-split-dataframe-column-into-multiple-columns/> 2019-12-25 14:26:31
vin Javascript & JQuery # Javascript & JQuery

标签（空格分隔）： javascript

---

## JQuery选择器

<https://blog.csdn.net/qq_38225558/article/details/83780618>

## chrome 脚本编辑器

<https://www.jianshu.com/p/87adbf88e2e3>
<https://www.cnblogs.com/liun1994/p/7265828.html>
