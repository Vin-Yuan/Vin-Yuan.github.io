---
title: hive
mathjax: true
date: 2024-06-04 15:11:19
categories:
tags:
---

## 查看表的信息

可以看到表的 location, tableType等
根据Table Type值可以知道表是内部表还是外部表

```python
describe extended tablename
desc formatted tablename;
```

查看表分区

```bash
>>show partitions ${table_name}
```

<!-- more -->
## Hive 输出日志分析

```shell
……
Partition xxx{dt=2019-11-12} stats: [numFiles=6, numRows=15677698, totalSize=1281943134, rawDataSize=1266265436]
……
```

numRows 是本次hive执行生成的数据量，这里要注意，实际查看这个数字时遇到一个现象：
hql-1:   insert overwrite 3 line on partition (dt = '2019-11-11')
hql-2:   insert into 2 line on partition (dt = '2019-11-11')
在执行hql-2的时候，日志最后显示的numRows=5, 原本的理解时这里应该是2，从我角度理解来看这应该是指hive会重新整理数据，整理的数据包括上一次已经插入表中的3条数据，所以最后产生的数据是5。

## 创建表

### create

```SQL
drop table ${prod_stat};
create table ${prod_stat}
(
    sku        string,
    cnt        string
)
PARTITIONED BY (dt string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS orc
LOCATION '$tbl_hdfs'
TBLPROPERTIES ("orc.compress"="SNAPPY");
```

创建双字段分区表：

```python
create table if not exists ${overlap_table_name}
(
    expo_uv string,
    clk_pv string,
)
partitioned by (dt string, contain_in_skus boolean)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
location 'hdfs://ns1013/user/recsys/recpro/tmpr.db/xxx'

#插入数据

insert overwrite table ${overlap_table_name} partition(dt='${dt}')
select ....
```

### 删除表

```SQL
truncate table table_name;
alter table table_name drop partition (partition_name='yyyy-mm-dd')
```

#### 除hive内部表，不会删除数据的方法探索

<https://blog.csdn.net/qomoman/article/details/50516560>
亲测第一种方法可行

##### 方法一：将内部表改成外部表

```hive
alter table table_name set TBLPROPERTIES('EXTERNAL'='TRUE');//内部表转外部表
drop table table_name
```

##### 方法二

1. 将表的名字改掉
`alter table table_name rename to table_name_temp`
2. 将表的数据所存放的目录改掉
`hadoop   fs -mv  /user/hive/warehouse/table_name_temp /user/hive/warehouse/table_name`
3. `drop table table_name`

#### 删除分区

```hive
 alter table tmpr.table drop partition(dt>='2021-01-05', dt<='2021-03-08')
```

### 内部表和外部表区别

1 删除内表时，内表数据会一并删除；
2 删除外表时，外表数据依旧存在。

### like

```bash
hql="drop table ${hp_few_sku};
create table ${hp_few_sku}
like ${hp_table}
location '${location}'
;
```

### alter

```SQL
ALTER TABLE name RENAME TO new_name
ALTER TABLE name ADD COLUMNS (col_spec[, col_spec ...])
ALTER TABLE name DROP [COLUMN] column_name
ALTER TABLE name CHANGE column_name new_name new_type
ALTER TABLE name REPLACE COLUMNS (col_spec[, col_spec ...])
```

### 创建表可以包含的关键字

```hive
CREATE [EXTERNAL] TABLE [IF NOT EXISTS] table_name
   [(col_name data_type [COMMENT col_comment], ...)]:指定表的名称和表的具体列信息。
   [COMMENT table_comment] :表的描述信息。
   [PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)]:表的分区信息。
   [CLUSTERED BY (col_name, col_name, ...) 
   [SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS]:表的桶信息。
   [ROW FORMAT row_format] :表的数据分割信息，格式化信息。
   [STORED AS file_format] :表数据的存储序列化信息。
   [LOCATION hdfs_path] :数据存储的文件夹地址信息。
```

## Insert

```hive
INSERT OVERWRITE [ LOCAL ] DIRECTORY directory_path
    [ ROW FORMAT row_format ] [ STORED AS file_format ]
    { VALUES ( { value | NULL } [ , ... ] ) [ , ( ... ) ] | query }
```

例子：

```SQL
INSERT OVERWRITE [LOCAL] DIRECTORY directory1 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' 
select_statement1;
```

### 生成数据常用config

```hive
set mapred.max.split.size=524288000;
set mapred.min.split.size=524288000;
set mapred.min.split.size.per.node=524288000;
set mapred.min.split.size.per.rack=524288000;
set hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
set hive.merge.mapfiles = true;
set hive.merge.mapredfiles = true;
set hive.merge.size.per.task = 256000000;
set hive.merge.smallfiles.avgsize = 104857600;
```

## 从本地导入数据到表中

load data inpath 'filepath' into table tablename
这里的filepath是hdfs路径，而非本地文本文件路径，所以最好将文本文件上传至hdfs目录, 然后再导入[【参考】](https://blog.csdn.net/xiongbingcool/article/details/82982099)

```hive
hdfs dfs -put 'local_file_path' 'hdfs://……/user/local/user.login_log'
```

导入本地数据

```hive
LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename [PARTITION (partcol1=val1, partcol2=val2 ...)]
```

这里需要注意的是如果使用的时：当使用hdf路径时不加local,即：`load data inpath hdfs://xxxx/data/`，并且在导入数据后会**源地址数据会被删除**
example

```hive
load data inpath 'hdfs://ns1007/user/recsys/recpro/yuanwenwu3/tmpr.feedback_bad_1200_in_15' overwrite into table tmpr.feedback_bad_1200_in_15 partition(dt='2019-09-03');
```

## 简略流程

```hive
# step-1 创建临时表
create table if not exists zx_user.temp_ly_sentitive_phone(phone STRING)
row format delimited
fields terminated by ','
stored as textfile;
# step-2 将本地数据导入临时表
LOAD DATA local INPATH '/home/zx_user/fmd5_0916_1021/sentitive_phone.csv' INTO TABLE zx_user.temp_ly_sentitive_phone
# step-3 验证
select * from zx_user.temp_ly_sentitive_phone_200917 limit 1;

```

## 查询SQL语句

### 按照排序截取指定位置的数据

场景：按照某一字段排序，然后选择第排序1000位置的数据

```SQL
select * from 
(
    select name,
    cnt,
    row_number() over (order by cnt desc) as rank 
    from tmpr.yuan_test
) a 
where a.rank = 3;
```

## 上传文件

```bash
hadoop fs -put /home/user/train.tar.gz 'hdfs://ns1013/user/.../train'
```

## 下载文件

```bash
hadoop fs -get 'hdfs://ns1013/user/.../train' /home/user/train.tar.gz
```

## hadoop 查看文件大小

```bash
hadoop fs -du -s -h hdfs://ns1013/0530/train/data/
# -h humanity
```

## hadoop 更改组权限

hadoop fs -chgrp -R user1

## LEFT JOIN 后面接AND和接WHERE的区别

参考博客[例子](https://blog.csdn.net/henrrywan/article/details/90207961)

```hive
[nd1:21000] default> select * from test_user;
+----+------+-----+---------+
| id | name | age | classid | 
+----+------+-----+---------+
| 1  | 张三 | 20  | 1       |
| 2  | 李四 | 21  | 1       |
| 3  | 王五 | 22  | 2       | 
| 4  | Lucy | 18  | 3       |
| 5  | Jack | 18  | 3       |
| 6  | Tom  | 18  | 3       |
+----+------+-----+---------+

[nd1:21000] default> select * from test_class;
+----+----------+
| id | classname|
+----+----------+
| 1  | class101 |
| 2  | class102 |
| 3  | class103 |
| 4  | class104 |
+----+----------+
```

### 带ON和AND的SQL查询

```hive
select t1.id,t1.name,t2.classname from test_user t1
left join test_class t2 
on t1.classid=t2.id and t1.id=1

+----+------+----------+
| id | name | classname|
+----+------+----------+
| 1  | 张三 | class101 | 
| 2  | 李四 | NULL     | 
| 3  | 王五 | NULL     | 
| 4  | Lucy | NULL     |
| 5  | Jack | NULL     | 
| 6  | Tom  | NULL     | 
+----+------+----------+

--这里我们把on后面的查询条件放在括号里面，结果一致。
select t1.id,t1.name,t2.classname from test_user t1
left join test_class  t2 
on (t1.classid = t2.id and t1.id = 1)
```

对于left join，**on条件是在生成临时表时使用的条件(参照上面的博客，这里用来过滤t2表的数据，而t1的数据全部显示，也就是说关联上了就显示t2数据，关联不上就显示NULL)**，它不管on中的条件是否为真，都会返回左边表中的记录，on后面的只作为关联条件。

```hive
select t1.id,t1.name,t2.classname from test_user t1
left join test_class t2
on t1.classid=t2.id
where t1.id=1

+----+------+----------+
| id | name | classname| 
+----+------+----------+
| 1  | 张三 | class101 |
+----+------+----------+

```

**where条件是在临时表生成好后，再对临时表进行过滤的条件。这时已经没有left join的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉**
其实比较好的写法是：

```hive
select u.id, u.name, c.classname from (select * from test_user where id = 1) u left join test_class c on u.classid=c.id;
```

尽量使用left join-on-where标准写法**，on只出现连接字段**，过滤条件放在where里面。

对于inner join 则没有这个顾虑，在join的过程正过滤掉数据有利于提高效率
主要结论如下：

- LEFT JOIN 后面如果只接ON查询，会显示所有左表的数据，右表中的数据是否显示取决于后面的查询条件
- LEFT JOIN 后面接WHERE查询，会根据WHERE条件先对数据进行过滤
- LEFT JOIN 后面条件查询，条件一定不要恒为真，否则会出现笛卡尔积
- LEFT JOIN 后面条件查询，条件一定不要恒为假，否则查询结果中右表数据始终为NULL
- RIGHT JOIN 和LEFT JOIN 特性相同，INNER JOIN没这个特殊性，不管条件是放在ON中，还是放在WHERE中，返回的结果集是相同的

## Rack & Node & cluster

参考[回答](https://www.quora.com/What-is-the-rack-in-a-Hadoop-cluster-How-does-it-work)

一个node就是一台电脑，一堆nodes的存储空间被称作rack。通常一个rack是30-40个有实际物理存储空间的nodes组成的，且这些nodes位置比较近，且连接于同一个network switch(交换机)。基于这个结构可知：同一rack中的任意两个nodes之间通信的带宽大于不同ranck中的nodes。一个Hadoop Cluster就是racks的集合。
>Whenever any data is stored in hdfs ,it is replicated (default 3) in which two copies are in same rack and one outside the rack. So that if whole rack goes down then also data can be retrived.

(上面这一点即Data protection against rack failure）

### Rack Awareness

In a large cluster of Hadoop, in order to improve the network traffic while reading/writing HDFS file, **namenode chooses the datanode which is closer to the same rack or nearby rack to Read/Write request**. Namenode achieves rack information by maintaining the rack id’s of each datanode. This concept that chooses closer datanodes based on the rack information is called **Rack Awareness** in Hadoop.

### node

A node in hadoop simply means a computer that can be used for processing and storing. There are two types of nodes in hadoop Name node and Data node. It is called as a node as all these computers are interconnected.

### [NameNode and DataNode](https://www.quora.com/What-is-the-difference-between-Namenode-+-Datanode-Jobtracker-+-Tasktracker-Combiners-Shufflers-and-Mappers+Reducers-in-their-technical-functionality-and-physically-ie-whether-they-are-on-the-same-machine-in-a-cluster-while-running-a-job)

Technical Sense: NameNode stores MetaData(No of Blocks, On Which Rack which DataNode the data is stored and other details) about the data being stored in DataNodes whereas the DataNode stores the actual Data.

Physical Sense: In a multinode cluster NameNode and DataNodes are usually on different machines. There is only one NameNode in a cluster and many DataNodes; Thats why we call NameNode as a single point of failure. Although There is a Secondary NameNode (SNN) that can exist on different machine which doesn't actually act as a NameNode but stores the image of primary NameNode at certain checkpoint and is used as backup to restore NameNode.

### JobTracker And TaskTracker

Technical Sense: JobTracker is a master which creates and runs the job. **JobTracker which can run on the NameNode allocates the job to TaskTrackers which run on DataNodes**; TaskTrackers run the tasks and report the status of task to JobTracker.

Physical Sense: The JobTracker runs on MasterNode aka NameNode whereas TaskTrackers run on DataNodes.

## [hive中控制map reduce数量方法](https://blog.csdn.net/zhong_han_jun/article/details/50814246)

- map:

set mapred.max.split.size=256000000; -- 决定每个map处理的最大的文件大小，单位为B
set mapred.min.split.size.per.node=1; -- 节点中可以处理的最小的文件大小
set mapred.min.split.size.per.rack=1; -- 机架中可以处理的最小的文件大小

如果mapred.max.split.size = 240x1024x1024=240M

1. 假设有两个文件大小分别为(256M,280M)被分配到节点A，那么会启动两个map，剩余的文件大小为10MB和35MB因为每个大小都不足241MB会先做保留
2. 根据参数set mapred.min.split.size.per.node看剩余的大小情况并进行合并,如果值为1，表示a中每个剩余文件都会自己起一个map，这里会起两个，如果设置为大于45*1024*1024则会合并成一个块，并产生一个map
如果mapred.min.split.size.per.node为10*1024*1024，那么在这个节点上一共会有4个map，处理的大小为(245MB,245MB,10MB,10MB，10MB，10MB)，余下9MB
如果mapred.min.split.size.per.node为45*1024*1024，那么会有三个map，处理的大小为(245MB,245MB,45MB)
实际中mapred.min.split.size.per.node无法准确地设置成45*1024*1024，会有剩余并保留带下一步进行判断处理
3. 对2中余出来的文件与其它节点余出来的文件根据mapred.min.split.size.per.rack大小进行判断是否合并，对再次余出来的文件独自产生一个map处理

- reduce:
方法1 set mapred.reduce.tasks=10; -- 设置reduce的数量
方法2 set hive.exec.reducers.bytes.per.reducer=1073741824 -- 每个reduce处理的数据量,默认1GB

其他更细的优化参考下面两个连接未完待续
[资料1](https://www.cnblogs.com/xd502djj/p/3799432.html)
[资料2](https://www.cnblogs.com/swordfall/p/11037539.html) 2019-09-04 18:09:24