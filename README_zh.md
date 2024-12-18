# 中文文档

## 新增配置

根据README.md中描述，在执行`make menuconfig`需要添加配置

```
CONFIG_HTMM
```

现已将该配置写在`linux/arch`目录下的`Kconfig`配置文件中

无需再次修改生成的文件

## 实验配置

测试脚本存放于`my_test_script`目录之下

### 对比实验的配置

开启`AUTONUMA`：

```

echo 1 | sudo tee /proc/sys/kernel/numa_balancing
```

查看`AUTONUMA`:

```
cat /proc/sys/kernel/numa_balancing
```

### 测试脚本

采用PageRank来作为测试工具，分为`pagerank.py`和`large_pagerank.py`两个测试文件

