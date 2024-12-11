import numpy as np
from scipy.sparse import csr_matrix
import os
import psutil
import time
import random
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import pandas as pd

def record_memory_usage(interval, memory_records, stop_event):
    """
    定期记录内存使用的线程函数，包括RSS和百分比。
    :param interval: 记录间隔时间（秒）
    :param memory_records: 用于存储内存使用记录的共享列表
    :param stop_event: 用于停止线程的事件
    """
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        memory_info = process.memory_info()
        memory_percent = psutil.virtual_memory().percent  # 获取系统内存使用百分比
        memory_records.append((time.time(), memory_info.rss / 1024 / 1024, memory_percent))  # 转为MB
        time.sleep(interval)


def pagerank_sparse(graph, alpha=0.85, max_iter=100, tol=1e-6):
    """
    高效PageRank实现，适合稀疏矩阵的大规模图。
    """
    nodes = list(set(u for edge in graph for u in edge))
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    row, col = zip(*[(node_index[u], node_index[v]) for u, v in graph])
    data = np.ones(len(graph))
    adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))

    # 归一化（行归一化处理出度）
    row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # 避免分母为0
    norm_adj_matrix = adj_matrix.multiply(1 / row_sums[:, None])

    # 初始化PageRank值
    rank = np.ones(n) / n
    teleport = np.ones(n) / n

    # 迭代计算
    for iter_num in range(max_iter):
        new_rank = alpha * norm_adj_matrix.T.dot(rank) + (1 - alpha) * teleport

        # 判断是否收敛
        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            break
        rank = new_rank

    return {node: rank[node_index[node]] for node in nodes}


def monitor_and_run(graph, output_dir, record_interval=2, num_nodes=1000, num_edges=1000):
    """
    运行PageRank并记录内存和时间数据，同时绘制内存使用折线图。
    """
    # 获取当前时间戳，创建文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 内存记录
    memory_records = []
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=record_memory_usage, args=(record_interval, memory_records, stop_event))
    memory_thread.start()

    # 运行PageRank算法
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem = process.memory_info().rss

    try:
        ranks = pagerank_sparse(graph)
    finally:
        # 停止内存记录线程
        stop_event.set()
        memory_thread.join()

    end_time = time.time()
    end_mem = process.memory_info().rss

    # 记录性能指标
    metrics = {
        "execution_time": end_time - start_time,
        "memory_peak": (end_mem - start_mem) / 1024 / 1024,  # 转换为MB
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }

    # 保存性能指标
    metrics_file = os.path.join(run_dir, f"pagerank_metrics_{timestamp}.txt")
    with open(metrics_file, "w") as f:
        f.write("PageRank Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 保存内存使用记录
    if memory_records:
        memory_file = os.path.join(run_dir, f"memory_records_{timestamp}.txt")
        with open(memory_file, "w") as f:
            f.write("Timestamp,Memory_MB,Memory_Percent\n")
            for record in memory_records:
                f.write(f"{record[0]},{record[1]:.2f},{record[2]:.2f}\n")

        # 绘制折线图
        timestamps, memory_usage, memory_percent = zip(*memory_records)
        elapsed_times = [t - start_time for t in timestamps]

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(elapsed_times, memory_usage, marker='o', label='Memory Usage (MB)')
        plt.xlabel('Elapsed Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(elapsed_times, memory_percent, marker='x', color='orange', label='Memory Usage (%)')
        plt.xlabel('Elapsed Time (s)')
        plt.ylabel('Memory Usage (%)')
        plt.title('Memory Percent Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"memory_plot_{timestamp}.png"))
        plt.close()

    print(f"Metrics and memory usage data saved to {run_dir}")
    return metrics, ranks


def generate_graph(num_nodes, num_edges):
    """
    随机生成图
    """
    edges = [(random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)) for _ in range(num_edges)]
    return edges


def run_multiple_experiments(output_dir, node_sizes, edge_factor=1):
    """
    运行多个不同节点和边数的实验，并保存到一个Excel文件中
    """
    all_metrics = []
    for num_nodes in node_sizes:
        num_edges = num_nodes * edge_factor  # 边数与节点数的比例
        print(f"Running experiment with {num_nodes} nodes and {num_edges} edges")
        graph = generate_graph(num_nodes, num_edges)
        metrics, ranks = monitor_and_run(graph, output_dir, num_nodes=num_nodes, num_edges=num_edges)
        all_metrics.append(metrics)

    # 保存所有实验结果到Excel
    df = pd.DataFrame(all_metrics)
    excel_file = os.path.join(output_dir, "pagerank_results.xlsx")
    df.to_excel(excel_file, index=False)
    print(f"All experiments completed. Results saved to {excel_file}")


# 设置实验的节点数
node_sizes = [100000, 500000, 1000000, 5000000, 10000000]

# 输出结果目录
output_dir = "pagerank_results"
os.makedirs(output_dir, exist_ok=True)

# 执行多个实验并保存结果
run_multiple_experiments(output_dir, node_sizes)


