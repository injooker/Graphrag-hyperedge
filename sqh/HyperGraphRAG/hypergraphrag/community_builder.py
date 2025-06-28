import numpy as np
import json
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Tuple, List

async def get_all_hyperedge_embeddings(hyperedge_vdb) -> Tuple[List[str], np.ndarray]:
    """提取所有超边的 ID 和其嵌入向量"""
    all_entries = await hyperedge_vdb.get_all_vectors()  # 你应在 storage 中实现该方法
    ids = []
    embeddings = []
    for id_, entry in all_entries.items():
        ids.append(id_)
        embeddings.append(entry["embedding"])
    return ids, np.array(embeddings)


def cluster_hyperedges(ids: List[str], embeddings: np.ndarray, num_clusters: int = 10):
    """对超边向量做聚类，返回社区编号 -> 超边 ID 映射"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_map = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_map[label].append(ids[idx])

    return cluster_map, kmeans.cluster_centers_  # 返回中心点用于查询阶段


async def cluster_and_index_hyperedges(
    hyperedge_vdb,
    num_clusters: int = 10,
    output_path: str = "hyperedge_community_index.json",
    center_path: str = "hyperedge_community_centers.npy"
):
    ids, embeddings = await get_all_hyperedge_embeddings(hyperedge_vdb)
    cluster_map, centers = cluster_hyperedges(ids, embeddings, num_clusters)

    # 保存为 JSON 索引
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, indent=2, ensure_ascii=False)

    # 保存聚类中心向量（后续 query 用）
    np.save(center_path, centers)

    print(f"✅ 超边社区划分完成，保存至：{output_path}")
    return cluster_map