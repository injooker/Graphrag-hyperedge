import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # 1. 从向量数据库中检索 top-k 个 hyperedge（第一跳）
    results = await hyperedges_vdb.query(keywords, top_k=query_param.top_k)
    if not results:
        return "", "", ""

    # 2. 获取 hyperedge1 的详细信息
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["hyperedge_name"]) for r in results]
    )

    edge_datas = [
        {"hyperedge": r["hyperedge_name"], "rank": r["distance"], **e}
        for r, e in zip(results, edge_datas) if e is not None
    ]

    if not edge_datas:
        logger.warning("Some edges are missing, maybe the storage is damaged")
        return "", "", ""

    # 3. 排序 + 截断
    edge_datas = sorted(edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["hyperedge"],
        max_token_size=query_param.max_token_for_global_context,
    )

    # 4. 收集 hyperedge1 → entities
    related_entities = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(e["hyperedge"]) for e in edge_datas]
    )

    # 5. 建立 hyperedge1 → entity → hyperedge2 路径
    second_hop_hyperedges = set()
    entity_names = set()
    for entity_list in related_entities:
        for ent_type, ent_name in entity_list:
            if not ent_name.startswith("hyperedge"):  # 筛出实体节点
                entity_names.add(ent_name)

    # 6. 实体 → hyperedge2 反向跳
    for ent_name in entity_names:
        neighbors = await knowledge_graph_inst.get_node_edges(ent_name)
        for t, n in neighbors:
            if n.startswith("hyperedge"):
                second_hop_hyperedges.add(n)

    # 7. 获取 hyperedge2 的详细信息
    edge2_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(h) for h in second_hop_hyperedges]
    )
    edge2_data = [
        {"hyperedge": h, "rank": 9999, **e}
        for h, e in zip(second_hop_hyperedges, edge2_data)
        if e is not None
    ]

    # 8. 合并 1-hop 和 2-hop edge 数据
    edge_datas.extend(edge2_data)

    # 9. 每个 hyperedge 找到相关节点
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(e["hyperedge"]) for e in edge_datas]
    )
    edge_datas = [
        {**e, "related_nodes": "|".join([n[1] for n in nodes])}
        for e, nodes in zip(edge_datas, all_related_nodes)
    ]

    # 10. 查找相关实体
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )

    # 11. 查找相关文本块
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    logger.info(
        f"2-hop query uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    # 12. 构建 CSV 上下文
    relations_section_list = [["id", "hyperedge", "related_entities"]]
    for i, e in enumerate(edge_datas):
        relations_section_list.append([i, e["hyperedge"], e["related_nodes"]])
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append([
            i,
            n["entity_name"],
            n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN")
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context