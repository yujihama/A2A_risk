import importlib
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import yaml

from .edge import Edge
from .condition import RuleCondition, LLMCondition
from .node_base import Node

logger = logging.getLogger(__name__)


CONDITION_TYPE_MAP = {
    "rule": RuleCondition,
    "llm": LLMCondition,
}


def _import_obj(path: str):
    """Import dotted path and return attribute"""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid import path: {path}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def load_graph(yml_path: str | Path):  # noqa: ANN001
    """YAML から nodes & edges を生成するユーティリティ

    Returns: (nodes_dict, edges_list, start_node_id)
    """
    path_obj = Path(yml_path)
    data = yaml.safe_load(path_obj.read_text(encoding="utf-8"))

    nodes_data = data.get("nodes", [])
    edges_data = data.get("edges", [])
    start_node = data.get("start") or (nodes_data[0]["id"] if nodes_data else None)

    nodes: Dict[str, Node] = {}
    for n in nodes_data:
        impl_path = n.get("impl")
        node_id = n.get("id")
        cls = _import_obj(impl_path)
        node_instance: Node = cls()  # type: ignore[call-arg]
        node_instance.id = node_id
        nodes[node_id] = node_instance

    edges: List[Edge] = []
    for e in edges_data:
        cond_type = e.get("type", "rule")
        cond_cls = CONDITION_TYPE_MAP.get(cond_type)
        if not cond_cls:
            logger.warning("Unknown condition type %s, skipping edge", cond_type)
            continue
        if cond_type == "rule":
            cond = cond_cls(e["expr"])
        else:  # llm
            cond = cond_cls(prompt=e["prompt"], priority=e.get("priority", 10))
        edges.append(Edge(src=e["src"], dst=e["dst"], condition=cond, priority=getattr(cond, "priority", 0)))

    return nodes, edges, start_node 