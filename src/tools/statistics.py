"""Utility functions for working with label statistics and sorting."""

from enum import Enum
from typing import Any, Dict, List

from langchain_core.tools import tool


class Order(str, Enum):
    """Sorting order for collection utilities."""

    ASC = "asc"
    DESC = "desc"


@tool
def invert_index(id2labels: Dict[int, List[str]]) -> Dict[str, List[int]]:
    """Agent tool: invert mapping of item ids to labels into labels to ids.

    Args:
        id2labels: Mapping of item identifiers to the labels they carry.
    """
    label2ids: Dict[str, List[int]] = {}
    for item_id, labels in id2labels.items():
        for label in labels:
            ids = label2ids.setdefault(label, [])
            if item_id not in ids:
                ids.append(item_id)
    return label2ids


@tool
def sort_by_len(mapping: Dict[str, List[Any]], order: Order) -> List[str]:
    """Agent tool: sort mapping keys by the length of their lists.

    Args:
        mapping: Mapping whose keys should be sorted.
        order: Sorting direction (`Order.ASC` or `Order.DESC`).
    """
    if order == Order.ASC:
        return sorted(mapping, key=lambda key: (len(mapping[key]), key))
    if order == Order.DESC:
        return sorted(mapping, key=lambda key: (-len(mapping[key]), key))
    raise ValueError(f"unsupported order: {order}")
