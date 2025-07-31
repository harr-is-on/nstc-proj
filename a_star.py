# a_star.py
"""
A* path-planning module for RMFS grid layouts
--------------------------------------------

* 通用：只需傳入 numpy 佈局矩陣、起點、終點即可回傳最短路徑。
* 可自訂各格子類型的移動成本 (DEFAULT_COSTS)。
* 只允許上下左右 4-方向移動（無對角）。
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import heapq
import numpy as np

Coord = Tuple[int, int]       # (row, col)

# ---------- 1.  走行成本設定（0 = obstacle, 其餘可自行調整） ----------
DEFAULT_COSTS: Dict[int, float] = {
    1: 0.8,   # turn road
    2: 1.0,   # sub-road  (稍高成本 → 模擬巷道較窄或擁擠)
    3: 1.0,   # main road
    4: 1.2,   # picking station
    5: 1.2,   # charge station
    6: 1.2,   # queue charge
    7: 1.2,   # charge leave
    8: 1.2,   # queue pick
    # 0 不列入：代表障礙／貨架 block
}

# ---------- 2.  A* 主函式 ---------- #
def a_star(
    layout: np.ndarray,
    start: Coord,
    goal: Coord,
    costs: Optional[Dict[int, float]] = None,
) -> List[Coord]:
    """
    使用 A* 搜尋 layout 上的最短路徑。
    
    Parameters
    ----------
    layout : np.ndarray
        倉庫佈局矩陣 (H × W)；0 = 障礙，其餘為可走格子類型代碼。
    start, goal : Coord
        起點與終點座標 (row, col) 皆以 layout 索引為準。
    costs : dict, optional
        {cell_code: move_cost}；若為 None 則沿用 DEFAULT_COSTS。
    
    Returns
    -------
    path : List[Coord]
        從 start 到 goal（含首尾）的座標序列；若無路徑則回傳 []。
    """
    if costs is None:
        costs = DEFAULT_COSTS

    # 檢查起終點是否合法
    if not _is_walkable(layout, start, costs) or not _is_walkable(layout, goal, costs):
        raise ValueError("Start 或 goal 為障礙格，無法尋徑！")

    # open_set 優先佇列：[(f, g, node)]
    open_heap: List[Tuple[float, float, Coord]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from: Dict[Coord, Coord] = {}          # 用於最後重建路徑
    g_score: Dict[Coord, float] = {start: 0.0}  # 目前最佳 g 值

    while open_heap:
        f_curr, g_curr, current = heapq.heappop(open_heap)

        # 找到目標－－重建路徑
        if current == goal:
            return _reconstruct_path(came_from, current)

        for nbr in _neighbors(layout, current, costs):
            tentative_g = g_curr + _move_cost(layout, nbr, costs)

            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + _heuristic(nbr, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nbr))

    # 若 while 結束仍未 return → 無路徑
    return []


# ---------- 3.  工具函式 ---------- #
def _heuristic(a: Coord, b: Coord) -> float:
    """曼哈頓距離 (4-方向格)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(
    layout: np.ndarray,
    node: Coord,
    costs: Dict[int, float],
) -> List[Coord]:
    """回傳合法的鄰居座標（4-方向且可通行）"""
    H, W = layout.shape
    r, c = node
    nbrs: List[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and _is_walkable(layout, (nr, nc), costs):
            nbrs.append((nr, nc))
    return nbrs


def _is_walkable(layout: np.ndarray, coord: Coord, costs: Dict[int, float]) -> bool:
    """判斷該格是否可走（非 0 且有定義成本）"""
    cell_code = int(layout[coord])
    return cell_code in costs


def _move_cost(layout: np.ndarray, coord: Coord, costs: Dict[int, float]) -> float:
    """走到指定格子的成本"""
    return costs[int(layout[coord])]


def _reconstruct_path(
    came_from: Dict[Coord, Coord],
    current: Coord,
) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path