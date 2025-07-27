import heapq
import math
from typing import List, Tuple, Optional, Set
import numpy as np

Coord = Tuple[int, int]

def euclidean_distance(pos1: Coord, pos2: Coord) -> float:
    """計算兩點之間的歐幾里得距離。"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_adjacent_aisle(pos: Coord, warehouse_matrix: np.ndarray) -> Optional[Coord]:
    """
    尋找給定位置旁邊的第一個可用走道格。
    這對於將機器人從貨架或工作站移到路徑上至關重要。

    :param pos: 當前位置 (例如貨架)。
    :param warehouse_matrix: 倉庫佈局。
    :return: 旁邊的走道座標，如果找不到則返回 None。
    """
    rows, cols = warehouse_matrix.shape
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and warehouse_matrix[nr, nc] == 0:
            return (nr, nc)
    return None


def plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles: Optional[List[Coord]] = None, forbidden_cells: Optional[Set[Coord]] = None):
    """
    A* pathfinding algorithm for routing in a warehouse.

    參數:
        start_pos (tuple): Start position (row, col).
        target_pos (tuple): Target position (row, col).
        warehouse_matrix (np.ndarray): Warehouse layout (0=aisle, 1=shelf, ...).
        dynamic_obstacles (list, optional): A list of coordinates currently occupied by other robots.
        forbidden_cells (set, optional): A set of coordinates that should be treated as walls for this specific path plan.

    回傳:
        list of tuple: List of positions representing the path, or None if no path is found.
    """
    rows, cols = warehouse_matrix.shape

    # Create a combined set of obstacles for efficient lookup
    if forbidden_cells is None:
        forbidden_cells = set()

    def neighbors(pos: Coord) -> List[Coord]:
        r, c = pos
        candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        valid_neighbors = []
        for nr, nc in candidates:
            if 0 <= nr < rows and 0 <= nc < cols:
                # Check dynamic obstacles
                if dynamic_obstacles and (nr, nc) in dynamic_obstacles and (nr, nc) != target_pos:
                    continue

                # Check forbidden cells provided by the caller
                if (nr, nc) in forbidden_cells and (nr, nc) != target_pos:
                    continue

                # Check static warehouse layout. All non-shelf/station cells are traversable.
                cell_type = warehouse_matrix[nr, nc]
                if cell_type in [0, 4, 5, 6, 7] or (nr, nc) == target_pos:
                    valid_neighbors.append((nr, nc))
        return valid_neighbors

    def heuristic(pos):
        return abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])

    open_list = [(heuristic(start_pos), 0, start_pos, [])]  # (f_score, g_score, pos, path)
    closed_set = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        # 關鍵修正：如果這個節點已經被處理過（在 closed_set 中），就跳過。
        # 這是因為優先佇列可能包含同一個節點的多個條目（透過不同路徑到達），
        # 我們只需要處理成本最低（即第一次從佇列中取出）的那一個。
        if current in closed_set:
            continue

        if current == target_pos:
            # 完整的路徑是 path + [current]，但我們需要的是從下一步開始的路徑。
            # 因此，我們返回從第二個元素開始的切片。
            return (path + [current])[1:]

        # 將當前節點加入已處理集合
        closed_set.add(current)

        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue
            new_g = g + 1
            new_f = new_g + heuristic(neighbor)
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [current]))

    return None  # No path found
