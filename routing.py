"""
路徑規劃策略模組 (範本)

這個檔案提供了基礎實作範例 (A* 演算法)。
參考並替換此處的 `plan_route` 函式來做出路徑規劃策略。
"""

import heapq
import math
from typing import List, Tuple, Optional, Set, Dict
import numpy as np

# --- 型別別名，方便閱讀 ---
Coord = Tuple[int, int]

def euclidean_distance(pos1: Coord, pos2: Coord) -> float:
    """【輔助函式】計算兩點之間的歐幾里得距離。"""
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


def plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles: Optional[List[Coord]] = None, forbidden_cells: Optional[Set[Coord]] = None, cost_map: Optional[Dict[Coord, int]] = None):
    """【核心策略函式】
    為機器人規劃一條從起點到終點的路徑。
    
    **你的演算法會收到什麼資訊 (參數)：**
    - `start_pos`, `target_pos`: 規劃路徑的起點與終點。
    - `warehouse_matrix`: 靜態的倉庫地圖。您可以根據其中的代號判斷哪些格子是可通行的 (例如，代號為 0, 4, 5, 6, 7 的是走道或特殊區域)。
    - `dynamic_obstacles`: 其他機器人目前的位置。您的演算法應避免路徑經過這些點。
    - `forbidden_cells`: 此次規劃中「絕對不能」經過的格子。這由主引擎根據機器人當前任務決定。
    - `cost_map`: 一個「建議」繞行的區域地圖。移動到這些格子的成本較高，您的演算法可以利用此資訊找出更有效率或更安全的路線，但並非強制禁止。

    **你的演算法需要提供什麼結果 (回傳值)：**
    - 一個座標列表 `List[Coord]`：代表從「下一步」到終點的路徑。
      例如：若從 (0,0) 到 (0,2)，應返回 `[(0,1), (0,2)]`。
    - `None`：如果找不到任何可行的路徑。

    **你的演算法「不需要」處理的：**
    - 機器人的狀態、電量、任務細節等。
    - 碰撞管理或路權協調 (這由 `congestion_model.py` 處理)。
    - 實際移動機器人 (主引擎會根據你回傳的路徑來執行)。
    ---
    """
    # 調試信息：記錄路徑規劃的參數
    if forbidden_cells:
        print(f"🗺️ 路徑規劃: {start_pos} -> {target_pos}, 禁止區域: {forbidden_cells}")
    else:
        print(f"🗺️ 路徑規劃: {start_pos} -> {target_pos}")
    rows, cols = warehouse_matrix.shape

    # --- A* 演算法的初始化 ---
    if forbidden_cells is None:
        forbidden_cells = set()
    if cost_map is None:
        cost_map = {}

    def neighbors(pos: Coord) -> List[Coord]:
        r, c = pos
        candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)] # 四個方向
        valid_neighbors = []
        for nr, nc in candidates:
            if 0 <= nr < rows and 0 <= nc < cols:
                # 檢查動態障礙物 (除非它是我們的最終目標)
                if dynamic_obstacles and (nr, nc) in dynamic_obstacles and (nr, nc) != target_pos:
                    continue

                # 檢查呼叫者提供的絕對禁止區域 (除非它是我們的最終目標)
                if (nr, nc) in forbidden_cells and (nr, nc) != target_pos:
                    continue

                # 檢查靜態倉庫佈局。所有非障礙物的格子都是可通行的。
                cell_type = warehouse_matrix[nr, nc]
                if cell_type in [0, 4, 5, 6, 7] or (nr, nc) == target_pos:
                    valid_neighbors.append((nr, nc))
        return valid_neighbors

    def heuristic(pos):
        # 啟發函式 (Heuristic): 使用曼哈頓距離，這在網格地圖上通常很有效。
        return abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])

    # --- A* 演算法主體 ---
    # open_list 是一個優先佇列，儲存待探索的節點。
    # 格式: (f_score, g_score, position, path_so_far)
    open_list = [(heuristic(start_pos), 0, start_pos, [])]  # (f_score, g_score, pos, path)
    # closed_set 儲存已經探索過的節點，避免重複計算。
    closed_set = set()

    while open_list:
        # 從優先佇列中取出 f_score 最低的節點
        f, g, current, path = heapq.heappop(open_list)

        if current in closed_set:
            continue

        # 如果到達目標，重建並返回路徑
        if current == target_pos:
            # 根據「合約」，我們需要返回從「下一步」開始的路徑。
            return (path + [current])[1:]

        closed_set.add(current)

        # 探索所有有效的鄰居節點
        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue
            
            # 計算移動到鄰居的成本 (g_score)
            move_cost = cost_map.get(neighbor, 1)
            new_g = g + move_cost
            # 計算 f_score = g_score + h_score
            new_f = new_g + heuristic(neighbor)
            # 將鄰居節點加入優先佇列
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [current]))

    return None  # 如果 open_list 為空仍未找到路徑，則表示無解
