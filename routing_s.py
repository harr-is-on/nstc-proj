"""
路徑規劃策略模組 - S-Shape 策略實作

這個檔案實作了 S-Shape 路徑規劃策略，適配自 old_rs/s_shape_d.py。
主要函式 `plan_route` 遵循原始範本的輸入輸出格式要求。

S-Shape 策略運作原理：
===================

1. **道路類型定義**：
   - Main Roads: 水平走道 (horizontal aisles: [0, 1, 6, 7, 12, 13])
   - Sub Roads: 垂直走道 (vertical aisles: [0, 1, 4, 7, 10, 13, 14])
   - Turn Points: main road 與 sub road 的交叉點

2. **S-Shape 路徑規劃流程**：
   Step 1: 如果不在 turn point，先垂直移動到最近的 turn point
   Step 2: 水平移動到目標 sub road 所在的列
   Step 3: 沿當前方向撿完該 sub road 中的所有貨物
   Step 4: 繼續往該方向走到底部 turn point
   Step 5: 重複 Step 1-4 直到撿完所有貨物

3. **適配策略**：
   - 保持原始 plan_route 函式簽名不變
   - 透過 cost_map 參數傳遞額外的 S-shape 相關資訊
   - 使用全域狀態管理避免重複計算完整路徑
   - 支援單點任務回退到 A* 演算法

4. **輸入輸出格式**：
   - 輸入: start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map
   - 輸出: 從「下一步」到終點的路徑列表，例如：[(0,1), (0,2), (1,2)]
"""

import heapq
import math
from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from warehouse_layout import (
    is_turn_point,
    find_nearest_turn_point
)

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
    """【核心策略函式】- S-Shape 策略實作
    為機器人規劃一條從起點到終點的路徑。
    
    **你的演算法會收到什麼資訊 (參數)：**
    - `start_pos`, `target_pos`: 規劃路徑的起點與終點。
    - `warehouse_matrix`: 靜態的倉庫地圖。您可以根據其中的代號判斷哪些格子是可通行的 (例如，代號為 0, 4, 5, 6, 7 的是走道或特殊區域)。
    - `dynamic_obstacles`: 其他機器人目前的位置。您的演算法應避免路徑經過這些點。
    - `forbidden_cells`: 此次規劃中「絕對不能」經過的格子。這由主引擎根據機器人當前任務決定。
    - `cost_map`: 一個「建議」繞行的區域地圖。移動到這些格子的成本較高，您的演算法可以利用此資訊找出更有效率或更安全的路線，但並非強制禁止。
                  **特殊用法**: 當 cost_map 包含 's_shape_picks' 鍵時，啟用 S-shape 策略。

    **你的演算法需要提供什麼結果 (回傳值)：**
    - 一個座標列表 `List[Coord]`：代表從「下一步」到終點的路徑。
      例如：若從 (0,0) 到 (0,2)，應返回 `[(0,1), (0,2)]`。
    - `None`：如果找不到任何可行的路徑。

    **你的演算法「不需要」處理的：**
    - 機器人的狀態、電量、任務細節等。
    - 碰撞管理或路權協調 (這由 `congestion_model.py` 處理)。
    - 實際移動機器人 (主引擎會根據你回傳的路徑來執行)。
    
    **S-Shape 策略說明：**
    當 cost_map 包含 's_shape_picks' 時，演算法會：
    1. 檢查是否有多個撿貨點需要 S-shape 路徑規劃
    2. 如果有，計算完整的 S-shape 路徑並快取
    3. 根據當前 start_pos 和 target_pos 返回路徑的適當段落
    4. 如果不適用 S-shape，回退到標準 A* 演算法
    ---
    """
    # 調試信息：記錄路徑規劃的參數
    print(f"S-Shape 路徑規劃: {start_pos} -> {target_pos}")
    
    # 初始化參數
    if forbidden_cells is None:
        forbidden_cells = set()
    if cost_map is None:
        cost_map = {}
    if dynamic_obstacles is None:
        dynamic_obstacles = []

    # 檢查是否使用 S-shape 策略
    if 's_shape_picks' in cost_map and len(cost_map['s_shape_picks']) > 1:
        pick_locations = cost_map['s_shape_picks']
        print(f" 啟用 S-shape 策略，撿貨點: {pick_locations}")
        
        # 生成快取鍵值
        cache_key = get_robot_key(start_pos, pick_locations)
        
        # 檢查快取
        if cache_key not in _s_shape_cache:
            # 計算完整的 S-shape 路徑
            full_path = plan_s_shape_complete_route(start_pos, pick_locations, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
            if full_path:
                _s_shape_cache[cache_key] = {
                    "full_path": full_path,
                    "picks": pick_locations.copy()
                }
                print(f" 快取 S-shape 路徑，共 {len(full_path)} 步")
            else:
                print(" S-shape 路徑規劃失敗，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        
        # 從快取中取得路徑並返回適當段落
        cached_data = _s_shape_cache[cache_key]
        full_path = cached_data["full_path"]
        
        try:
            # 找到起點在完整路徑中的位置
            start_idx = full_path.index(start_pos)
            
            # 找到終點在完整路徑中的位置
            if target_pos in full_path[start_idx:]:
                end_idx = full_path.index(target_pos, start_idx)
                # 返回從下一步到終點的路徑段
                result_path = full_path[start_idx + 1:end_idx + 1]
                print(f"返回 S-shape 路徑段: {len(result_path)} 步")
                return result_path if result_path else None
            else:
                print("目標點不在 S-shape 路徑中，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        except ValueError:
            print("起點不在 S-shape 路徑中，回退到 A* 演算法")
            return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
    
    # 不使用 S-shape 策略，使用標準 A* 演算法
    return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)


def plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map):
    """標準 A* 路徑規劃演算法 (原始實作)"""
    rows, cols = warehouse_matrix.shape

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
    open_list = [(heuristic(start_pos), 0, start_pos, [])]  # (f_score, g_score, pos, path)
    closed_set = set()

    while open_list:
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
            move_cost = cost_map.get(neighbor, 1) if isinstance(cost_map.get(neighbor), int) else 1
            new_g = g + move_cost
            # 計算 f_score = g_score + h_score
            new_f = new_g + heuristic(neighbor)
            # 將鄰居節點加入優先佇列
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [current]))

    return None  # 如果 open_list 為空仍未找到路徑，則表示無解


# --- S-shape 策略全域狀態管理 ---
# 儲存每個機器人的 S-shape 路徑狀態
# 格式: robot_position_key -> {"full_path": [...], "picks_remaining": [...], "current_target": Coord}
_s_shape_cache = {}

def get_robot_key(start_pos: Coord, picks: List[Coord]) -> str:
    """生成機器人狀態的唯一鍵值"""
    picks_str = "_".join([f"{p[0]}-{p[1]}" for p in sorted(picks)])
    return f"{start_pos[0]}-{start_pos[1]}_{picks_str}"

def clear_s_shape_cache():
    """清除所有 S-shape 快取"""
    global _s_shape_cache
    _s_shape_cache = {}

def plan_s_shape_complete_route(start_pos: Coord, pick_locations: List[Coord], warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord], cost_map: Dict) -> List[Coord]:
    """
    實作完整 S-shape 路徑規劃

    此版本為純正的 S-Shape 策略，會依序由左至右清掃所有需要作業的巷道。

    S-Shape 策略步驟：
    1. 找出所有需要撿貨的巷道 (sub roads)。
    2. 依巷道順序 (由左至右) 進行排序。
    3. 交替上下方向，清掃每個巷道內的所有貨物。
    4. 從巷道一端進入，另一端離開，形成 S 形路徑。
    5. 重複直到所有巷道清掃完畢。

    返回包含起點的完整路徑
    """
    if not pick_locations:
        return [start_pos]

    path = [start_pos]
    curr = start_pos

    # 1. 找出所有需要撿貨的巷道並排序
    remaining_picks = pick_locations.copy()
    aisles_to_visit = sorted(list(set(p[1] for p in remaining_picks)))

    print(f" 開始純正 S-shape 路徑計算，起點: {start_pos}，目標巷道: {aisles_to_visit}")

    # 2. 交替清掃方向，1=向下, -1=向上
    sweep_direction = 1

    for aisle_col in aisles_to_visit:
        print(f"\n  清掃巷道: {aisle_col}, 方向: {'下' if sweep_direction == 1 else '上'}")

        # 3. 決定入口和出口轉彎點
        if sweep_direction == 1: # 向下掃
            entry_turn = find_nearest_turn_point((0, aisle_col), 'any')
            exit_turn = find_nearest_turn_point((warehouse_matrix.shape[0]-1, aisle_col), 'any')
        else: # 向上掃
            entry_turn = find_nearest_turn_point((warehouse_matrix.shape[0]-1, aisle_col), 'any')
            exit_turn = find_nearest_turn_point((0, aisle_col), 'any')

        # 從當前位置移動到入口轉彎點
        if curr != entry_turn:
            print(f"  → 前往入口: {entry_turn}")
            segment = a_star_internal_path(curr, entry_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment and len(segment) > 1:
                path.extend(segment[1:])
            curr = entry_turn
        
        # 4. 找出該巷道內的所有撿貨點，並根據清掃方向排序
        aisle_picks = sorted(
            [p for p in remaining_picks if p[1] == aisle_col],
            key=lambda p: p[0],
            reverse=(sweep_direction == -1)
        )
        
        # 逐一撿貨
        for pick_pos in aisle_picks:
            segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment:
                if len(segment) > 1:
                    path.extend(segment[1:])
                curr = pick_pos
                # 從剩餘清單中移除已撿的貨物
                if pick_pos in remaining_picks:
                    remaining_picks.remove(pick_pos)
                print(f"     撿貨完成: {pick_pos}")
            else:
                print(f"    無法到達撿貨點: {pick_pos}")
                # 同樣移除無法到達的點，避免重複嘗試
                if pick_pos in remaining_picks:
                    remaining_picks.remove(pick_pos)
        
        # 5. 撿完後，移動到出口轉彎點
        if curr != exit_turn:
            print(f"  → 前往出口: {exit_turn}")
            segment = a_star_internal_path(curr, exit_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment and len(segment) > 1:
                path.extend(segment[1:])
            curr = exit_turn

        # 6. 反轉清掃方向，為下一個巷道做準備
        sweep_direction *= -1
    
    print(f"🎉 S-shape 路徑計算完成，總長度: {len(path)}")
    return path

def a_star_internal_path(start: Coord, goal: Coord, warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord]) -> List[Coord]:
    """A* 路徑搜尋，專用於 S-shape 內部路徑規劃"""
    if start == goal:
        return [start]
    
    rows, cols = warehouse_matrix.shape
    
    def neighbors(pos: Coord) -> List[Coord]:
        r, c = pos
        candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        valid = []
        for nr, nc in candidates:
            if 0 <= nr < rows and 0 <= nc < cols:
                # 檢查動態障礙物
                if (nr, nc) in dynamic_obstacles and (nr, nc) != goal:
                    continue
                # 檢查禁止區域
                if (nr, nc) in forbidden_cells and (nr, nc) != goal:
                    continue
                # 檢查倉庫佈局
                cell_type = warehouse_matrix[nr, nc]
                if cell_type in [0, 4, 5, 6, 7] or (nr, nc) == goal:
                    valid.append((nr, nc))
        return valid
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    open_list = [(heuristic(start), 0, start, [start])]
    closed_set = set()
    
    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        
        if current in closed_set:
            continue
            
        if current == goal:
            return path
            
        closed_set.add(current)
        
        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue
            new_g = g + 1
            new_f = new_g + heuristic(neighbor)
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))
    
    return []  # 無路徑

# --- 使用範例和測試函式 ---

def example_usage():
    """
    使用範例：如何呼叫 S-shape 路徑規劃
    
    要啟用 S-shape 策略，需要在 cost_map 中包含 's_shape_picks' 鍵：
    
    ```python
    from warehouse_layout import create_warehouse_layout
    
    # 建立倉庫佈局
    warehouse_matrix, _ = create_warehouse_layout()
    
    # 單點路徑規劃（使用 A*）
    path = plan_route((1, 1), (5, 8), warehouse_matrix)
    
    # 多點 S-shape 路徑規劃
    cost_map_with_s_shape = {
        's_shape_picks': [(2, 4), (5, 4), (8, 7), (3, 10)]  # 多個撿貨點
    }
    
    # 從起點到第一個撿貨點
    path_segment = plan_route(
        start_pos=(1, 1), 
        target_pos=(2, 4), 
        warehouse_matrix=warehouse_matrix,
        cost_map=cost_map_with_s_shape
    )
    ```
    
    輸出範例：
    - 單點：[(1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)]
    - S-shape 路徑段：根據 S-shape 完整路徑返回適當段落
    """
    pass
