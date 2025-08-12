"""
路徑規劃策略模組 - 混合策略實作

這個檔案實作了混合路徑規劃策略，適配自原始 routing_m.py。
主要函式 `plan_route` 遵循原始範本的輸入輸出格式要求。

混合策略運作原理：
==================

1. **道路類型定義**：
   - Main Roads: 水平走道 (horizontal aisles: [0, 1, 6, 7, 12, 13])
   - Sub Roads: 垂直走道 (vertical aisles: [0, 1, 4, 7, 10, 13, 14])
   - Turn Points: main road 與 sub road 的交叉點

2. **智慧策略選擇機制**：
   - 緊鄰貨物 (距離 ≤ 閾值): 一次撿完所有貨物
   - 多個貨物 (≥3個): 完整穿越策略，選擇最佳出口
   - 少量貨物 (1-2個): 入口側策略，撿完即返回
   - 順路檢查: 返程時檢查主幹道附近的貨物

3. **混合策略流程**：
   Step 1: 選擇距離最近的撿貨點
   Step 2: 移動到轉彎點（如果需要）
   Step 3: 水平移動到目標 sub road
   Step 4: 根據巷道內貨物分布智慧選擇策略
   Step 5: 執行對應的撿貨策略
   Step 6: 順路檢查和路徑優化
   Step 7: 重複直到撿完所有貨物

4. **適配策略**：
   - 保持原始 plan_route 函式簽名不變
   - 透過 cost_map 參數傳遞額外的混合策略相關資訊
   - 使用全域狀態管理避免重複計算完整路徑
   - 支援單點任務回退到 A* 演算法

5. **輸入輸出格式**：
   - 輸入: start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map
   - 輸出: 從「下一步」到終點的路徑列表，例如：[(0,1), (0,2), (1,2)]
"""

import heapq
import math
from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from warehouse_layout import (
    is_turn_point, find_nearest_turn_point
)
from routing import plan_route as plan_route_a_star # 匯入基礎 A* 演算法並重新命名

# --- 型別別名，方便閱讀 ---
Coord = Tuple[int, int]

def euclidean_distance(pos1: Coord, pos2: Coord) -> float:
    """【輔助函式】計算兩點之間的歐幾里得距離。"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def manhattan_distance(pos1: Coord, pos2: Coord) -> int:
    """計算兩點之間的曼哈頓距離"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

def nearest_station(curr: Coord, stations: List[Coord]) -> Coord:
    """找到最近的站點"""
    return min(stations, key=lambda s: manhattan_distance(curr, s))

def pick_exit_based_on_next(curr: Coord, col: int, warehouse_matrix: np.ndarray, next_target: Coord) -> Coord:
    """出口選擇：根據下一目標方向選擇最佳出口"""
    # 找到該列的上下轉彎點
    turn_up = find_nearest_turn_point((0, col), 'any')   # 上出口
    turn_down = find_nearest_turn_point((warehouse_matrix.shape[0]-1, col), 'any') # 下出口
    
    if turn_up and turn_down:
        # 選擇距離下一目標較近的出口
        return turn_up if next_target[0] < curr[0] else turn_down
    elif turn_up:
        return turn_up
    elif turn_down:
        return turn_down
    else:
        return curr

def plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles: Optional[List[Coord]] = None, forbidden_cells: Optional[Set[Coord]] = None, cost_map: Optional[Dict[Coord, int]] = None):
    """【核心策略函式】- 混合策略實作
    為機器人規劃一條從起點到終點的路徑。
    
    **你的演算法會收到什麼資訊 (參數)：**
    - `start_pos`, `target_pos`: 規劃路徑的起點與終點。
    - `warehouse_matrix`: 靜態的倉庫地圖。您可以根據其中的代號判斷哪些格子是可通行的 (例如，代號為 0, 4, 5, 6, 7 的是走道或特殊區域)。
    - `dynamic_obstacles`: 其他機器人目前的位置。您的演算法應避免路徑經過這些點。
    - `forbidden_cells`: 此次規劃中「絕對不能」經過的格子。這由主引擎根據機器人當前任務決定。
    - `cost_map`: 一個「建議」繞行的區域地圖。移動到這些格子的成本較高，您的演算法可以利用此資訊找出更有效率或更安全的路線，但並非強制禁止。
                  **特殊用法**: 當 cost_map 包含 'composite_picks' 鍵時，啟用混合策略。

    **你的演算法需要提供什麼結果 (回傳值)：**
    - 一個座標列表 `List[Coord]`：代表從「下一步」到終點的路徑。
      例如：若從 (0,0) 到 (0,2)，應返回 `[(0,1), (0,2)]`。
    - `None`：如果找不到任何可行的路徑。

    **你的演算法「不需要」處理的：**
    - 機器人的狀態、電量、任務細節等。
    - 碰撞管理或路權協調 (這由 `congestion_model.py` 處理)。
    - 實際移動機器人 (主引擎會根據你回傳的路徑來執行)。
    
    **混合策略說明：**
    當 cost_map 包含 'composite_picks' 時，演算法會：
    1. 檢查是否有多個撿貨點需要混合策略路徑規劃
    2. 如果有，計算完整的混合策略路徑並快取
    3. 根據當前 start_pos 和 target_pos 返回路徑的適當段落
    4. 如果不適用混合策略，回退到標準 A* 演算法
    ---
    """
    # 調試信息：記錄路徑規劃的參數
    print(f"🗺️ 混合策略路徑規劃: {start_pos} -> {target_pos}")
    
    # 初始化參數
    if forbidden_cells is None:
        forbidden_cells = set()
    if cost_map is None:
        cost_map = {}
    if dynamic_obstacles is None:
        dynamic_obstacles = []

    # 檢查是否使用混合策略
    if 'composite_picks' in cost_map and len(cost_map['composite_picks']) > 1:
        pick_locations = cost_map['composite_picks']
        neighbor_threshold = cost_map.get('neighbor_threshold', 2)  # 緊鄰閾值
        print(f" 啟用混合策略，撿貨點: {pick_locations}，緊鄰閾值: {neighbor_threshold}")
        
        # 生成快取鍵值
        cache_key = get_robot_key(start_pos, pick_locations, neighbor_threshold)
        
        # 檢查快取
        if cache_key not in _composite_cache:
            # 計算完整的混合策略路徑
            full_path = plan_composite_complete_route(start_pos, pick_locations, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
            if full_path:
                _composite_cache[cache_key] = {
                    "full_path": full_path,
                    "picks": pick_locations.copy()
                }
                print(f" 快取混合策略路徑，共 {len(full_path)} 步")
            else:
                print(" 混合策略路徑規劃失敗，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        
        # 從快取中取得路徑並返回適當段落
        cached_data = _composite_cache[cache_key]
        full_path = cached_data["full_path"]
        
        try:
            # 找到起點在完整路徑中的位置
            start_idx = full_path.index(start_pos)
            
            # 找到終點在完整路徑中的位置
            if target_pos in full_path[start_idx:]:
                end_idx = full_path.index(target_pos, start_idx)
                # 返回從下一步到終點的路徑段
                result_path = full_path[start_idx + 1:end_idx + 1]
                print(f" 返回混合策略路徑段: {len(result_path)} 步")
                return result_path if result_path else None
            else:
                print("目標點不在混合策略路徑中，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        except ValueError:
            print(" 起點不在混合策略路徑中，回退到 A* 演算法")
            return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
    
    # 不使用混合策略，使用標準 A* 演算法
    return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)


# --- 混合策略全域狀態管理 ---
# 儲存每個機器人的混合策略路徑狀態
_composite_cache = {}

def get_robot_key(start_pos: Coord, picks: List[Coord], threshold: int = 2) -> str:
    """生成機器人狀態的唯一鍵值"""
    picks_str = "_".join([f"{p[0]}-{p[1]}" for p in sorted(picks)])
    return f"{start_pos[0]}-{start_pos[1]}_{picks_str}_t{threshold}"

def clear_composite_cache():
    """清除所有混合策略快取"""
    global _composite_cache
    _composite_cache = {}

def plan_composite_complete_route(start_pos: Coord, pick_locations: List[Coord], warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord], cost_map: Dict) -> List[Coord]:
    """
    實作完整混合策略路徑規劃
    
    基於原始 routing_m.py 的邏輯，適配到 nstc-proj-main 框架：
    
    混合策略步驟：
    1. 選擇距離最近的撿貨點
    2. 移動到轉彎點（如果需要）
    3. 水平移動到目標 sub road
    4. 根據巷道內貨物分布智慧選擇策略：
       - 緊鄰貨物 (距離 ≤ 閾值): 一次撿完所有貨物
       - 多個貨物 (≥3個): 完整穿越策略，選擇最佳出口
       - 少量貨物 (1-2個): 入口側策略，撿完即返回
    5. 順路檢查: 返程時檢查主幹道附近的貨物
    6. 重複直到撿完所有貨物
    
    返回包含起點的完整路徑
    """
    if not pick_locations:
        return [start_pos]
    
    remaining = pick_locations.copy()
    path = [start_pos]
    curr = start_pos
    neighbor_threshold = cost_map.get('neighbor_threshold', 2)
    
    print(f" 開始混合策略路徑計算，起點: {start_pos}，撿貨點: {pick_locations}")
    
    while remaining:
        # 1. 選擇距離最近的撿貨點
        target = min(remaining, key=lambda p: manhattan_distance(curr, p))
        tr, tc = target
        print(f"  → 目標撿貨點: {target}")
        
        # 2. 如果不在 turn point，先移動到最近的 turn point
        if not is_turn_point(curr):
            turn_point = find_nearest_turn_point(curr)
            if turn_point and turn_point != curr:
                print(f"  → 移動到轉彎點: {turn_point}")
                segment = a_star_internal_path(curr, turn_point, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                if segment:
                    if len(segment) > 1:
                        path.extend(segment[1:])
                    curr = turn_point
        
        # 3. 水平移動到目標 sub road 所在列
        if curr[1] != tc:
            horizontal_target = (curr[0], tc)
            print(f"  → 水平移動到: {horizontal_target}")
            segment = a_star_internal_path(curr, horizontal_target, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment:
                if len(segment) > 1:
                    path.extend(segment[1:])
                curr = horizontal_target
        
        # 4. 分析巷道內貨物並決定撿取策略
        aisle_orders = sorted([p for p in remaining if p[1] == tc], key=lambda p: p[0])
        picked_now = []
        
        if len(aisle_orders) >= 2:
            # 判斷距離分布
            col_range = max(aisle_orders, key=lambda p: p[0])[0] - min(aisle_orders, key=lambda p: p[0])[0]
            
            if col_range <= neighbor_threshold:
                # 緊鄰策略：一次撿完
                print(f"  🎯 緊鄰策略：一次撿完 {len(aisle_orders)} 個貨物，範圍: {col_range}")
                seq = aisle_orders if curr[0] <= aisle_orders[0][0] else list(reversed(aisle_orders))
                for pick_pos in seq:
                    segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                    if segment:
                        if len(segment) > 1:
                            path.extend(segment[1:])
                        curr = pick_pos
                        picked_now.apend(curr)
                        print(f"     撿貨完成: {curr}")
            elif len(aisle_orders) >= 3:
                # 完整穿越策略
                print(f"   完整穿越策略：撿完 {len(aisle_orders)} 個貨物")
                seq = aisle_orders if curr[0] <= aisle_orders[0][0] else list(reversed(aisle_orders))
                for pick_pos in seq:
                    segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                    if segment:
                        if len(segment) > 1:
                            path.extend(segment[1:])
                        curr = pick_pos
                        picked_now.append(curr)
                        print(f"     撿貨完成: {curr}")
                
                # 選擇最佳出口（基於下一目標）
                remaining_after_picks = [p for p in remaining if p not in aisle_orders]
                if remaining_after_picks:
                    next_target = min(remaining_after_picks, key=lambda p: manhattan_distance(curr, p))
                    exit_turn = pick_exit_based_on_next(curr, tc, warehouse_matrix, next_target)
                    print(f"  → 基於下一目標 {next_target}，選擇出口: {exit_turn}")
                    segment = a_star_internal_path(curr, exit_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                    if segment:
                        if len(segment) > 1:
                            path.extend(segment[1:])
                        curr = exit_turn
            else:
                # 入口側策略
                # 此策略適用於巷道內有2個且分佈較遠的貨物。修正了原先只撿一個就返回的缺陷。
                print(f"   入口側策略：撿完 {len(aisle_orders)} 個貨物後返回入口")
                # 決定撿貨順序
                seq = aisle_orders if curr[0] <= aisle_orders[0][0] else list(reversed(aisle_orders))
                for pick_pos in seq:
                    segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                    if segment:
                        if len(segment) > 1:
                            path.extend(segment[1:])
                        curr = pick_pos
                        picked_now.append(curr)
                        print(f"     撿貨完成: {curr}")
                
                # 撿完該巷道的所有目標後，再返回入口轉彎點
                entry_turn = find_nearest_turn_point(curr)
                if entry_turn and entry_turn != curr:
                    segment = a_star_internal_path(curr, entry_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                    if segment:
                        if len(segment) > 1:
                            path.extend(segment[1:])
                        curr = entry_turn
        else:
            # 單一貨物
            print(f"   單一貨物策略")
            pick_pos = aisle_orders[0]
            segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment and len(segment) > 1:
                path.extend(segment[1:])
                curr = pick_pos
                picked_now.append(curr)
                print(f"     撿貨完成: {curr}")
            
            # 返回入口轉彎點
            entry_turn = find_nearest_turn_point(curr)
            if entry_turn and entry_turn != curr:
                segment = a_star_internal_path(curr, entry_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                if segment:
                    if len(segment) > 1:
                        path.extend(segment[1:])
                    curr = entry_turn
        
        # 移除已完成的貨物
        for p in picked_now:
            if p in remaining:
                remaining.remove(p)
        
        # 5. 順路檢查：返程時檢查主幹道附近的貨物
        nearby_picks = []
        for p in remaining[:]:
            if manhattan_distance(curr, p) <= neighbor_threshold:
                nearby_picks.append(p)
        
        if nearby_picks:
            print(f"  🛤️ 順路檢查：發現 {len(nearby_picks)} 個附近貨物")
            for p in nearby_picks:
                segment = a_star_internal_path(curr, p, warehouse_matrix, dynamic_obstacles, forbidden_cells)
                if segment:
                    if len(segment) > 1:
                        path.extend(segment[1:])
                    curr = p
                    remaining.remove(p)
                    print(f"     順路撿貨: {p}")
    
    print(f" 混合策略路徑計算完成，總長度: {len(path)}")
    return path

def a_star_internal_path(start: Coord, goal: Coord, warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord]) -> List[Coord]:
    """
    A* 路徑搜尋，專用於混合策略內部路徑規劃。
    此版本調用共享的 A* 實作，並將結果包裝成包含起點的完整路徑。
    """
    if start == goal:
        return [start]
    
    # 基礎 A* 演算法返回的是從「下一步」開始的路徑段
    path_segment = plan_route_a_star(start, goal, warehouse_matrix, dynamic_obstacles, forbidden_cells, None)
    
    if path_segment:
        # 將起點加到路徑開頭，以符合內部邏輯的預期格式
        return [start] + path_segment
    
    return []  # 如果找不到路徑

# --- 使用範例和測試函式 ---

def example_usage():
    """
    使用範例：如何呼叫混合策略路徑規劃
    
    要啟用混合策略，需要在 cost_map 中包含 'composite_picks' 鍵：
    
    ```python
    from warehouse_layout import create_warehouse_layout
    
    # 建立倉庫佈局
    warehouse_matrix, _ = create_warehouse_layout()
    
    # 單點路徑規劃（使用 A*）
    path = plan_route((1, 1), (5, 8), warehouse_matrix)
    
    # 多點混合策略路徑規劃
    cost_map_with_composite = {
        'composite_picks': [(2, 4), (5, 4), (8, 7), (3, 10)],  # 多個撿貨點
        'neighbor_threshold': 2  # 緊鄰閾值
    }
    
    # 從起點到第一個撿貨點
    path_segment = plan_route(
        start_pos=(1, 1), 
        target_pos=(2, 4), 
        warehouse_matrix=warehouse_matrix,
        cost_map=cost_map_with_composite
    )
    ```
    
    輸出範例：
    - 單點：[(1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)]
    - 混合策略路徑段：根據智慧策略選擇返回適當段落
    """
    pass
