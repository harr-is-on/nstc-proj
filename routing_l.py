

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
    
    rows, cols = warehouse_matrix.shape
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and warehouse_matrix[nr, nc] == 0:
            return (nr, nc)
    return None


def plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles: Optional[List[Coord]] = None, forbidden_cells: Optional[Set[Coord]] = None, cost_map: Optional[Dict[Coord, int]] = None):
    
    # 調試信息：記錄路徑規劃的參數
    print(f"🗺️ Largest Gap 路徑規劃: {start_pos} -> {target_pos}")
    
    # 初始化參數
    if forbidden_cells is None:
        forbidden_cells = set()
    if cost_map is None:
        cost_map = {}
    if dynamic_obstacles is None:
        dynamic_obstacles = []

    # 檢查是否使用 Largest Gap 策略
    if 'largest_gap_picks' in cost_map and len(cost_map['largest_gap_picks']) > 1:
        pick_locations = cost_map['largest_gap_picks']
        print(f"🔄 啟用 Largest Gap 策略，撿貨點: {pick_locations}")
        
        # 生成快取鍵值
        cache_key = get_robot_key(start_pos, pick_locations)
        
        # 檢查快取
        if cache_key not in _largest_gap_cache:
            # 計算完整的 Largest Gap 路徑
            full_path = plan_largest_gap_complete_route(start_pos, pick_locations, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
            if full_path:
                _largest_gap_cache[cache_key] = {
                    "full_path": full_path,
                    "picks": pick_locations.copy()
                }
                print(f"💾 快取 Largest Gap 路徑，共 {len(full_path)} 步")
            else:
                print("❌ Largest Gap 路徑規劃失敗，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        
        # 從快取中取得路徑並返回適當段落
        cached_data = _largest_gap_cache[cache_key]
        full_path = cached_data["full_path"]
        
        try:
            # 找到起點在完整路徑中的位置
            start_idx = full_path.index(start_pos)
            
            # 找到終點在完整路徑中的位置
            if target_pos in full_path[start_idx:]:
                end_idx = full_path.index(target_pos, start_idx)
                # 返回從下一步到終點的路徑段
                result_path = full_path[start_idx + 1:end_idx + 1]
                print(f"📍 返回 Largest Gap 路徑段: {len(result_path)} 步")
                return result_path if result_path else None
            else:
                print("⚠️ 目標點不在 Largest Gap 路徑中，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        except ValueError:
            print("⚠️ 起點不在 Largest Gap 路徑中，回退到 A* 演算法")
            return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
    
    # 不使用 Largest Gap 策略，使用標準 A* 演算法
    return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)


# --- Largest Gap 策略全域狀態管理 ---
# 儲存每個機器人的 Largest Gap 路徑狀態
# 格式: robot_position_key -> {"full_path": [...], "picks_remaining": [...], "current_target": Coord}
_largest_gap_cache = {}

def get_robot_key(start_pos: Coord, picks: List[Coord]) -> str:
    """生成機器人狀態的唯一鍵值"""
    picks_str = "_".join([f"{p[0]}-{p[1]}" for p in sorted(picks)])
    return f"{start_pos[0]}-{start_pos[1]}_{picks_str}"

def clear_largest_gap_cache():
    """清除所有 Largest Gap 快取"""
    global _largest_gap_cache
    _largest_gap_cache = {}

def plan_largest_gap_complete_route(start_pos: Coord, pick_locations: List[Coord], warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord], cost_map: Dict) -> List[Coord]:
    
    if not pick_locations:
        return [start_pos]
    
    remaining_picks = pick_locations.copy()
    path = [start_pos]
    curr = start_pos
    
    print(f"🔄 開始「最近巷道優先」路徑計算，起點: {start_pos}，撿貨點: {pick_locations}")
    
    while remaining_picks:
        # 1. 找到包含最近撿貨點的巷道
        nearest_pick = min(remaining_picks, key=lambda p: manhattan_distance(curr, p))
        target_aisle_col = nearest_pick[1]
        print(f"\n  → 目標巷道: {target_aisle_col} (因最近點 {nearest_pick})")

        # 2. 找到該巷道的入口轉彎點
        entry_turn = find_nearest_turn_point(curr)
        target_entry_turn = (entry_turn[0], target_aisle_col)

        # 3. 移動到入口轉彎點
        if curr != target_entry_turn:
            print(f"  → 前往巷道入口: {target_entry_turn}")
            segment = a_star_internal_path(curr, target_entry_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment and len(segment) > 1:
                path.extend(segment[1:])
            curr = target_entry_turn

        # 4. 找出該巷道內的所有撿貨點，並按距離排序
        aisle_picks_to_do = sorted(
            [p for p in remaining_picks if p[1] == target_aisle_col],
            key=lambda p: manhattan_distance(curr, p)
        )
        
        print(f"  → 清理巷道內 {len(aisle_picks_to_do)} 個貨物: {aisle_picks_to_do}")
        
        # 5. 逐一撿貨 (進出式)
        picked_in_aisle = []
        for pick_pos in aisle_picks_to_do:
            segment = a_star_internal_path(curr, pick_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment:
                if len(segment) > 1:
                    path.extend(segment[1:])
                curr = pick_pos
                picked_in_aisle.append(pick_pos)
                print(f"    ✅ 撿貨完成: {pick_pos}")
            else:
                print(f"    ❌ 無法到達撿貨點: {pick_pos}")

        # 6. 撿完後，返回入口轉彎點
        if curr != target_entry_turn:
            print(f"  → 返回巷道入口: {target_entry_turn}")
            segment = a_star_internal_path(curr, target_entry_turn, warehouse_matrix, dynamic_obstacles, forbidden_cells)
            if segment and len(segment) > 1:
                path.extend(segment[1:])
            curr = target_entry_turn

        # 7. 從剩餘列表中移除已完成的貨物
        remaining_picks = [p for p in remaining_picks if p not in picked_in_aisle]

    print(f"🎉 「最近巷道優先」路徑計算完成，總長度: {len(path)}")
    return path

def a_star_internal_path(start: Coord, goal: Coord, warehouse_matrix: np.ndarray, dynamic_obstacles: List[Coord], forbidden_cells: Set[Coord]) -> List[Coord]:
    """A* 路徑搜尋，專用於 Largest Gap 內部路徑規劃"""
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

