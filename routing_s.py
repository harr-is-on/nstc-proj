
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
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

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
    print(f"🗺️ S-Shape 路徑規劃: {start_pos} -> {target_pos}")
    
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
        print(f"🔄 啟用 S-shape 策略，撿貨點: {pick_locations}")
        
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
                print(f"💾 快取 S-shape 路徑，共 {len(full_path)} 步")
            else:
                print("❌ S-shape 路徑規劃失敗，回退到 A* 演算法")
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
                print(f"📍 返回 S-shape 路徑段: {len(result_path)} 步")
                return result_path if result_path else None
            else:
                print("⚠️ 目標點不在 S-shape 路徑中，回退到 A* 演算法")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        except ValueError:
            print("⚠️ 起點不在 S-shape 路徑中，回退到 A* 演算法")
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
    
    if not pick_locations:
        return [start_pos]

    path = [start_pos]
    curr = start_pos

    # 1. 找出所有需要撿貨的巷道並排序
    remaining_picks = pick_locations.copy()
    aisles_to_visit = sorted(list(set(p[1] for p in remaining_picks)))

    print(f"🔄 開始純正 S-shape 路徑計算，起點: {start_pos}，目標巷道: {aisles_to_visit}")

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
                print(f"    ✅ 撿貨完成: {pick_pos}")
            else:
                print(f"    ❌ 無法到達撿貨點: {pick_pos}")
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

