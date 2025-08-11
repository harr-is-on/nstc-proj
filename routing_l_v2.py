"""
路徑規劃策略模組 - Largest Gap v2 (線性掃描)

這個檔案實作了 Largest Gap v2 路徑規劃策略。
主要函式 `plan_route` 遵循與系統中其他策略模組相同的輸入輸出格式。

L-v2 策略運作原理：
========================
1.  **觸發方式**:
    - 在呼叫 `plan_route` 時，於 `cost_map` 字典中提供一個鍵 `'l_v2_picks'`，其值為所有撿貨點的列表。

2.  **路徑規劃流程**:
    - **首次呼叫**:
        1.  `reorder_task_items`: 根據線性掃描（C 型路徑）演算法，對所有撿貨點進行全域排序。
        2.  `plan_l_v2_complete_route`: 根據排好的順序，使用基礎 A* 演算法逐段規劃出從起點到所有撿貨點的完整路徑。
        3.  **快取**: 將計算出的完整路徑儲存在記憶體中。
    - **後續呼叫**:
        1.  直接從快取中讀取完整路徑。
        2.  根據當前的 `start_pos` 和 `target_pos`，返回路徑中的對應路段。

3.  **回退機制**:
    - 如果 `cost_map` 中未提供 `'l_v2_picks'`，或者路徑規劃失敗，則自動回退使用標準的 A* 演算法。
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from routing import plan_route as base_plan_route, euclidean_distance, find_adjacent_aisle

# --- 型別別名與常數 ---
Coord = Tuple[int, int]
AISLE_CODES = {0, 7}  # 0=走道, 7=撿貨出口

# =============================================================================
# SECTION 1: L-v2 核心邏輯 (排序與輔助函式)
# =============================================================================

def in_upper(y: int) -> bool: return 0 <= y <= 6
def in_lower(y: int) -> bool: return 7 <= y <= 13

def classify_group(y: int) -> str:
    if 0 <= y <= 3: return "upper_back"
    if 4 <= y <= 6: return "upper_front"
    if 7 <= y <= 9: return "lower_front"
    return "lower_back"

def half_from_group(g: str) -> str:
    return "upper" if g.startswith("upper_") else "lower"

def get_access_point(wm: np.ndarray, shelf: Coord) -> Optional[Coord]:
    """回傳貨位左右相鄰的走道(AP)；只取可通行(0/7)。"""
    r, c = shelf
    rows, cols = wm.shape
    if c-1 >= 0 and wm[r, c-1] in AISLE_CODES: return (r, c-1)
    if c+1 < cols and wm[r, c+1] in AISLE_CODES: return (r, c+1)
    return None

def build_index(items: List[Coord], wm: np.ndarray):
    """半區→走道x→各組貨位清單；以及每個貨位的 AP 對應表。"""
    aisles: Dict[str, Dict[int, Dict[str, List[Coord]]]] = {"upper": {}, "lower": {}}
    ap_of: Dict[Coord, Coord] = {}
    for it in items:
        ap = get_access_point(wm, it)
        if ap is None:
            continue
        ap_of[it] = ap
        ax = ap[1]
        g = classify_group(it[0])
        half = half_from_group(g)
        if ax not in aisles[half]:
            aisles[half][ax] = {"upper_front": [], "upper_back": [], "lower_front": [], "lower_back": []}
        aisles[half][ax][g].append(it)
    return {"aisles": aisles, "ap_of": ap_of}

def pick_through_aisle(half: str, half_dict: Dict[int, Dict[str, List[Coord]]]) -> Optional[int]:
    """走道分數：AP 數量 +（同時有 front 與 back 再 +2）；同分取 x 較大。"""
    if not half_dict: return None
    best_x, best_score = None, -1
    for x, groups in half_dict.items():
        n_ap = len(groups["upper_front"]) + len(groups["upper_back"]) + len(groups["lower_front"]) + len(groups["lower_back"])
        if half == "upper":
            has_front = len(groups["upper_front"]) > 0
            has_back  = len(groups["upper_back"])  > 0
        else:
            has_front = len(groups["lower_front"]) > 0
            has_back  = len(groups["lower_back"])  > 0
        score = n_ap + (2 if (has_front and has_back) else 0)
        if (score > best_score) or (score == best_score and (best_x is None or x > best_x)):
            best_x, best_score = x, score
    return best_x

def order_same_end(group: str, items: List[Coord]) -> List[Coord]:
    if group == "upper_back":   return sorted(items, key=lambda p: p[0])
    if group == "upper_front":  return sorted(items, key=lambda p: -p[0])
    if group == "lower_front":  return sorted(items, key=lambda p: p[0])
    return sorted(items, key=lambda p: -p[0])

def order_through_along_direction(half: str, groups: Dict[str, List[Coord]], back_to_front: bool) -> List[Coord]:
    pts: List[Coord] = []
    if half == "upper":
        pts += groups.get("upper_back", [])
        pts += groups.get("upper_front", [])
        return sorted(pts, key=lambda p: p[0]) if back_to_front else sorted(pts, key=lambda p: p[0], reverse=True)
    else:
        pts += groups.get("lower_front", [])
        pts += groups.get("lower_back", [])
        return sorted(pts, key=lambda p: p[0]) if not back_to_front else sorted(pts, key=lambda p: p[0], reverse=True)

def scan_non_through(half: str, group_key: str, half_dict: Dict[int, Dict[str, List[Coord]]],
                     through_x: int, mode: str) -> List[Coord]:
    xs_all = [x for x, g in half_dict.items() if x != through_x and len(g.get(group_key, [])) > 0]
    if not xs_all: return []
    ordered_items: List[Coord] = []

    if mode == 'towards_through_from_farthest':
        start_x = max(xs_all, key=lambda x: abs(x - through_x))
        dir_sign = 1 if start_x < through_x else -1
        xs_side = [x for x in xs_all if (x - through_x) * dir_sign < 0]
        xs_side_sorted = sorted(xs_side, key=lambda x: abs(x - through_x), reverse=True)
        for x in xs_side_sorted:
            ordered_items += order_same_end(group_key, half_dict[x][group_key])
        return ordered_items

    left_candidates  = [x for x in xs_all if x < through_x]
    right_candidates = [x for x in xs_all if x > through_x]
    dist_left  = (through_x - max(left_candidates)) if left_candidates else float('inf')
    dist_right = (min(right_candidates) - through_x) if right_candidates else float('inf')

    if dist_left <= dist_right:
        xs_side = sorted(left_candidates, key=lambda x: (through_x - x))
    else:
        xs_side = sorted(right_candidates, key=lambda x: (x - through_x))

    for x in xs_side:
        ordered_items += order_same_end(group_key, half_dict[x][group_key])
    return ordered_items

def reorder_task_items(robot_start: Coord, shelf_locations: List[Coord], wm: np.ndarray) -> List[Coord]:
    """依照線性掃描規則產生貨位訪問順序（僅回傳貨位，不含 AP）。"""
    if not shelf_locations: return []
    idx = build_index(shelf_locations, wm)
    aisles = idx["aisles"]

    start_half = "upper" if in_upper(robot_start[0]) else "lower"
    other_half = "lower" if start_half == "upper" else "upper"

    def make_half_order(half: str, phase: str) -> List[Coord]:
        hd = aisles[half]
        if not hd: return []
        through_x = pick_through_aisle(half, hd)
        ordered: List[Coord] = []

        if half == "upper":
            back_key, front_key = "upper_back", "upper_front"
        else:
            back_key, front_key = "lower_back", "lower_front"

        if phase == "first":
            ordered += scan_non_through(half, back_key,  hd, through_x, mode='towards_through_from_farthest')
            if through_x is not None:
                ordered += order_through_along_direction(half, hd[through_x], back_to_front=True)
            ordered += scan_non_through(half, front_key, hd, through_x, mode='outward_from_through_nearest')
        else:
            ordered += scan_non_through(half, front_key, hd, through_x, mode='towards_through_from_farthest')
            if through_x is not None:
                ordered += order_through_along_direction(half, hd[through_x], back_to_front=False)
            ordered += scan_non_through(half, back_key,  hd, through_x, mode='outward_from_through_nearest')

        return ordered

    first_half  = make_half_order(start_half, phase="first")
    second_half = make_half_order(other_half,  phase="second")
    return first_half + second_half

# =============================================================================
# SECTION 2: 策略整合與調度 (快取、完整路徑生成、主函式)
# =============================================================================

_l_v2_cache = {}

def get_l_v2_cache_key(start_pos: Coord, picks: List[Coord]) -> str:
    """生成 L-v2 策略快取的唯一鍵值"""
    picks_str = "_".join([f"{p[0]}-{p[1]}" for p in sorted(picks)])
    return f"l_v2_{start_pos[0]}-{start_pos[1]}_{picks_str}"

def clear_l_v2_cache():
    """清除 L-v2 策略的快取"""
    global _l_v2_cache
    _l_v2_cache = {}

def plan_l_v2_complete_route(start_pos: Coord, pick_locations: List[Coord], wm: np.ndarray,
                             dynamic_obstacles: Optional[List[Coord]], forbidden_cells: Optional[Set[Coord]], cost_map: Optional[Dict]):
    """根據 L-v2 排序結果，生成完整的 A* 路徑。"""
    idx = build_index(pick_locations, wm)
    ap_of = idx["ap_of"]
    ordered_shelves = reorder_task_items(start_pos, pick_locations, wm)

    path = [start_pos]
    curr = start_pos
    for shelf in ordered_shelves:
        ap = ap_of.get(shelf)
        if not ap: continue

        # 使用基礎 A* 演算法規劃路段
        segment = base_plan_route(curr, ap, wm, dynamic_obstacles, forbidden_cells, cost_map)
        if segment is None:
            print(f"❌ L-v2: 無法從 {curr} 規劃到 {ap}")
            return None  # 路徑規劃失敗

        # base_plan_route 返回的是從下一步開始的路徑，所以將其附加到目前路徑
        path.extend(segment)
        curr = ap

    return path

def plan_route(start_pos: Coord, target_pos: Coord, warehouse_matrix: np.ndarray,
               dynamic_obstacles: Optional[List[Coord]] = None,
               forbidden_cells: Optional[set] = None,
               cost_map: Optional[Dict] = None):
    """
    【核心策略函式】- L-v2 策略實作
    透過 `cost_map` 中的 `'l_v2_picks'` 鍵來觸發。
    """
    if cost_map is None: cost_map = {}

    # 檢查是否啟用 L-v2 策略
    if 'l_v2_picks' in cost_map and len(cost_map['l_v2_picks']) > 1:
        pick_locations = cost_map['l_v2_picks']
        print(f"🗺️ 啟用 L-v2 策略，撿貨點: {pick_locations}")

        cache_key = get_l_v2_cache_key(start_pos, pick_locations)

        if cache_key not in _l_v2_cache:
            full_path = plan_l_v2_complete_route(start_pos, pick_locations, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
            if full_path:
                _l_v2_cache[cache_key] = {"full_path": full_path}
                print(f"💾 快取 L-v2 策略路徑，共 {len(full_path)} 步")
            else:
                print("❌ L-v2 策略路徑規劃失敗，回退到 A* 演算法")
                return base_plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)

        # 從快取中取得路徑並返回適當段落
        cached_data = _l_v2_cache[cache_key]
        full_path = cached_data["full_path"]

        try:
            start_idx = full_path.index(start_pos)
            if target_pos in full_path[start_idx:]:
                end_idx = full_path.index(target_pos, start_idx)
                result_path = full_path[start_idx + 1:end_idx + 1]
                print(f"📍 返回 L-v2 策略路徑段: {len(result_path)} 步")
                return result_path if result_path else None
        except ValueError:
            # 如果起點或終點不在預計路徑中，也回退到 A*
            pass

        print(f"⚠️ 目標點不在 L-v2 快取路徑中，回退到 A* 演算法")

    # 預設行為：使用標準 A* 演算法
    return base_plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)