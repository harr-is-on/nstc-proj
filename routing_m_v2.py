"""
路徑規劃策略模組 - Composite v2 (複合策略)

這個檔案實作了 Composite v2 路徑規劃策略。
主要函式 `plan_route` 遵循與系統中其他策略模組相同的輸入輸出格式。

Composite v2 策略運作原理：
========================
1.  **觸發方式**:
    - 在呼叫 `plan_route` 時，於 `cost_map` 字典中提供一個鍵 `'m_v2_picks'`，其值為所有撿貨點的列表。

2.  **路徑規劃流程**:
    - **首次呼叫**:
        1.  `reorder_task_items`: 依 Composite 策略重排任務中的各貨位順序。此策略會動態決定在每個走道使用貫穿（S-shape）或同端進出（Return）。
        2.  `plan_m_v2_complete_route`: 根據排好的順序，使用基礎 A* 演算法逐段規劃出完整路徑。
        3.  **快取**: 將計算出的完整路徑儲存在記憶體中。
    - **後續呼叫**:
        1.  直接從快取中讀取完整路徑。
        2.  根據當前的 `start_pos` 和 `target_pos`，返回路徑中的對應路段。

3.  **回退機制**:
    - 如果 `cost_map` 中未提供 `'m_v2_picks'`，或者路徑規劃失敗，則自動回退使用標準的 A* 演算法。
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np

# === 相依性函式 ===
from routing import plan_route as base_plan_route, euclidean_distance, find_adjacent_aisle
# 從 L-v2 模組中重用輔助函式，修正了原始的 test_routing_l 依賴
from routing_l_v2 import (
    build_index, order_same_end, order_through_along_direction,
    in_upper, classify_group, half_from_group
)

# --- 型別別名與常數 ---
Coord = Tuple[int, int]
GroupDict = Dict[str, List[Coord]]

# =============================================================================
# SECTION 1: M-v2 核心邏輯 (排序與輔助函式)
# =============================================================================

def reorder_task_items(robot_start: Coord,
                       shelf_locations: List[Coord],
                       wm: np.ndarray) -> List[Coord]:
    """
    依 Composite 策略，重排整張任務的貨位訪問順序。
    """
    if not shelf_locations:
        return []

    idx = build_index(shelf_locations, wm)
    aisles = idx["aisles"]

    def half_bounds(half: str) -> Tuple[int, int]:
        return (6, 0) if half == "upper" else (7, 13)

    def rows_in_half(groups: GroupDict, half: str) -> List[int]:
        keys = ("upper_front", "upper_back") if half == "upper" else ("lower_front", "lower_back")
        return [r for k in keys for r, _c in groups.get(k, [])]

    def farthest_depth_from_side(half: str, groups: GroupDict, side: str) -> int:
        rows = rows_in_half(groups, half)
        if not rows: return 0
        front_row, back_row = half_bounds(half)
        if side == "front":
            return max(abs(front_row - r) for r in rows)
        else:
            return max(abs(r - back_row) for r in rows)

    def has_both_sides(half: str, groups: GroupDict) -> bool:
        if half == "upper":
            return bool(groups.get("upper_front")) and bool(groups.get("upper_back"))
        else:
            return bool(groups.get("lower_front")) and bool(groups.get("lower_back"))

    def order_return_both_groups(half: str, cur_side: str, groups: GroupDict) -> List[Coord]:
        if half == "upper": g_front, g_back = "upper_front", "upper_back"
        else: g_front, g_back = "lower_front", "lower_back"

        seq: List[Coord] = []
        if cur_side == "front":
            if groups.get(g_front): seq += order_same_end(g_front, groups[g_front])
            if groups.get(g_back): seq += order_same_end(g_back, groups[g_back])
        else:
            if groups.get(g_back): seq += order_same_end(g_back, groups[g_back])
            if groups.get(g_front): seq += order_same_end(g_front, groups[g_front])
        return seq

    def decide_through(half: str, groups_now: GroupDict, cur_side: str, groups_next: Optional[GroupDict]) -> bool:
        d_cur_front = farthest_depth_from_side(half, groups_now, cur_side)
        if d_cur_front == 0: return False

        cost_through_now = d_cur_front
        cost_return_now  = 2 * d_cur_front

        def expected_next_cost(start_side: str) -> int:
            if not groups_next: return 0
            return farthest_depth_from_side(half, groups_next, start_side)

        next_cost_if_through = expected_next_cost("back" if cur_side == "front" else "front")
        next_cost_if_return  = expected_next_cost(cur_side)

        score_through = cost_through_now + next_cost_if_through
        score_return  = cost_return_now  + next_cost_if_return

        front_row, back_row = half_bounds(half)
        half_len = abs(front_row - back_row)
        deep_threshold = max(1, half_len // 2)

        if has_both_sides(half, groups_now) and d_cur_front >= deep_threshold:
            return True

        return score_through < score_return

    def sweep_one_half(half: str, first_half: bool) -> List[Coord]:
        ordered: List[Coord] = []
        hd: Dict[int, GroupDict] = aisles[half]
        if not hd: return ordered

        xs = sorted(hd.keys(), reverse=first_half)
        cur_side = "front"

        for i, ax in enumerate(xs):
            groups_now = hd[ax]
            groups_next = hd.get(xs[i + 1]) if i + 1 < len(xs) else None
            if not rows_in_half(groups_now, half): continue

            use_through = decide_through(half, groups_now, cur_side, groups_next)

            if use_through:
                ordered += order_through_along_direction(half, groups_now, back_to_front=(cur_side == "back"))
                cur_side = "back" if cur_side == "front" else "front"
            else:
                ordered += order_return_both_groups(half, cur_side, groups_now)
        return ordered

    start_half = "upper" if in_upper(robot_start[0]) else "lower"
    other_half = "lower" if start_half == "upper" else "upper"

    first  = sweep_one_half(start_half, first_half=True)
    second = sweep_one_half(other_half,  first_half=False)

    return first + second

# =============================================================================
# SECTION 2: 策略整合與調度 (快取、完整路徑生成、主函式)
# =============================================================================

_m_v2_cache = {}

def get_m_v2_cache_key(start_pos: Coord, picks: List[Coord]) -> str:
    """生成 M-v2 策略快取的唯一鍵值"""
    picks_str = "_".join([f"{p[0]}-{p[1]}" for p in sorted(picks)])
    return f"m_v2_{start_pos[0]}-{start_pos[1]}_{picks_str}"

def clear_m_v2_cache():
    """清除 M-v2 策略的快取"""
    global _m_v2_cache
    _m_v2_cache = {}

def plan_m_v2_complete_route(start_pos: Coord, pick_locations: List[Coord], wm: np.ndarray,
                             dynamic_obstacles: Optional[List[Coord]], forbidden_cells: Optional[Set[Coord]], cost_map: Optional[Dict]):
    """根據 M-v2 排序結果，生成完整的 A* 路徑。"""
    idx = build_index(pick_locations, wm)
    ap_of = idx["ap_of"]
    ordered_shelves = reorder_task_items(start_pos, pick_locations, wm)

    path = [start_pos]
    curr = start_pos
    for shelf in ordered_shelves:
        ap = ap_of.get(shelf)
        if not ap: continue

        segment = base_plan_route(curr, ap, wm, dynamic_obstacles, forbidden_cells, cost_map)
        if segment is None:
            print(f"❌ M-v2: 無法從 {curr} 規劃到 {ap}")
            return None

        path.extend(segment)
        curr = ap

    return path

def plan_route(start_pos: Coord, target_pos: Coord, warehouse_matrix: np.ndarray,
               dynamic_obstacles: Optional[List[Coord]] = None,
               forbidden_cells: Optional[set] = None,
               cost_map: Optional[Dict] = None):
    """
    【核心策略函式】- M-v2 策略實作
    透過 `cost_map` 中的 `'m_v2_picks'` 鍵來觸發。
    """
    if cost_map is None: cost_map = {}

    if 'm_v2_picks' in cost_map and len(cost_map['m_v2_picks']) > 1:
        pick_locations = cost_map['m_v2_picks']
        print(f"🗺️ 啟用 M-v2 策略，撿貨點: {pick_locations}")

        cache_key = get_m_v2_cache_key(start_pos, pick_locations)

        if cache_key not in _m_v2_cache:
            full_path = plan_m_v2_complete_route(start_pos, pick_locations, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
            if full_path:
                _m_v2_cache[cache_key] = {"full_path": full_path}
                print(f"💾 快取 M-v2 策略路徑，共 {len(full_path)} 步")
            else:
                print("❌ M-v2 策略路徑規劃失敗，回退到 A* 演算法")
                return base_plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)

        cached_data = _m_v2_cache[cache_key]
        full_path = cached_data["full_path"]

        try:
            start_idx = full_path.index(start_pos)
            if target_pos in full_path[start_idx:]:
                end_idx = full_path.index(target_pos, start_idx)
                result_path = full_path[start_idx + 1:end_idx + 1]
                print(f"📍 返回 M-v2 策略路徑段: {len(result_path)} 步")
                return result_path if result_path else None
        except ValueError:
            pass

        print(f"⚠️ 目標點不在 M-v2 快取路徑中，回退到 A* 演算法")

    return base_plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)