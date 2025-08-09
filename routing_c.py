# composite_routing.py
"""
Composite（複合）撿貨路徑策略
-------------------------------------------------------
目標：
- 在每個「有撿貨的走道（垂直走道 x）」上，動態決定使用：
  - S-shape（貫穿，traverse / through），或
  - Return（同端進出，return）
- 決策要考慮「穿越會改變目前所處的主幹道側（front/back）」，
  因而影響下一條走道的進入位置與成本 → 採用 1-步前視（look-ahead）評估。
- 依倉庫特性，形成 C 型撿貨：第一半區（upper 或 lower）由右→左，
  第二半區由左→右，確保掃完第二半區後自然回到右側的撿貨站。

整合方式：
- 本模組對外提供：
  1) `plan_route(...)`：僅轉呼叫現有 routing.py 的 A*，保持與主引擎相容。
  2) `reorder_task_items(...)`：依 Composite 策略「重排任務中的各貨位順序」。
     你只要在任務分配時，先用它重排 `task["shelf_locations"]`，模擬就會按此順序逐點前進。

實作重點（不重複定義）：
- AP 映射與分組（upper/lower、front/back）沿用你 `test_routing_l.py` 的工具函式。
- A* 與輔助函式沿用 `routing.py`（`plan_route`、`euclidean_distance`、`find_adjacent_aisle`）。

參考原理（實作在程式內中文註解）：
- Composite/Combined 政策結合 S-shape 與 Return，並用狀態（front/rear）在 block 內遞推
  （本實作用簡化版 DP 概念 → 1-step look-ahead 做近似，降低計算量）。
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

# === 重用既有函式：A* 與距離/找走道輔助 ===
from routing import plan_route as base_plan_route, euclidean_distance, find_adjacent_aisle

# === 重用你 test_routing_l.py 的 AP 與分組工具 ===
from test_routing_l import (
    build_index,                # 生成：{ "aisles": {"upper": {ax: {...}}, "lower": {...}}, "ap_of": {shelf: ap} }
    order_same_end,             # 同端進出時，某組(group)內的順序（避免走道內折返）
    order_through_along_direction,  # 貫穿時，依行進方向排序（先碰到 AP 先撿）
    in_upper, classify_group, half_from_group
)

Coord = Tuple[int, int]
GroupDict = Dict[str, List[Coord]]

# 走道可通行碼（沿用 test_routing_l 的概念）
AISLE_CODES = {0, 7}  # 0=走道, 7=撿貨出口

# === 與 main.py 相容：提供 plan_route，實際轉給 base(A*) ===
def plan_route(start_pos: Coord, target_pos: Coord, warehouse_matrix: np.ndarray,
               dynamic_obstacles: Optional[List[Coord]] = None,
               forbidden_cells: Optional[set] = None,
               cost_map: Optional[Dict[Coord, int]] = None):
    """與主系統相容的路徑規劃入口：轉呼叫 routing.py 的 A*。"""
    return base_plan_route(start_pos, target_pos, warehouse_matrix,
                           dynamic_obstacles=dynamic_obstacles,
                           forbidden_cells=forbidden_cells,
                           cost_map=cost_map)

# =========================
# Composite 的「重排訪問順序」核心
# =========================

def reorder_task_items(robot_start: Coord,
                       shelf_locations: List[Coord],
                       wm: np.ndarray) -> List[Coord]:
    """
    依 Composite 策略，重排整張任務的貨位訪問順序（回傳純「貨位」清單，不含 AP）。
    - 先掃「機器人所在半區」→ 再掃另一半區。
    - 第一半區：由右往左；第二半區：由左往右（形成 C 型路線，利於回右側撿貨站）。
    - 每個有貨的走道，動態決定用 through（貫穿）或 return（同端進出）。
    - 走道內部順序遵循「先碰到 AP 先撿」原則，避免走道內折返。
    """
    if not shelf_locations:
        return []

    idx = build_index(shelf_locations, wm)
    aisles = idx["aisles"]       # {"upper": {ax: {"upper_back": [...], ...}}, "lower": {...}}
    ap_of = idx["ap_of"]         # { (r, c_shelf): (r, c_ap) }

    def half_bounds(half: str) -> Tuple[int, int]:
        # 定義上下半區邊界（front/back 的「邊界 row」）
        # upper: rows 0..6 → front=6, back=0
        # lower: rows 7..13 → front=7, back=13
        return (6, 0) if half == "upper" else (7, 13)

    def rows_in_half(groups: GroupDict, half: str) -> List[int]:
        # 萃取該半區所有貨位的「row」
        if half == "upper":
            keys = ("upper_front", "upper_back")
        else:
            keys = ("lower_front", "lower_back")
        pts = []
        for k in keys:
            for (r, _c) in groups.get(k, []):
                pts.append(r)
        return pts

    def farthest_depth_from_side(half: str, groups: GroupDict, side: str) -> int:
        """計算：若從指定 side(front/back) 進入此走道，要到達最遠貨位需走的「縱向深度」。
        這是 aisle 內的近似成本：through≈depth、return≈2*depth。"""
        rows = rows_in_half(groups, half)
        if not rows:
            return 0
        front_row, back_row = half_bounds(half)
        if side == "front":
            # 從 front 邊界往內（upper: 6->0 降；lower: 7->13 升）
            return max(abs(front_row - r) for r in rows)
        else:
            return max(abs(r - back_row) for r in rows)

    def has_both_sides(half: str, groups: GroupDict) -> bool:
        if half == "upper":
            return bool(groups.get("upper_front")) and bool(groups.get("upper_back"))
        else:
            return bool(groups.get("lower_front")) and bool(groups.get("lower_back"))

    def order_return_both_groups(half: str, cur_side: str, groups: GroupDict) -> List[Coord]:
        """同端進出：先撿「進入端」那一側的組，再撿另一側。排序遵循每組的單向掃描規則。"""
        if half == "upper":
            g_front, g_back = "upper_front", "upper_back"
        else:
            g_front, g_back = "lower_front", "lower_back"

        seq: List[Coord] = []
        if cur_side == "front":
            if groups.get(g_front):
                seq += order_same_end(g_front, groups[g_front])
            if groups.get(g_back):
                # 回程朝 cur_side 移動，對 back 組用其單向順序即可（方向與回程一致）
                seq += order_same_end(g_back, groups[g_back])
        else:  # cur_side == "back"
            if groups.get(g_back):
                seq += order_same_end(g_back, groups[g_back])
            if groups.get(g_front):
                seq += order_same_end(g_front, groups[g_front])
        return seq

    def order_through(half: str, from_side: str, groups: GroupDict) -> List[Coord]:
        """貫穿：沿行進方向排序，保證「先碰到 AP 先撿」。"""
        # test_routing_l 的邏輯：back_to_front=True 代表由 back→front；False 代表 front→back
        back_to_front = (from_side == "back")
        return order_through_along_direction(half, groups, back_to_front=back_to_front)

    def decide_through(half: str,
                       groups_now: GroupDict,
                       cur_side: str,
                       groups_next: Optional[GroupDict]) -> bool:
        """
        是否在「當前走道」採用貫穿？
        規則（綜合文獻與實務，並做 1-步前視）：
        1) 若當前走道兩側皆有貨（front/back 都有），更傾向貫穿（Combined/Composite 的經典情形）。
        2) 局部距離：through ≈ depth，return ≈ 2*depth（同一側進出）。
        3) 1-step look-ahead：比較「透過貫穿翻到對側」對下一走道進入深度的影響。
        4) Tie-break：若 depth 很深（接近半區長度的一半以上），偏向 through。
        """
        # 沒貨則無意義
        d_cur_front = farthest_depth_from_side(half, groups_now, cur_side)
        if d_cur_front == 0:
            return False

        # 規則 1：兩側都有 → 強勢候選
        both_sides = has_both_sides(half, groups_now)

        # 當前 aisle 的局部成本
        cost_through_now = d_cur_front               # through 只走到最遠點（從 cur_side 到對側）
        cost_return_now  = 2 * d_cur_front           # return 去最遠點再回頭

        # 1-step look-ahead：假設下一走道會選「就當下側起算的最佳（min(depth, 2*depth)）」。
        # 這裡比較的只是 “起始側不同” 帶來的差異（因為 through 會翻面）。
        def expected_next_cost(start_side: str) -> int:
            if not groups_next:
                return 0
            d_next = farthest_depth_from_side(half, groups_next, start_side)
            # 局部最佳：min(through≈d_next, return≈2*d_next) = d_next
            return d_next

        next_cost_if_through = expected_next_cost("back" if cur_side == "front" else "front")
        next_cost_if_return  = expected_next_cost(cur_side)

        # 規則 4：深度閾值（半區長度的一半）
        front_row, back_row = half_bounds(half)
        half_len = abs(front_row - back_row)
        deep_threshold = max(1, half_len // 2)  # 走道很深時，就算看不到未來，也偏向貫穿

        # 綜合評分（可微調 gamma 作為翻面的保守度）
        gamma = 0  # 你想更保守，就設為正值（避免頻繁翻面）
        score_through = cost_through_now + next_cost_if_through + gamma
        score_return  = cost_return_now  + next_cost_if_return

        if both_sides and d_cur_front >= deep_threshold:
            # 強力觸發：當前兩側皆有且很深 → 貫穿
            return True

        # 一般情況：比較加總成本
        return score_through < score_return

    def sweep_one_half(half: str, first_half: bool) -> List[Coord]:
        """
        在指定半區（upper/lower）做一次掃描。
        - 第一半區：走道 x 從右→左
        - 第二半區：走道 x 從左→右
        - 初始在「front」側（靠近中線），符合你地圖與起點習慣。
        """
        ordered: List[Coord] = []
        hd: Dict[int, GroupDict] = aisles[half]
        if not hd:
            return ordered

        xs = sorted(hd.keys(), reverse=first_half)  # 第一半區右→左；第二半區左→右
        cur_side = "front"  # 依你地圖設定，從中間主幹道側啟動

        for i, ax in enumerate(xs):
            groups_now = hd[ax]
            groups_next = hd[xs[i + 1]] if i + 1 < len(xs) else None
            if not rows_in_half(groups_now, half):
                continue

            use_through = decide_through(half, groups_now, cur_side, groups_next)

            if use_through:
                ordered += order_through(half, cur_side, groups_now)
                # 貫穿後翻面
                cur_side = "back" if cur_side == "front" else "front"
            else:
                ordered += order_return_both_groups(half, cur_side, groups_now)
                # return 不翻面

        return ordered

    # 半區順序：起點在哪，就先掃哪一半（upper→lower 或 lower→upper）
    start_half = "upper" if in_upper(robot_start[0]) else "lower"
    other_half = "lower" if start_half == "upper" else "upper"

    first  = sweep_one_half(start_half, first_half=True)   # 右→左
    second = sweep_one_half(other_half,  first_half=False) # 左→右

    return first + second
