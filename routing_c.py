# composite_routing.py
"""
Composite（複合）撿貨路徑策略
-------------------------------------------------------
目的（高階）：
  - 以「走道為單位」動態選擇：
      • S-shape（貫穿，through）：同一條走道前後兩側都處理，並「翻面」到另一側主幹道。
      • Return（同端進出，return）：只從當前側進去，處理完兩側（有貨的一側在回程處理），最後回到原側主幹道。
  - 決策考慮「穿越會改變目前所處主幹道側（front/back）」，會影響下一條走道的進入成本。
    因此使用 1-step look-ahead：估計「下一條走道」在不同起始側時的深度差異。
  - 形成 C 型撿貨：第一半區（upper 或 lower）從右→左，第二半區從左→右，確保掃完後自然回到右側撿貨站。

整合方式（本模組對外提供）：
  1) plan_route(...)：單純代理 routing.py 的 A*，完全相容主引擎呼叫（回傳「逐格座標」或 None）。
  2) reorder_task_items(...)：依本 Composite 策略「重排任務中的貨位順序」。
     - 輸入：機器人起點 robot_start、任務貨位清單 shelf_locations、倉庫矩陣 wm。
     - 輸出：一個「貨位座標列表 List[Coord]」，代表建議的撿貨順序（僅貨位，不含 AP）。
       主程式可用這個順序逐點規劃路徑（建議走各貨位的 AP，不真正走進貨架）。

不重複定義：
  - AP 映射、分組（upper/lower、front/back）沿用你現有的工具（harr.routing_l / test_routing_l）。
  - A* 與距離輔助沿用 routing.py（plan_route, euclidean_distance, find_adjacent_aisle）。

關鍵直覺：
  - 「走道內的成本」用「最遠貨位的縱向深度 depth」近似：
      through ≈ depth
      return  ≈ 2 * depth   （因為要走到最遠點再回頭）
  - 1-step look-ahead：比較若選 through（翻面）或 return（不翻面），「下一條走道」的啟始側不同，所造成的深度差異。
    最後以「當前成本 + 下一條預估成本」做出選擇；必要時用深度閾值做 tie-break（當前走道很深且兩側皆有貨時，偏向 through）。
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

# === 既有 A* 與輔助（只代理呼叫，不重複定義） ===
from routing import plan_route as base_plan_route, euclidean_distance, find_adjacent_aisle

# === AP 與分組工具（不重複定義） ===
# 你環境使用 harr.routing_l；若在本專案原結構，請改為 from test_routing_l import ...
from harr.routing_l import (
    build_index,                # -> {"aisles": {"upper": {ax: groups}, "lower": {...}}, "ap_of": {...}}
    order_same_end,             # 同端進出時，單一「組」內的順序（避免在走道內折返）
    order_through_along_direction,  # 貫穿時，沿行進方向排序：「先碰到 AP 先撿」
    in_upper, classify_group, half_from_group
)

Coord = Tuple[int, int]
GroupDict = Dict[str, List[Coord]]

# （說明用）走道可通行碼：與 test_routing_l 一致
AISLE_CODES = {0, 7}  # 0=走道, 7=撿貨出口（兩者都允許通行）


# === 與 main.py 相容：提供 plan_route，實際轉給 A* ===
def plan_route(start_pos: Coord, target_pos: Coord, warehouse_matrix: np.ndarray,
               dynamic_obstacles: Optional[List[Coord]] = None,
               forbidden_cells: Optional[set] = None,
               cost_map: Optional[Dict[Coord, int]] = None):
    """
    【與主系統相容的路徑規劃入口】
    參數：
      - start_pos: 起點座標
      - target_pos: 目標座標
      - warehouse_matrix: 倉庫矩陣（0/1/2/...）
      - dynamic_obstacles: 動態障礙（其他機器人座標）
      - forbidden_cells: 此次規劃中絕對禁止踩踏的格
      - cost_map: 各格移動成本（未提供則預設 1）
    回傳：
      - List[Coord]：從「下一步」開始的逐格路徑（A* 找到的可通行軌跡），例如 [(r1,c1), (r2,c2), ...]
      - None：找不到路徑
    """
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
    依「Composite 策略」重排整張任務的貨位訪問順序（僅回傳貨位，不含 AP）。
    掃描規則：
      1) 先掃「機器人所在半區」，再掃另一半區。
      2) 第一半區：走道 x 由右→左；第二半區：走道 x 由左→右（形成 C 型撿貨，利於回右側撿貨站）。
      3) 每個「有貨的走道」在進入前會決策：
           - through：貫穿；處理完後「翻面」（front↔back）。
           - return ：同端進出；處理完後仍在「原側」。
      4) 走道內部遵守「先碰到 AP 先撿」（避免走道內折返）。
    參數：
      - robot_start: 機器人起點（row, col）
      - shelf_locations: 任務貨位清單（row, col）
      - wm: 倉庫矩陣
    回傳：
      - List[Coord]：建議撿貨順序（貨位本身）；主程式應以各貨位的 AP（由 build_index 提供）串接 A* 路徑。
    """
    if not shelf_locations:
        return []

    # 依貨位生成索引（分半區、分走道、分前/後組；以及每個貨位對應的 AP）
    idx = build_index(shelf_locations, wm)
    aisles = idx["aisles"]       # {"upper": {ax: {"upper_back": [...], ...}}, "lower": {...}}
    # ap_of = idx["ap_of"]       # 本函式只決定貨位順序，AP 在外部組裝路徑時使用

    # ---- 輔助：半區邊界（定義 front/back 的「靠近中線 row」與「靠外 row」）----
    # upper: rows 0..6  →  front=6（靠中線）, back=0（靠倉庫上緣）
    # lower: rows 7..13 →  front=7（靠中線）, back=13（靠倉庫下緣）
    def half_bounds(half: str) -> Tuple[int, int]:
        return (6, 0) if half == "upper" else (7, 13)

    # ---- 輔助：取出某半區內、某走道 groups 的所有 row 值（用來估算深度）----
    def rows_in_half(groups: GroupDict, half: str) -> List[int]:
        keys = ("upper_front", "upper_back") if half == "upper" else ("lower_front", "lower_back")
        pts: List[int] = []
        for k in keys:
            for (r, _c) in groups.get(k, []):
                pts.append(r)
        return pts

    # ---- 輔助：估算「走道內最遠貨位深度」→ 作為走道內成本近似 ----
    # 直覺：從指定 side（front/back）進入該走道，最遠要深入幾格？
    #   through ≈ depth               （深入到最遠點後從另一側離開）
    #   return  ≈ 2 * depth           （深入到最遠點還要原路返回）
    def farthest_depth_from_side(half: str, groups: GroupDict, side: str) -> int:
        rows = rows_in_half(groups, half)
        if not rows:
            return 0
        front_row, back_row = half_bounds(half)
        if side == "front":
            return max(abs(front_row - r) for r in rows)
        else:
            return max(abs(r - back_row) for r in rows)

    # ---- 輔助：判斷此走道的兩側是否都有貨（Composite 的「典型貫穿候選」）----
    def has_both_sides(half: str, groups: GroupDict) -> bool:
        if half == "upper":
            return bool(groups.get("upper_front")) and bool(groups.get("upper_back"))
        else:
            return bool(groups.get("lower_front")) and bool(groups.get("lower_back"))

    # ---- 輔助：Return（同端進出）時，走道內的撿取順序 ----
    # 規則：先處理「進入端」那一側的組，再處理另一側；每個組內使用單向掃描順序（order_same_end）
    # 這能確保不在走道內折返，並且回到原側主幹道
    def order_return_both_groups(half: str, cur_side: str, groups: GroupDict) -> List[Coord]:
        if half == "upper":
            g_front, g_back = "upper_front", "upper_back"
        else:
            g_front, g_back = "lower_front", "lower_back"

        seq: List[Coord] = []
        if cur_side == "front":
            if groups.get(g_front): seq += order_same_end(g_front, groups[g_front])
            if groups.get(g_back):  seq += order_same_end(g_back,  groups[g_back])
        else:
            if groups.get(g_back):  seq += order_same_end(g_back,  groups[g_back])
            if groups.get(g_front): seq += order_same_end(g_front, groups[g_front])
        return seq

    # ---- 輔助：Through（貫穿）時，走道內的撿取順序 ----
    # 由當前側朝另一側前進；內部順序由 order_through_along_direction 負責確保「先碰到 AP 先撿」
    def order_through(half: str, from_side: str, groups: GroupDict) -> List[Coord]:
        back_to_front = (from_side == "back")  # True: back→front；False: front→back
        return order_through_along_direction(half, groups, back_to_front=back_to_front)

    # ---- 決策：當前走道是否採用 Through？（核心 1-step look-ahead）----
    def decide_through(half: str,
                       groups_now: GroupDict,
                       cur_side: str,
                       groups_next: Optional[GroupDict]) -> bool:
        """
        計分思路（皆為「相對」比較，並非真實距離）：
          • d_cur_front = farthest_depth_from_side(half, groups_now, cur_side)
            - 當前走道的「深入深度」。
          • 當前成本：
              cost_through_now = d_cur_front
              cost_return_now  = 2 * d_cur_front
          • 下一走道的預估成本（只看深度，不管它之後是否 return/through，因為 min ≈ depth）：
              next_cost_if_through = depth(next, start_side = 翻面之後的側)
              next_cost_if_return  = depth(next, start_side = 維持原側)
          • 總分：
              score_through = cost_through_now + next_cost_if_through + gamma
              score_return  = cost_return_now  + next_cost_if_return
            其中 gamma 可作為保守參數（>0 時降低翻面的頻率）。
          • 強勢條件（tie-break）：若當前走道「兩側皆有貨」且「深度很深（≥半區長度/2）」→ 直接選 through。
        回傳：
          - True  表示採用 Through
          - False 表示採用 Return
        """
        # 1) 無貨：沒有意義，直接不 through
        d_cur_front = farthest_depth_from_side(half, groups_now, cur_side)
        if d_cur_front == 0:
            return False

        # 2) 記錄是否兩側皆有貨（經典 Composite 情形）
        both_sides = has_both_sides(half, groups_now)

        # 3) 當前走道的局部成本（近似）
        cost_through_now = d_cur_front
        cost_return_now  = 2 * d_cur_front

        # 4) 1-step look-ahead：比較「翻面/不翻面」對下一走道的起始側影響
        def expected_next_cost(start_side: str) -> int:
            if not groups_next:  # 沒有下一走道，視為 0
                return 0
            d_next = farthest_depth_from_side(half, groups_next, start_side)
            # 下一走道的最佳近似成本 ≈ d_next（因為 min(through≈d_next, return≈2*d_next) = d_next）
            return d_next

        next_cost_if_through = expected_next_cost("back" if cur_side == "front" else "front")
        next_cost_if_return  = expected_next_cost(cur_side)

        # 5) tie-break：若當前走道夠深 + 兩側皆有貨，偏向 through
        front_row, back_row = half_bounds(half)
        half_len = abs(front_row - back_row)
        deep_threshold = max(1, half_len // 2)

        gamma = 0  # 若你想降低翻面頻率，可調成正值
        score_through = cost_through_now + next_cost_if_through + gamma
        score_return  = cost_return_now  + next_cost_if_return

        if both_sides and d_cur_front >= deep_threshold:
            return True  # 強勢條件觸發

        # 6) 一般情況：比較加總成本
        return score_through < score_return

    # ---- 在指定半區掃描所有有貨走道，按左右方向組裝貨位順序 ----
    def sweep_one_half(half: str, first_half: bool) -> List[Coord]:
        """
        half: "upper" 或 "lower"
        first_half: True → 第一半區（右→左），False → 第二半區（左→右）
        回傳：此半區內，依決策展開後的「貨位順序（List[Coord]）」
        """
        ordered: List[Coord] = []
        hd: Dict[int, GroupDict] = aisles[half]
        if not hd:
            return ordered

        xs = sorted(hd.keys(), reverse=first_half)
        cur_side = "front"  # 起始在「中線側」

        for i, ax in enumerate(xs):
            groups_now = hd[ax]
            groups_next = hd[xs[i + 1]] if i + 1 < len(xs) else None
            if not rows_in_half(groups_now, half):
                # 此走道沒有貨，跳過
                continue

            use_through = decide_through(half, groups_now, cur_side, groups_next)

            if use_through:
                # 貫穿：沿行進方向撿 → 翻面（front↔back）
                ordered += order_through(half, cur_side, groups_now)
                cur_side = "back" if cur_side == "front" else "front"
            else:
                # 同端進出：先撿進入側組，回程撿另一側組 → 不翻面
                ordered += order_return_both_groups(half, cur_side, groups_now)

        return ordered

    # ---- 半區順序：起點在哪，就先掃哪一半（upper→lower 或 lower→upper）----
    start_half = "upper" if in_upper(robot_start[0]) else "lower"
    other_half = "lower" if start_half == "upper" else "upper"

    # 第一半區：右→左；第二半區：左→右（C 型）
    first  = sweep_one_half(start_half, first_half=True)
    second = sweep_one_half(other_half,  first_half=False)

    # 回傳「整體貨位順序」（給外部組裝 AP 路徑與 A*）
    return first + second
