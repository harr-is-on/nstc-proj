
from typing import List, Tuple, Dict, Optional
import numpy as np
from routing import plan_route as base_plan_route
from routing import euclidean_distance, find_adjacent_aisle

"""Largest Gap演算法: 依貨位分佈，動態決定揀貨順序，確保利用同端進出撿貨策略之優勢
運算邏輯：
1.將倉庫區分成上下兩區塊，區塊間區分成front/back兩組
2.每個塊會有一個「貫穿走道」，用來連接兩端(front/back)的貨位
3.假設機器人出發點在倉庫最右側(假設剛結束充電任務/揀貨任務)
4.機器人會環繞倉庫進行類似C形揀貨路徑(右→左→右)，確保最後趟次方向向右可以順路撿貨回到撿貨區
5.檢貨順序決定:如果機器人起始位置在UPPER，則順序為 UPPER BACK - UPPER FRONT - LOWER FRONT - LOWER BACK ，反之則 LOWER BACK - LOWER FRONT - UPPER FRONT - UPPER BACK
6.貫穿走道的選擇: 在確定揀貨順序後，機器人會選擇最適合的貫穿走道，以最小化總路徑成本，貫穿走道的評分機制，主要考慮以下幾點：
   - AP 數量：貫穿走道上可通行的 AP 數量越多，得分越高(值得採取貫穿取貨，一條龍撿貨到底)
   - 同時有 front 與 back：若貫穿走道同時連接 front 與 back，則額外加分(由於頭尾皆有貨物，機器人可以在同一趟次完成撿貨)
   - 走道位置：走道位置越靠近機器人起始位置，得分越高
7.除了貫穿走道以外的貨物，機器人皆採取同側進出撿貨策略
"""
Coord = Tuple[int, int]
AISLE_CODES = {0, 7}  # 0=走道, 7=撿貨出口

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

# ---- 單走道內的撿取順序（同端進出）----
def order_same_end(group: str, items: List[Coord]) -> List[Coord]:
    if group == "upper_back":   return sorted(items, key=lambda p: p[0])   # 0→3
    if group == "upper_front":  return sorted(items, key=lambda p: -p[0])  # 6→4
    if group == "lower_front":  return sorted(items, key=lambda p: p[0])   # 7→9
    return sorted(items, key=lambda p: -p[0])                              # lower_back: 13→10

# ---- 貫穿走道：沿行進方向排序（確保走到誰先、就先撿誰）----
def order_through_along_direction(half: str, groups: Dict[str, List[Coord]], back_to_front: bool) -> List[Coord]:
    pts: List[Coord] = []
    if half == "upper":
        pts += groups.get("upper_back", [])
        pts += groups.get("upper_front", [])
        # 第一半區貫穿：back->front，用 y 由小到大
        return sorted(pts, key=lambda p: p[0]) if back_to_front else sorted(pts, key=lambda p: p[0], reverse=True)
    else:
        pts += groups.get("lower_front", [])
        pts += groups.get("lower_back", [])
        # 第二半區貫穿：front->back，用 y 由小到大
        return sorted(pts, key=lambda p: p[0]) if not back_to_front else sorted(pts, key=lambda p: p[0], reverse=True)

# ---- 非貫穿走道掃描：從「邊緣」單向掃描 ----
def scan_non_through(half: str, group_key: str, half_dict: Dict[int, Dict[str, List[Coord]]],
                     through_x: int, mode: str) -> List[Coord]:
    """
    mode:
      - 'towards_through_from_farthest': 從距離 through 最遠的走道開始，朝 through 方向掃描
      - 'outward_from_through_nearest': 從距離 through 最近的一側開始，往外掃描（單側）
    """
    xs_all = [x for x, g in half_dict.items() if x != through_x and len(g.get(group_key, [])) > 0]
    if not xs_all:
        return []
    ordered_items: List[Coord] = []

    if mode == 'towards_through_from_farthest':
        start_x = max(xs_all, key=lambda x: abs(x - through_x))
        dir_sign = 1 if start_x < through_x else -1
        xs_side = [x for x in xs_all if (x - through_x) * dir_sign < 0]
        xs_side_sorted = sorted(xs_side, key=lambda x: abs(x - through_x), reverse=True)  # 遠→近
        for x in xs_side_sorted:
            ordered_items += order_same_end(group_key, half_dict[x][group_key])
        return ordered_items

    # outward_from_through_nearest
    left_candidates  = [x for x in xs_all if x < through_x]
    right_candidates = [x for x in xs_all if x > through_x]
    dist_left  = (through_x - max(left_candidates)) if left_candidates else None
    dist_right = (min(right_candidates) - through_x) if right_candidates else None

    if dist_left is not None and (dist_right is None or dist_left <= dist_right):
        xs_side = sorted(left_candidates, key=lambda x: (through_x - x))  # 近→遠
    else:
        xs_side = sorted(right_candidates, key=lambda x: (x - through_x))  # 近→遠

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

        if phase == "first":  # 第一半區
            ordered += scan_non_through(half, back_key,  hd, through_x, mode='towards_through_from_farthest')
            if through_x is not None:
                ordered += order_through_along_direction(half, hd[through_x], back_to_front=True)
            ordered += scan_non_through(half, front_key, hd, through_x, mode='outward_from_through_nearest')
        else:  # 第二半區
            ordered += scan_non_through(half, front_key, hd, through_x, mode='towards_through_from_farthest')
            if through_x is not None:
                ordered += order_through_along_direction(half, hd[through_x], back_to_front=False)
            ordered += scan_non_through(half, back_key,  hd, through_x, mode='outward_from_through_nearest')

        return ordered

    first_half  = make_half_order(start_half, phase="first")
    second_half = make_half_order(other_half,  phase="second")
    return first_half + second_half

# ==== 測試/繪圖用：組裝完整路徑（永不走進貨架格；只在 AP 間移動） ====
def build_full_path_debug(robot_start: Coord,
                          shelf_locations: List[Coord],
                          wm: np.ndarray,
                          station_layout: Dict) -> List[Coord]:
    """
    組裝完整測試路徑：
    - 走到 batch 的入口 AP
    - 每個貨位改成走到「該貨位的 AP」（不進入貨架格）
    - 非貫穿走道最後退回入口 AP
    - 最後接到最近撿貨站的隊列入口（真實模擬請由引擎呼叫 set_path_to_dropoff）
    """
    if not shelf_locations:
        return []
    idx = build_index(shelf_locations, wm)
    aisles = idx["aisles"]
    ap_of  = idx["ap_of"]

    def half_of(p: Coord) -> str: return "upper" if in_upper(p[0]) else "lower"

    through = {
        "upper": pick_through_aisle("upper", aisles["upper"]) if aisles["upper"] else None,
        "lower": pick_through_aisle("lower", aisles["lower"]) if aisles["lower"] else None,
    }

    ordered = reorder_task_items(robot_start, shelf_locations, wm)

    full_path: List[Coord] = []
    cur = robot_start
    i = 0
    n = len(ordered)
    while i < n:
        cur_item = ordered[i]
        cur_half = half_of(cur_item)
        cur_ax = ap_of[cur_item][1]
        is_through = (through[cur_half] is not None and cur_ax == through[cur_half])

        # 收集同「走道 + 組」的連續貨位
        gname = classify_group(cur_item[0])
        batch = [cur_item]
        j = i + 1
        while j < n:
            nxt = ordered[j]
            if ap_of[nxt][1] == cur_ax and classify_group(nxt[0]) == gname:
                batch.append(nxt); j += 1
            else:
                break

        entry_ap = ap_of[batch[0]]

        # 1) 走到入口 AP（可通行）
        seg = base_plan_route(cur, entry_ap, wm, forbidden_cells=None)
        if seg is None: return []
        full_path += seg; cur = entry_ap

        # 2) 逐貨位：只走到該貨位 AP（不走進貨架）
        for shelf in batch:
            ap = ap_of[shelf]
            if cur != ap:
                seg = base_plan_route(cur, ap, wm, forbidden_cells=None)
                if seg is None: return []
                full_path += seg; cur = ap
            # 在此位置完成取貨（無需進一步移動）

        # 3) 非貫穿：硬性退回入口 AP
        if not is_through and cur != entry_ap:
            seg = base_plan_route(cur, entry_ap, wm, forbidden_cells=None)
            if seg is None: return []
            full_path += seg; cur = entry_ap

        i = j

    # 4) 測試模式：連到最近撿貨站的隊列入口（真模擬由引擎自動 set_path_to_dropoff）
    stations = station_layout.get("picking_stations", [])
    if stations:
        best = min(stations, key=lambda s: euclidean_distance(cur, tuple(s["pos"])))
        queue_spots = [tuple(q) for q in best.get("queue", [])]
        if queue_spots:
            entry = queue_spots[-1]  # 入口取最遠
            forbid = set(queue_spots[:-1])
            start_pos_for_route = find_adjacent_aisle(cur, wm) or cur
            # 若當前在非走道、先走回走道
            if start_pos_for_route != cur:
                seg0 = base_plan_route(cur, start_pos_for_route, wm, forbidden_cells=None)
                if seg0: full_path += seg0; cur = start_pos_for_route
            seg = base_plan_route(cur, entry, wm, forbidden_cells=forbid)
            if seg: full_path += seg

    return full_path

# re-export（主程式仍可 import 本模組的 plan_route）
def plan_route(start_pos: Coord, target_pos: Coord, warehouse_matrix: np.ndarray,
               forbidden_cells: Optional[set] = None):
    return base_plan_route(start_pos, target_pos, warehouse_matrix, forbidden_cells=forbidden_cells)


