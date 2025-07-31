from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
from a_star import a_star, DEFAULT_COSTS

Coord = Tuple[int, int]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_nearest(layout: np.ndarray, start_r: int, col: int, target_codes: set[int]) -> Optional[int]:
    H = layout.shape[0]
    for d in range(1, H):
        for nr in (start_r - d, start_r + d):
            if 0 <= nr < H and layout[nr, col] in target_codes:
                return nr
    return None

def nearest_station(curr: Coord, stations: List[Coord]) -> Coord:
    return min(stations, key=lambda s: manhattan(curr, s))

def plan_route_largest_gap_custom(
    layout: np.ndarray,
    start: Coord,
    picks: List[Coord],
    picking_stations: List[Coord],
    que_pick_zones: List[Coord],
    costs: Optional[Dict[int, float]] = None
) -> List[Coord]:
    if costs is None:
        costs = DEFAULT_COSTS

    remaining = picks.copy()
    path: List[Coord] = [start]
    curr = start

    while remaining:
        # 1. 找最近的未取 order
        remaining.sort(key=lambda p: manhattan(curr, p))
        target = remaining[0]
        tr, tc = target

        # 2. 若不在 turn/main road，先垂直移動到最近 turn point
        if layout[curr[0], curr[1]] not in (1, 3):
            turn_r = find_nearest(layout, curr[0], curr[1], {1})
            if turn_r is not None:
                seg = a_star(layout, curr, (turn_r, curr[1]), costs)
                path += seg[1:]
                curr = (turn_r, curr[1])

        # 3. 水平移動到 subroad
        if curr[1] != tc:
            seg = a_star(layout, curr, (curr[0], tc), costs)
            path += seg[1:]
            curr = (curr[0], tc)

        # 4. 垂直移動至 order
        if curr[0] != tr:
            seg = a_star(layout, curr, (tr, tc), costs)
            path += seg[1:]
            curr = (tr, tc)

        # ✅ 完成一筆撿貨
        remaining.remove(curr)

        # 5. 檢查是否還有下一個 order 在同 col 的 subroad 且可直接延伸
        next_same_col = [p for p in remaining if p[1] == curr[1]]

        if next_same_col:
            next_p = min(next_same_col, key=lambda p: manhattan(curr, p))
            dir_to_next = np.sign(next_p[0] - curr[0])

            if dir_to_next != 0:
                blocked = False
                for r in range(curr[0] + dir_to_next, next_p[0], dir_to_next):
                    if layout[r, curr[1]] == 0:
                        blocked = True
                        break
                if not blocked:
                    continue  # 不退回，直接進行下一次循環

        # 否則才退回最近 turn point
        turn_r = find_nearest(layout, curr[0], curr[1], {1})
        if turn_r is not None:
            seg = a_star(layout, curr, (turn_r, curr[1]), costs)
            path += seg[1:]
            curr = (turn_r, curr[1])

    # 6. 全部撿完後 → 導航至 que_pick，再進入 pick_station
    end_station = nearest_station(curr, picking_stations)
    que_pick_pos = min(que_pick_zones, key=lambda q: manhattan(q, end_station))

    seg = a_star(layout, curr, que_pick_pos, costs)
    path += seg[1:]
    curr = que_pick_pos

    seg = a_star(layout, curr, end_station, costs)
    path += seg[1:]

    return path
