from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
from a_star import a_star, DEFAULT_COSTS

Coord = Tuple[int, int]

def find_nearest(layout: np.ndarray, start_r: int, col: int, target_codes: set[int]) -> Optional[int]:
    for offset in range(1, layout.shape[0]):
        for r in [start_r - offset, start_r + offset]:
            if 0 <= r < layout.shape[0] and layout[r, col] in target_codes:
                return r
    return None

def find_nearest_station(curr: Coord, stations: List[Coord]) -> Coord:
    return min(stations, key=lambda p: abs(p[0] - curr[0]) + abs(p[1] - curr[1]))

def plan_route_s_shape_dynamic(
    layout: np.ndarray,
    start: Coord,
    picks: List[Coord],
    que_pick_zones: List[Coord],
    picking_stations: List[Coord],
    costs: Optional[Dict[int, float]] = None
) -> List[Coord]:
    if costs is None:
        costs = DEFAULT_COSTS

    remaining = picks.copy()
    path: List[Coord] = [start]
    curr = start

    while remaining:
        # 1. 找最近貨物
        remaining.sort(key=lambda p: abs(p[0] - curr[0]) + abs(p[1] - curr[1]))
        target = remaining[0]

        # 2. 若不在 main road or turn point，先垂直移動至最近 turn point
        if layout[curr[0], curr[1]] not in (1, 3):
            turn_r = find_nearest(layout, curr[0], curr[1], {1})
            if turn_r is not None:
                segment = a_star(layout, curr, (turn_r, curr[1]), costs)
                path.extend(segment[1:])
                curr = (turn_r, curr[1])

        # 3. 水平移動到目標 subroad 所在 col
        if curr[1] != target[1]:
            segment = a_star(layout, curr, (curr[0], target[1]), costs)
            path.extend(segment[1:])
            curr = (curr[0], target[1])

        # 4. 沿目前方向揀完該 subroad（僅揀符合方向的 order）
        direction = -1 if target[0] < curr[0] else 1
        cr, cc = curr

        aisle_orders = sorted(
            [p for p in remaining if p[1] == cc and (p[0] - cr) * direction >= 0],
            key=lambda p: p[0],
            reverse=(direction == -1)
        )

        for p in aisle_orders:
            segment = a_star(layout, curr, p, costs)
            path.extend(segment[1:])
            curr = p
            remaining.remove(p)

        # 5. 繼續往該方向走到底部 turn point
        r = curr[0]
        while True:
            next_r = r + direction
            if not (0 <= next_r < layout.shape[0]):
                break
            if layout[next_r, cc] == 1:
                segment = a_star(layout, curr, (next_r, cc), costs)
                path.extend(segment[1:])
                curr = (next_r, cc)
                break
            r = next_r

    # 6. 所有貨物撿完後 → 前往 que_pick → picking station
    end_station = find_nearest_station(curr, picking_stations)
    que_pick_pos = min(que_pick_zones, key=lambda q: abs(q[0] - end_station[0]) + abs(q[1] - end_station[1]))

    segment = a_star(layout, curr, que_pick_pos, costs)
    path.extend(segment[1:])
    curr = que_pick_pos

    segment = a_star(layout, curr, end_station, costs)
    path.extend(segment[1:])

    return path
