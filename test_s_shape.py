import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Optional, Set

# 型別別名
Coord = Tuple[int, int]

class ImprovedSShapePathPlanner:
    def __init__(self, warehouse_matrix: np.ndarray):
        self.wm = warehouse_matrix
        self.rows, self.cols = warehouse_matrix.shape
        self.current_items: List[Coord] = []
        self.visited: Set[Coord] = set()
        self.touch_count = 0
        self.zone = None
        self.start_zone = None
        self.zone_sequence = []

    def determine_zone(self, pos: Coord) -> str:
        return 'upper' if pos[0] <= 6 else 'lower'

    def get_scan_direction(self, zone: str, is_first_zone: bool) -> str:
        if self.start_zone == 'upper':
            return 'left' if zone == 'upper' else 'right'
        else:
            return 'left' if zone == 'lower' else 'right'

    def items_in_zone(self, zone: str = None) -> List[Coord]:
        if zone is None:
            zone = self.zone
        if zone == 'upper':
            return [i for i in self.current_items if i not in self.visited and i[0] <= 6]
        else:
            return [i for i in self.current_items if i not in self.visited and i[0] >= 7]

    def scan_next(self, cur_x: int, items: List[Coord], dir: str) -> Optional[Coord]:
        if not items:
            return None
        if dir == 'left':
            cands = [i for i in items if i[1] <= cur_x]
            return max(cands, key=lambda i: i[1]) if cands else None
        else:
            cands = [i for i in items if i[1] >= cur_x]
            return min(cands, key=lambda i: i[1]) if cands else None

    def gen_access(self, item: Coord) -> List[Coord]:
        r, c = item
        aps = []
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < self.cols and self.wm[r, nc] == 0:
                aps.append((r, nc))
        if not aps:
            for dr in (-1, 1):
                nr = r + dr
                if 0 <= nr < self.rows and self.wm[nr, c] == 0:
                    aps.append((nr, c))
        return aps

    def gen_relays_for(self, item: Coord) -> Dict[str, List[Coord]]:
        aps = self.gen_access(item)
        groups = {'upper': [], 'lower': []}
        for ay, ax in aps:
            if 1 <= ay <= 6:
                for ry in (1, 6):
                    if self.wm[ry, ax] == 0:
                        groups['upper'].append((ry, ax))
            elif 7 <= ay <= 12:
                for ry in (7, 12):
                    if self.wm[ry, ax] == 0:
                        groups['lower'].append((ry, ax))
        groups['upper'] = list(set(groups['upper']))
        groups['lower'] = list(set(groups['lower']))
        return groups

    def get_zone_entry_point(self, target_zone: str, from_pos: Coord) -> Optional[Coord]:
        if target_zone == 'lower':
            relay_rows = [7, 12]
        else:
            relay_rows = [1, 6]

        candidates = []
        for ry in relay_rows:
            for x in range(self.cols):
                if self.wm[ry, x] == 0:
                    candidates.append((ry, x))

        if not candidates:
            return None

        if self.start_zone == 'upper' and target_zone == 'lower':
            return min(candidates, key=lambda p: (p[0], p[1]))
        elif self.start_zone == 'lower' and target_zone == 'upper':
            return min(candidates, key=lambda p: (-p[0], p[1]))

        return min(candidates, key=lambda p: self.euclidean_distance(from_pos, p))

    def euclidean_distance(self, p1: Coord, p2: Coord) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def nearest(self, pos: Coord, pts: List[Coord]) -> Optional[Coord]:
        return min(pts, key=lambda p: self.euclidean_distance(pos, p)) if pts else None

    def paired(self, relay: Coord) -> Optional[Coord]:
        ry, rx = relay
        if ry in (1, 6):
            py = 6 if ry == 1 else 1
        elif ry in (7, 12):
            py = 12 if ry == 7 else 7
        else:
            return None
        return (py, rx) if self.wm[py, rx] == 0 else None

    def check_all_items_visited(self) -> bool:
        return len(self.visited) >= len(self.current_items)

    def astar(self, start: Coord, goal: Coord,
              dyn: Optional[List[Coord]] = None,
              forbid: Optional[Set[Coord]] = None) -> Optional[List[Coord]]:
        if forbid is None:
            forbid = set()
        rows, cols = self.rows, self.cols

        def nbrs(p):
            r, c = p
            res = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dyn and (nr, nc) in dyn and (nr, nc) != goal:
                        continue
                    if (nr, nc) in forbid and (nr, nc) != goal:
                        continue
                    if self.wm[nr, nc] in (0, 4, 5, 6, 7) or (nr, nc) == goal:
                        res.append((nr, nc))
            return res

        def h(p): return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        open_ = [(h(start), 0, start, [])]
        closed = set()
        while open_:
            f, g, cur, path = heapq.heappop(open_)
            if cur in closed:
                continue
            if cur == goal:
                return (path + [cur])[1:]
            closed.add(cur)
            for n in nbrs(cur):
                if n not in closed:
                    heapq.heappush(open_, (g + 1 + h(n), g + 1, n, path + [cur]))
        return None

    def execute_items_only(self, start: Coord, items: List[Coord],
                           dyn: Optional[List[Coord]] = None,
                           forbid: Optional[Set[Coord]] = None) -> Optional[List[Coord]]:
        print(f"=== 改進的 S-Shape 開始 (起始位置: {start}) ===")

        self.current_items = items
        self.visited.clear()
        self.touch_count = 0
        pos = start
        path: List[Coord] = []

        self.zone = self.determine_zone(pos)
        self.start_zone = self.zone
        self.zone_sequence = [self.zone]

        print(f"📍 起始區域: {self.start_zone}")
        print(f"📋 待揀貨物分布: 上半區 {len(self.items_in_zone('upper'))} 個, "
              f"下半區 {len(self.items_in_zone('lower'))} 個")

        while not self.check_all_items_visited():
            zone_items = self.items_in_zone()
            if not zone_items:
                other_zone = 'upper' if self.zone == 'lower' else 'lower'
                other_items = self.items_in_zone(other_zone)
                if not other_items:
                    print("✅ 所有貨物已揀取完成")
                    break
                print(f"🔄 切換區域: {self.zone} -> {other_zone}")
                entry_point = self.get_zone_entry_point(other_zone, pos)
                if entry_point:
                    seg = self.astar(pos, entry_point, dyn, forbid)
                    if seg is not None:
                        path += seg
                        pos = entry_point
                        self.zone = other_zone
                        self.zone_sequence.append(self.zone)
                        self.touch_count = 0
                        print(f"📍 已進入 {other_zone} 區域，入口點: {entry_point}")
                    else:
                        print(f"❌ 無法導航至 {other_zone} 區域入口點")
                        break
                else:
                    print(f"❌ 找不到進入 {other_zone} 區域的入口點")
                    break
                continue

            is_first_zone = (self.zone == self.start_zone)
            direction = self.get_scan_direction(self.zone, is_first_zone)
            nxt = self.scan_next(pos[1], zone_items, direction)
            if not nxt:
                odir = 'right' if direction == 'left' else 'left'
                nxt = self.scan_next(pos[1], zone_items, odir)
                if nxt:
                    direction = odir
                else:
                    continue

            same_row_items = [i for i in zone_items if i[0] == nxt[0]]
            same_row_items.sort(key=lambda p: p[1], reverse=(direction == 'left'))

            for item in same_row_items:
                if item in self.visited:
                    continue

                print(f"🎯 處理目標: {item} (同巷道, 掃描方向: {direction})")

                relays = self.gen_relays_for(item)[self.zone]
                aps = self.gen_access(item)

                if not relays or not aps:
                    print(f"⚠️ 無法為 {item} 生成導航點，跳過")
                    self.visited.add(item)
                    continue

                rlay = self.nearest(pos, relays)
                ap = min(aps, key=lambda p: self.euclidean_distance(pos, p))

                seg = self.astar(pos, rlay, dyn, forbid)
                if seg is not None:
                    path += seg
                    pos = rlay
                    self.touch_count += 1

                    if self.touch_count % 2 == 1:
                        pr = self.paired(rlay)
                        if pr:
                            seg2 = self.astar(pos, pr, dyn, forbid)
                            if seg2:
                                if ap in seg2:
                                    idx = seg2.index(ap)
                                    path += seg2[:idx + 1]
                                    self.visited.add(item)
                                    print(f"✅ 已揀取: {item} (路徑中)")
                                    if self.check_all_items_visited():
                                        print("🎉 所有貨物揀取完成！")
                                        pos = ap
                                        break
                                    path += seg2[idx + 1:]
                                    pos = pr
                                else:
                                    path += seg2
                                    pos = pr
                                    seg3 = self.astar(pos, ap, dyn, forbid)
                                    if seg3:
                                        path += seg3
                                        pos = ap
                                        self.visited.add(item)
                                        print(f"✅ 已揀取: {item}")
                                        if self.check_all_items_visited():
                                            print("🎉 所有貨物揀取完成！")
                                            break
                                    else:
                                        print(f"⚠️ 無法導航至 {item} 的 access point，跳過")
                                        self.visited.add(item)
                            else:
                                print(f"⚠️ 無法導航至配對 relay point，跳過 {item}")
                                self.visited.add(item)
                        else:
                            print(f"⚠️ 無配對 relay point，跳過 {item}")
                            self.visited.add(item)
                    else:
                        seg2 = self.astar(pos, ap, dyn, forbid)
                        if seg2:
                            path += seg2
                            pos = ap
                            self.visited.add(item)
                            print(f"✅ 已揀取: {item}")
                            if self.check_all_items_visited():
                                print("🎉 所有貨物揀取完成！")
                                break
                        else:
                            print(f"⚠️ 無法直接導航至 {item} 的 access point，跳過")
                            self.visited.add(item)
                else:
                    print(f"❌ 無法找到到達 {item} 的路徑，跳過")
                    self.visited.add(item)
                    continue

        print(f"=== S-Shape 結束 ===")
        print(f"📊 訪問 {len(self.visited)}/{len(self.current_items)} 個貨物")
        print(f"📍 最終位置: {pos}")
        print(f"📏 路徑總長度: {len(path)} 步")

        return path


    def execute_with_robot(self, robot, items: List[Coord],
                          dyn: Optional[List[Coord]] = None,
                          forbid: Optional[Set[Coord]] = None) -> Dict:
        """
        與機器人對象整合的 S-Shape 路徑規劃
        在揀貨完成後會自動調用 robot.set_path_to_dropoff()
        """
        print(f"=== S-Shape (機器人整合版) 開始 ===")
        print(f"🤖 機器人 ID: {robot.id if hasattr(robot, 'id') else 'N/A'}")
        
        # 執行改進的 S-Shape 路徑規劃
        path_to_items = self.execute_items_only(robot.position, items, dyn, forbid)
        
        if not path_to_items:
            print("❌ 無法規劃 S-Shape 路徑")
            return None
        
        # 找到最後的實際位置
        last_actual_pos = path_to_items[-1] if path_to_items else robot.position
        print(f"🎯 揀貨完成位置: {last_actual_pos}")
        
        # 自動設定交貨路徑
        try:
            from warehouse_layout import get_station_locations
            from routing import find_adjacent_aisle, plan_route
            
            stations = get_station_locations()['picking_stations']
            best_station = min(stations, 
                             key=lambda s: self.euclidean_distance(last_actual_pos, s['pos']))
            best_queue_spot = best_station['queue'][-1]
            
            print(f"📍 選定交貨站: {best_station['pos']}, 排隊入口: {best_queue_spot}")
            
            # 從貨架移到鄰近的走道
            start_pos_for_dropoff = find_adjacent_aisle(last_actual_pos, self.wm)
            if not start_pos_for_dropoff:
                print(f"⚠️ 無法在 {last_actual_pos} 附近找到走道")
                start_pos_for_dropoff = last_actual_pos
            
            # 規劃到交貨站的路徑
            dropoff_path = plan_route(
                start_pos_for_dropoff,
                best_queue_spot,
                self.wm,
                dynamic_obstacles=dyn,
                forbidden_cells=forbid
            )
            
            if dropoff_path and hasattr(robot, 'set_path_to_dropoff'):
                print(f"🚚 設定交貨路徑")
                robot.set_path_to_dropoff(dropoff_path, best_station['pos'])
                print(f"✅ 交貨路徑設定完成")
            
        except Exception as e:
            print(f"⚠️ 設定交貨路徑時發生錯誤: {e}")
        
        # 返回詳細資訊
        return {
            'picking_path': path_to_items,
            'picking_start': robot.position,
            'picking_end': last_actual_pos,
            'picking_steps': len(path_to_items) if path_to_items else 0,
            'visited_items': len(self.visited),
            'total_items': len(items),
            'zone_sequence': self.zone_sequence,
            'start_zone': self.start_zone
        }
