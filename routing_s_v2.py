"""
路徑規劃策略模組 - 改良式 S-Shape 策略

此檔案實作了您自訂的改良式 S-Shape 策略，並已整合至現有系統框架。
"""

import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Optional, Set

# --- 從通用模組匯入，確保一致性 ---
from routing import plan_route as plan_route_a_star, euclidean_distance

# --- 型別別名 ---
Coord = Tuple[int, int]

# --- 全域路徑快取 ---
_s_shape_cache = {}

def clear_s_shape_cache():
    """清除 S-Shape 策略的全域路徑快取。"""
    global _s_shape_cache
    _s_shape_cache = {}

class ImprovedSShapePathPlanner:
    """
    改良式 S-Shape 策略的核心邏輯。
    這個類別封裝了區域判斷、路徑生成和貨物訪問的演算法。
    """
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
        """根據 Y 座標判斷所在區域 (上半部/下半部)。"""
        return 'upper' if pos[0] <= 6 else 'lower'

    def get_scan_direction(self, zone: str, is_first_zone: bool) -> str:
        """根據起始區域和目前區域決定掃描方向 (向左/向右)。"""
        if self.start_zone == 'upper':
            return 'left' if zone == 'upper' else 'right'
        else:
            return 'left' if zone == 'lower' else 'right'

    def items_in_zone(self, zone: str = None) -> List[Coord]:
        """取得指定區域內所有未訪問的貨物。"""
        if zone is None:
            zone = self.zone
        if zone == 'upper':
            return [i for i in self.current_items if i not in self.visited and i[0] <= 6]
        else:
            return [i for i in self.current_items if i not in self.visited and i[0] >= 7]

    def scan_next(self, cur_x: int, items: List[Coord], dir: str) -> Optional[Coord]:
        """根據掃描方向，從候選貨物中找出下一個目標。"""
        if not items:
            return None
        if dir == 'left':
            cands = [i for i in items if i[1] <= cur_x]
            return max(cands, key=lambda i: i[1]) if cands else None
        else:
            cands = [i for i in items if i[1] >= cur_x]
            return min(cands, key=lambda i: i[1]) if cands else None

    def gen_access(self, item: Coord) -> List[Coord]:
        """生成貨物的可訪問點 (相鄰的走道)。"""
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
        """為貨物生成位於主幹道的中繼點。"""
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
        """計算進入目標區域的最佳入口點。"""
        relay_rows = [7, 12] if target_zone == 'lower' else [1, 6]
        candidates = [(ry, x) for ry in relay_rows for x in range(self.cols) if self.wm[ry, x] == 0]
        if not candidates:
            return None
        # 根據起始區和目標區決定入口選擇邏輯，確保S形路徑
        if self.start_zone == 'upper' and target_zone == 'lower':
            return min(candidates, key=lambda p: (p[0], p[1]))
        elif self.start_zone == 'lower' and target_zone == 'upper':
            return min(candidates, key=lambda p: (-p[0], p[1]))
        return self.nearest(from_pos, candidates)

    def nearest(self, pos: Coord, pts: List[Coord]) -> Optional[Coord]:
        """從列表中找出距離給定點最近的點。"""
        return min(pts, key=lambda p: euclidean_distance(pos, p)) if pts else None

    def paired(self, relay: Coord) -> Optional[Coord]:
        """找到主幹道上與給定中繼點配對的另一個中繼點。"""
        ry, rx = relay
        py = {1: 6, 6: 1, 7: 12, 12: 7}.get(ry)
        return (py, rx) if py and self.wm[py, rx] == 0 else None

    def check_all_items_visited(self) -> bool:
        """檢查是否所有貨物都已被訪問。"""
        return len(self.visited) >= len(self.current_items)

    def execute_items_only(self, start: Coord, items: List[Coord],
                           dyn: Optional[List[Coord]] = None,
                           forbid: Optional[Set[Coord]] = None) -> Optional[List[Coord]]:
        """
        執行完整的多點撿貨路徑規劃，返回包含所有步驟的完整路徑。
        """
        print(f"=== 改進的 S-Shape 開始 (起始位置: {start}) ===")
        self.current_items = items
        self.visited.clear()
        self.touch_count = 0
        pos = start
        path: List[Coord] = [start] # 路徑應包含起點

        self.zone = self.determine_zone(pos)
        self.start_zone = self.zone
        self.zone_sequence = [self.zone]

        print(f" 起始區域: {self.start_zone}")
        print(f" 待揀貨物分布: 上半區 {len(self.items_in_zone('upper'))} 個, 下半區 {len(self.items_in_zone('lower'))} 個")

        while not self.check_all_items_visited():
            zone_items = self.items_in_zone()
            if not zone_items:
                other_zone = 'upper' if self.zone == 'lower' else 'lower'
                if not self.items_in_zone(other_zone):
                    print(" 所有貨物已在規劃中，結束路徑生成。")
                    break
                
                print(f"🔄 切換區域: {self.zone} -> {other_zone}")
                entry_point = self.get_zone_entry_point(other_zone, pos)
                if not entry_point:
                    print(f" 找不到進入 {other_zone} 區域的入口點，路徑規劃終止。")
                    return None # 嚴重錯誤，無法繼續
                
                seg = plan_route_a_star(pos, entry_point, self.wm, dyn, forbid)
                if seg:
                    path.extend(seg)
                    pos = entry_point
                    self.zone = other_zone
                    self.zone_sequence.append(self.zone)
                    self.touch_count = 0
                    print(f" 已進入 {other_zone} 區域，入口點: {entry_point}")
                else:
                    print(f" 無法導航至 {other_zone} 區域入口點，路徑規劃終止。")
                    return None # 嚴重錯誤
                continue

            is_first_zone = (self.zone == self.start_zone)
            direction = self.get_scan_direction(self.zone, is_first_zone)
            nxt = self.scan_next(pos[1], zone_items, direction)
            if not nxt:
                direction = 'right' if direction == 'left' else 'left'
                nxt = self.scan_next(pos[1], zone_items, direction)
                if not nxt:
                    # 如果雙向都掃描不到，代表此區域已完成
                    self.visited.update(zone_items)
                    continue

            same_aisle_items = sorted([i for i in zone_items if i[1] == nxt[1]], key=lambda p: p[0])

            for item in same_aisle_items:
                if item in self.visited:
                    continue

                print(f" 處理目標: {item} (掃描方向: {direction})")
                relays = self.gen_relays_for(item)[self.zone]
                aps = self.gen_access(item)

                if not relays or not aps:
                    print(f" 無法為 {item} 生成導航點，標記為已訪問並跳過。")
                    self.visited.add(item)
                    continue

                rlay = self.nearest(pos, relays)
                ap = self.nearest(pos, aps)

                seg_to_relay = plan_route_a_star(pos, rlay, self.wm, dyn, forbid)
                if not seg_to_relay:
                    print(f" 無法找到到達中繼點 {rlay} 的路徑，跳過 {item}。")
                    self.visited.add(item)
                    continue
                
                path.extend(seg_to_relay)
                pos = rlay
                self.touch_count += 1

                # 核心S-Shape邏輯：在主幹道間移動並訪問貨物
                paired_relay = self.paired(rlay)
                if self.touch_count % 2 == 1 and paired_relay:
                    seg_across = plan_route_a_star(pos, paired_relay, self.wm, dyn, forbid)
                    if seg_across:
                        # 檢查訪問點是否在橫穿路徑上
                        try:
                            idx = seg_across.index(ap)
                            path.extend(seg_across[:idx+1])
                            pos = ap
                            self.visited.add(item)
                            print(f" 已揀取: {item} (在橫穿路徑上)")
                            # 繼續路徑的剩餘部分
                            if idx + 1 < len(seg_across):
                                path.extend(seg_across[idx+1:])
                                pos = paired_relay
                        except ValueError:
                            # 不在路徑上，先走完再訪問
                            path.extend(seg_across)
                            pos = paired_relay
                
                # 如果貨物還沒被訪問，規劃從當前位置到訪問點的路徑
                if item not in self.visited:
                    seg_to_ap = plan_route_a_star(pos, ap, self.wm, dyn, forbid)
                    if seg_to_ap:
                        path.extend(seg_to_ap)
                        pos = ap
                        self.visited.add(item)
                        print(f" 已揀取: {item}")
                    else:
                        print(f" 無法從 {pos} 導航至 {item} 的訪問點 {ap}，跳過。")
                        self.visited.add(item) # 標記為已訪問避免死循環

        print(f"=== S-Shape 結束 ===")
        print(f" 訪問 {len(self.visited)}/{len(self.current_items)} 個貨物")
        print(f" 最終位置: {pos}")
        print(f" 路徑總長度: {len(path)} 步")
        return path


def plan_route(start_pos, target_pos, warehouse_matrix, dynamic_obstacles: Optional[List[Coord]] = None, forbidden_cells: Optional[Set[Coord]] = None, cost_map: Optional[Dict[str, any]] = None):
    """
    【核心策略函式】- 改良式 S-Shape 策略實作
    為機器人規劃一條從起點到終點的路徑。
    
    此函式為系統的統一入口點，它會根據 `cost_map` 的內容決定是否啟用
    改良式 S-Shape 演算法。
    """
    print(f"  改良式 S-Shape 策略處理中: {start_pos} -> {target_pos}")
    
    if cost_map and 's_shape_picks' in cost_map and len(cost_map['s_shape_picks']) > 1:
        pick_locations = cost_map['s_shape_picks']
        print(f" 啟用改良式 S-Shape 策略，共 {len(pick_locations)} 個撿貨點。")
        
        planner = ImprovedSShapePathPlanner(warehouse_matrix)
        
        # 使用元組作為快取鍵，因為列表不可哈希
        cache_key = (tuple(sorted(map(tuple, pick_locations))), start_pos)
        
        if cache_key in _s_shape_cache:
            full_path = _s_shape_cache[cache_key]
            print(" 從快取中讀取完整路徑。")
        else:
            full_path = planner.execute_items_only(start_pos, pick_locations, dynamic_obstacles, forbidden_cells)
            if full_path:
                _s_shape_cache[cache_key] = full_path
                print(" 新的完整路徑已計算並快取。")
            else:
                print(" 改良式 S-Shape 策略無法生成完整路徑。")

        if not full_path:
            print(" S-Shape 策略無路徑，回退至標準 A* 演算法。")
            return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)

        try:
            # 確保 start_pos 在路徑的最前端
            if full_path[0] != start_pos:
                 print(f"警告: 完整路徑的起點 {full_path[0]} 與請求的起點 {start_pos} 不符。")
                 # 這是預期行為，因為路徑是從機器人當前位置開始的
            
            start_idx = 0 # 完整路徑總是從 start_pos 開始

            # 找到目標點在路徑中的索引
            if target_pos in full_path:
                end_idx = full_path.index(target_pos)
                # 返回從下一步到目標點的路徑片段
                result_path = full_path[start_idx + 1 : end_idx + 1]
                print(f"📍 返回 S-Shape 路徑片段，共 {len(result_path)} 步。")
                return result_path if result_path else None
            else:
                print(f" 目標點 {target_pos} 不在計算出的 S-Shape 路徑中，回退至標準 A*。")
                return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
        
        except ValueError:
            print(f" 處理路徑時發生錯誤，回退至標準 A*。")
            return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells)
    
    # 如果不符合 S-Shape 策略的觸發條件，使用標準 A* 演算法
    print(" 未觸發 S-Shape (單點任務或無指定撿貨點)，使用標準 A* 演算法。")
    return plan_route_a_star(start_pos, target_pos, warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)