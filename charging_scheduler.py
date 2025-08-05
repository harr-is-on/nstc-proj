from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import math
from robot_and_initial_state import Robot

Coord = Tuple[int, int]

class ChargingScheduler:
    """
    智能充電排程模組
    - 選擇最小可接納時間的充電站
    - 支援預約佔位與釋放
    """
    def __init__(
        self,
        charge_stations_info: List[Dict],
        charge_rate: int,
        plan_route_func: Optional[Callable] = None
    ):
        """
        :param charge_stations_info: 來自 SimulationEngine.station_layout['charge_stations'] 的站點資訊列表。
                                     每個 dict 包含 'id', 'pos', 'queue', 'exit'.
        :param charge_rate: 充電速率 (每個時間步增量)。
        :param plan_route_func: 路徑規劃函數，如果為 None 則需要在使用前設定。
        """
        # 一次性快取所有充電站資訊，key = station_id
        self.stations_info: Dict[str, Dict] = {s['id']: s for s in charge_stations_info}
        self.charge_rate: int = charge_rate
        # 路徑規劃函數
        self.plan_route_func = plan_route_func
        # reservations: station_id -> List of tuples (robot_id, eta, finish_time)
        self.reservations: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)

    def set_plan_route_func(self, plan_route_func: Callable):
        """
        設定路徑規劃函數。這允許在初始化後動態設定路徑規劃策略。
        
        :param plan_route_func: 路徑規劃函數
        """
        self.plan_route_func = plan_route_func

    def _travel_time(
        self,
        robot: Robot,
        entry: Coord,
        warehouse_matrix
    ) -> int:
        """
        估算從 robot.position 到 entry 的行程時間 (時間步數)。
        使用 plan_route() 回傳的路徑長度，並考慮 robot.move_speed。
        """
        if self.plan_route_func is None:
            print(f"警告：ChargingScheduler 的路徑規劃函數未設定")
            return math.inf
        
        try:
            path = self.plan_route_func(robot.position, entry, warehouse_matrix)
            if path is None or len(path) == 0:
                return math.inf
            # time steps = ceil(steps / move_speed)
            return math.ceil(len(path) / robot.move_speed)
        except Exception as e:
            print(f"路徑規劃計算錯誤: {e}")
            return math.inf

    def _remaining_time(
        self,
        station_id: str,
        current_time: int
    ) -> int:
        """
        計算指定站點在 current_time 後的總占用時間。
        包括正在充電與已預約但尚未開始或完成的時間。
        """
        if station_id not in self.reservations:
            return 0
        
        remaining = 0
        for robot_id, eta, finish in self.reservations.get(station_id, []):
            if finish > current_time:
                # 若預計結束時間晚於 current_time，剩餘占用為 finish - max(current_time, eta)
                remaining += finish - max(current_time, eta)
        return remaining

    def choose_station(
        self,
        robot: Robot,
        current_time: int,
        warehouse_matrix
    ) -> Optional[Tuple[Dict, Coord, int]]:
        """
        根據「最小可接納時間 = remaining_time - travel_time」選擇站點。
        回傳: (station_info, entry_pos, travel_time)
        如果所有站點均無法到達，回傳 None。
        """
        if not self.stations_info:
            print("警告：沒有可用的充電站")
            return None
        
        if robot is None:
            print("錯誤：機器人對象為 None")
            return None
        
        best_info = None
        best_entry = None
        best_travel = 0
        best_score = math.inf

        for station_id, info in self.stations_info.items():
            # 檢查站點資訊完整性
            if 'queue' not in info or not info['queue']:
                print(f"警告：充電站 {station_id} 沒有排隊區資訊")
                continue
                
            entry_pos = info['queue'][-1]
            travel = self._travel_time(robot, entry_pos, warehouse_matrix)
            if travel == math.inf:
                continue  # 無路徑
            remaining = self._remaining_time(station_id, current_time)
            wait_time = remaining - travel
            # 選擇最小 wait_time，如相同則選 travel 較小者
            if wait_time < best_score or (wait_time == best_score and travel < best_travel):
                best_score = wait_time
                best_info = info
                best_entry = entry_pos
                best_travel = travel

        if best_info:
            return best_info, best_entry, best_travel
        return None

    def reserve_station(
        self,
        robot: Robot,
        station_id: str,
        current_time: int,
        warehouse_matrix
    ) -> bool:
        """
        當機器人決定前往並佔位時呼叫。
        計算 ETA (抵達時間) 與預計完成充電時間，加入 reservations。
        
        :return: True 如果預約成功，False 如果失敗
        """
        if robot is None or station_id not in self.stations_info:
            print(f"錯誤：無效的機器人或充電站 ID {station_id}")
            return False
        
        # 檢查是否已經有預約
        existing_reservations = self.reservations.get(station_id, [])
        for existing_robot_id, _, _ in existing_reservations:
            if existing_robot_id == robot.id:
                print(f"警告：機器人 {robot.id} 已經預約了充電站 {station_id}")
                return False
        
        info = self.stations_info[station_id]
        if 'queue' not in info or not info['queue']:
            print(f"錯誤：充電站 {station_id} 沒有排隊區資訊")
            return False
            
        entry_pos = info['queue'][-1]
        travel = self._travel_time(robot, entry_pos, warehouse_matrix)
        
        if travel == math.inf:
            print(f"錯誤：機器人 {robot.id} 無法到達充電站 {station_id}")
            return False
        
        eta = current_time + travel
        # 計算充滿電所需的時間步
        remaining_charge = max(0, robot.full_charge_level - robot.battery_level)
        if self.charge_rate <= 0:
            print("錯誤：充電速率必須大於 0")
            return False
            
        charge_steps = math.ceil(remaining_charge / self.charge_rate)
        finish_time = eta + charge_steps
        # 記錄預約
        self.reservations[station_id].append((robot.id, eta, finish_time))
        return True

    def release_station(
        self,
        robot_id: str,
        station_id: str
    ) -> bool:
        """
        機器人完成/取消充電後呼叫，從 reservations 移除其預約。
        
        :return: True 如果成功移除預約，False 如果沒有找到預約
        """
        if not robot_id or not station_id:
            print("錯誤：機器人 ID 或充電站 ID 不能為空")
            return False
            
        if station_id not in self.reservations:
            print(f"警告：充電站 {station_id} 沒有任何預約記錄")
            return False
        
        original_count = len(self.reservations[station_id])
        self.reservations[station_id] = [
            rec for rec in self.reservations[station_id] if rec[0] != robot_id
        ]
        
        removed_count = original_count - len(self.reservations[station_id])
        if removed_count == 0:
            print(f"警告：機器人 {robot_id} 在充電站 {station_id} 沒有預約記錄")
            return False
        elif removed_count > 1:
            print(f"警告：移除了機器人 {robot_id} 的 {removed_count} 個重複預約")
        
        return True

    def get_station_status(self, station_id: str) -> Dict:
        """
        獲取指定充電站的狀態資訊。
        
        :param station_id: 充電站 ID
        :return: 包含預約資訊的字典
        """
        if station_id not in self.stations_info:
            return {"error": f"充電站 {station_id} 不存在"}
        
        reservations = self.reservations.get(station_id, [])
        return {
            "station_id": station_id,
            "total_reservations": len(reservations),
            "reservations": reservations,
            "station_info": self.stations_info[station_id]
        }

    def clean_expired_reservations(self, current_time: int) -> int:
        """
        清理已過期的預約記錄。
        
        :param current_time: 當前時間
        :return: 清理的預約數量
        """
        cleaned_count = 0
        for station_id in list(self.reservations.keys()):
            original_reservations = self.reservations[station_id]
            # 保留尚未完成的預約（finish_time > current_time）
            self.reservations[station_id] = [
                (robot_id, eta, finish) for robot_id, eta, finish in original_reservations
                if finish > current_time
            ]
            
            removed = len(original_reservations) - len(self.reservations[station_id])
            cleaned_count += removed
            
            if removed > 0:
                print(f"清理了充電站 {station_id} 的 {removed} 個過期預約")
        
        return cleaned_count
