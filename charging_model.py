# charging_model.py
from typing import List, TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from robot_and_initial_state import Robot


# --- 充電站相關設定 ---
# 將所有充電相關的定義集中於此
CHARGING_STATION_CONFIG: Dict[str, Union[int, float]] = {
    "capacity": 1,                # 充電站可同時容納的機器人數量
    "charge_rate": 5,             # 每個時間步為機器人增加的電量
    "charging_threshold": 20,     # 當電量低於此百分比時，機器人會尋求充電
    "full_charge_level": 100,     # 機器人充電到此電量後會離開
}

class ChargingStation:
    def __init__(self, capacity: int, charge_rate: int, **kwargs):
        """
        初始化 ChargingStation。
        
        :param capacity: 充電站同時可容納的機器人數量。
        :param charge_rate: 每個時間步為機器人增加的電量。
        :param kwargs: 接受其他設定參數，以保持向前相容性。
        """
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.queue: List['Robot'] = []      # 等待充電的 robot queue
        self.charging: List['Robot'] = []   # 目前正在充電的 robot list

    def request_charging(self, robot: 'Robot'):
        """
        機器人請求充電。
        如果充電站有空位，機器人立即開始充電。
        如果沒有，則進入等待隊列。
        """
        # 避免重複請求
        if robot not in self.charging and robot not in self.queue:
            if len(self.charging) < self.capacity:
                self.charging.append(robot)
                robot.start_charging()
            else:
                self.queue.append(robot)
                robot.wait_for_charge()
                print(f"⏳ 充電站已滿。機器人 {robot.id} 進入等待隊列。")

    def update(self) -> List['Robot']:
        """
        每個模擬時間步調用此方法，以更新所有充電中機器人的狀態。
        - 為充電中的機器人充電。
        - 將充滿電的機器人移出。
        - 讓隊列中的機器人遞補。

        :return: 一個包含在此時間步完成充電的機器人列表。
        """
        finished_charging = []
        for robot in self.charging:
            # 命令機器人充電，並檢查是否充滿
            if robot.charge(self.charge_rate):
                finished_charging.append(robot)

        # 將充滿電的機器人移出充電列表。
        # 狀態更新的責任轉移到主模擬引擎，以更好地處理後續移動。
        for robot in finished_charging:
            self.charging.remove(robot)

        # 從隊列中遞補空位
        while len(self.charging) < self.capacity and self.queue:
            next_robot = self.queue.pop(0)
            self.charging.append(next_robot)
            next_robot.start_charging()
        
        return finished_charging

    def is_robot_charging(self, robot: 'Robot') -> bool:
        return robot in self.charging

    def in_queue(self, robot: 'Robot') -> bool:
        return robot in self.queue
