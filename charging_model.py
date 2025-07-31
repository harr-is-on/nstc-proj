"""
充電策略模組 (範本)(先到先得的排隊系統)。
你可以透過修改並替換 `ChargingStation` 類別來實現您的充電策略。
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from robot_and_initial_state import Robot

class ChargingStation:
    """【核心策略類別】
    代表一個充電站的運作邏輯。這個基礎版本實作了一個簡單的先到先得 (FIFO) 排隊系統。
    """
    def __init__(self, capacity: int, charge_rate: int, **kwargs: any):
        """【初始化方法】
        初始化 ChargingStation。
        任何新的充電策略類別都必須有這個 `__init__` 方法來接收基本設定。
        
        :param capacity: 充電站同時可容納的機器人數量。
        :param charge_rate: 每個時間步為機器人增加的電量。
        :param kwargs: 接受其他設定參數，以保持向前相容性。
        """
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.queue: List['Robot'] = []      # 等待充電的機器人隊列
        self.charging: List['Robot'] = []   # 目前正在充電的機器人列表

    def request_charging(self, robot: 'Robot'):
        """【核心互動方法】
        當一個機器人到達充電站時，主模擬引擎會呼叫此方法。
        
        **你的策略在此決定：**
        - 是否接受這個充電請求。
        - 如果接受，是立即開始充電，還是將其放入等待隊列。
        - 你就需要呼叫機器人對應的方法來更新其狀態 (例如 `robot.start_charging()`)。

        在這個範例中：
        如果充電站有空位，機器人立即開始充電；否則進入等待隊列。
        """
        # 避免同一個機器人被重複加入
        if robot not in self.charging and robot not in self.queue:
            if len(self.charging) < self.capacity:
                self.charging.append(robot)
                robot.start_charging()
            else:
                self.queue.append(robot)
                robot.wait_for_charge()
                print(f"⏳ 充電站已滿。機器人 {robot.id} 進入等待隊列。")

    def update(self) -> List['Robot']:
        """【核心更新方法】
        主模擬引擎會在「每個時間步」呼叫此方法，以驅動充電站的內部邏輯。

        **你的策略在這邊實作：**
        - 更新正在充電的機器人的電量。
        - 判斷哪些機器人已經充滿電。
        - 管理等待隊列，例如讓排在最前面的機器人遞補空出的充電位。
        
        :return: 【必要回傳】一個列表，其中包含「在此時間步」剛好完成充電的機器人。
                 主引擎需要這個列表來處理後續邏輯。
        """
        finished_charging = []
        # 為充電中的機器人充電，並找出已完成的
        for robot in self.charging:
            if robot.charge(self.charge_rate): # charge() 方法會回傳是否已充滿
                finished_charging.append(robot)

        # 將已充滿電的機器人從充電列表中移除。
        # 狀態更新的責任轉移到主模擬引擎，以便更好地處理後續的移動邏輯。
        for robot in finished_charging:
            self.charging.remove(robot)

        # 從隊列中遞補空位
        while len(self.charging) < self.capacity and self.queue:
            next_robot = self.queue.pop(0)
            self.charging.append(next_robot)
            next_robot.start_charging()
        
        return finished_charging

    def is_robot_charging(self, robot: 'Robot') -> bool:
        """【輔助方法】檢查指定的機器人是否正在充電。"""
        return robot in self.charging

    def in_queue(self, robot: 'Robot') -> bool:
        """【輔助方法】檢查指定的機器人是否在等待隊列中。"""
        return robot in self.queue
