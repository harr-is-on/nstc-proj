"""
充電策略模組 (範本)(先到先得的排隊系統)。
你可以透過修改並替換 `ChargingStation` 類別來實現您的充電策略。
"""

from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from robot_and_initial_state import Robot

class ChargingStation:
    """【核心策略類別】
    代表一個充電站的運作邏輯。這個基礎版本實作了一個簡單的先到先得 (FIFO) 排隊系統。
    """
    def __init__(self, capacity: int, charge_rate: int, enable_dynamic_charging: bool, dynamic_states: List[Dict], default_full_charge_level: int, full_charge_level: int, **kwargs: any):
        """【初始化方法】
        初始化 ChargingStation。
        任何新的充電策略類別都必須有這個 `__init__` 方法來接收基本設定。
        
        :param capacity: 充電站同時可容納的機器人數量。
        :param charge_rate: 每個時間步為機器人增加的電量。
        :param enable_dynamic_charging: 是否啟用動態充電。
        :param dynamic_states: 動態充電的狀態規則。
        :param default_full_charge_level: 悠閒狀態的充電目標。
        :param full_charge_level: 禁用動態充電時的固定充電目標。
        :param kwargs: 接受其他設定參數，以保持向前相容性。
        """
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.queue: List['Robot'] = []      # 等待充電的機器人隊列
        self.charging: List['Robot'] = []   # 目前正在充電的機器人列表

        # --- 動態充電設定 ---
        self.enable_dynamic_charging = enable_dynamic_charging
        if self.enable_dynamic_charging:
            # 按閾值排序，確保從最嚴格的條件 (最少閒置機器人) 開始檢查
            self.dynamic_states = sorted(dynamic_states, key=lambda x: x['max_idle_robots'])
            self.default_charge_level = default_full_charge_level
            self.default_state_name = kwargs.get('default_state_name', 'Idle')
            print(" 已啟用動態充電策略。")
        else:
            # 若禁用，則使用固定的充電水平
            self.static_full_charge_level = full_charge_level
            print(f" 已停用動態充電，將使用固定充電目標: {self.static_full_charge_level}%。")


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

    def _get_current_state_info(self, idle_robot_count: int) -> Dict[str, Any]:
        """根據閒置機器人數量，從規則中找出當前的狀態資訊 (名稱和充電目標)。"""
        if not self.enable_dynamic_charging:
            return {"name": "Fixed", "level": self.static_full_charge_level}

        for state in self.dynamic_states:
            if idle_robot_count <= state['max_idle_robots']:
                return {"name": state.get('name', 'Unknown'), "level": state['full_charge_level']}
        
        # 如果所有條件都不滿足，使用預設的悠閒狀態值
        return {"name": self.default_state_name, "level": self.default_charge_level}

    def get_current_state_name(self, idle_robot_count: int) -> str:
        """【輔助方法】根據閒置機器人數量，返回當前運作節奏的名稱。"""
        return self._get_current_state_info(idle_robot_count)["name"]

    def _get_current_target_level(self, idle_robot_count: int) -> float:
        """【內部方法】根據閒置機器人數量，從規則中找出當前的充電目標電量。"""
        return self._get_current_state_info(idle_robot_count)["level"]
    def update(self, idle_robot_count: int) -> List['Robot']:
        """【核心更新方法】
        主模擬引擎會在「每個時間步」呼叫此方法，以驅動充電站的內部邏輯。

        **你的策略在這邊實作：**
        - 更新正在充電的機器人的電量。
        - 根據當前系統忙碌程度 (idle_robot_count) 判斷哪些機器人已達充電目標。
        - 管理等待隊列，例如讓排在最前面的機器人遞補空出的充電位。
        
        :param idle_robot_count: 【必要輸入】當前系統中閒置的機器人總數。
        :return: 【必要回傳】一個列表，其中包含「在此時間步」剛好完成充電的機器人。
                 主引擎需要這個列表來處理後續邏輯。
        """
        current_target_level = self._get_current_target_level(idle_robot_count)

        finished_charging = []
        # 為充電中的機器人充電，並根據動態目標找出已完成的
        for robot in list(self.charging): # 迭代副本以允許在迴圈中修改
            robot.charge(self.charge_rate)
            if robot.battery_level >= current_target_level:
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
