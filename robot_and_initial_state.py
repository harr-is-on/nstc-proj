import random
import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, List, Union

# --- 參數設定區 ---

# --- 機器人相關設定 ---
ROBOT_CONFIG = {
    "num_robots": 5,
    "move_speed": 1,              # 機器人每個時間步移動的格數
    "pickup_duration": 2,         # 機器人撿貨所需的時間步
    "dropoff_duration": 1,        # 機器人交貨所需的時間步
    "initial_battery": (20, 100), # 機器人初始電量的隨機範圍 (最小值, 最大值)
    "energy_per_step": 1,         # 機器人每移動一步消耗的電量
    "replan_wait_threshold": 3,   # 機器人因擁塞等待多久後會嘗試重新規劃路徑
}

# --- 模擬與任務相關設定 ---
SIMULATION_CONFIG = {
    "num_initial_tasks": 5,      # 模擬開始時生成的初始任務數量
    "target_tasks_completed": 300, # 【新】模擬運行的目標任務完成數
    "max_simulation_steps_safety_limit": 50000, # 【新】為防止無限迴圈，設定一個極大的安全步數上限
    "task_generation_interval": 1, # 每 n 個時間步生成一個新任務 (調快以確保有足夠任務)
    "task_locations_range": (1, 3), # 每個任務包含的貨架地點數量的隨機範圍 (最小值, 最大值)
}

Coord = Tuple[int, int]

class RobotStatus(Enum):
    """定義機器人所有可能的狀態，以提高程式碼的穩固性。"""
    IDLE = "idle"
    MOVING_TO_SHELF = "moving_to_shelf"
    PICKING = "picking"
    MOVING_TO_DROPOFF = "moving_to_dropoff"
    DROPPING_OFF = "dropping_off"
    MOVING_TO_CHARGE = "moving_to_charge"
    WAITING_FOR_CHARGE = "waiting_for_charge"
    WAITING_IN_QUEUE = "waiting_in_queue" # 新增：在物理排隊區等待的狀態
    CHARGING = "charging"


class Robot:
    """
    代表倉庫中的單一機器人，包含其所有屬性與行為。
    """
    def __init__(
        self,
        robot_id: str,
        initial_position: Coord,
        battery_level: int = 100,
        move_speed: int = 1,
        pickup_duration: int = 2,
        dropoff_duration: int = 2,
        charging_threshold: int = 20,
        full_charge_level: int = 100,
        energy_per_step: int = 1,
        replan_wait_threshold: int = 3
    ):
        # 基本屬性 / 狀態
        self.id = robot_id
        self.position = initial_position
        self.task: Optional[Dict] = None
        self.status: RobotStatus = RobotStatus.IDLE
        
        # 搬運作業相關屬性
        self.carrying_item: bool = False
        self.pickup_duration: int = pickup_duration # 撿貨所需的時間步
        self.pickup_timer: int = 0 # 撿貨計時器
        self.dropoff_duration: int = dropoff_duration # 交貨所需的時間步
        self.dropoff_timer: int = 0 # 交貨計時器

        # 移動 / 追蹤狀態
        self.path: List[Coord] = []
        self.move_speed: int = move_speed
        self.target_station_pos: Optional[Coord] = None # 記住最終要去的站點
        self.wait_time: int = 0 # 因壅塞等待的時間

        # 電池 / 充電狀態資料
        self.battery_level: int = battery_level
        self.charging_status: bool = False
        self.charging_threshold: int = charging_threshold # 低於此電量需充電
        self.full_charge_level: int = full_charge_level # 充電到此電量即停止
        self.energy_per_step: int = energy_per_step # 每步消耗的電量
        self.replan_wait_threshold: int = replan_wait_threshold # 等待重新規劃的閾值

    @property
    def next_position(self) -> Optional[Coord]:
        """計算屬性：從路徑中取得下一個要移動的座標，但不移動。"""
        if not self.path:
            return None
        return self.path[0]

    def __repr__(self) -> str:
        """提供一個方便開發者閱讀的字串表示法。"""
        return (f"Robot(id={self.id}, pos={self.position}, "
                f"status='{self.status.value}', battery={self.battery_level})")

    def assign_task(self, task: Dict, path: List[Coord]):
        """為機器人指派新任務與路徑。"""
        if self.status != RobotStatus.IDLE:
            raise RuntimeError(f"無法指派任務給狀態為 '{self.status.value}' 的機器人 {self.id}")
        self.task = task
        self.path = path
        self.status = RobotStatus.MOVING_TO_SHELF
        print(f"任務 {task.get('task_id')} 已指派給 {self.id}")

    def move_to_next_step(self) -> int:
        """
        將機器人沿著路徑移動，最多移動 self.move_speed 步。
        並根據移動的步數消耗電量。
        
        :return: 實際移動的步數。
        """
        if not self.path:
            return 0
        
        steps_to_move = min(self.move_speed, len(self.path))
        for _ in range(steps_to_move):
            self.position = self.path.pop(0)
        self.battery_level -= (steps_to_move * self.energy_per_step)
        return steps_to_move

    def clear_task(self):
        """將機器人的任務相關狀態重設為閒置。"""
        self.task = None
        self.path = []
        self.pickup_timer = 0
        self.dropoff_timer = 0
        self.carrying_item = False
        self.target_station_pos = None # 重設目標站點
        self.status = RobotStatus.IDLE

    def wait_for_charge(self):
        """將機器人狀態設為等待充電。"""
        # 只有剛到達充電站的機器人才能排隊
        if self.status == RobotStatus.MOVING_TO_CHARGE:
            self.status = RobotStatus.WAITING_FOR_CHARGE
        else:
            print(f"警告：機器人 {self.id} 在非預期狀態 '{self.status.value}' 下嘗試等待充電。")

    def go_charge(self, path: List[Coord], station_pos: Coord):
        """設定前往充電站的路徑並更新狀態。"""
        if self.status != RobotStatus.IDLE:
            print(f"警告: 只有閒置機器人 {self.id} 才能前往充電。")
            return
        self.path = path
        self.target_station_pos = station_pos # 記住目標充電站
        self.status = RobotStatus.MOVING_TO_CHARGE
        print(f"🔌 機器人 {self.id} 電量低，前往充電站。")

    def start_charging(self):
        """開始充電，更新機器人狀態。"""
        self.status = RobotStatus.CHARGING
        self.target_station_pos = None # 到達充電站，清除目標
        self.charging_status = True
        print(f"🔋 機器人 {self.id} 開始充電。")

    def stop_charging(self):
        """停止充電，將機器人狀態重設為閒置。"""
        self.battery_level = self.full_charge_level
        self.charging_status = False
        self.status = RobotStatus.IDLE
        print(f"✅ 機器人 {self.id} 充電完畢，恢復閒置狀態。")

    def charge(self, amount: int) -> bool:
        """
        為機器人充電。
        :param amount: 要增加的電量。
        :return: 如果電量已滿，返回 True，否則返回 False。
        """
        if self.battery_level >= self.full_charge_level:
            return True # 已經充滿
        
        self.battery_level += amount
        if self.battery_level >= self.full_charge_level:
            self.battery_level = self.full_charge_level
            return True  # 充電完成
        return False  # 仍在充電

    def start_picking(self):
        """開始撿貨，更新機器人狀態並設定計時器。"""
        # 只有在前往貨架且已到達目的地時才能開始撿貨
        if self.status == RobotStatus.MOVING_TO_SHELF and not self.path:
            self.status = RobotStatus.PICKING
            self.pickup_timer = self.pickup_duration
            print(f"📦 機器人 {self.id} 在位置 {self.position} 開始撿貨 (耗時: {self.pickup_duration} 步)。")
        else:
            print(f"警告: 機器人 {self.id} 在非預期狀態 '{self.status.value}' 或未到達貨架時嘗試撿貨。")

    def pick_item(self) -> bool:
        """
        執行一個時間步的撿貨動作。
        :return: 如果撿貨完成，返回 True，否則返回 False。
        """
        if self.status != RobotStatus.PICKING:
            return False

        # 如果計時器還在跑，就減一
        if self.pickup_timer > 0:
            self.pickup_timer -= 1
        
        # 如果計時器跑完了，就代表撿貨已完成
        if self.pickup_timer == 0:
            self.carrying_item = True
            return True
        
        # 計時器還沒跑完
        return False

    def set_path_to_dropoff(self, path: List[Coord], station_pos: Coord):
        """在撿貨完成後，設定前往交貨站的路徑。"""
        if self.status != RobotStatus.PICKING:
             print(f"警告: 機器人 {self.id} 在非撿貨狀態 '{self.status.value}' 下嘗試設定交貨路徑。")
             return
        self.path = path
        self.target_station_pos = station_pos # 記住目標交貨站
        self.status = RobotStatus.MOVING_TO_DROPOFF
        print(f"🚚 機器人 {self.id} 撿貨完畢，前往交貨站。")

    def start_dropping_off(self):
        """開始交貨，更新機器人狀態並設定計時器。"""
        if self.status == RobotStatus.MOVING_TO_DROPOFF and not self.path:
            self.status = RobotStatus.DROPPING_OFF
            self.dropoff_timer = self.dropoff_duration
            print(f"📥 機器人 {self.id} 在位置 {self.position} 開始交貨 (耗時: {self.dropoff_duration} 步)。")
        else:
            print(f"警告: 機器人 {self.id} 在非預期狀態 '{self.status.value}' 或未到達交貨站時嘗試交貨。")

    def drop_off_item(self) -> bool:
        """
        執行一個時間步的交貨動作。
        :return: 如果交貨完成，返回 True，否則返回 False。
        """
        if self.status == RobotStatus.DROPPING_OFF and self.dropoff_timer > 0:
            self.dropoff_timer -= 1
            if self.dropoff_timer == 0:
                self.carrying_item = False # 交貨完成，不再攜帶物品
                return True  # 交貨完成
        return False  # 仍在交貨


def initialize_robots(
    warehouse_matrix: np.ndarray,
    robot_config: Dict[str, Union[int, Tuple[int, int]]],
    charging_config: Dict[str, Union[int, float]]
) -> Dict[str, Robot]:
    """
    在有效的走道上隨機初始化指定數量的 Robot 物件。

    :param warehouse_matrix: 倉庫佈局矩陣。
    :param robot_config: 包含機器人設定的字典，例如：
                         {'num_robots': 5, 'initial_battery': (80, 100), ...}
    :param charging_config: 包含充電相關設定的字典。
    :return: 一個 Robot 物件的字典。
    """
    num_rows, num_cols = warehouse_matrix.shape

    # 找出所有有效的走道位置 (數值為 0)
    aisle_positions: List[Coord] = [
        (r, c) for r in range(num_rows) for c in range(num_cols) 
        if warehouse_matrix[r, c] == 0 # 確保只在純走道上生成
    ]

    num_robots = robot_config.get("num_robots", 5)
    # 穩固的錯誤處理
    if len(aisle_positions) < num_robots:
        raise ValueError(f"無法放置 {num_robots} 個機器人，有效的走道位置只有 {len(aisle_positions)} 個。")

    # 隨機選取不重複的起始位置
    initial_positions = random.sample(aisle_positions, num_robots)

    def get_battery():
        battery_config = robot_config.get("initial_battery", 100)
        if isinstance(battery_config, int):
            return battery_config
        elif isinstance(battery_config, tuple) and len(battery_config) == 2:
            return random.randint(*battery_config)
        raise TypeError("battery_config 必須是整數或一個包含兩個整數的元組 (min, max)。")

    # 從 config 中提取機器人初始化所需的參數
    robot_params = {
        "move_speed": robot_config.get("move_speed", 1),
        "pickup_duration": robot_config.get("pickup_duration", 2),
        "dropoff_duration": robot_config.get("dropoff_duration", 2),
        "charging_threshold": charging_config.get("charging_threshold", 20),
        "full_charge_level": charging_config.get("full_charge_level", 100),
        "energy_per_step": robot_config.get("energy_per_step", 1),
        "replan_wait_threshold": ROBOT_CONFIG.get("replan_wait_threshold", 3),
    }

    # 建立一個 Robot 物件的字典
    robots: Dict[str, Robot] = {
        f"R{i+1}": Robot(robot_id=f"R{i+1}", initial_position=pos, 
                         battery_level=get_battery(), **robot_params)
        for i, pos in enumerate(initial_positions)
    }
    return robots

if __name__ == '__main__':
    print("This module contains the Robot class and robot initialization logic.")