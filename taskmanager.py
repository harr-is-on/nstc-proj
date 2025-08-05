import random
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Set
import numpy as np

# The routing function will be passed in, decoupling this module from a specific implementation.
from robot_and_initial_state import Robot, RobotStatus, Coord, SIMULATION_CONFIG

if TYPE_CHECKING:
    pass
# Type aliases
Coord = Tuple[int, int]
Task = Dict[str, any]

class TaskManager:
    """
    管理待處理的任務佇列，並根據需求生成新的隨機任務。
    它還負責將任務分配給最合適的可用機器人。
    """
    def __init__(self, warehouse_matrix: np.ndarray):
        self.task_queue: List[Task] = []
        self.next_task_id: int = 1
        # 從倉庫佈局中提取所有貨架的座標，用於生成隨機任務
        self.shelf_coords: List[Coord] = [
            tuple(coord) for coord in np.argwhere(warehouse_matrix == 1)
        ]

    def generate_random_task(self) -> Optional[Task]:
        """
        創建一個隨機的撿貨任務，並將其加入佇列。
        任務會根據 SIMULATION_CONFIG 中的設定，從可用的貨架位置中隨機選擇數個地點。

        :return: 生成的任務字典，如果沒有可用貨架則返回 None。
        """
        min_loc, max_loc = SIMULATION_CONFIG.get("task_locations_range", (1, 1))
        num_locations = random.randint(min_loc, max_loc)

        if not self.shelf_coords or len(self.shelf_coords) < num_locations:
            print("警告: 倉庫中沒有貨架可供生成任務。")
            return None

        # 隨機選擇不重複的多個貨架位置
        shelf_locations = random.sample(self.shelf_coords, num_locations)

        task = {
            "task_id": self.next_task_id,
            "shelf_locations": shelf_locations, # 現在是一個地點列表
            "original_locations": list(shelf_locations) # 複製一份原始列表以供日誌記錄
        }
        self.task_queue.append(task)
        self.next_task_id += 1
        
        # 格式化輸出，使其更易讀
        locations_str = ', '.join(map(str, shelf_locations))
        print(f"✨ 已生成新任務 {task['task_id']} (共 {num_locations} 個點)，目標貨架: {locations_str}")
        return task

    def assign_pending_tasks(self, robots: Dict[str, 'Robot'], warehouse_matrix: np.ndarray, plan_route_func, forbidden_cells_for_tasks: Optional[Set[Coord]] = None):
        """
        將佇列中的任務分配給任何可用的閒置機器人。
        採用先到先得的分配策略，而非尋找最近的機器人。

        :param robots: 當前所有機器人物件的字典。
        :param warehouse_matrix: 倉庫佈局，用於路徑規劃。
        :param plan_route_func: 用於規劃路徑的函數。
        :param forbidden_cells_for_tasks: 在為任務規劃路徑時應避開的格子集合 (例如，排隊區)。
        """
        if not self.task_queue:
            return

        # 找出所有閒置且電量充足的機器人
        available_robots = [
            r for r in robots.values() 
            if r.status == RobotStatus.IDLE and r.battery_level > r.charging_threshold
        ]
        
        if not available_robots:
            return # 沒有可用的機器人

        # 隨機打亂機器人順序，避免每次都分配給同一個機器人 (例如 R1)
        random.shuffle(available_robots)

        unassigned_tasks = []
        # 遍歷任務佇列中的每一項任務
        for task in self.task_queue:
            # 如果還有可用的機器人
            if available_robots:
                # 取出一個可用的機器人來分配任務
                robot_to_assign = available_robots.pop(0)
                # 規劃到任務列表中的第一個貨架
                target_pos = task["shelf_locations"][0]
                
                path = plan_route_func(robot_to_assign.position, target_pos, warehouse_matrix, forbidden_cells=forbidden_cells_for_tasks)
                
                if path:
                    # 如果路徑規劃成功，則分配任務
                    robot_to_assign.assign_task(task, path)
                else:
                    # 如果路徑規劃失敗，將機器人和任務都放回待處理列表
                    print(f"⚠️ 無法為機器人 {robot_to_assign.id} 規劃到任務 {task['task_id']} 的路徑。")
                    available_robots.append(robot_to_assign) # 將機器人放回可用列表的末尾
                    unassigned_tasks.append(task)
            else:
                # 如果沒有可用的機器人了，保留剩餘的任務
                unassigned_tasks.append(task)
        
        # 更新任務佇列，只保留未被分配的任務
        self.task_queue = unassigned_tasks

    def get_queue_size(self) -> int:
        """Return the number of unassigned tasks in the queue."""
        return len(self.task_queue)