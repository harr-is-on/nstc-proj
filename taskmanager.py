import random
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Set
import numpy as np

from imerge.other.routing import plan_route, euclidean_distance
from robot_and_initial_state import Robot, RobotStatus, Coord

if TYPE_CHECKING:
    pass # 現在為空，因為導入已移至外面
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

    def generate_random_task(self, num_locations: int = 1) -> Optional[Task]:
        """
        創建一個隨機的撿貨任務，並將其加入佇列。
        任務會從可用的貨架位置中隨機選擇一個或多個地點。

        :param num_locations: 該任務包含的貨架地點數量。
        :return: 生成的任務字典，如果沒有可用貨架則返回 None。
        """
        if not self.shelf_coords:
            print("警告: 倉庫中沒有貨架可供生成任務。")
            return None

        # 支援多個撿貨點 (S-shape 策略)
        available_shelves = self.shelf_coords.copy()
        shelf_locations = []
        
        for _ in range(min(num_locations, len(available_shelves))):
            shelf_location = random.choice(available_shelves)
            shelf_locations.append(shelf_location)
            available_shelves.remove(shelf_location)  # 避免重複選擇

        task = {
            "task_id": self.next_task_id,
            "shelf_locations": shelf_locations,  # 修改為複數形式支援多點
            "use_s_shape": len(shelf_locations) > 1,  # 標記是否使用 S-shape
        }
        self.task_queue.append(task)
        self.next_task_id += 1
        print(f"✨ 已生成新任務 {task['task_id']}，目標貨架: {shelf_locations}")
        return task

    def assign_pending_tasks(self, robots: Dict[str, 'Robot'], warehouse_matrix: np.ndarray, forbidden_cells_for_tasks: Optional[Set[Coord]] = None):
        """
        將佇列中的任務分配給任何可用的閒置機器人。
        採用先到先得的分配策略，而非尋找最近的機器人。

        :param robots: 當前所有機器人物件的字典。
        :param warehouse_matrix: 倉庫佈局，用於路徑規劃。
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
                
                # 支援新的多點任務格式
                if "shelf_locations" in task:
                    target_pos = task["shelf_locations"][0]  # 第一個撿貨點作為初始目標
                else:
                    # 相容舊格式
                    target_pos = task.get("shelf_location")
                
                path = plan_route(robot_to_assign.position, target_pos, warehouse_matrix, forbidden_cells=forbidden_cells_for_tasks)
                
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