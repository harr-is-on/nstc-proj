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
        try:
            # 驗證配置
            task_range = SIMULATION_CONFIG.get("task_locations_range", (1, 1))
            if not isinstance(task_range, tuple) or len(task_range) != 2:
                print("錯誤: task_locations_range 配置格式不正確，使用預設值 (1, 1)")
                task_range = (1, 1)
                
            min_loc, max_loc = task_range
            if min_loc <= 0 or max_loc <= 0 or min_loc > max_loc:
                print(f"錯誤: task_locations_range 配置值不合理 {task_range}，使用預設值 (1, 1)")
                min_loc, max_loc = 1, 1
                
            num_locations = random.randint(min_loc, max_loc)

            # 驗證貨架數據
            if not hasattr(self, 'shelf_coords') or self.shelf_coords is None:
                print("錯誤: 貨架座標數據未初始化")
                return None
                
            if not self.shelf_coords:
                print("警告: 倉庫中沒有貨架可供生成任務。")
                return None
                
            if len(self.shelf_coords) < num_locations:
                print(f"警告: 可用貨架數量 ({len(self.shelf_coords)}) 少於需求數量 ({num_locations})，調整為 {len(self.shelf_coords)} 個地點")
                num_locations = len(self.shelf_coords)

            # 隨機選擇不重複的多個貨架位置
            try:
                shelf_locations = random.sample(self.shelf_coords, num_locations)
            except ValueError as e:
                print(f"錯誤: 無法從貨架中採樣 {num_locations} 個位置: {e}")
                return None

            # 確保任務 ID 的唯一性
            if hasattr(self, 'next_task_id') and isinstance(self.next_task_id, int):
                task_id = self.next_task_id
            else:
                print("警告: next_task_id 未正確初始化，重設為 1")
                self.next_task_id = 1
                task_id = 1

            task = {
                "task_id": task_id,
                "shelf_locations": shelf_locations, # 現在是一個地點列表
                "original_locations": list(shelf_locations), # 複製一份原始列表以供日誌記錄
                "retry_count": 0,  # 新增：重試次數計數器
                "max_retries": 3,  # 新增：最大重試次數
                "created_at": len(self.task_queue)  # 新增：創建時的任務佇列長度（用於調試）
            }
            self.task_queue.append(task)
            self.next_task_id += 1
            
            # 格式化輸出，使其更易讀
            locations_str = ', '.join(map(str, shelf_locations))
            print(f"✨ 已生成新任務 {task['task_id']} (共 {num_locations} 個點)，目標貨架: {locations_str}")
            return task
            
        except Exception as e:
            print(f"錯誤: 生成任務時發生異常: {e}")
            return None

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

        # 參數驗證
        if not robots:
            print("警告: 機器人字典為空")
            return
            
        if warehouse_matrix is None:
            print("錯誤: 倉庫矩陣為 None")
            return
            
        if plan_route_func is None:
            print("錯誤: 路徑規劃函數為 None")
            return

        # 找出所有閒置且電量充足的機器人
        available_robots = []
        for robot in robots.values():
            if robot is None:
                print("警告: 發現 None 機器人對象")
                continue
            if hasattr(robot, 'status') and hasattr(robot, 'battery_level') and hasattr(robot, 'charging_threshold'):
                if robot.status == RobotStatus.IDLE and robot.battery_level > robot.charging_threshold:
                    available_robots.append(robot)
            else:
                print(f"警告: 機器人 {getattr(robot, 'id', 'unknown')} 缺少必要屬性")
        
        if not available_robots:
            return # 沒有可用的機器人

        # 使用更好的隨機化方法
        if len(available_robots) > 1:
            # 多次打亂確保真正隨機
            for _ in range(3):
                random.shuffle(available_robots)
        
        print(f"📋 找到 {len(available_robots)} 個可用機器人，準備分配 {len(self.task_queue)} 個任務")

        unassigned_tasks = []
        failed_tasks = []  # 新增：超過重試次數的任務
        
        # 遍歷任務佇列中的每一項任務
        for task in self.task_queue:
            # 檢查任務是否已經超過最大重試次數
            if task.get("retry_count", 0) >= task.get("max_retries", 3):
                print(f"❌ 任務 {task['task_id']} 已超過最大重試次數 ({task.get('max_retries', 3)})，將被放棄")
                failed_tasks.append(task)
                continue
            
            # 如果還有可用的機器人
            if available_robots:
                # 取出一個可用的機器人來分配任務
                robot_to_assign = available_robots.pop(0)
                # 規劃到任務列表中的第一個貨架
                target_pos = task["shelf_locations"][0]
                
                try:
                    path = plan_route_func(robot_to_assign.position, target_pos, warehouse_matrix, forbidden_cells=forbidden_cells_for_tasks)
                    
                    if path and len(path) > 0:
                        # 如果路徑規劃成功，則分配任務
                        try:
                            robot_to_assign.assign_task(task, path)
                            # 重設重試次數（成功分配後）
                            task["retry_count"] = 0
                            print(f"✅ 成功分配任務 {task['task_id']} 給機器人 {robot_to_assign.id}")
                        except Exception as e:
                            print(f"錯誤: 分配任務時發生異常: {e}")
                            # 任務分配失敗，增加重試次數
                            task["retry_count"] = task.get("retry_count", 0) + 1
                            available_robots.append(robot_to_assign)
                            unassigned_tasks.append(task)
                    else:
                        # 如果路徑規劃失敗，增加重試次數
                        task["retry_count"] = task.get("retry_count", 0) + 1
                        print(f"⚠️ 無法為機器人 {robot_to_assign.id} 規劃到任務 {task['task_id']} 的路徑。(重試次數: {task['retry_count']}/{task.get('max_retries', 3)})")
                        available_robots.append(robot_to_assign) # 將機器人放回可用列表的末尾
                        unassigned_tasks.append(task)
                except Exception as e:
                    print(f"錯誤: 路徑規劃時發生異常: {e}")
                    task["retry_count"] = task.get("retry_count", 0) + 1
                    available_robots.append(robot_to_assign)
                    unassigned_tasks.append(task)
            else:
                # 如果沒有可用的機器人了，保留剩餘的任務
                unassigned_tasks.append(task)
        
        # 更新任務佇列，只保留未被分配的任務（不包括失敗的任務）
        self.task_queue = unassigned_tasks
        
        # 記錄統計資訊
        successful_assignments = len([r for r in robots.values() if hasattr(r, 'task') and r.task is not None])
        if failed_tasks:
            print(f"📈 本輪放棄了 {len(failed_tasks)} 個無法完成的任務")
        if successful_assignments > 0:
            print(f"📋 本輪成功分配 {successful_assignments} 個任務")

    def get_queue_size(self) -> int:
        """Return the number of unassigned tasks in the queue."""
        return len(self.task_queue)
        
    def get_task_statistics(self) -> Dict:
        """
        獲取任務統計資訊。
        
        :return: 包含任務統計的字典
        """
        total_tasks = len(self.task_queue)
        retry_counts = {}
        
        for task in self.task_queue:
            retry_count = task.get("retry_count", 0)
            retry_counts[retry_count] = retry_counts.get(retry_count, 0) + 1
        
        return {
            "total_pending_tasks": total_tasks,
            "retry_distribution": retry_counts,
            "next_task_id": self.next_task_id
        }
    
    def clean_stale_tasks(self, max_age: int = 1000) -> int:
        """
        清理過舊的任務（根據創建時的佇列長度判斷）。
        
        :param max_age: 任務的最大年齡（佇列長度差）
        :return: 清理的任務數量
        """
        if not self.task_queue:
            return 0
            
        current_queue_length = len(self.task_queue)
        stale_tasks = []
        clean_tasks = []
        
        for task in self.task_queue:
            created_at = task.get("created_at", 0)
            age = current_queue_length - created_at
            if age > max_age:
                stale_tasks.append(task)
                print(f"🗑️ 清理過舊任務 {task['task_id']} (年齡: {age})")
            else:
                clean_tasks.append(task)
        
        self.task_queue = clean_tasks
        return len(stale_tasks)