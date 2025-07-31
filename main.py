"""
倉儲機器人模擬的主入口點。

這個腳本定義了核心的 SimulationEngine 並運行整個模擬。
"""

import json
from typing import Tuple, List, Dict, Optional

# 從專案中匯入模組
from warehouse_layout import create_warehouse_layout, get_station_locations
from robot_and_initial_state import (
    Robot,
    RobotStatus, # 引入機器人狀態列舉
    initialize_robots,
    ROBOT_CONFIG,
    SIMULATION_CONFIG
)
from strategy_config import ROUTING_STRATEGY
from charging_config import CHARGING_STRATEGY, CHARGING_STATION_CONFIG
from taskmanager import TaskManager
from congestion_model import CongestionManager
from visualization import Visualizer
from performance_logger import PerformanceLogger

Coord = Tuple[int, int]
Task = Dict[str, any]


# --- 策略選擇 ---
# 根據 strategy_config.py 中的設定，此區塊會動態匯入對應的模組。
# 這是當您想要更換策略時，唯一需要修改程式碼的地方。

print(f"--- 正在載入策略 ---")
print(f"  路徑規劃: {ROUTING_STRATEGY}")
print(f"  充電: {CHARGING_STRATEGY}")

# 匯入路徑規劃策略
if ROUTING_STRATEGY == 'routing':
    from routing import plan_route, euclidean_distance, find_adjacent_aisle
else:
    # 這裡是為了未來擴充性預留的。
    # 如果您在 strategy_config.py 中設定了新的策略名稱，
    # 您需要在這裡加入對應的 `elif` 條件來匯入您的新模組。
    # 例如:
    # elif ROUTING_STRATEGY == 'my_new_routing':
    #     from my_new_routing import plan_route, ...
    raise NotImplementedError(f"路徑規劃策略 '{ROUTING_STRATEGY}' 尚未被實作。")

# 匯入充電策略
if CHARGING_STRATEGY == 'charging_model':
    from charging_model import ChargingStation
else:
    raise NotImplementedError(f"充電策略 '{CHARGING_STRATEGY}' 尚未被實作。")


class SimulationEngine:
    """
    倉儲模擬的核心引擎。
    它負責協調所有組件、管理主迴圈並記錄效能。
    """
    def __init__(self):
        # --- 初始化各個組件 ---
        self.warehouse_matrix, _ = create_warehouse_layout()
        self.robots = initialize_robots(self.warehouse_matrix, ROBOT_CONFIG, CHARGING_STATION_CONFIG)
        self.task_manager = TaskManager(self.warehouse_matrix)
        self.charging_station = ChargingStation(**CHARGING_STATION_CONFIG)
        self.congestion_manager = CongestionManager()
        self.visualizer = Visualizer(self.warehouse_matrix, list(self.robots.values()))
        self.performance_logger = PerformanceLogger()

        # --- Get station locations for routing ---
        self.station_layout = get_station_locations()
        # 為了方便查找，將站點資訊扁平化
        self.picking_stations_info = self.station_layout['picking_stations']
        self.charge_stations_info = self.station_layout['charge_stations']
        # 為了與舊邏輯相容（例如尋找最近的站點）
        self.picking_stations = [s['pos'] for s in self.picking_stations_info]
        self.charge_stations = [s['pos'] for s in self.charge_stations_info]
        # 為了方便處理出口邏輯
        self.picking_exits = {s['pos']: s['exit'] for s in self.picking_stations_info}
        self.charge_exits = {s['pos']: s['exit'] for s in self.charge_stations_info}

        # 建立一個包含所有排隊區格子的集合，用於快速查找
        self.all_queue_spots = set()
        for station_list in self.station_layout.values():
            for station_info in station_list:
                self.all_queue_spots.update(station_info['queue'])

        # --- 模擬控制參數 ---
        self.max_simulation_steps = SIMULATION_CONFIG.get("max_simulation_steps", 500)
        self.task_generation_interval = SIMULATION_CONFIG.get("task_generation_interval", 25)

        # --- 生成初始任務 ---
        for i in range(SIMULATION_CONFIG.get("num_initial_tasks", 5)):
            self.task_manager.generate_random_task()

    def find_available_queue_entry(self, station_info: Dict) -> Optional[Coord]:
        """
        檢查一個站點的排隊區入口 (最遠的那格) 是否可用。
        """
        entry_point = station_info['queue'][-1] # 將隊列的最後一格定義為唯一入口

        # 取得所有目前被佔用，或已被其他機器人預訂為路徑終點的位置
        occupied_or_targeted = {r.position for r in self.robots.values()}
        for r in self.robots.values():
            if r.path:
                occupied_or_targeted.add(r.path[-1])

        # 如果入口點沒有被佔用或被預訂，則返回該入口點
        return entry_point if entry_point not in occupied_or_targeted else None

    def _update_moving_robot(self, robot: Robot, approved_robot_ids: set):
        """處理處於移動狀態的機器人。"""
        if robot.id in approved_robot_ids:
            battery_before_move = robot.battery_level
            distance_moved = robot.move_to_next_step()
            energy_consumed = battery_before_move - robot.battery_level
            self.performance_logger.log_distance_traveled(robot.id, distance_moved)
            self.performance_logger.log_energy_usage(robot.id, energy_consumed)
            robot.wait_time = 0  # 成功移動後，重設等待時間

            # 檢查是否到達目的地
            if not robot.path:
                if robot.status == RobotStatus.MOVING_TO_SHELF:
                    robot.start_picking()
                elif robot.status in [RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
                    if robot.position != robot.target_station_pos:
                        # 如果還沒到最終站點，代表它到達了排隊區
                        robot.status = RobotStatus.WAITING_IN_QUEUE
                        print(f"🚶 機器人 {robot.id} 到達排隊區 {robot.position}，開始排隊。")
                    else:
                        # 如果已到達最終站點
                        if robot.status == RobotStatus.MOVING_TO_DROPOFF:
                            robot.start_dropping_off()
                        elif robot.status == RobotStatus.MOVING_TO_CHARGE:
                            self.charging_station.request_charging(robot)
        else:
            # 機器人被阻擋，增加等待時間
            robot.wait_time += 1
            print(f"🚧 機器人 {robot.id} 在 {robot.position} 被阻擋 (等待時間: {robot.wait_time})")

            if robot.position in self.all_queue_spots:
                print(f"🔄 機器人 {robot.id} 在排隊時被阻擋，重設狀態為 WAITING_IN_QUEUE。")
                robot.status = RobotStatus.WAITING_IN_QUEUE
                robot.path = []
            elif robot.wait_time > robot.replan_wait_threshold:
                self._try_replanning_path(robot)

    def _try_replanning_path(self, robot: Robot):
        """當機器人等待過久時，嘗試為其重新規劃路徑。"""
        print(f"🤔 機器人 {robot.id} 等待過久，嘗試重新規劃路徑...")
        
        dynamic_obstacles = [r.position for r in self.robots.values() if r.id != robot.id]
        final_destination = robot.path[-1]
        forbidden_cells = set()
        cost_map = {}

        # 建立成本地圖，讓機器人傾向於避開其他機器人周圍的區域
        for r in self.robots.values():
            if r.id != robot.id:
                (br, bc) = r.position
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        cost_map[(br + dr, bc + dc)] = 5

        if robot.status == RobotStatus.MOVING_TO_SHELF:
            forbidden_cells = self.all_queue_spots
        elif robot.status in [RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
            station_list = self.picking_stations_info if robot.status == RobotStatus.MOVING_TO_DROPOFF else self.charge_stations_info
            station_info = next((s for s in station_list if s['pos'] == robot.target_station_pos), None)

            if station_info:
                entry_point = station_info['queue'][-1]
                final_destination = entry_point
                forbidden_cells = self.all_queue_spots - {entry_point}
            else:
                forbidden_cells = self.all_queue_spots
        
        new_path = plan_route(robot.position, final_destination, self.warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
        
        if new_path:
            print(f"🗺️ 機器人 {robot.id} 找到新路徑！")
            robot.path = new_path
            robot.wait_time = 0
        else:
            print(f"❌ 機器人 {robot.id} 找不到替代路徑，將在下一輪再試。")

    def _update_action_robot(self, robot: Robot, time_step: int):
        """處理正在執行動作 (撿貨、交貨) 的機器人。"""
        if robot.status == RobotStatus.PICKING:
            if robot.pick_item():
                completed_shelf = robot.task['shelf_locations'].pop(0)
                print(f"👍 機器人 {robot.id} 在 {completed_shelf} 完成撿貨。")

                if robot.task['shelf_locations']:
                    self._plan_path_to_next_shelf(robot, completed_shelf)
                else:
                    self._plan_path_to_dropoff(robot, completed_shelf)

        elif robot.status == RobotStatus.DROPPING_OFF:
            if robot.drop_off_item():
                print(f"✅ 機器人 {robot.id} 完成任務 {robot.task['task_id']} 的交貨。")
                self.performance_logger.log_task_completion(time_step)
                exit_pos = self.picking_exits.get(robot.position)
                if exit_pos:
                    robot.position = exit_pos
                    print(f"🤖 機器人 {robot.id} 交貨後移動至出口 {exit_pos}。")
                else:
                    print(f"⚠️ 機器人 {robot.id} 在交貨站 {robot.position} 找不到指定的出口！")
                robot.clear_task()


    def _update_queueing_robot(self, robot: Robot, spots_targeted_in_queue_logic: set):
        """處理在隊列中等待的機器人。"""
        target_station_pos = robot.target_station_pos
        station_info = next((s for s in self.picking_stations_info + self.charge_stations_info if s['pos'] == target_station_pos), None)

        if not station_info:
            print(f"錯誤：機器人 {robot.id} 正在排隊，但找不到其目標站點 {target_station_pos}！")
            robot.clear_task()
            return

        try:
            current_queue_index = station_info['queue'].index(robot.position)
        except ValueError:
            return

        next_spot_in_line = station_info['pos'] if current_queue_index == 0 else station_info['queue'][current_queue_index - 1]

        # 檢查前方是否被佔用或已被預定
        occupied = any(r.position == next_spot_in_line for r in self.robots.values() if r.id != robot.id)
        targeted = next_spot_in_line in spots_targeted_in_queue_logic
        if occupied or targeted:
            # 原地等待，不做任何事
            return

        # 如果要進入的是站點本身，檢查站點是否可用
        if next_spot_in_line == station_info['pos']:
            station_is_available = len(self.charging_station.charging) < self.charging_station.capacity if "CS" in station_info['id'] else True
            if not station_is_available:
                return

        # 規劃路徑時，將其他機器人位置視為動態障礙
        dynamic_obstacles = [r.position for r in self.robots.values() if r.id != robot.id]
        path_to_next_spot = plan_route(robot.position, next_spot_in_line, self.warehouse_matrix, dynamic_obstacles=dynamic_obstacles)
        if path_to_next_spot:
            print(f"👍 機器人 {robot.id} 從 {robot.position} 向前移動至 {next_spot_in_line}")
            robot.path = path_to_next_spot
            robot.status = RobotStatus.MOVING_TO_CHARGE if "CS" in station_info['id'] else RobotStatus.MOVING_TO_DROPOFF
            spots_targeted_in_queue_logic.add(next_spot_in_line)
        else:
            print(f"⚠️ 機器人 {robot.id} 在隊列中找不到前往下一格 {next_spot_in_line} 的路徑！")
            # 保持 WAITING_IN_QUEUE 狀態，不切換

    def _update_idle_robot(self, robot: Robot):
        """處理閒置的機器人，主要是檢查是否需要充電。"""
        self.performance_logger.log_robot_idle_time(robot.id, 1)
        if robot.battery_level <= robot.charging_threshold:
            best_station, best_queue_spot, _ = self._find_closest_available_station(robot.position, self.charge_stations_info)
            if best_station and best_queue_spot:
                path = plan_route(robot.position, best_queue_spot, self.warehouse_matrix)
                if path:
                    robot.go_charge(path, best_station['pos'])
                else:
                    print(f"⚠️ 機器人 {robot.id} 在 {robot.position} 找不到前往充電排隊區 {best_queue_spot} 的路徑！")
            else:
                print(f"⏳ 機器人 {robot.id} 需要充電，但所有充電站入口都忙碌中。")

    def _find_closest_available_station(self, pos: Coord, station_list: List[Dict]) -> Tuple[Optional[Dict], Optional[Coord], float]:
        """尋找最近且入口可用的站點。"""
        best_station, best_queue_spot, min_dist = None, None, float('inf')
        for station_info in station_list:
            queue_spot = self.find_available_queue_entry(station_info)
            if queue_spot:
                dist = euclidean_distance(pos, station_info['pos'])
                if dist < min_dist:
                    min_dist, best_station, best_queue_spot = dist, station_info, queue_spot
        return best_station, best_queue_spot, min_dist

    def _plan_path_to_next_shelf(self, robot: Robot, completed_shelf: Coord):
        """規劃路徑到任務中的下一個貨架。"""
        next_shelf = robot.task['shelf_locations'][0]
        print(f"...任務 {robot.task['task_id']} 未完成，機器人 {robot.id} 前往下一站: {next_shelf}")

        start_pos_for_route = find_adjacent_aisle(robot.position, self.warehouse_matrix)
        if not start_pos_for_route:
            print(f"⚠️ 機器人 {robot.id} 在貨架 {robot.position} 旁找不到可用的走道！")
            robot.clear_task()
            return
        
        path = plan_route(start_pos_for_route, next_shelf, self.warehouse_matrix)
        if path:
            robot.position = start_pos_for_route
            robot.path = path
            robot.status = RobotStatus.MOVING_TO_SHELF
        else:
            print(f"⚠️ 機器人 {robot.id} 在 {start_pos_for_route} 找不到前往下一個貨架 {next_shelf} 的路徑！將在原地等待。")
            robot.task['shelf_locations'].insert(0, completed_shelf)

    def _plan_path_to_dropoff(self, robot: Robot, completed_shelf: Coord):
        """在所有撿貨點完成後，規劃路徑到交貨站。"""
        print(f"🎉 機器人 {robot.id} 完成任務 {robot.task['task_id']} 的所有撿貨點。")
        best_station, best_queue_spot, _ = self._find_closest_available_station(robot.position, self.picking_stations_info)
        
        if not (best_station and best_queue_spot):
            print(f"⏳ 機器人 {robot.id} 撿貨完畢，但所有交貨站入口忙碌中，將在原地等待。")
            robot.task['shelf_locations'].insert(0, completed_shelf)
            return
        
        start_pos_for_route = find_adjacent_aisle(robot.position, self.warehouse_matrix)
        path = plan_route(start_pos_for_route, best_queue_spot, self.warehouse_matrix)
        if path:
            print(f"🤖 機器人 {robot.id} 從貨架移至走道 {start_pos_for_route}。")
            robot.position = start_pos_for_route
            robot.set_path_to_dropoff(path, best_station['pos'])
        else:
            print(f"⚠️ 機器人 {robot.id} 在 {start_pos_for_route} 找不到前往排隊區 {best_queue_spot} 的路徑！將在原地等待。")
            robot.task['shelf_locations'].insert(0, completed_shelf)

    def run(self):
        """主模擬迴圈。"""
        time_step = 0
        while time_step < self.max_simulation_steps:
            time_step += 1
            print(f"\n--- Time Step: {time_step} ---")

            # --- 1. Generate and Assign Tasks ---
            # This set will prevent multiple queueing robots from targeting the same empty spot in the same timestep.
            spots_targeted_in_queue_logic = set() # 這個集合用來防止多個排隊中的機器人在同一時間步搶佔同一個空位

            if time_step % self.task_generation_interval == 0:
                self.task_manager.generate_random_task()
            self.task_manager.assign_pending_tasks(self.robots, self.warehouse_matrix, plan_route, forbidden_cells_for_tasks=self.all_queue_spots)

            # --- 2. 協調機器人移動以避免碰撞 ---
            approved_robot_ids = self.congestion_manager.coordinate_moves(list(self.robots.values()))

            # --- 3. 更新機器人狀態與動作 ---
            for robot in self.robots.values():
                # --- Handle moving robots ---
                if robot.status in [RobotStatus.MOVING_TO_SHELF, RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
                    if robot.id in approved_robot_ids:
                        battery_before_move = robot.battery_level
                        distance_moved = robot.move_to_next_step()
                        energy_consumed = battery_before_move - robot.battery_level
                        self.performance_logger.log_distance_traveled(robot.id, distance_moved)
                        self.performance_logger.log_energy_usage(robot.id, energy_consumed)
                        robot.wait_time = 0 # 成功移動後，重設等待時間

                        # Check for arrival at destination
                        if not robot.path:
                            if robot.status == RobotStatus.MOVING_TO_SHELF:
                                robot.start_picking()
                            # 當機器人到達目的地時，檢查它是否到達了最終站點
                            elif robot.status in [RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
                                if robot.position != robot.target_station_pos:
                                    # 如果還沒到最終站點，代表它到達了排隊區
                                    robot.status = RobotStatus.WAITING_IN_QUEUE
                                    print(f"🚶 機器人 {robot.id} 到達排隊區 {robot.position}，開始排隊。")
                                else:
                                    # 如果已到達最終站點
                                    if robot.status == RobotStatus.MOVING_TO_DROPOFF:
                                        robot.start_dropping_off()
                                    elif robot.status == RobotStatus.MOVING_TO_CHARGE:
                                        self.charging_station.request_charging(robot)
                    else:
                        # Robot is blocked, increment wait time
                        robot.wait_time += 1
                        print(f"🚧 機器人 {robot.id} 在 {robot.position} 被阻擋 (等待時間: {robot.wait_time})")

                        # **核心修正**：如果機器人在排隊區被阻擋，將其狀態重設回「排隊中」
                        # 這樣它在下一輪才能重新評估是否可以前進，而不是卡在「移動中」的狀態。
                        if robot.position in self.all_queue_spots:
                            print(f"🔄 機器人 {robot.id} 在排隊時被阻擋，重設狀態為 WAITING_IN_QUEUE。")
                            robot.status = RobotStatus.WAITING_IN_QUEUE
                            robot.path = [] # 清除為這次失敗移動所規劃的路徑
                        
                        # **新增：智慧繞路邏輯**
                        # 如果機器人不是在排隊，且等待時間過長，則嘗試重新規劃路徑
                        elif robot.wait_time > robot.replan_wait_threshold:
                            print(f"🤔 機器人 {robot.id} 等待過久，嘗試重新規劃路徑...")
                            
                            # 將其他所有機器人的位置視為動態障礙物
                            dynamic_obstacles = [r.position for r in self.robots.values() if r.id != robot.id]
                            final_destination = robot.path[-1] # 預設的最終目標是原路徑的終點
                            forbidden_cells = set() # 預設沒有禁止通行的區域

                            # 新增：建立成本地圖，讓機器人傾向於避開其他機器人周圍的區域
                            cost_map = {}
                            for r in self.robots.values():
                                if r.id != robot.id:
                                    # 將其他機器人周圍的格子成本提高
                                    (br, bc) = r.position
                                    for dr in [-1, 0, 1]:
                                        for dc in [-1, 0, 1]:
                                            cost_map[(br + dr, bc + dc)] = 5 # 設定懲罰性成本

                            # **強化規則：根據機器人當前的任務，嚴格限制其可通行的區域**
                            if robot.status == RobotStatus.MOVING_TO_SHELF:
                                # 如果是去撿貨，則所有排隊區都禁止通行
                                forbidden_cells = self.all_queue_spots
                            
                            elif robot.status in [RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
                                # 如果是去站點，則必須從指定的入口進入。
                                # 為了強制這一點，我們將所有非入口的排隊區格子都設為禁止通行。
                                station_list = self.picking_stations_info if robot.status == RobotStatus.MOVING_TO_DROPOFF else self.charge_stations_info
                                station_info = next((s for s in station_list if s['pos'] == robot.target_station_pos), None)

                                if station_info:
                                    entry_point = station_info['queue'][-1] # 取得該站點的唯一入口
                                    final_destination = entry_point # 確保重新規劃的目標是這個入口點
                                    # 禁止所有排隊區，除了它自己的目標入口點
                                    forbidden_cells = self.all_queue_spots - {entry_point}
                                else:
                                    # 備用策略: 如果找不到站點資訊，則禁止所有排隊區
                                    forbidden_cells = self.all_queue_spots
                            
                            new_path = plan_route(robot.position, final_destination, self.warehouse_matrix, dynamic_obstacles, forbidden_cells, cost_map)
                            
                            if new_path:
                                print(f"🗺️ 機器人 {robot.id} 找到新路徑！")
                                robot.path = new_path
                                robot.wait_time = 0 # 找到新路徑後，重設等待時間
                            else:
                                print(f"❌ 機器人 {robot.id} 找不到替代路徑，將在下一輪再試。")

                # --- Handle non-moving, action-based states ---
                elif robot.status == RobotStatus.PICKING:
                    if robot.pick_item():
                        # 撿貨完成。從任務列表中移除剛剛完成的貨架點。
                        completed_shelf = robot.task['shelf_locations'].pop(0)
                        print(f"👍 機器人 {robot.id} 在 {completed_shelf} 完成撿貨。")

                        # 檢查任務中是否還有其他貨架點需要前往
                        if robot.task['shelf_locations']:
                            # --- 前往任務中的下一個貨架點 ---
                            next_shelf = robot.task['shelf_locations'][0]
                            print(f"...任務 {robot.task['task_id']} 未完成，機器人 {robot.id} 前往下一站: {next_shelf}")

                            # 找到旁邊的走道格作為路徑規劃的起點
                            start_pos_for_route = find_adjacent_aisle(robot.position, self.warehouse_matrix)
                            if not start_pos_for_route:
                                print(f"⚠️ 機器人 {robot.id} 在貨架 {robot.position} 旁找不到可用的走道！")
                                robot.clear_task() # 卡住了，重設
                                continue
                            
                            path = plan_route(start_pos_for_route, next_shelf, self.warehouse_matrix)
                            if path:
                                robot.position = start_pos_for_route
                                robot.path = path
                                robot.status = RobotStatus.MOVING_TO_SHELF # 狀態變回「前往貨架」
                            else:
                                print(f"⚠️ 機器人 {robot.id} 在 {start_pos_for_route} 找不到前往下一個貨架 {next_shelf} 的路徑！將在原地等待。")
                                # 將剛完成的貨架點加回去，以便下一輪重試
                                robot.task['shelf_locations'].insert(0, completed_shelf)

                        else:
                            # --- 所有貨架點都已完成，現在前往交貨站 ---
                            print(f"🎉 機器人 {robot.id} 完成任務 {robot.task['task_id']} 的所有撿貨點。")
                            # 步驟 1: 尋找一個可用的交貨站入口。
                            best_station, best_queue_spot, min_dist = None, None, float('inf')
                            for station_info in self.picking_stations_info:
                                queue_spot = self.find_available_queue_entry(station_info)
                                if queue_spot:
                                    dist = euclidean_distance(robot.position, station_info['pos'])
                                    if dist < min_dist:
                                        min_dist, best_station, best_queue_spot = dist, station_info, queue_spot
                            
                            if not (best_station and best_queue_spot):
                                print(f"⏳ 機器人 {robot.id} 撿貨完畢，但所有交貨站入口忙碌中，將在原地等待。")
                                # 將剛完成的貨架點加回去，以便下一輪重試
                                robot.task['shelf_locations'].insert(0, completed_shelf)
                                continue
                            
                            start_pos_for_route = find_adjacent_aisle(robot.position, self.warehouse_matrix)
                            path = plan_route(start_pos_for_route, best_queue_spot, self.warehouse_matrix)
                            if path:
                                print(f"🤖 機器人 {robot.id} 從貨架移至走道 {start_pos_for_route}。")
                                robot.position = start_pos_for_route
                                robot.set_path_to_dropoff(path, best_station['pos'])
                            else:
                                print(f"⚠️ 機器人 {robot.id} 在 {start_pos_for_route} 找不到前往排隊區 {best_queue_spot} 的路徑！將在原地等待。")
                                robot.task['shelf_locations'].insert(0, completed_shelf)

                elif robot.status == RobotStatus.DROPPING_OFF:
                    if robot.drop_off_item():
                        print(f"✅ 機器人 {robot.id} 完成任務 {robot.task['task_id']} 的交貨。")
                        self.performance_logger.log_task_completion(time_step)

                        # 交貨完成後，將機器人移至指定的出口區
                        exit_pos = self.picking_exits.get(robot.position)
                        if exit_pos:
                            robot.position = exit_pos
                            print(f"🤖 機器人 {robot.id} 交貨後移動至出口 {exit_pos}。")
                        else:
                            # 這是一個邊界情況，但我們仍然要記錄下來
                            print(f"⚠️ 機器人 {robot.id} 在交貨站 {robot.position} 找不到指定的出口！")
                        
                        robot.clear_task() # 完成所有動作後，才將機器人重設為閒置
                
                elif robot.status == RobotStatus.WAITING_IN_QUEUE:
                    # 檢查排隊中的機器人是否可以前進
                    target_station_pos = robot.target_station_pos
                    station_info = next((s for s in self.picking_stations_info + self.charge_stations_info if s['pos'] == target_station_pos), None)

                    if not station_info:
                        print(f"錯誤：機器人 {robot.id} 正在排隊，但找不到其目標站點 {target_station_pos}！")
                        robot.clear_task()
                        continue

                    # 找到機器人在隊列中的當前索引
                    try:
                        current_queue_index = station_info['queue'].index(robot.position)
                    except ValueError:
                        # 如果機器人不在隊列中，可能是因為它正在從隊列移動到站點，或者出現了錯誤。
                        # 在這種情況下，我們暫時跳過它，讓主要的移動邏輯來處理。
                        continue

                    # 確定機器人面前的下一個位置 (如果在隊首，下一個位置就是工作站本身)
                    next_spot_in_line = station_info['pos'] if current_queue_index == 0 else station_info['queue'][current_queue_index - 1]

                    # 檢查下一個位置是否被其他機器人佔據
                    if any(r.position == next_spot_in_line for r in self.robots.values() if r.id != robot.id):
                        continue # 前方有其他機器人，原地等待

                    # 新增檢查：確認此空位是否在本輪已被其他排隊機器人預定
                    if next_spot_in_line in spots_targeted_in_queue_logic:
                        continue # 前方的空位剛被預定，原地等待

                    # 如果前方位置是工作站，需要額外檢查工作站是否可用
                    if next_spot_in_line == station_info['pos']:
                        station_is_available = False
                        if "CS" in station_info['id']:  # 如果是充電站
                            if len(self.charging_station.charging) < self.charging_station.capacity:
                                station_is_available = True
                        else:  # 如果是撿貨站
                            station_is_available = True
                        if not station_is_available:
                            continue # 工作站不可用 (例如充電位已滿)，原地等待
                    
                    # 如果執行到這裡，代表前方是空的，可以前進
                    print(f"👍 機器人 {robot.id} 從 {robot.position} 向前移動至 {next_spot_in_line}")
                    path_to_next_spot = plan_route(robot.position, next_spot_in_line, self.warehouse_matrix)
                    if path_to_next_spot:
                        robot.path = path_to_next_spot
                        robot.status = RobotStatus.MOVING_TO_CHARGE if "CS" in station_info['id'] else RobotStatus.MOVING_TO_DROPOFF
                        # 標記此空位已被預定
                        spots_targeted_in_queue_logic.add(next_spot_in_line)
                    else:
                        print(f"⚠️ 機器人 {robot.id} 在隊列中找不到前往下一格 {next_spot_in_line} 的路徑！")

                elif robot.status == RobotStatus.IDLE:
                    self.performance_logger.log_robot_idle_time(robot.id, 1)
                    # Check if it needs to charge
                    if robot.battery_level <= robot.charging_threshold:
                        # 尋找最近的可用充電站排隊區
                        best_station, best_queue_spot, min_dist = None, None, float('inf')
                        for station_info in self.charge_stations_info:
                            queue_spot = self.find_available_queue_entry(station_info)
                            if queue_spot:
                                dist = euclidean_distance(robot.position, station_info['pos'])
                                if dist < min_dist:
                                    min_dist, best_station, best_queue_spot = dist, station_info, queue_spot
                        
                        if best_station and best_queue_spot:
                            path = plan_route(robot.position, best_queue_spot, self.warehouse_matrix)
                            if path:
                                robot.go_charge(path, best_station['pos'])
                            else:
                                print(f"⚠️ 機器人 {robot.id} 在 {robot.position} 找不到前往充電排隊區 {best_queue_spot} 的路徑！")
                        else:
                            print(f"⏳ 機器人 {robot.id} 需要充電，但所有充電站入口都忙碌中。")

            # --- 4. Update Charging Station ---
            finished_charging_robots = self.charging_station.update()
            for robot in finished_charging_robots:
                robot.stop_charging() # 更新機器人狀態為閒置，並將電量充滿

                # 將剛充完電的機器人移到指定的出口區
                exit_pos = self.charge_exits.get(robot.position)
                if exit_pos:
                    robot.position = exit_pos
                    print(f"🤖 機器人 {robot.id} 充電後移動至出口 {exit_pos}。")
                else:
                    print(f"⚠️ 機器人 {robot.id} 在充電站 {robot.position} 找不到指定的出口！")

            # --- 5. 視覺化呈現 ---
            self.visualizer.draw(time_step)

        # --- 模擬結束 ---
        print(f"\n--- 模擬在 {self.max_simulation_steps} 時間步後結束 ---")
        print("\n--- 效能報告 ---")
        report = self.performance_logger.report()
        print(json.dumps(report, indent=2))

        self.visualizer.show()

if __name__ == "__main__":
    print("🚀 正在啟動倉儲模擬...")
    engine = SimulationEngine()
    engine.run()
    print("✅ 模擬結束。")