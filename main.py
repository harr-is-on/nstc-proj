"""
Main entry point for the Warehouse Robot Simulation.

This single script defines the core SimulationEngine and runs the simulation.
"""

import random
import json
import math
from typing import Tuple, List, Dict, Optional

# Import modules from the project
from warehouse_layout import create_warehouse_layout, get_station_locations
from robot_and_initial_state import (
    Robot,
    RobotStatus, # 引入機器人狀態列舉
    initialize_robots,
    ROBOT_CONFIG,
    SIMULATION_CONFIG
)
from taskmanager import TaskManager
from routing import plan_route, euclidean_distance, find_adjacent_aisle
from charging_model import ChargingStation, CHARGING_STATION_CONFIG
from congestion_model import CongestionManager # 引入新的擁塞管理器
from visualization import Visualizer
from performance_logger import PerformanceLogger

Coord = Tuple[int, int]
Task = Dict[str, any]

class SimulationEngine:
    """
    The core engine for the warehouse simulation.
    It orchestrates all components, manages the main loop, and logs performance.
    """
    def __init__(self):
        # --- Initialize components ---
        self.warehouse_matrix, _ = create_warehouse_layout()
        self.robots = initialize_robots(self.warehouse_matrix, ROBOT_CONFIG, CHARGING_STATION_CONFIG)
        self.task_manager = TaskManager(self.warehouse_matrix)
        self.charging_station = ChargingStation(**CHARGING_STATION_CONFIG)
        self.congestion_manager = CongestionManager() # 初始化新的擁塞管理器
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
        # 為了處理出口邏輯
        self.picking_exits = {s['pos']: s['exit'] for s in self.picking_stations_info}
        self.charge_exits = {s['pos']: s['exit'] for s in self.charge_stations_info}

        # 建立一個包含所有排隊區格子的集合，用於快速查找
        self.all_queue_spots = set()
        for station_list in self.station_layout.values():
            for station_info in station_list:
                self.all_queue_spots.update(station_info['queue'])

        # --- Simulation control parameters ---
        self.max_simulation_steps = SIMULATION_CONFIG.get("max_simulation_steps", 500)
        self.task_generation_interval = SIMULATION_CONFIG.get("task_generation_interval", 25)

        # --- Generate initial tasks ---
        for i in range(SIMULATION_CONFIG.get("num_initial_tasks", 5)):
            self.task_manager.generate_random_task()

    def find_available_queue_entry(self, station_info: Dict) -> Optional[Coord]:
        """
        檢查一個站點的排隊區入口 (最遠的那格) 是否可用。
        """
        entry_point = station_info['queue'][-1] # 定義隊列的最後一格為唯一入口

        # 取得所有目前被佔用或已被預訂為路徑終點的位置
        occupied_or_targeted = {r.position for r in self.robots.values()}
        for r in self.robots.values():
            if r.path:
                occupied_or_targeted.add(r.path[-1])

        # 如果入口點沒有被佔用或預訂，則返回該入口點
        return entry_point if entry_point not in occupied_or_targeted else None

    def run(self):
        """Main simulation loop."""
        time_step = 0
        while time_step < self.max_simulation_steps:
            time_step += 1
            print(f"\n--- Time Step: {time_step} ---")

            # --- 1. Generate and Assign Tasks ---
            # This set will prevent multiple queueing robots from targeting the same empty spot in the same timestep.
            spots_targeted_in_queue_logic = set()

            if time_step % self.task_generation_interval == 0:
                self.task_manager.generate_random_task()
            self.task_manager.assign_pending_tasks(self.robots, self.warehouse_matrix, forbidden_cells_for_tasks=self.all_queue_spots)

            # --- 2. Coordinate Robot Moves to Avoid Collision ---
            approved_robot_ids = self.congestion_manager.coordinate_moves(list(self.robots.values()))

            # --- 3. Update Robot States and Actions ---
            for robot in self.robots.values():
                # --- Handle moving robots ---
                if robot.status in [RobotStatus.MOVING_TO_SHELF, RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]:
                    if robot.id in approved_robot_ids:
                        battery_before_move = robot.battery_level
                        distance_moved = robot.move_to_next_step()
                        energy_consumed = battery_before_move - robot.battery_level
                        self.performance_logger.log_distance_traveled(robot.id, distance_moved)
                        self.performance_logger.log_energy_usage(robot.id, energy_consumed)
                        robot.wait_time = 0 # Reset wait time on successful move

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

                        # **核心修正**：如果機器人在排隊區被阻擋，將其狀態重設回排隊中
                        # 這樣它在下一輪才能重新評估是否可以前進，而不是卡在「移動中」的狀態。
                        if robot.position in self.all_queue_spots:
                            print(f"🔄 機器人 {robot.id} 在排隊時被阻擋，重設狀態為 WAITING_IN_QUEUE。")
                            robot.status = RobotStatus.WAITING_IN_QUEUE
                            robot.path = [] # 清除為此次失敗移動所規劃的路徑
                        
                        # **新增：智慧繞路邏輯**
                        # 如果機器人不是在排隊，且等待時間過長，則嘗試重新規劃路徑
                        elif robot.wait_time > robot.replan_wait_threshold:
                            print(f"🤔 機器人 {robot.id} 等待過久，嘗試重新規劃路徑...")
                            
                            # 將其他所有機器人的位置視為動態障礙物
                            dynamic_obstacles = [r.position for r in self.robots.values() if r.id != robot.id]
                            final_destination = robot.path[-1] # 預設目標是原路徑的終點
                            forbidden_cells = set() # 預設沒有禁止區域

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
                                    entry_point = station_info['queue'][-1]
                                    final_destination = entry_point # 確保重新規劃的目標是入口點
                                    # 禁止所有排隊區，除了它自己的目標入口點
                                    forbidden_cells = self.all_queue_spots - {entry_point}
                                else:
                                    # Fallback: 如果找不到站點資訊，則禁止所有排隊區
                                    forbidden_cells = self.all_queue_spots
                            
                            new_path = plan_route(robot.position, final_destination, self.warehouse_matrix, dynamic_obstacles, forbidden_cells)
                            
                            if new_path:
                                print(f"🗺️ 機器人 {robot.id} 找到新路徑！")
                                robot.path = new_path
                                robot.wait_time = 0 # 重設等待時間
                            else:
                                print(f"❌ 機器人 {robot.id} 找不到替代路徑，將在下一輪再試。")

                # --- Handle non-moving, action-based states ---
                elif robot.status == RobotStatus.PICKING:
                    if robot.pick_item():
                        # 揀貨完成。現在需要為它規劃下一步。
                        
                        # 步驟 1: 尋找一個可用的交貨站入口。
                        best_station, best_queue_spot, min_dist = None, None, float('inf')
                        for station_info in self.picking_stations_info:
                            queue_spot = self.find_available_queue_entry(station_info)
                            if queue_spot:
                                dist = euclidean_distance(robot.position, station_info['pos'])
                                if dist < min_dist:
                                    min_dist, best_station, best_queue_spot = dist, station_info, queue_spot
                        
                        # 如果找不到可用的入口，則機器人留在原地等待，下一輪再試。
                        if not (best_station and best_queue_spot):
                            print(f"⏳ 機器人 {robot.id} 撿貨完畢，但所有交貨站入口忙碌中，將在原地等待。")
                            continue

                        # 步驟 2: 找到旁邊的走道格作為路徑規劃的起點。
                        start_pos_for_route = find_adjacent_aisle(robot.position, self.warehouse_matrix)
                        if not start_pos_for_route:
                            print(f"⚠️ 機器人 {robot.id} 在貨架 {robot.position} 旁找不到可用的走道！")
                            robot.clear_task() # 卡住了，重設
                            continue

                        # 步驟 3: 從走道位置規劃到入口。
                        path = plan_route(start_pos_for_route, best_queue_spot, self.warehouse_matrix)
                        
                        # 步驟 4: 只有在路徑規劃成功後，才真正移動機器人並設定新狀態。
                        if path:
                            print(f"🤖 機器人 {robot.id} 撿貨完畢，從貨架移至走道 {start_pos_for_route}。")
                            robot.position = start_pos_for_route
                            robot.set_path_to_dropoff(path, best_station['pos'])
                        else:
                            print(f"⚠️ 機器人 {robot.id} 在 {start_pos_for_route} 找不到前往排隊區 {best_queue_spot} 的路徑！將在原地等待。")

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
                            # 這是個邊界情況，但我們仍然要記錄下來
                            print(f"⚠️ 機器人 {robot.id} 在交貨站 {robot.position} 找不到指定的出口！")
                        
                        robot.clear_task() # 現在才將機器人重設為閒置
                
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
                        # 在這種情況下，我們暫時跳過它，讓主移動邏輯處理。
                        continue

                    # 確定機器人面前的下一個位置 (如果在隊首，下一個位置就是工作站本身)
                    next_spot_in_line = station_info['pos'] if current_queue_index == 0 else station_info['queue'][current_queue_index - 1]

                    # 檢查下一個位置是否被其他機器人佔據
                    if any(r.position == next_spot_in_line for r in self.robots.values() if r.id != robot.id):
                        continue # 前方有機器人，原地等待

                    # 新增檢查：確認此空位是否在本輪已被其他排隊機器人預定
                    if next_spot_in_line in spots_targeted_in_queue_logic:
                        continue # 前方空位剛被預定，原地等待

                    # 如果前方位置是工作站，需要額外檢查工作站是否可用
                    if next_spot_in_line == station_info['pos']:
                        station_is_available = False
                        if "CS" in station_info['id']:  # 如果是充電站
                            if len(self.charging_station.charging) < self.charging_station.capacity:
                                station_is_available = True
                        else:  # 如果是撿貨站
                            station_is_available = True
                        if not station_is_available:
                            continue # 工作站不可用 (例如充電已滿)，原地等待
                    
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
                            print(f"⏳ 機器人 {robot.id} 需要充電，但所有充電站入口忙碌中。")

            # --- 4. Update Charging Station ---
            finished_charging_robots = self.charging_station.update()
            for robot in finished_charging_robots:
                robot.stop_charging() # 更新機器人狀態為閒置並充滿電

                # 將剛充完電的機器人移到指定的出口區
                exit_pos = self.charge_exits.get(robot.position)
                if exit_pos:
                    robot.position = exit_pos
                    print(f"🤖 機器人 {robot.id} 充電後移動至出口 {exit_pos}。")
                else:
                    print(f"⚠️ 機器人 {robot.id} 在充電站 {robot.position} 找不到指定的出口！")

            # --- 5. Visualization ---
            self.visualizer.draw(time_step)

        # --- Simulation End ---
        print(f"\n--- 模擬在 {self.max_simulation_steps} 時間步後結束 ---")
        print("\n--- Performance Report ---")
        report = self.performance_logger.report()
        print(json.dumps(report, indent=2))

        self.visualizer.show()

if __name__ == "__main__":
    print("🚀 Starting Warehouse Simulation...")
    engine = SimulationEngine()
    engine.run()
    print("✅ Simulation Finished.")