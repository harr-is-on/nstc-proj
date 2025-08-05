# performance_logger.py

from collections import defaultdict
from typing import Dict, Any, Union
import json

class PerformanceLogger:
    """
    一個用於記錄和計算多機器人模擬效能指標的類別。

    這個日誌記錄器會追蹤整體的系統效能以及個別機器人的貢獻。
    追蹤的指標：
    - Makespan (總完成時間)
    - 所有及個別機器人的總閒置時間
    - 所有及個別機器人的總移動距離
    - 所有及個別機器人的總能耗
    """

    def __init__(self):
        """初始化 PerformanceLogger。"""
        # 最後一個任務完成的時間點，代表 makespan。
        self._makespan = 0.0

        # 追蹤已完成的任務總數
        self._tasks_completed = 0

        # 一個用來儲存每個機器人詳細統計資料的字典。
        # 使用 defaultdict 可以輕鬆處理新加入的機器人，無需檢查其是否存在。
        # 結構: {'robot_id': {'idle_time': float, 'distance': float, 'energy': float}}
        self.robot_stats = defaultdict(lambda: defaultdict(float))

    def log_task_completion(self, finish_time: Union[int, float]):
        """
        記錄任務的完成時間以計算 makespan。
        makespan 是指最晚完成任務的時間點。

        參數:
            finish_time (Union[int, float]): 任務完成時的模擬時間。
        """
        if finish_time > self._makespan:
            self._makespan = float(finish_time)
        self._tasks_completed += 1

    def log_robot_idle_time(self, robot_id: str, idle_time: Union[int, float]):
        """
        增加特定機器人的閒置時間。

        參數:
            robot_id (str): 機器人的唯一識別碼。
            idle_time (Union[int, float]): 要增加的閒置時間長度。
        """
        self.robot_stats[robot_id]['idle_time'] += float(idle_time)

    def log_distance_traveled(self, robot_id: str, distance: Union[int, float]):
        """
        增加特定機器人移動的距離。

        參數:
            robot_id (str): 機器人的唯一識別碼。
            distance (Union[int, float]): 要增加的移動距離。
        """
        self.robot_stats[robot_id]['distance'] += float(distance)

    def log_energy_usage(self, robot_id: str, energy_used: Union[int, float]):
        """
        增加特定機器人消耗的能量。

        參數:
            robot_id (str): 機器人的唯一識別碼。
            energy_used (Union[int, float]): 要增加的能量消耗量。
        """
        self.robot_stats[robot_id]['energy'] += float(energy_used)

    def get_makespan(self) -> float:
        """返回目前的 makespan。"""
        return self._makespan

    def get_tasks_completed(self) -> int:
        """返回已完成的任務總數。"""
        return self._tasks_completed

    def _calculate_total(self, metric_key: str) -> float:
        """一個輔助函式，用於計算給定指標的總和。"""
        return sum(stats.get(metric_key, 0.0) for stats in self.robot_stats.values())

    def get_total_idle_time(self) -> float:
        """返回所有機器人的閒置時間總和。"""
        return self._calculate_total('idle_time')

    def get_total_distance_traveled(self) -> float:
        """返回所有機器人移動的距離總和。"""
        return self._calculate_total('distance')

    def get_total_energy_usage(self) -> float:
        """返回所有機器人消耗的能量總和。"""
        return self._calculate_total('energy')

    def get_robot_stats(self, robot_id: str) -> Dict[str, float]:
        """
        返回單一機器人的所有統計數據字典。
        
        參數:
            robot_id (str): 機器人的 ID。
        
        返回:
            包含指定機器人統計數據的字典副本。
        """
        # 直接存取 defaultdict 會自動處理不存在的 key，再轉換為一般 dict 返回副本。
        return dict(self.robot_stats[robot_id])
        
    def report(self) -> Dict[str, Any]:
        """
        產生所有效能指標的摘要報告。

        返回:
            一個包含總體指標和個別機器人指標的字典。
        """
        # 將 defaultdict 轉換為一般的 dict，讓報告更簡潔。
        per_robot_report = {
            robot_id: dict(stats) for robot_id, stats in self.robot_stats.items()
        }

        summary = {
            'overall_metrics': {
                'makespan': self.get_makespan(),
                'tasks_completed': self.get_tasks_completed(),
                'total_idle_time': self.get_total_idle_time(),
                'total_distance_traveled': self.get_total_distance_traveled(),
                'total_energy_usage': self.get_total_energy_usage(),
            },
            'per_robot_metrics': per_robot_report
        }
        return summary

    def reset(self):
        """將所有記錄器資料重設為初始狀態。"""
        self._makespan = 0.0
        self._tasks_completed = 0
        self.robot_stats.clear()
        print("PerformanceLogger 已被重設。")

# --- 範例用法 ---
if __name__ == '__main__':
    import random

    print("--- PerformanceLogger 範例用法 ---")

    # 1. 初始化記錄器
    logger = PerformanceLogger()

    # --- 2. 模擬一個簡單情境 ---
    robot_ids = ['R1', 'R2', 'R3']
    simulation_steps = 100
    
    print(f"\n正在為 {len(robot_ids)} 個機器人執行 {simulation_steps} 個時間步的簡單模擬...")

    for current_time in range(1, simulation_steps + 1):
        # 在每個時間步，每個機器人執行某些動作
        for robot_id in robot_ids:
            action = random.choice(['move', 'idle', 'work'])

            if action == 'move':
                # 機器人移動，記錄距離和能耗
                distance_moved = random.uniform(0.5, 1.5)
                energy_used = distance_moved * 1.2  # 假設能耗與距離成正比
                logger.log_distance_traveled(robot_id, distance_moved)
                logger.log_energy_usage(robot_id, energy_used)
            
            elif action == 'idle':
                # 機器人閒置，記錄閒置時間
                idle_duration = 1  # 假設每個時間步為 1 個時間單位
                logger.log_robot_idle_time(robot_id, idle_duration)
            
            # 'work' 動作是一個佔位符，不會對這些特定指標產生影響
    
    # 模擬一些在不同時間完成的任務
    # 在真實系統中，這會在機器人完成其分配的任務時被呼叫。
    logger.log_task_completion(finish_time=85)
    logger.log_task_completion(finish_time=98)  # 這會將 makespan 更新為最新的時間
    logger.log_task_completion(finish_time=92)  # 這個會被忽略，因為它不是最新的時間

    # --- 3. 取得並印出個別指標 ---
    print("\n--- 個別指標 ---")
    print(f"總完成時間 (Makespan): {logger.get_makespan()}")
    print(f"總閒置時間: {logger.get_total_idle_time():.2f}")
    print(f"總移動距離: {logger.get_total_distance_traveled():.2f}")
    print(f"總能耗: {logger.get_total_energy_usage():.2f}")
    print(f"R1 的統計數據: {logger.get_robot_stats('R1')}")


    # --- 4. 產生並印出最終報告 ---
    print("\n--- 最終摘要報告 ---")
    final_report = logger.report()

    # 使用 json.dumps 以更易讀的格式印出報告字典
    print(json.dumps(final_report, indent=2, ensure_ascii=False))

    # --- 5. 重設記錄器以進行新的模擬 ---
    print("\n")
    logger.reset()
    print("\n記錄器重設後的狀態:")
    print(json.dumps(logger.report(), indent=2, ensure_ascii=False))
