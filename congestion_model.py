from typing import List, Dict, Tuple, Set, TYPE_CHECKING
from robot_and_initial_state import Robot, RobotStatus

if TYPE_CHECKING:
    pass

Coord = Tuple[int, int]

class CongestionManager:
    """
    負責協調機器人移動以避免碰撞的中央管理器。
    它採用基於優先級的儲存格預留策略。
    """
    def __init__(self):
        """初始化擁塞管理器。"""
        pass

    def coordinate_moves(self, robots: List['Robot']) -> Set[str]:
        """
        協調一個時間步內所有機器人的移動。

        此方法決定哪些機器人可以在當前時間步安全地移動，以避免碰撞。
        策略：
        1. 靜止的機器人會預留它們當前的位置。
        2. 移動中的機器人會根據優先級嘗試預留它們的下一個位置。
        3. 優先級規則：攜帶物品的機器人 > 前往充電的機器人 > 其他。

        :param robots: 當前模擬中所有機器人物件的列表。
        :return: 一個包含被批准移動的機器人 ID 的集合。
        """
        approved_moves: Set[str] = set()
        # 這個集合將儲存「下一個」時間步中所有被佔用的位置
        future_reserved_cells: Set[Coord] = set()

        moving_robots = [r for r in robots if r.status in [RobotStatus.MOVING_TO_SHELF, RobotStatus.MOVING_TO_DROPOFF, RobotStatus.MOVING_TO_CHARGE]]
        stationary_robots = [r for r in robots if r not in moving_robots]

        # 1. 靜止的機器人將在下一個時間步繼續佔用它們當前的位置
        for r in stationary_robots:
            future_reserved_cells.add(r.position)

        # 2. 根據優先級對移動中的機器人進行排序
        # 規則: (攜帶物品, 前往充電) -> True > False。所以 (True, False) > (False, True) > (False, False)
        moving_robots.sort(key=lambda r: (r.carrying_item, r.status == RobotStatus.MOVING_TO_CHARGE), reverse=True)

        # 3. 根據優先級決定移動中的機器人的命運
        for robot in moving_robots:
            next_pos = robot.next_position
            # 檢查目標位置是否已被預留
            if next_pos and next_pos not in future_reserved_cells:
                # 如果未被預留，則批准移動，並為下一個時間步預留其目標位置
                approved_moves.add(robot.id)
                future_reserved_cells.add(next_pos)
            else:
                # 如果移動被拒絕，機器人將停在原地。
                # 因此，為下一個時間步預留其「當前」位置，以防被其他機器人撞上。
                future_reserved_cells.add(robot.position)

        return approved_moves
