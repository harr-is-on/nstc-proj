import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from typing import Tuple, Dict, List

Coord = Tuple[int, int]
ShelfDict = Dict[Coord, List]

# --- 單一事實來源 (Single Source of Truth) for Station Layout ---
# 將每個站點與其排隊區和出口區明確關聯
STATION_LAYOUT = {
    "charge_stations": [
        {"id": "CS1", "pos": (0, 14), "queue": [(0, 13), (0, 12), (0, 11)], "exit": (1, 14)},
        {"id": "CS2", "pos": (13, 14), "queue": [(13, 13), (13, 12), (13, 11)], "exit": (12, 14)}
    ],
    "picking_stations": [
        {
            "id": "PS1", "pos": (2, 14), 
            "queue": [(3, 14), (4, 14), (5, 14)], # 隊列由近到遠
            "exit": (2, 13)
        },
        {
            "id": "PS2", "pos": (11, 14), 
            "queue": [(10, 14), (9, 14), (8, 14)], # 隊列由近到遠
            "exit": (11, 13)
        }
    ]
}

# 代號定義
# 0=走道, 1=貨架, 2=撿貨站, 3=充電站, 4=充電排隊區, 5=充電出口, 6=撿貨排隊區, 7=撿貨出口
CELL_CODES = {
    "aisle": 0, "shelf": 1, "picking": 2, "charge": 3,
    "charge_queue": 4, "charge_exit": 5,
    "picking_queue": 6, "picking_exit": 7
}


def create_warehouse_layout() -> Tuple[np.ndarray, ShelfDict]:
    """
    創建一個固定的倉儲佈局及其對應的貨架資料結構。

    這個函數封裝了倉庫的特定設計，確保每次呼叫它時，
    都會返回完全相同的預定義佈局。

    返回:
        一個包含以下內容的元組：
        - warehouse_matrix (np.ndarray): 一個代表倉庫的網格。
          (0=走道, 1=貨架, 2=撿貨站, 3=充電站,
           4=充電排隊區, 5=充電出口,
           6=撿貨排隊區, 7=撿貨出口)
        - shelf_levels_dict (dict): 一個將貨架座標映射到其儲存層級的字典。
    """
    # --- 固定的倉儲配置 ---
    num_rows = 14
    num_cols = 15
    vertical_aisles = [0, 1, 4, 7, 10, 13, 14]
    horizontal_aisles = [0, 1, 6, 7, 12, 13]
    shelf_levels = 4

    # --- 配置結束 ---

    # 使用 numpy 的廣播功能更有效率地創建矩陣
    warehouse_matrix = np.ones((num_rows, num_cols), dtype=int)
    warehouse_matrix[horizontal_aisles, :] = 0
    warehouse_matrix[:, vertical_aisles] = 0

    # 使用字典推導式更有效率地創建貨架字典
    shelf_coords = np.argwhere(warehouse_matrix == 1)
    shelf_levels_dict = {tuple(coord): [[] for _ in range(shelf_levels)] for coord in shelf_coords}

    # 從單一來源設定所有特殊區域
    for station_list in STATION_LAYOUT.values():
        for station_info in station_list:
            station_type = "picking" if "PS" in station_info["id"] else "charge"
            r, c = station_info["pos"]
            warehouse_matrix[r, c] = CELL_CODES[station_type]
            for qr, qc in station_info["queue"]:
                warehouse_matrix[qr, qc] = CELL_CODES[station_type + "_queue"]
            er, ec = station_info["exit"]
            warehouse_matrix[er, ec] = CELL_CODES[station_type + "_exit"]


    return warehouse_matrix, shelf_levels_dict

def get_station_locations() -> Dict[str, List[Coord]]:
    """
    返回倉庫中所有工作站的固定座標。
    這是一個輔助函數，讓主程式可以方便地獲取這些位置，而無需重新解析矩陣。

    返回:
        一個包含所有站點、排隊區和出口區詳細資訊的字典。
    """
    # 直接返回定義好的佈局常數
    return STATION_LAYOUT

def plot_warehouse(warehouse_matrix: np.ndarray):
    """
    使用 Matplotlib 將倉儲佈局視覺化。
    """
    num_rows, num_cols = warehouse_matrix.shape
    
    # 為所有儲存格類型定義顏色和標籤
    color_map = {
        0: ('#FFFFFF', 'Aisle'), 1: ('#A5A5A5', 'Shelf'), 2: ('#FF9800', 'Picking Station'),
        3: ('#2196F3', 'Charge Station'), 4: ('#ADD8E6', 'Charge Queue'), 5: ('#00008B', 'Charge Exit'),
        6: ('#FFFACD', 'Picking Queue'), 7: ('#FFA500', 'Picking Exit')
    }
    
    # 根據 color_map 創建 ListedColormap 和圖例
    colors = [color_map[i][0] for i in sorted(color_map.keys())]
    labels = [color_map[i][1] for i in sorted(color_map.keys())]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    cmap = mcolors.ListedColormap(colors)    
    bounds = np.arange(-0.5, 8.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(warehouse_matrix, cmap=cmap, norm=norm)

    # 增加網格線和標籤以提高清晰度
    ax.set_xticks(np.arange(num_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(num_cols))
    ax.set_yticks(np.arange(num_rows))

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Warehouse Layout", fontsize=16)
    plt.show()

if __name__ == '__main__':
    # 這個區塊只在直接執行此腳本時運行。
    # 這樣可以在不影響其他檔案匯入的情況下進行測試和視覺化。
    # 單獨跑這張程式碼時可以有倉庫圖出現 run main.py時不會有
    
    print("Generating and testing warehouse layout...")
    matrix, shelves = create_warehouse_layout()

    print("\nWarehouse Matrix Created (shape: {}):".format(matrix.shape))
    print(matrix)
    print("\nNumber of shelves created: {}".format(len(shelves)))

    print("\nDisplaying warehouse plot for verification...")
    plot_warehouse(matrix)
