"""
策略設定檔：用於選擇可插拔的策略模組。

要切換路徑規劃策略，您只需要：
1. 建立一個新的策略檔案 (例如 `my_new_routing.py`)。
2. 確保您的新檔案遵循與 `routing.py` 相同的介面 (例如，包含一個 `plan_route` 函式)。
3. 將下方的 `ROUTING_STRATEGY` 變數值修改為您的新檔案名稱 (不含 .py)。

主程式會自動載入它，無需修改 main.py。
"""

# --- Pluggable Strategy Configuration ---

# --- 路徑規劃策略 (Routing Strategy) ---
# 在此處填寫您想使用的路徑規劃模組的「檔案名稱」(不含 .py)。
# 'routing': 預設的 A* 演算法實作。
ROUTING_STRATEGY = 'routing'
