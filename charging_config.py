from typing import Dict, Union, Any, List

# --- 充電策略選擇 (Charging Strategy Selection) ---
# 在此處填寫您想使用的充電策略模組的「檔案名稱」(不含 .py)。
# 'charging_model': 預設的充電站模型。
# 'smart_charging_model': (範例) 一個考慮電價的智慧充電模型。
CHARGING_STRATEGY = 'charging_model'


# --- 充電站相關設定 ---
# 將所有充電相關的定義集中於此
CHARGING_STATION_CONFIG: Dict[str, Union[int, float, bool, List[Dict[str, Any]]]] = {
    "capacity": 1,                # 充電站可同時容納的機器人數量
    "charge_rate": 5,             # 每個時間步為機器人增加的電量
    "charging_threshold": 30,     # 當電量低於此百分比時，機器人會尋求充電

    # --- 【新】動態充電策略設定 ---
    "enable_dynamic_charging": True, # 設為 True 來啟用以下動態規則

    # 狀態規則列表 (請由最忙碌到次忙碌排序)
    # max_idle_robots: 當閒置機器人數量「小於或等於」此值時，套用此規則。
    "dynamic_states": [
        # Busy State: When idle robots <= 0, charge to 70%
        {"max_idle_robots": 0, "full_charge_level": 70, "name": "Busy"},
        # Normal State: When idle robots <= 1, charge to 85%
        {"max_idle_robots": 1, "full_charge_level": 85, "name": "Normal"},
    ],
    # Idle State (Default): When all other conditions are not met (i.e., idle robots >= 2), charge to 100%
    "default_state_name": "Idle",
    "default_full_charge_level": 100,
    "full_charge_level": 80,      # (備用) 當 enable_dynamic_charging 為 False 時使用的固定值
}
