from layout import build_layout
from s_shape_d import plan_route_s_shape_dynamic
from largest_gap_n import plan_route_largest_gap_custom
from combine import plan_route_composite_custom  # <-- 新增
from visualize import visualize_route
import matplotlib.pyplot as plt
from typing import Tuple

Coord = Tuple[int, int]

def main():
    wl = build_layout()
    layout = wl.layout_matrix
    start: Coord = (7, 7)
    picks = [(5,10),(2,10),(11,4),(10,4),(5,4),(2,4),(8,7)]

    # ---------- S-shape ----------
    path_s = plan_route_s_shape_dynamic(layout,start,picks.copy(),
                                        que_pick_zones=wl.que_pick,
                                        picking_stations=wl.picking_stations)
    visualize_route(layout,path_s,picks,save_path="s_shape.png")
    plt.close('all')


    # ---------- Largest-Gap ----------
    path_lg = plan_route_largest_gap_custom(layout,start,picks.copy(),
                                            picking_stations=wl.picking_stations,
                                            que_pick_zones=wl.que_pick)
    visualize_route(layout,path_lg,picks,save_path="largest_gap.png")
    plt.close('all')
   
    # ---------- Composite (簡易) ----------
    path_c = plan_route_composite_custom(layout,start,picks.copy(),
                                         que_pick_zones=wl.que_pick,
                                         picking_stations=wl.picking_stations)
    visualize_route(layout,path_c,picks,save_path="composite_simple.png")
    plt.close('all')
  

if __name__ == "__main__":
    main()
