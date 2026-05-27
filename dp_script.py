import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import joblib
from gp_surrogate import train_surrogate, check_trimp_feasibility

# --- call once, up front ---
gpr, x_scaler, y_scaler = train_surrogate()

# 
T_max = 140
K_CTL = 1 - np.exp(-1 / 42)   # ≈ 0.0235
K_ATL = 1 - np.exp(-1 / 7)    # ≈ 0.1331

# need to discritize all of these
state = [CTL_t, ATL_t]
action = [dist, pace, elev]

# check these with the literature
CTL_{t+1} = CTL_t + (mu_trimp - CTL_t) * K_CTL
ATL_{t+1} = ATL_t + (mu_trimp - ATL_t) * K_ATL

# mu_trimp = GP model(action) - aka query GP

constraints = []
constraints += [ATL_next < r_max - fatigue cap]
constraints += [weekly distance increase < 10%]
constraints += [f(t) > 0]
constraints += [T <= T_max]
constraints += [rest_day_last >= t - 3] 

objective =  time2goalCTL # AKA cost function

possible_distances_array = np.linspace(0, 42, 43)
possible_pace_array = np.linspace(4, 7, 19)
possible_elevations_array = np.linspace(0, 300, 7)


# loop 1: 
#     V = np.full((n_ctl, n_atl), np.inf)
#     policy = np.zeros((n_ctl, n_atl), dtype=int)

#     possible actions = matrix of possible run combos based on distance,pace, elevation arrays
#     for action in actions 
#         trimp = query GP with action
#         save actions and trimp that give feasible trimp
#         ctl, atl = equation w feasible trimp
        
#         if action(distance) == distance_goal & CTL, ATL == CTL_goal, ATL_goal:
#             R = 1
#         else:
#             R = 0

#         expected_value += R + penalty*V[]
    
#     best_val = max(best_val, expected_value)
# V_next[r, c] = best_val

ctl_grid = np.arange(10, 66, 1)  # [10, 11, 12, ..., 65]  → 56 bins
atl_grid = np.arange(5, 76, 1)   # [5, 6, 7, ..., 75]     → 71 bins

n_ctl = len(ctl_grid)
n_atl = len(atl_grid)

action_array = np.array(list(product(possible_distances_array, possible_pace_array, possible_elevations_array)))
V = np.full((n_ctl, n_atl), np.inf)

for t in range(T_max, 0, -1):
    for ctl in ctl_grid:
        for atl in atl_grid:
            best_cost = np.inf
            
            # --- rest day ---
            mu_trimp = 0
            ctl_n = ctl + (0 - ctl) * K_CTL   # decays toward zero
            atl_n = atl + (0 - atl) * K_ATL   # decays toward zero

            ctl_n_idx = np.argmin(np.abs(ctl_grid - ctl_n))
            atl_n_idx = np.argmin(np.abs(atl_grid - atl_n))

            cost = 1 + V[ctl_n_idx, atl_n_idx]
            if cost < best_cost:
                best_cost = cost
                best_action = "rest"

            # --- run actions ---
            for action in action_array:
                ok, mu_trimp, std_trimp = check_trimp_feasibility(gpr, action, x_scaler, y_scaler)

                if not ok:
                    continue

                ctl_n = ctl + (mu_trimp - ctl) * K_CTL
                atl_n = atl + (mu_trimp - atl) * K_ATL

                ctl_n = np.clip(ctl_n, ctl_grid[0], ctl_grid[-1])
                atl_n = np.clip(atl_n, atl_grid[0], atl_grid[-1])

                ctl_n_idx = np.argmin(np.abs(ctl_grid - ctl_n))
                atl_n_idx = np.argmin(np.abs(atl_grid - atl_n))

                cost = 1 + V[ctl_n_idx, atl_n_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
            
            V[ctl_idx, atl_idx] = best_cost
            policy[ctl_idx, atl_idx] = best_action


# for t in range(T_max, 0, -1):          # backward in time
#     for each CTL bin:
#         for each ATL bin:
#             best_cost = inf
#             for each action (including rest):
#                 trimp = GP query
#                 check constraints
#                 compute next CTL, ATL
#                 find nearest grid indices for next state
#                 cost = 1 + V[next_ctl_idx, next_atl_idx]
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_action = action
#             V[ctl_idx, atl_idx] = best_cost
#             policy[ctl_idx, atl_idx] = best_action



# # State: (CTL_idx, ATL_idx, consec_run_days, day_of_week)
# # V[CTL, ATL, consec, dow] = min days to goal from this state

# # Base case
# for all (CTL, ATL) where CTL >= CTL_goal and TSB >= TSB_min:
#     V[CTL, ATL, ...] = 0

# # Bottom-up: iterate backwards from goal or forwards from day 0
# for each state (CTL, ATL, consec, dow):
#     best_cost = inf
    
#     # Option 1: rest
#     CTL_next, ATL_next = transition(CTL, ATL, trimp=0)
#     best_cost = min(best_cost, 1 + V[CTL_next, ATL_next, 0, (dow+1)%7])
    
#     # Option 2: run
#     for action in feasible_actions:  # pre-filtered by check_trimp_feasibility
#         trimp_mu = gp_predict(action)
#         CTL_next, ATL_next = transition(CTL, ATL, trimp_mu)
        
#         if constraints_satisfied(ATL_next, weekly_load, consec):
#             cost = 1 + V[CTL_next, ATL_next, consec+1, (dow+1)%7]
#             best_cost = min(best_cost, cost)
    
#     V[CTL, ATL, consec, dow] = best_cost
#     policy[CTL, ATL, consec, dow] = best_action