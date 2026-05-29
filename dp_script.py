import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import joblib
from gp_surrogate import train_surrogate, check_trimp_feasibility


# --- load surrogate (once) ---
gpr, x_scaler, y_scaler = train_surrogate()

# --- constants ---
K_CTL = 1 - np.exp(-1 / 42)
K_ATL = 1 - np.exp(-1 / 7)
CTL_GOAL = 60
atl_max = 75
T_max = 140
MAX_CONSEC_RUNS = 2   # force rest after this many consecutive run days
DIST_RAMP   = 3.0     # max km a run may exceed the recent earned distance ceiling
DMAX_SEED   = 5.0     # day-1 ceiling (first run capped at DMAX_SEED + DIST_RAMP)

# --- define action space ---
# possible_distances_array = np.linspace(0, 42, 43)
# possible_pace_array = np.linspace(4, 7, 19)
# possible_elevations_array = np.linspace(0, 300, 7)

possible_distances_array  = np.linspace(2, 42, 19)   # capped at data support
possible_pace_array       = np.linspace(4, 7, 6)
possible_elevations_array = np.linspace(0, 300, 6)

action_array = np.array(list(product(possible_distances_array, possible_pace_array, possible_elevations_array)))

# --- define grids ---
ctl_grid    = np.arange(10, 65, 1)   # up to CTL_GOAL + buffer
atl_grid    = np.arange(5,  76, 1)
consec_grid = np.arange(0, MAX_CONSEC_RUNS + 1)  # 0, 1, 2
# earned distance ceiling: the longest run done so far (seed .. max action distance).
# the grid points are the achievable distances themselves, so a run always lands on one.
dmax_grid   = np.unique(np.concatenate(([DMAX_SEED], possible_distances_array)))

n_ctl    = len(ctl_grid)
n_atl    = len(atl_grid)
n_consec = len(consec_grid)
n_dmax   = len(dmax_grid)

# --- init value function and policy ---
# V[i, j, k, l] = min days to goal from (ctl_grid[i], atl_grid[j], consec_grid[k], dmax_grid[l])
V      = np.full((n_ctl, n_atl, n_consec, n_dmax), np.inf)
policy = np.full((n_ctl, n_atl, n_consec, n_dmax), None, dtype=object)

# --- terminal condition ---
for i, ctl in enumerate(ctl_grid):
    if ctl >= CTL_GOAL:
        V[i, :, :, :] = 0


print("Pre-computing TRIMP for all actions...")
trimp_cache = {}
for action in action_array:
    ok, mu_trimp, std_trimp = check_trimp_feasibility(gpr, action, x_scaler, y_scaler)
    if ok and mu_trimp > 0:
        trimp_cache[tuple(action)] = mu_trimp
feasible_actions = [(action, trimp) for action, trimp in trimp_cache.items()]
print(f"  {len(feasible_actions)} feasible actions out of {len(action_array)}")
joblib.dump(trimp_cache, "trimp_cache.pkl")

# --- precompute next-state indices for all actions × all (CTL, ATL) states ---
# trimp_vals[a] = TRIMP for feasible action a
trimp_vals   = np.array([t for _, t in feasible_actions])   # (A,)
action_list  = [a for a, _ in feasible_actions]             # list of tuples, length A
action_dist  = np.array([a[0] for a in action_list])        # (A,) distance (km) of each action

# distance-ramp constraint — shape (n_dmax, A): True if action's distance is within
# DIST_RAMP km of the current earned ceiling.
dist_ok = (action_dist[None, :] <= dmax_grid[:, None] + DIST_RAMP)      # (n_dmax, A)

# next earned ceiling for every (ceiling, action): max(current ceiling, run distance).
# every run distance is itself a dmax_grid point, so searchsorted gives an exact index.
dmax_after_run   = np.maximum(dmax_grid[:, None], action_dist[None, :])  # (n_dmax, A)
dmax_next_run_idx = np.searchsorted(dmax_grid, dmax_after_run)           # (n_dmax, A)
dmax_next_run_idx = np.clip(dmax_next_run_idx, 0, n_dmax - 1)

# next CTL/ATL for every (state, action) combo — shape (n_ctl, A) and (n_atl, A)
ctl_next_run = ctl_grid[:, None] + (trimp_vals[None, :] - ctl_grid[:, None]) * K_CTL  # (n_ctl, A)
atl_next_run = atl_grid[:, None] + (trimp_vals[None, :] - atl_grid[:, None]) * K_ATL  # (n_atl, A)

# clip and convert to grid indices (grids are integer steps so we can round directly)
ctl_next_run_idx = np.clip(np.round(ctl_next_run - ctl_grid[0]).astype(int), 0, n_ctl - 1)
atl_next_run_idx = np.clip(np.round(atl_next_run - atl_grid[0]).astype(int), 0, n_atl - 1)

# ATL constraint masks — shape (n_atl, A): True means action is feasible from that ATL
atl_next_run_vals = atl_next_run                                        # (n_atl, A)
atl_ok            = (atl_next_run_vals <= atl_max)                      # absolute cap
ramp_ok           = (atl_next_run_vals <= atl_grid[:, None] + 8)        # ramp constraint (max +8 ATL/day)
run_feasible      = atl_ok & ramp_ok                                    # (n_atl, A)

# goal-reaching mask — shape (n_ctl, A): True means this action reaches CTL_GOAL
reaches_goal = ctl_next_run >= CTL_GOAL                                 # (n_ctl, A)

# precompute rest-day next indices (same for every iteration)
ctl_rest = ctl_grid + (0 - ctl_grid) * K_CTL
atl_rest = atl_grid + (0 - atl_grid) * K_ATL
ctl_rest_idx = np.clip(np.round(ctl_rest - ctl_grid[0]).astype(int), 0, n_ctl - 1)
atl_rest_idx = np.clip(np.round(atl_rest - atl_grid[0]).astype(int), 0, n_atl - 1)


for iteration in range(T_max):
    V_prev = V.copy()

    for i, ctl in enumerate(ctl_grid):
        for j, atl in enumerate(atl_grid):
            for k, consec in enumerate(consec_grid):
                for l, dmax in enumerate(dmax_grid):

                    if ctl >= CTL_GOAL:
                        V[i, j, k, l] = 0
                        continue

                    # --- rest day --- (earned ceiling l is monotonic, so it carries over)
                    rest_cost   = 1 + V_prev[ctl_rest_idx[i], atl_rest_idx[j], 0, l]
                    best_cost   = rest_cost
                    best_action = "rest"

                    # --- run actions (blocked if at max consecutive runs) ---
                    if consec < MAX_CONSEC_RUNS:
                        consec_n = consec + 1
                        # feasible = ATL ramp ok AND distance within DIST_RAMP of ceiling
                        feasible_mask = run_feasible[j] & dist_ok[l]    # (A,)

                        # actions that reach the goal cost exactly 1
                        goal_mask = reaches_goal[i] & feasible_mask     # (A,)
                        if goal_mask.any():
                            best_cost   = 1
                            best_action = action_list[np.argmax(goal_mask)]
                        else:
                            # look up V_prev for all feasible non-goal actions at once
                            nonfeasible = ~feasible_mask
                            ci = ctl_next_run_idx[i]                    # (A,)
                            ai = atl_next_run_idx[j]                    # (A,)
                            di = dmax_next_run_idx[l]                   # (A,)
                            costs = 1 + V_prev[ci, ai, consec_n, di]    # (A,)
                            costs[nonfeasible] = np.inf
                            best_a = int(np.argmin(costs))
                            if costs[best_a] < best_cost:
                                best_cost   = costs[best_a]
                                best_action = action_list[best_a]

                    V[i, j, k, l]      = best_cost
                    policy[i, j, k, l] = best_action

    changed = np.sum(V != V_prev)
    print(f"Iteration {iteration+1}: {changed} states updated")

    if changed == 0:
        print(f"Converged after {iteration+1} iterations!")
        break

print(f"All done!")
np.save("V.npy", V)
np.save("policy.npy", policy, allow_pickle=True)
np.save("dmax_grid.npy", dmax_grid)
print("Saved V.npy, policy.npy, dmax_grid.npy, trimp_cache.pkl")




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