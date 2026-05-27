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
CTL_GOAL = 50
atl_max = 75
T_max = 140

# --- define grids ---
ctl_grid = np.arange(10, 56, 2)
atl_grid = np.arange(5, 76, 1)

n_ctl = len(ctl_grid)
n_atl = len(atl_grid)

# --- define action space ---
# possible_distances_array = np.linspace(0, 42, 43)
# possible_pace_array = np.linspace(4, 7, 19)
# possible_elevations_array = np.linspace(0, 300, 7)

possible_distances_array = np.linspace(0, 40, 10)
possible_pace_array = np.linspace(4, 7, 6)
possible_elevations_array = np.linspace(0, 300, 3)

action_array = np.array(list(product(possible_distances_array, possible_pace_array, possible_elevations_array)))

#  -- init value fuinction and policy --
V = np.full((n_ctl, n_atl), np.inf)
policy = np.full((n_ctl, n_atl), None, dtype=object)

# --- terminal condition ---
for i, ctl in enumerate(ctl_grid):
    if ctl >= CTL_GOAL:
        V[i, :] = 0


print("Pre-computing TRIMP for all actions...")
trimp_cache = {}
for action in action_array:
    ok, mu_trimp, std_trimp = check_trimp_feasibility(gpr, action, x_scaler, y_scaler)
    if ok and mu_trimp > 0:
        trimp_cache[tuple(action)] = mu_trimp
feasible_actions = [(action, trimp) for action, trimp in trimp_cache.items()]
print(f"  {len(feasible_actions)} feasible actions out of {len(action_array)}")








# --- diagnostic ---
print("\n=== DIAGNOSTIC ===")

# 1. TRIMP range
trimps = [t for _, t in feasible_actions]
print(f"Feasible action TRIMP range: min={min(trimps):.1f}, max={max(trimps):.1f}")

# 2. Can we ever reach CTL_GOAL continuously?
print(f"\nCTL_GOAL = {CTL_GOAL}")
print(f"K_CTL = {K_CTL:.4f}")
print(f"Max TRIMP = {max(trimps):.1f}")
for ctl_test in [35, 40, 45, 48, 49]:
    ctl_n = ctl_test + (max(trimps) - ctl_test) * K_CTL
    print(f"  From CTL={ctl_test}, best run -> CTL_next={ctl_n:.3f}  {'✓ REACHES GOAL' if ctl_n >= CTL_GOAL else '✗'}")

# 3. How many days of max effort to reach goal from current CTL?
print(f"\nSimulating max effort every day from CTL=37:")
ctl_sim = 37.0
for day in range(1, 200):
    ctl_sim = ctl_sim + (max(trimps) - ctl_sim) * K_CTL
    if ctl_sim >= CTL_GOAL:
        print(f"  Reaches CTL_GOAL={CTL_GOAL} on day {day}  (CTL={ctl_sim:.2f})")
        break
    if day % 20 == 0:
        print(f"  Day {day}: CTL={ctl_sim:.2f}")
else:
    print(f"  Never reaches CTL_GOAL={CTL_GOAL} in 200 days. Final CTL={ctl_sim:.2f}")

# 4. Check terminal states in V
print(f"\nTerminal states in V (V==0): {np.sum(V==0)}")
print(f"CTL grid values at terminal states: {ctl_grid[np.where(np.any(V==0, axis=1))[0]]}")

# 5. Check first iteration manually for one state
print(f"\nManual check — state CTL=48, ATL=20:")
ctl_test, atl_test = 48.0, 20.0
reachable_goal = []
for action, mu_trimp in feasible_actions:
    ctl_n = ctl_test + (mu_trimp - ctl_test) * K_CTL
    atl_n = atl_test + (mu_trimp - atl_test) * K_ATL
    if ctl_n >= CTL_GOAL:
        reachable_goal.append((action, mu_trimp, ctl_n))
print(f"  Actions that transition into goal: {len(reachable_goal)}")
if reachable_goal:
    for a, t, cn in reachable_goal[:3]:
        print(f"    TRIMP={t:.1f} -> CTL_next={cn:.3f}")
else:
    print(f"  None — max reachable CTL_next = {max(ctl_test + (t - ctl_test)*K_CTL for _,t in feasible_actions):.3f}")

print(f"Max distance in action space: {possible_distances_array.max()}")
print(f"Max TRIMP with 20km cap: {max(trimps):.1f}")

print("=== END DIAGNOSTIC ===\n")





for iteration in range(T_max):
    V_prev = V.copy()
    for i, ctl in enumerate(ctl_grid):
        for j, atl in enumerate(atl_grid):

            if ctl >= CTL_GOAL:
                V[i, j] = 0
                continue

            best_cost = np.inf
            best_action = None
            
            # --- rest day ---
            ctl_n = ctl + (0 - ctl) * K_CTL   # decays toward zero
            atl_n = atl + (0 - atl) * K_ATL   # decays toward zero
            
            ctl_n = np.clip(ctl_n, ctl_grid[0], ctl_grid[-1])
            atl_n = np.clip(atl_n, atl_grid[0], atl_grid[-1])

            ctl_n_idx = np.argmin(np.abs(ctl_grid - ctl_n))
            atl_n_idx = np.argmin(np.abs(atl_grid - atl_n))

            cost = 1 + V_prev[ctl_n_idx, atl_n_idx]
            if cost < best_cost:
                best_cost = cost
                best_action = "rest"

            # --- run actions ---
            for action, mu_trimp in feasible_actions:

                ctl_n = ctl + (mu_trimp - ctl) * K_CTL
                atl_n = atl + (mu_trimp - atl) * K_ATL

                if atl_n > atl_max:
                    continue

                # ramp constraint — ATL proxy for weekly load increase
                if atl_n > atl + 20:
                    continue

                if ctl_n >= CTL_GOAL:
                    cost = 1
                    if cost < best_cost:
                        best_cost = cost
                        best_action = action
                    continue


                ctl_n = np.clip(ctl_n, ctl_grid[0], ctl_grid[-1])
                atl_n = np.clip(atl_n, atl_grid[0], atl_grid[-1])

                ctl_n_idx = np.argmin(np.abs(ctl_grid - ctl_n))
                atl_n_idx = np.argmin(np.abs(atl_grid - atl_n))

                cost = 1 + V_prev[ctl_n_idx, atl_n_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
            
            V[i, j] = best_cost
            policy[i, j] = best_action

    # check convergence
    finite_mask = np.isfinite(V) & np.isfinite(V_prev)
    changed = np.sum(V[finite_mask] != V_prev[finite_mask])
    print(f"Iteration {iteration+1}: {changed} states updated")
    
    if changed == 0:
        print(f"Converged after {iteration+1} iterations!")
        break

print(f"All done!")
print(f"V found, V = {V}")
print(f"Best policy found, policy = {policy}")


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