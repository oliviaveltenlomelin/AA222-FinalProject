import numpy as np
import pandas as pd
import joblib

# --- constants (must match dp_script.py) ---
K_CTL           = 1 - np.exp(-1 / 42)
K_ATL           = 1 - np.exp(-1 / 7)
CTL_GOAL        = 60
MAX_CONSEC_RUNS = 2

ctl_grid    = np.arange(10, 65, 1)   # must match dp_script.py
atl_grid    = np.arange(5,  76, 1)
consec_grid = np.arange(0, MAX_CONSEC_RUNS + 1)
dmax_grid   = np.load("dmax_grid.npy")   # earned distance-ceiling bins (from dp_script.py)
DMAX_SEED   = 5.0                        # day-1 ceiling (must match dp_script.py)

# --- load saved DP outputs ---
V           = np.load("V.npy")
policy      = np.load("policy.npy", allow_pickle=True)
trimp_cache = joblib.load("trimp_cache.pkl")

# --- set your starting fitness here ---
ctl_start = 13.0
atl_start = 13.0

# --- rollout ---
ctl, atl, consec = ctl_start, atl_start, 0
dmax = DMAX_SEED                         # longest run done so far (monotonic ceiling)
plan = []

for day in range(200):
    ctl_idx    = np.argmin(np.abs(ctl_grid - ctl))
    atl_idx    = np.argmin(np.abs(atl_grid - atl))
    consec_idx = np.argmin(np.abs(consec_grid - consec))
    dmax_idx   = np.argmin(np.abs(dmax_grid - dmax))

    # goal is terminal in the DP (policy is None there); use the same grid
    # snapping the policy was built on so we report success instead of "stuck"
    if ctl_grid[ctl_idx] >= CTL_GOAL:
        print(f"Reached CTL goal on day {day+1}! (CTL={ctl:.1f})")
        break

    action = policy[ctl_idx, atl_idx, consec_idx, dmax_idx]

    if action is None:
        print(f"Day {day+1}: stuck in unreachable state (CTL={ctl:.1f}, ATL={atl:.1f}, consec={consec}, dmax={dmax:.1f})")
        break

    if action == "rest":
        dist, pace, elev, trimp = 0.0, 0.0, 0.0, 0.0
        ctl    = ctl + (0 - ctl) * K_CTL
        atl    = atl + (0 - atl) * K_ATL
        consec = 0
        # rest does not change the earned distance ceiling
    else:
        dist, pace, elev = action
        trimp  = trimp_cache[tuple(action)]
        ctl    = ctl + (trimp - ctl) * K_CTL
        atl    = atl + (trimp - atl) * K_ATL
        consec = consec + 1
        dmax   = max(dmax, dist)         # ceiling rises to the longest run done

    tsb = ctl - atl
    plan.append({
        "day":    day + 1,
        "action": "rest" if action == "rest" else f"{dist:.1f}km @ {pace:.2f}min/km +{elev:.0f}m",
        "trimp":  round(trimp, 1),
        "ctl":    round(ctl, 2),
        "atl":    round(atl, 2),
        "tsb":    round(tsb, 2),
    })

    if ctl >= CTL_GOAL:
        print(f"Reached CTL goal on day {day+1}!")
        break
else:
    print("Did not reach CTL goal within 200 days.")

df = pd.DataFrame(plan)
print(df.to_string(index=False))
df.to_csv("training_plan.csv", index=False)
print("Saved training_plan.csv")
