import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler



# ==============================================================================
#  CONFIGURATION
# ==============================================================================

CSV_PATH    = "gp_ready_runs.csv"   # output from compute_ctl_atl.py

# CTL/ATL dropped as inputs — they don't predict TRIMP independently, and belong in the DP state/transition function, not the surrogate.
TARGET_COL    = "trimp" # Relative Effort - cleaner signal that HR for predicting training load.
FEATURE_COLS  = ["distance_km", "pace_min_per_km", "elevation_gain_m"] # found that distance really affects TRIMP. Pace and elevation are lightly correlated (due to nature of data)
FEATURE_NAMES = ["Distance (km)", "Pace (min/km)", "Elevation (m)"]

# Kernel hyperparameters
LENGTH_SCALE_0      = 1.0           # larger values here correlate to smoother functions
LENGTH_SCALE_BOUNDS = (1e-3, 1e3)   # sklearn's optimizer takes over and finds best value for length scale within these bounds
N_RESTARTS          = 10
NOISE_LEVEL         = 1e-1          # Relative Effort has real variance — keep noise generous

USE_CONSTANT_MEAN   = False

N_PLOT_PTS          = 300

MAX_PACE_MIN_PER_KM = 10.0          # Pace sanity filter

# Tune this to match your athlete's weekly load targets.
# A typical easy run for this athlete is ~15-20 RE; a long run ~50-80.
TRIMP_MAX = 200.0                    # DP constraint: maximum allowable predicted Relative Effort per run.


# ==============================================================================
#  DATA LOADING
# ==============================================================================

def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (n, 3) array  [distance_km, pace_min_per_km, elevation_gain_m]
    y : (n,)   Relative Effort (TRIMP)
    """
    print(f"Loading '{path}' ...")
    df = pd.read_csv(path, parse_dates=["date"])

    # Drop rows missing TRIMP or pace artifacts
    df = df.dropna(subset=[TARGET_COL])                     # when TRIMP is missing
    df = df[df["pace_min_per_km"] <= MAX_PACE_MIN_PER_KM]   # when max pace is exceeded
    df = df[df[TARGET_COL] > 0]                             # drop zero-effort entries

    # Fill missing elevation with elevation=0 (flat run assumption)
    df["elevation_gain_m"] = df["elevation_gain_m"].fillna(0.0)

    # Prints how many usable activities we have
    print(f"  {len(df):,} clean rows  "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    
    print("--")

    # Prints TRIMP mean, std, and range
    print(f"  TRIMP — mean: {df[TARGET_COL].mean():.1f}, "
          f"std: {df[TARGET_COL].std():.1f}, "
          f"range: {df[TARGET_COL].min():.0f}–{df[TARGET_COL].max():.0f}")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    return X, y


# ==============================================================================
#  KERNEL FACTORY
# ==============================================================================

def build_kernel(n_features: int,
                 length_scale_0: float = 1.0,
                 length_scale_bounds: tuple = (1e-3, 1e3),
                 noise_level: float = 1e-1):
    """
    ARD RBF kernel with WhiteKernel noise.
    One length-scale per input so the GP learns that distance matters
    more than pace for predicting Relative Effort.
    """
    ls0 = np.full(n_features, length_scale_0) # tldr [1.0, 1.0, 1.0] for distance, pace, elevation: Given each feature a length-scale that controls how quickly the similarity between two points decays as the features change

    rbf = RBF(length_scale=ls0,
              length_scale_bounds=length_scale_bounds) # initializes the RBF kernel object with length-scales and search bounds

    noise = WhiteKernel(noise_level=noise_level,
                        noise_level_bounds=(1e-3, 1e1)) # configures the noise kernel object before fitting

    # if we didnt want to use a zero mean, we would add that in here

    return rbf + noise


# ==============================================================================
#  SURROGATE MODEL
# ==============================================================================

def build_gp_surrogate(X_train: np.ndarray,
                       y_train: np.ndarray,
                       label: str = "TRIMP",
                       n_restarts: int = 10,
                       length_scale_0: float = 1.0,
                       length_scale_bounds: tuple = (1e-3, 1e3),
                       noise_level: float = 1e-1,
                       use_constant_mean: bool = False):
    """
    Fit a GP surrogate to predict Relative Effort from run features.

    Returns
    -------
    gpr      : fitted GaussianProcessRegressor
    x_scaler : fitted StandardScaler for X
    y_scaler : fitted StandardScaler for y
    """

    # Scaling TLDR: since distance, pace, and elevations have simply different numbers, this is scaling to 0-1 so that elevation doesn't domminate just because it has a higher raw number  
    # -----
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_train)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    # -----

    kernel = build_kernel(
        n_features          = X_train.shape[1],
        length_scale_0      = length_scale_0,
        length_scale_bounds = length_scale_bounds,
        noise_level         = noise_level
    )

    gpr = GaussianProcessRegressor(
        kernel              = kernel,
        n_restarts_optimizer= n_restarts,
        normalize_y         = False,
        alpha               = 0.0,
    )

    print("--")
    print(f"Fitting GP surrogate ({label}) on {len(y_train)} points ...")
    gpr.fit(X_scaled, y_scaled)
    print("--")
    print(f"  Optimised kernel : {gpr.kernel_} where order is [distance, pace, elevation]")
    print("--")
    # This tests kernels against each other, swap RBF for Matern or smth of the sort to see. If the number becomes less negative, that is a better kernel
    print(f"  Log-marginal-likelihood: {gpr.log_marginal_likelihood_value_:.4f}")

    try:
        ls_values = gpr.kernel_.k1.length_scale
    except AttributeError:
        ls_values = gpr.kernel_.k1.k2.length_scale
    for name, ls_val in zip(FEATURE_NAMES, np.atleast_1d(ls_values)):
        print(f"    length-scale  {name:<22}: {ls_val:.4f}")

    return gpr, x_scaler, y_scaler


# ==============================================================================
#  PREDICTION HELPER
# ==============================================================================

def gp_predict(gpr: GaussianProcessRegressor,
               X_new: np.ndarray,
               x_scaler: StandardScaler,
               y_scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X_new : (m, 3)  [distance_km, pace_min_per_km, elevation_gain_m]
            in original (unscaled) units
    Returns
    -------
    mu  : (m,) posterior mean TRIMP
    std : (m,) posterior std  TRIMP
    """
    # Scaling for the purposes of feeding into the GP
    X_scaled           = x_scaler.transform(X_new)
    mu_scaled, std_scaled = gpr.predict(X_scaled, return_std=True)

    sigma_y = y_scaler.scale_[0]
    mu_y    = y_scaler.mean_[0]
    mu  = mu_scaled * sigma_y + mu_y
    std = std_scaled * sigma_y
    return mu, std


# ==============================================================================
#  SLICE PLOTS
# ==============================================================================

# Only a visual aid for confirmation
def plot_gp_slices(gpr, X_train_orig, y_train_orig,
                   x_scaler, y_scaler,
                   feature_names=None, target_name="Relative Effort (TRIMP)",
                   n_pts=300, save_path="gp_surrogate_slices.png"):
    """
    One subplot per input feature, others held at training-set mean.
    """
    n_features = X_train_orig.shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    anchor = X_train_orig.mean(axis=0)

    fig, axes = plt.subplots(1, n_features,
                             figsize=(5 * n_features, 4),
                             sharey=False)
    if n_features == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        sweep = np.linspace(X_train_orig[:, d].min(),
                            X_train_orig[:, d].max(), n_pts)
        X_query = np.tile(anchor, (n_pts, 1))
        X_query[:, d] = sweep

        mu, std = gp_predict(gpr, X_query, x_scaler, y_scaler)
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std

        ax.scatter(X_train_orig[:, d], y_train_orig,
                   alpha=0.2, s=10, zorder=5, label="Training data")
        ax.plot(sweep, mu, label="GP mean")
        ax.fill_between(sweep, lower, upper, alpha=0.3, label="95% CI")
        ax.axvline(anchor[d], color="gray", linestyle=":", linewidth=1,
                   label="Other inputs @ mean")

        fixed_parts = [f"{feature_names[j]}={anchor[j]:.1f}"
                       for j in range(n_features) if j != d]
        ax.set_xlabel(feature_names[d])
        ax.set_ylabel(target_name)
        ax.set_title(f"Slice: {feature_names[d]}\n"
                     f"(fixed: {', '.join(fixed_parts)})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"GP Surrogate — {target_name}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved to '{save_path}'")


# ==============================================================================
#  FEASIBILITY CHECK  — call this inside your DP action loop
# ==============================================================================
# Self explanatory
def check_trimp_feasibility(gpr, action: np.ndarray,
                             x_scaler: StandardScaler,
                             y_scaler: StandardScaler,
                             trimp_max: float = TRIMP_MAX,
                             confidence: float = 1.96) -> tuple[bool, float, float]:
    x = np.array(action).reshape(1, -1)
    mu, std = gp_predict(gpr, x, x_scaler, y_scaler)
    mu, std = float(mu[0]), float(std[0])
    feasible = (mu + confidence * std) < trimp_max
    return feasible, mu, std


# ----------- CALL FUNCTION -----------

def train_surrogate(model_path: str = "gpr_surrogate.pkl"):
    """
    Fits the GP surrogate and returns everything the DP needs.
    Saves the fitted model to disk so subsequent calls load instantly.

    Returns
    -------
    gpr      : fitted GaussianProcessRegressor
    x_scaler : fitted StandardScaler for X
    y_scaler : fitted StandardScaler for y
    """
    import os
    if os.path.exists(model_path):
        print(f"Loading cached surrogate from '{model_path}' ...")
        return joblib.load(model_path)

    X, y = load_data(CSV_PATH)

    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(len(y))
    split = int(0.8 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    gpr, x_scaler, y_scaler = build_gp_surrogate(
        X_train, y_train, label="Relative Effort",
        n_restarts=N_RESTARTS, length_scale_0=LENGTH_SCALE_0,
        length_scale_bounds=LENGTH_SCALE_BOUNDS, noise_level=NOISE_LEVEL,
        use_constant_mean=USE_CONSTANT_MEAN,
    )

    joblib.dump((gpr, x_scaler, y_scaler), model_path)
    print(f"  Saved surrogate to '{model_path}'")
    return gpr, x_scaler, y_scaler


# ==============================================================================
#  MAIN
# ==============================================================================

# if __name__ == "__main__":

#     # 1. Load data
#     X, y = load_data(CSV_PATH)
#     print(f"  Features : {FEATURE_COLS}")
#     print(f"  X shape  : {X.shape}")

#     # 2. Train / test split
#     rng   = np.random.default_rng(seed=42)  # seed=42 means that the shuffling is the same every time - i.e. we can rerun this and the same results will be produced
#     idx   = rng.permutation(len(y))         # Shuffing our data points so that it does just train/test on the same set every time
#     split = int(0.8 * len(y))               # Splits 80/20, training/testing
#     X_train, X_test = X[idx[:split]], X[idx[split:]]
#     y_train, y_test = y[idx[:split]], y[idx[split:]]

#     # 3. Fit surrogate
#     gpr, x_scaler, y_scaler = build_gp_surrogate(
#         X_train, y_train, label="Relative Effort",
#         n_restarts          = N_RESTARTS,
#         length_scale_0      = LENGTH_SCALE_0,
#         length_scale_bounds = LENGTH_SCALE_BOUNDS,
#         noise_level         = NOISE_LEVEL,
#         use_constant_mean   = USE_CONSTANT_MEAN,
#     )

#     # 4. RMSE on held-out test set
#     mu_test, std_test = gp_predict(gpr, X_test, x_scaler, y_scaler)
#     rmse = np.sqrt(np.mean((mu_test - y_test) ** 2))
#     print(f"\n  Test RMSE : {rmse:.2f} RE units")
#     print(f"  Mean σ    : {std_test.mean():.2f} RE units")

#     # 4b. Validation
#     print("\n--- Validation ---")

#     # Calibration
#     within_95 = np.mean(np.abs(mu_test - y_test) < 1.96 * std_test)
#     print(f"Calibration: {within_95:.1%} of test points within 95% CI")

#     # Directional sanity — TRIMP should rise with distance and elevation
#     mean_pace = X_train[:, 1].mean()
#     mean_elev = X_train[:, 2].mean()
#     mean_dist = X_train[:, 0].mean()

#     _, short_re, _ = check_trimp_feasibility(gpr, [5.0,  mean_pace, mean_elev], x_scaler, y_scaler)
#     _, long_re,  _ = check_trimp_feasibility(gpr, [15.0, mean_pace, mean_elev], x_scaler, y_scaler)
#     _, flat_re,  _ = check_trimp_feasibility(gpr, [mean_dist, mean_pace, 0.0],   x_scaler, y_scaler)
#     _, hilly_re, _ = check_trimp_feasibility(gpr, [mean_dist, mean_pace, 200.0], x_scaler, y_scaler)
#     _, easy_re,  _ = check_trimp_feasibility(gpr, [mean_dist, 6.0, mean_elev],   x_scaler, y_scaler)
#     _, hard_re,  _ = check_trimp_feasibility(gpr, [mean_dist, 4.0, mean_elev],   x_scaler, y_scaler)

#     print(f"Short (5km)  vs Long (15km)  @ mean pace+elev : {short_re:.1f} vs {long_re:.1f} RE  ({'✓' if long_re > short_re else '✗'})")
#     print(f"Flat (0m)    vs Hilly (200m) @ mean dist+pace : {flat_re:.1f} vs {hilly_re:.1f} RE  ({'✓' if hilly_re > flat_re else '✗'})")
#     print(f"Easy (6:00)  vs Hard (4:00)  @ mean dist+elev : {easy_re:.1f} vs {hard_re:.1f} RE  ({'✓' if hard_re > easy_re else '✗'})")
#     print("------------------\n")

#     # 5. Slice plots
#     plot_gp_slices(gpr, X_train, y_train, x_scaler, y_scaler,
#                    feature_names=FEATURE_NAMES,
#                    target_name="Relative Effort (TRIMP)",
#                    save_path="gp_slices_trimp.png")
