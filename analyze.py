import polars as pl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

t = np.linspace(0, 1, 30)
lc_mean = np.sin(10 * t) / 2 + 1
sigma_obs = 0.1 * lc_mean
lc_obs = lc_mean + np.random.normal(loc=np.zeros_like(t), scale=sigma_obs)

lc_obs_norm = np.linalg.norm(lc_obs)

nsig = 2.0
n_each = 20
opacity = 0.5

methods = ["log-likelihood", "l2"]
cols = ["m", "c", "g"]

for method, col in zip(methods, cols):
    res = pl.read_parquet(f"irregular-{method}.parquet")

    print(
        f'{method} 5, 50, 95 percentile iterations: {np.percentile(res["iterations"].to_numpy(), [5, 50, 95])}'
    )

    lcs = res["lcs"].to_numpy().T.copy()
    # lcs /= np.linalg.norm(lcs, axis=0, keepdims=True)
    # lcs *= lc_obs_norm

    plt.plot(
        t,
        lcs[:, 0],
        c=col,
        alpha=opacity,
        label=f"{method}",
    )
    plt.plot(t, lcs[:, 1:n_each], c=col, alpha=opacity)
plt.plot(t, lc_mean, linewidth=2, label="True mean", c="k", alpha=1.0)
plt.fill_between(
    t,
    lc_mean - nsig * sigma_obs,
    lc_mean + nsig * sigma_obs,
    alpha=0.2,
    label=f"True uncertainty $\pm {nsig} \sigma$",
)
plt.scatter(t, lc_obs, s=25, label="Observations", c="k", marker="+")
plt.xlabel("Timestep")
plt.ylabel("Light curve value")
plt.title("Loss Functions Compared")
plt.legend()
plt.grid()
plt.show()
