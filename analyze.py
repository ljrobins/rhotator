import polars as pl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

t = np.linspace(0, 1, 30)
true_mean = np.sin(10 * t) / 2 + 1
lc_true = true_mean + np.random.normal(loc=np.zeros_like(t), scale=0.3)

n_each = 10
opacity = 0.2

for method in ["nnl", "l2"]:
    res = pl.read_parquet(f"saved_{method}.parquet")
    col = "c" if method == "nnl" else "m"

    plt.plot(
        t,
        res["lcs"].to_numpy()[0, :].T,
        c=col,
        alpha=opacity,
        label=f"Optimized: {method.upper()}",
    )
    plt.plot(t, res["lcs"].to_numpy()[1:n_each, :].T, c=col, alpha=opacity)
plt.scatter(t, lc_true, s=5, label="Observations", c="k")
plt.plot(t, true_mean, linewidth=2, label="True mean", c="k")
plt.xlabel("Timestep")
plt.ylabel("Light curve value")
plt.title("Loss Functions Compared")
plt.legend()
plt.grid()
plt.show()
