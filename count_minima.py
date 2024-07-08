import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

x = dict(np.load('saved.npz'))

x = pl.DataFrame(x).filter(pl.col('status') == 2).sort('fun')
print(x)

xk = x['xk'].to_numpy()
x0 = x['x0'].to_numpy()
hn = np.linalg.norm(x['jac'].to_numpy(), axis=1)
# hess_invs = x['hess_inv'].to_numpy()
# hn = np.linalg.norm(hess_invs, ord=np.inf, axis=(1,2))
# plt.hist(hn, bins=np.geomspace(1e-2,1e0))
# plt.xscale('log')
# plt.show()

print(', '.join([str(x) for x in xk[0]]))
# endd

plt.style.use('dark_background')
plt.scatter(xk[:,2], xk[:,3], s=1, alpha=0.05)
# plt.xlim([-2,2])
# plt.ylim([-2,2])
plt.show()