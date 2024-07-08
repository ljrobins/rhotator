import numpy as np
from scipy.special import ellipj, elliprj, elliprf, ellipeinc
import tater


def test_sn_cn_dn_am():
    x = 0.2
    k = 0.9
    residual = np.array(tater.sn_cn_dn_am(x, k)) - np.array(ellipj(x, k**2))
    assert np.all(np.abs(residual) < 1e-13)


print(tater.sn_cn_dn_am(0.2, 0.9))
print(ellipj(0.2, 0.9**2))

def test_rf():
    assert abs(tater.rf(1, 2, 4) - elliprf(1,2,4)) < 1e-13


def test_rj():
    assert abs(tater.rj(1,2,4,1.0) - elliprj(1,2,4,1.0)) < 1e-13

print(tater.rj(1,2,4,1.0), elliprj(1,2,4,1.0))
endd


def test_ellip_e_incomplete():
    assert abs(ellipeinc(0.4, 0.3**2) - tater.elliptic_inc_ek(0.4, 0.3)) < 1e-13

# %% 
# Elliptic integrals of the third kind


print("PIs")
print(tater.elliptic_pi_complete(0.5, 0.5))
print(tater.elliptic_pi_incomplete(0.1, 0.5, 0.5))


