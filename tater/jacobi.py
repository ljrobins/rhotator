import taichi as ti
from math import pi as PI

@ti.pyfunc
def agm(a, b):
    c = 1.0
    while c > 1e-13:
        an = 1/2*(a + b)
        bn = ti.sqrt(a*b)
        c = 1/2*(an - bn)
        a, b = an, bn
    return a

@ti.pyfunc
def sn_cn_dn_am(x, k):
    # https://dlmf.nist.gov/22.20#ii.p1
    a = ti.Vector([1.0, 0, 0, 0, 0, 0], dt=ti.f32)
    kp = ti.sqrt(1-k**2)
    b = ti.Vector([kp, 0, 0, 0, 0, 0], dt=ti.f32)
    c = ti.Vector([k, 0, 0, 0, 0, 0], dt=ti.f32)
    for n in range(len(a)-1):
        a[n+1] = 1/2*(a[n] + b[n])
        b[n+1] = ti.sqrt(a[n]*b[n])
        c[n+1] = 1/2*(a[n] - b[n])
    
    N = len(a)-2
    phi = ti.Vector([0, 0, 0, 0, 2**N*a[N]*x], dt=ti.f32)

    while N > 0:
        phi[N-1] = 1/2*(phi[N] + ti.asin(c[N]/a[N] * ti.sin(phi[N])))
        N -= 1

    sn = ti.sin(phi[0])
    cn = ti.cos(phi[0])
    dn = ti.cos(phi[0]) / ti.cos(phi[1] - phi[0])

    am = ti.asin(sn)

    K = PI / (2*agm(1.0,kp))  #TODO: fix with this https://dlmf.nist.gov/19.8#i.p1 to fufill https://dlmf.nist.gov/22.20

    n = ti.floor((x-K)/K/2)+1.0
    direction = -2*(n % 2)+1

    if n != 0.0:
        am = n*PI + direction*am

    return sn, cn, dn, am

@ti.pyfunc
def rf(x, y, z):
    for _ in range(4):
        lam = ti.sqrt(x)*ti.sqrt(y) + ti.sqrt(y)*ti.sqrt(z) + ti.sqrt(z)*ti.sqrt(x)
        x = (x+lam)/4
        y = (y+lam)/4
        z = (z+lam)/4
    A = (x + y + z) / 3
    dx = 1 - x / A
    dy = 1 - y / A
    dz = 1 - z / A
    E2 = dx*dy + dy*dz + dz*dx
    E3 = dx*dy*dz
    rf = 1 / ti.sqrt(A) * (
        1 - 1/10*E2 + 1/14*E3 + 1/24*E2**2 - 3/44*E2*E3 - 5/208*E2**3 + 3/104*E3**2 + 1/16*E2**2*E3
    )
    return rf


@ti.pyfunc
def rc(x, y):
    errtol = 1e-3
    xn = x
    yn = y

    sn = 0.0
    mu = 0.0
    while ( True ):

        mu = ( xn + yn + yn ) / 3.0
        sn = ( yn + mu ) / mu - 2.0

        if ( ti.abs ( sn ) < errtol ):
            break

        lamda = 2.0 * ti.sqrt ( xn ) * ti.sqrt ( yn ) + yn
        xn = ( xn + lamda ) * 0.25
        yn = ( yn + lamda ) * 0.25
    
    c1 = 1.0 / 7.0
    c2 = 9.0 / 22.0
    s = sn * sn * ( 0.3 + sn * ( c1 + sn * ( 0.375 + sn * c2 ) ) )
    value = ( 1.0 + s ) / ti.sqrt ( mu )
    return value


@ti.pyfunc
def rj(x, y, z, p):
    errtol = 1e-3
    epslon = 0.0
    xn = x
    yn = y
    zn = z
    pn = p
    sigma = 0.0
    power4 = 1.0

    xndev = 0.0
    yndev = 0.0
    zndev = 0.0
    pndev = 0.0
    mu = 0.0
    while True:
        mu = ( xn + yn + zn + pn + pn ) * 0.2
        xndev = ( mu - xn ) / mu
        yndev = ( mu - yn ) / mu
        zndev = ( mu - zn ) / mu
        pndev = ( mu - pn ) / mu
        epslon = ti.max ( ti.abs ( xndev ), \
        ti.max ( ti.abs ( yndev ), \
        ti.max ( ti.abs ( zndev ), ti.abs ( pndev ) ) ) )

        if epslon < errtol:
           break

        xnroot = ti.sqrt ( xn )
        ynroot = ti.sqrt ( yn )
        znroot = ti.sqrt ( zn )
        lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
        alfa = pn * ( xnroot + ynroot + znroot ) + xnroot * ynroot * znroot
        alfa = alfa * alfa
        beta = pn * ( pn + lamda ) * ( pn + lamda )
        value = rc( alfa, beta)
        sigma = sigma + power4 * value

        power4 = power4 * 0.25
        xn = ( xn + lamda ) * 0.25
        yn = ( yn + lamda ) * 0.25
        zn = ( zn + lamda ) * 0.25
        pn = ( pn + lamda ) * 0.25

    c1 = 3.0 / 14.0
    c2 = 1.0 / 3.0
    c3 = 3.0 / 22.0
    c4 = 3.0 / 26.0
    ea = xndev * ( yndev + zndev ) + yndev * zndev
    eb = xndev * yndev * zndev
    ec = pndev * pndev
    e2 = ea - 3.0 * ec
    e3 = eb + 2.0 * pndev * ( ea - ec )
    s1 = 1.0 + e2 * ( - c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3 )
    s2 = eb * ( 0.5 * c2 + pndev * ( - c3 - c3 + pndev * c4 ) )
    s3 = pndev * ea * ( c2 - pndev * c3 ) - c2 * pndev * ec
    value = 3.0 * sigma + power4 * ( s1 + s2 + s3 ) / ( mu * ti.sqrt ( mu ) )
    return value


@ti.pyfunc
def elliptic_pi_complete(n: float, ksquared: float) -> float:
    """Computes the complete elliptic integral of the third kind based on Carlson symmetric forms :math:`R_f, R_j`

    :param n: :math:`n` parameter of the integral
    :type n: float
    :param ksquared: :math:`k^2` parameter of the integral
    :type ksquared: float
    :return: :math:`\\pi(n, \\phi, k^2)`
    :rtype: float
    """

    return rf(0.0, 1 - ksquared, 1.0) + 1 / 3 * n * rj(0.0, 1 - ksquared, 1.0, 1.0 - n)

@ti.pyfunc
def elliptic_pi_incomplete(
    n: float, phi: float, ksquared: float
) -> float:
    """Computes the incomplete elliptic integral of the third kind based on Carlson symmetric forms :math:`R_f, R_j`

    :param n: :math:`n` parameter of the integral
    :type n: float [nx1]
    :param phi: :math:`\\phi`
    :type phi: float [nx1]
    :param ksquared: :math:`k^2` parameter of the integral
    :type ksquared: float [nx1]
    :return: :math:`\\Pi(n, \\phi, k^2)`
    :rtype: float [nx1]
    """
    c = ti.floor((phi + PI / 2) / PI)
    phi_shifted = phi - c * PI
    onemk2_sin2phi = 1 - ksquared * ti.sin(phi_shifted) ** 2
    cos2phi = ti.cos(phi_shifted) ** 2
    sin3phi = ti.sin(phi_shifted) ** 3
    n_sin2phi = n * ti.sin(phi_shifted) ** 2

    periodic_portion = 2 * c * elliptic_pi_complete(n, ksquared)

    return (
        ti.sin(phi_shifted) * rf(cos2phi, onemk2_sin2phi, 1.0)
        + 1 / 3 * n * sin3phi * rj(cos2phi, onemk2_sin2phi, 1.0, 1.0 - n_sin2phi)
        + periodic_portion
    )

@ti.pyfunc
def elliptic_inc_ek ( phi, k ):
  cp = ti.cos ( phi )
  sp = ti.sin ( phi )
  x = cp * cp
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp )
  z = 1.0

  value1 = rf ( x, y, z )
  value2 = rd ( x, y, z )

  value = sp * value1 - k ** 2 * sp ** 3 * value2 / 3.0
  return value

@ti.pyfunc
def rd ( x, y, z ):
  
  errtol = 1e-3

  xn = x
  yn = y
  zn = z
  sigma = 0.0
  power4 = 1.0

  while ( True ):

    mu = ( xn + yn + 3.0 * zn ) * 0.2
    xndev = ( mu - xn ) / mu
    yndev = ( mu - yn ) / mu
    zndev = ( mu - zn ) / mu
    epslon = max ( abs ( xndev ), max ( abs ( yndev ), abs ( zndev ) ) )

    if ( epslon < errtol ):
      c1 = 3.0 / 14.0
      c2 = 1.0 / 6.0
      c3 = 9.0 / 22.0
      c4 = 3.0 / 26.0
      ea = xndev * yndev
      eb = zndev * zndev
      ec = ea - eb
      ed = ea - 6.0 * eb
      ef = ed + ec + ec
      s1 = ed * ( - c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef )
      s2 = zndev  * ( c2 * ef + zndev * ( - c3 * ec + zndev * c4 * ea ) )
      value = 3.0 * sigma  + power4 * ( 1.0 + s1 + s2 ) / ( mu * ti.sqrt ( mu ) )

      return value

    xnroot = ti.sqrt ( xn )
    ynroot = ti.sqrt ( yn )
    znroot = ti.sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    sigma = sigma + power4 / ( znroot * ( zn + lamda ) )
    power4 = power4 * 0.25
    xn = ( xn + lamda ) * 0.25
    yn = ( yn + lamda ) * 0.25
    zn = ( zn + lamda ) * 0.25

@ti.pyfunc
def elliptic_inc_fm ( phi, m ):
  cp = ti.cos ( phi )
  sp = ti.sin ( phi )
  x = cp * cp
  y = 1.0 - m * sp ** 2
  z = 1.0
  value = rf (x, y, z)
  value = sp * value
  return value

