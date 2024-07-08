import taichi as ti
from .jacobi import sn_cn_dn_am, elliptic_pi_incomplete, elliptic_inc_fm

@ti.pyfunc
def quat_inv(q: ti.math.vec4) -> ti.math.vec4:
    """Finds the quaternion inverse (conjugate for unit quaternions)

    :param q: Input quaternion 
    :type q: ti.math.vec4
    :return: Inverse quaternion
    :rtype: ti.math.vec4
    """
    qinv = -q
    qinv[3] = -qinv[3]
    return qinv

@ti.pyfunc
def quat_add(q1: ti.math.vec4, q2: ti.math.vec4) -> ti.math.vec4:
    """Adds (multiplies) quaternions together such that quat_add(inv_quat(q1), q2) gives the rotation between q1 and q2

    :param q1: Quaternion 12
    :type q1: ti.math.vec4
    :param q2: Quaternion 23
    :type q2: ti.math.vec4
    :return: Quaternion 13
    :rtype: ti.math.vec4
    """
    q2dmat = ti.math.mat4(q2[3], q2[2], -q2[1], q2[0],
            -q2[2], q2[3], q2[0], q2[1],
            q2[1], -q2[0], q2[3], q2[2],
            -q2[0], -q2[1], -q2[2], q2[3])
    
    q12 = q2dmat @ q1
    return q12

@ti.pyfunc
def torque_free_attitude(
    quat0: ti.math.vec4,
    omega0: ti.math.vec3,
    omega: ti.math.vec3,
    itensor: ti.math.vec3,
    teval: float,
    ksquared: float,
    tau0: float,
    tau_dot: float,
    is_sam: bool,
) -> ti.math.vec4:
    """Analytically propagates orientation under torque-free rigid-body motion

    :param quat0: Quaternion from the body frame to inertial at initial time step
    :type quat0: ti.math.vec4
    :param omega: Body angular velocity vector at `teval` [rad/s]
    :type omega: ti.math.vec3
    :param itensor: Principal inertia tensor diagonal satisfying `itensor[0,0] <= itensor[1,1] <= itensor[2,2]` [kg m^2]
    :type itensor: ti.math.vec3
    :param teval: Time to output attitude soliution at, where :math:`t=0` corresponds to `quat0` and `omega[0,:]`
    :type teval: float
    :param ksquared: :math:`k^2` parameter from the angular velocity propagation
    :type ksquared: float
    :param tau0: Initial value of :math:`\tau` from the angular velocity propagation
    :type tau0: float
    :param tau_dot: Rate of change of :math:`\tau` from the angular velocity propagation
    :type tau_dot: float
    :param is_sam: Indicates the rotation mode. 0 = long-axis mode (LAM), 1 = short-axis mode (SAM)
    :type is_sam: bool
    :return: Propagated quaternions at `teval`
    :rtype: ti.math.vec4
    """

    # Compute angular momentum vector and direction
    hvec = itensor * omega
    hvec0 = itensor * omega0
    hmag = hvec.norm()
    hhat = hvec0.normalized()

    # separate itensor components
    ix = itensor[0]
    iy = itensor[1]
    iz = itensor[2]

    # separate out the angular velocity components
    wx = omega[0]
    wy = omega[1]
    wz = omega[2]

    # get initial nutation angle
    theta0 = 0.0
    if is_sam:
        theta0 = ti.acos(hhat[2])
    else:
        theta0 = ti.acos(hhat[0])
    

    (cT, sT, cT2, sT2) = (
        ti.cos(theta0),
        ti.sin(theta0),
        ti.cos(theta0 / 2),
        ti.sin(theta0 / 2),
    )

    # get initial spin angle
    psi0 = 0.0
    if is_sam:
        psi0 = ti.atan2(hhat[0] / sT, hhat[1] / sT)
    else:
        psi0 = ti.atan2(hhat[2] / sT, -hhat[1] / sT)

    (cS, sS) = (ti.cos(psi0), ti.sin(psi0))
    (cS2, sS2) = (ti.cos(psi0 / 2), ti.sin(psi0 / 2))

    # get initial precession angle
    A = (cT + 1) * cT2 * cS2 + sT * sT2 * (cS * cS2 + sS * sS2)
    B = -(cT + 1) * cT2 * sS2 + sT * sT2 * (cS * sS2 - sS * cS2)
    C = ti.sqrt(2 * (cT + 1))

    t_i = (B - ti.sqrt(ti.max(0, A**2 + B**2 - C**2))) / (A + C)
    phi0 = 4 * ti.atan2(t_i, 1.0)

    # compute the nutation and spin angles at each time step
    theta, psi = 0.0, 0.0
    if is_sam:
        theta = ti.acos(iz * wz / hmag)
        psi = ti.atan2(ix * wx, iy * wy)
    else:
        theta = ti.acos(ix * wx / hmag)
        psi = ti.atan2(iz * wz, -iy * wy)

    # get Lambda value used to compute phi
    phi_denom = 0.0
    if is_sam:
        phi_denom = -iz * (ix - iy) / (ix * (iy - iz))
    else:
        phi_denom = -ix * (iz - iy) / (iz * (iy - ix))

    k = ti.sqrt(ksquared)
    amp0 = sn_cn_dn_am(tau0, k)[3]
    phi_lambda = -elliptic_pi_incomplete(phi_denom, amp0, ksquared)

    # compute the precession angles
    ampt = sn_cn_dn_am(tau_dot * teval + tau0, k)[3]
    phi_pi = elliptic_pi_incomplete(phi_denom, ampt, ksquared)

    phi = 0.0
    if is_sam:
        phi = (
            phi0
            + hmag / iz * teval
            + hmag * (iz - ix) / (tau_dot * ix * iz) * (phi_pi + phi_lambda)
        )
    else:
        phi = (
            hmag / ix * teval
            + phi0
            + hmag * (ix - iz) / (tau_dot * ix * iz) * (phi_pi + phi_lambda)
        )

    quat_r = quat_r_from_eas(hhat, phi, theta, psi, is_sam)
    quat_bi = quat_add(quat0, quat_inv(quat_r))
    return quat_bi


@ti.pyfunc
def quat_r_from_eas(
    hhat: ti.math.vec3,
    phi: float,
    theta: float,
    psi: float,
    is_sam: bool,
) -> ti.math.vec4:
    """Computes quaternion for attitude solution from Euler angle sequence

    :param hhat: Inertial angular momentum direction
    :type hhat: ti.math.vec3
    :param phi: Spin angle [rad]
    :type phi: float
    :param theta: Nutation angle [rad]
    :type theta: float
    :param psi: Precession angle [rad]
    :type psi: float
    :param is_sam: Whether the rotation is short axis mode (SAM)
    :type is_sam: bool
    :return: Computed quaternions
    :rtype: ti.math.vec4
    """
    hx = hhat.x
    hy = hhat.y
    hz = hhat.z
    (st2, ct2) = (ti.sin(theta / 2), ti.cos(theta / 2))
    (sm, sp, cm, cp) = (
        ti.sin((phi - psi) / 2),
        ti.sin((phi + psi) / 2),
        ti.cos((phi - psi) / 2),
        ti.cos((phi + psi) / 2),
    )

    q = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    if is_sam:
        q = (
            1
            / ti.sqrt(2 * (hz + 1))
            * ti.math.vec4(
                    -(hz + 1) * st2 * cm - ct2 * (hx * sp - hy * cp),
                    -(hz + 1) * st2 * sm - ct2 * (hx * cp + hy * sp),
                    -(hz + 1) * ct2 * sp + st2 * (hx * cm + hy * sm),
                    (hz + 1) * ct2 * cp + st2 * (hy * cm - hx * sm),
            )
        )
    else:
        q = (
            1
            / ti.sqrt(2 * (hx + 1))
            * ti.math.vec4(
                    -(hx + 1) * ct2 * sp + st2 * (hz * cm - hy * sm),
                    (hx + 1) * st2 * sm + ct2 * (hz * cp - hy * sp),
                    -(hx + 1) * st2 * cm - ct2 * (hy * cp + hz * sp),
                    (hx + 1) * ct2 * cp - st2 * (hy * cm + hz * sm),
            )
        )
    return q


@ti.pyfunc
def torque_free_angular_velocity(
    omega0: ti.math.vec3,
    itensor: ti.math.vec3,
    teval: float,
):
    """Analytically propogates angular velocity assuming no torque is being applied

    :param omega0: Initial angular velocity in body frame [rad/s]
    :type omega0: ti.math.vec3
    :param itensor: Principal moments of inertia on the diagonalsatisfying `i[2] < i[1] < i[0]` [kg m^2]
    :type itensor: ti.math.vec3
    :param teval: Times to compute angular velocity for after initial state time [seconds]
    :type teval: float
    :return: Angular velocity [rad/s]; :math:`k^2` term of elliptic integral, :math:`\\tau_0`, :math:`\\dot{\\tau}`; short-axis model (SAM) flag
    :rtype: Tuple[ti.math.vec3, float, float, float, bool]
    """
    wx0 = omega0.x
    wy0 = omega0.y
    wz0 = omega0.z
    # Angular momentum
    hvec = itensor * omega0
    hmag = hvec.norm()

    # Rotational kinetic energy
    T = 0.5 * omega0.dot(hvec)

    idyn = hmag**2 / (2 * T)  # Dynamic moment of inertia
    we = 2 * T / hmag  # Effective angular velocity

    ix = itensor.x
    iy = itensor.y
    iz = itensor.z

    is_sam = True
    if idyn < iy:  # For long axis mode rotation
        is_sam = False
        (ix, iz) = (iz, ix)
        (wx0, wz0) = (wz0, wx0)

    ibig_neg = 2 * int(wz0 > 0) - 1  # 1 if not negative, -1 else

    tau_dot = we * ti.sqrt(idyn * (idyn - ix) * (iz - iy) / (ix * iy * iz))
    ksquared = (iy - ix) * (iz - idyn) / ((iz - iy) * (idyn - ix))
    cos_phi = ti.sqrt(ix * (iz - ix) / (idyn * (iz - idyn))) * wx0 / we
    sin_phi = ti.sqrt(iy * (iz - iy) / (idyn * (iz - idyn))) * wy0 / we

    psi = ibig_neg * ti.atan2(sin_phi, cos_phi)
    tau0 = elliptic_inc_fm(psi, ksquared)
    tau = tau_dot * teval + tau0
    (sn, cn, dn, _) = sn_cn_dn_am(tau, ti.sqrt(ksquared))


    wx = we * ti.sqrt(idyn * (iz - idyn) / (ix * (iz - ix))) * cn
    wy = ibig_neg * we * ti.sqrt(idyn * (iz - idyn) / (iy * (iz - iy))) * sn
    wz = ibig_neg * we * ti.sqrt(idyn * (idyn - ix) / (iz * (iz - ix))) * dn

    if not is_sam:
        (wz, wx) = (wx, wz)

    omega = ti.math.vec3(wx, wy, wz)
    return (omega, ksquared, tau0, tau_dot, is_sam)
