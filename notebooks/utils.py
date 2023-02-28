"""
These functions are all taken from the ILSI package.
Download https://github.com/ebeauce/ILSI if you are interested!
"""

import numpy as np
import sys

def A_phi_(principal_stresses, principal_directions):
    """Compute A_phi as defined by Simpson 1997.
    Parameters
    -----------
    principal_stresses: (3,) numpy.ndarray
        The three eigenvalues of the stress tensor, ordered
        from most compressive (sigma1) to least compressive (sigma3).
    principal_directions: (3, 3) numpy.ndarray
        The three eigenvectors of the stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    Returns
    --------
    A_phi: scalar, float
    """
    # first, find the stress axis closest to the vertical
    max_dip = 0.0
    for i in range(3):
        stress_dir = principal_directions[:, i]
        az, dip = get_bearing_plunge(stress_dir)
        if dip > max_dip:
            max_dip = dip
            vertical_stress_idx = i
    # n is the number of principal stresses larger than the "vertical" stress
    # n=0 for normal faulting
    # n=1 for strike-slip faulting
    # n=2 for reverse faulting
    n = np.sum(principal_stresses < principal_stresses[vertical_stress_idx])
    # compute the shape ratio
    R = R_(principal_stresses)
    # compute A_phi
    A_phi = (n + 0.5) + (-1.0) ** n * (0.5 - R)
    return A_phi


def R_(principal_stresses):
    """
    Computes the shape ratio R=(sig1-sig2)/(sig1-sig3).
    Parameters
    -----------
    pinricpal_stresses: numpy.ndarray or list
        Contains the three eigenvalues of the stress tensor
        ordered such that:
        `principal_stresses[0]` < `principal_stresses[1]` < `principal_stresses[2]`
        with `principal_stresses[0]` being the most compressional stress.
    Returns
    ---------
    shape_ratio: scalar float
    """
    return (principal_stresses[0] - principal_stresses[1]) / (
        principal_stresses[0] - principal_stresses[2]
    )


def forward_model(n_):
    """
    Build the forward modeling matrix ``G`` given a collection
    of fault normals.
    Parameters
    ------------
    n_: (n_earthquakes, 3) numpy.ndarray
        The i-th row n_ are the components of the i-th
        fault normal in the (north, west, up) coordinate
        system.
    Returns
    ---------
    G: (3 x n_earthquakes, 5) numpy.ndarray
        The forward modeling matrix giving the slip (shear stress)
        directions on the faults characterized by `n_`, given the 5
        elements of the deviatoric stress tensor.
    """
    n_earthquakes = n_.shape[0]
    G = np.zeros((n_earthquakes * 3, 5), dtype=np.float32)
    for i in range(n_earthquakes):
        ii = i * 3
        n1, n2, n3 = n_[i, :]
        G[ii + 0, 0] = n1 + n1 * n3**2 - n1**3
        G[ii + 0, 1] = n2 - 2.0 * n2 * n1**2
        G[ii + 0, 2] = n3 - 2.0 * n3 * n1**2
        G[ii + 0, 3] = n1 * n3**2 - n1 * n2**2
        G[ii + 0, 4] = -2.0 * n1 * n2 * n3
        G[ii + 1, 0] = n2 * n3**2 - n2 * n1**2
        G[ii + 1, 1] = n1 - 2.0 * n1 * n2**2
        G[ii + 1, 2] = -2.0 * n1 * n2 * n3
        G[ii + 1, 3] = n2 + n2 * n3**2 - n2**3
        G[ii + 1, 4] = n3 - 2.0 * n3 * n2**2
        G[ii + 2, 0] = n3**3 - n3 - n3 * n1**2
        G[ii + 2, 1] = -2.0 * n1 * n2 * n3
        G[ii + 2, 2] = n1 - 2.0 * n1 * n3**2
        G[ii + 2, 3] = n3**3 - n3 - n3 * n2**2
        G[ii + 2, 4] = n2 - 2.0 * n2 * n3**2
    return G


def normal_slip_vectors(strike, dip, rake, direction="inward"):
    """
    Determine the normal and the slip vectors of the
    focal mechanism defined by (strike, dip, rake).
    From Stein and Wysession 2002.
    N.B.: This is the normal of the FOOT WALL and the slip
    of the HANGING WALL w.r.t the foot wall. It means that the
    normal is an inward-pointing normal for the hanging wall,
    and an outward pointing-normal for the foot wall.
    The vectors are in the coordinate system (x1, x2, x3):
    x1: north
    x2: west
    x3: upward
    Parameters
    ------------
    strike: float
        Strike of the fault.
    dip: float
        Dip of the fault.
    rake: float
        Rake of the fault.
    direction: string, default to 'inward'
        If 'inward', returns the inward normal of the HANGING wall,
        which is the formula given in Stein and Wysession. Equivalently,
        this is the outward normal of the foot wall.
        If 'outward', returns the outward normal of the HANGING wall,
        or, equivalently, the inward normal of the hanging wall.
    Returns
    -----------
    n: (3) numpy.ndarray
        The fault normal.
    d: (3) numpy.ndarray
        The slip vector given as the direction of motion
        of the hanging wall w.r.t. the foot wall.
    """
    d2r = np.pi / 180.0
    strike = strike * d2r
    dip = dip * d2r
    rake = rake * d2r
    n = np.array(
        [-np.sin(dip) * np.sin(strike), -np.sin(dip) * np.cos(strike), np.cos(dip)]
    )
    if direction == "inward":
        # this formula already gives the inward-pointing
        # normal of the hanging wall
        pass
    elif direction == "outward":
        n *= -1.0
    else:
        print('direction should be either "inward" or "outward"')
        return
    # slip on the hanging wall
    d = np.array(
        [
            np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
            -np.cos(rake) * np.sin(strike)
            + np.sin(rake) * np.cos(dip) * np.cos(strike),
            np.sin(rake) * np.sin(dip),
        ]
    )
    return n, d


def round_cos(x):
    """Clip x so that it fits with the [-1,1] interval.
    If x is slightly outside the [-1,1] because of numerical
    imprecision, x is rounded, and can then be safely passed
    to arccos or arcsin. If x is truly outside of [-1,1], x
    is returned unchanged.
    Parameters
    -----------
    x: scalar, float
        Float variable that represents a cos or sin that
        is supposed to be within the [-1,1] interval.
    Returns
    -----------
    x_r: scalar, float
       A rounded version of x, if necessary.
    """
    if (abs(x) > 1.0) and (abs(x) < 1.005):
        return 1.0 * np.sign(x)
    else:
        return x


def strike_dip_rake(n, d):
    """
    Invert the relationships between strike/dip/rake
    and normal (n) and slip (d) vectors found in Stein.
    n and d are required to be given as the default format
    returned by normal_slip_vectors.
    Parameters
    -----------
    n: (3) numpy.ndarray
        The outward pointing normal of the FOOT wall.
    d: (3) numpy.ndarray
        The slip direction of the hanging wall w.r.t.
        the foot wall.
    Returns
    ---------
    strike: float
        Strike of the fault, in degress.
    dip: float
        Dip of the fault, in degrees.
    rake: float
        Rake of the fault, in degrees.
    """
    r2d = 180.0 / np.pi
    # ----------------
    # dip is straightforward:
    dip = np.arccos(round_cos(n[2]))
    sin_dip = np.sin(dip)
    if sin_dip != 0.0:
        # ----------------
        # strike is more complicated because it spans 0-360 degrees
        sin_strike = -n[0] / sin_dip
        cos_strike = -n[1] / sin_dip
        strike = np.arctan2(sin_strike, cos_strike)
        # ---------------
        # rake is even more complicated
        sin_rake = d[2] / sin_dip
        cos_rake = (d[0] - sin_rake * np.cos(dip) * sin_strike) / cos_strike
        rake = np.arctan2(sin_rake, cos_rake)
    else:
        print("Dip is zero! The strike and rake cannot be determined")
        # the solution is ill-defined, we can only
        # determine rake - strike
        cos_rake_m_strike = d[0]
        sin_rake_m_strike = d[1]
        rake_m_strike = np.arctan2(sin_rake_m_strike, cos_rake_m_strike)
        # fix arbitrarily the rake to zero
        rake = 0.0
        strike = -rake_m_strike
    return (strike * r2d) % 360.0, dip * r2d, (rake * r2d) % 360.0


def get_bearing_plunge(u, degrees=True, hemisphere="lower"):
    """
    The vectors are in the coordinate system (x1, x2, x3):
    x1: north
    x2: west
    x3: upward
    Parameters
    -----------
    u: (3) numpy.ndarray or list
        Vector for which we want the bearing (azimuth) and plunge.
    degrees: boolean, default to True
        If True, returns bearing and plunge in degrees.
        In radians otherwise.
    hemisphere: string, default to 'lower'
        Consider the intersection of the line defined by u
        with the lower hemisphere if `hemisphere` is 'lower', or
        with the upper hemisphere if `hemisphere` is 'upper'.
    Returns
    ---------
    bearing: float
        Angle between the north and the line.
    plunge: float
        Angle between the horizontal plane and the line.
    """
    r2d = 180.0 / np.pi
    if hemisphere == "lower" and u[2] > 0.0:
        # we need to consider the end of the line
        # that plunges downward and crosses the
        # lower hemisphere
        u = -1.0 * u
    elif hemisphere == "upper" and u[2] < 0.0:
        u = -1.0 * u
    # the trigonometric sense is the opposite of the azimuthal sense,
    # therefore we need a -1 multiplicative factor
    bearing = -1.0 * np.arctan2(u[1], u[0])
    # the plunge is measured downward from the end of the
    # line specified by the bearing
    # this formula is valid for p_axis[2] < 0
    plunge = np.arccos(round_cos(u[2])) - np.pi / 2.0
    if hemisphere == "upper":
        plunge *= -1.0
    if degrees:
        return (bearing * r2d) % 360.0, plunge * r2d
    else:
        return bearing, plunge


def stress_tensor_eigendecomposition(stress_tensor):
    """Compute the eigendecomposition of stress tensor.
    Parameters
    -----------
    stress_tensor: (3, 3) numpy.ndarray.
        The stress tensor for which to solve the
        eigenvalue problem.
    Returns
    -----------
    principal_stresses: (3,) numpy.ndarray.
        The three eigenvalues of the stress tensor, ordered
        from most compressive (sigma1) to least compressive (sigma3).
    principal_directions: (3, 3) numpy.ndarray.
        The three eigenvectors of the stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    """
    try:
        principal_stresses, principal_directions = np.linalg.eigh(stress_tensor)
    except np.linalg.LinAlgError:
        print(stress_tensor)
        sys.exit()
    # order = np.argsort(principal_stresses)[::-1]
    order = np.argsort(principal_stresses)
    # reorder from most compressive to most extensional
    # with tension positive convention
    # (note: principal_directions is the matrix a column-eigenvectors)
    principal_stresses = principal_stresses[order]
    principal_directions = check_right_handedness(principal_directions[:, order])
    return principal_stresses, principal_directions

def check_right_handedness(basis):
    """
    Make sure the matrix of column vectors forms
    a right-handed basis. This is particularly important
    when re-ordering the principal stress directions
    based on their eigenvalues.
    Parameters
    -----------
    basis: (3, 3) numpy.ndarray
        Matrix with column vectors that form the basis of interest.
    Returns
    ----------
    rh_basis: (3, 3) numpy.ndarray
        Matrix with column vectors that form the right-handed
        version of the input basis. One of the unit vectors
        might have been reversed in the process.
    """
    vector1 = basis[:, 0]
    vector2 = basis[:, 1]
    vector3 = np.cross(vector1, vector2)
    return np.stack([vector1, vector2, vector3], axis=1)