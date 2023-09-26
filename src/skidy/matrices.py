from __future__ import annotations

from sympy import Matrix, Identity, symbols, sin, cos, zeros, MutableDenseMatrix, Expr
import numpy as np
from numpy.typing import ArrayLike

def generalized_vectors(
    DOF: int, startindex: int=0, derivatives: bool=False
    ) -> tuple[MutableDenseMatrix, MutableDenseMatrix, MutableDenseMatrix]:
    """Generate symbolic generalized vectors q, qd and q2d and optional
    q3d and q4d.
    
    The symbols are named as follows:
        
        q0, q1, ....., qi for joint positions.
        dq0, dq1, ....., dqi for joint velocities.
        ddq0, ddq1, ....., ddqi for joint accelerations.
        dddq0, dddq1, ....., dddqi for joint jerks.
        ddddq0, ddddq1, ....., ddddqi for joint jounces.
        

    Args:
        DOF (int): Degrees of freedom.
        startindex (int, optional): Index of first joint. Defaults to 0.
        derivatives (bool, optional): return jerks and jounces too. 
            Defaults to False. 

    Returns:
        tuple(sympy.Matrix): 
            Generalized vectors: (q,dq,d2q) if derivative == False
            else: (q,dq,d2q,q3d,q4d) 
    """
    if DOF > 1:
        q = Matrix(symbols(" ".join(f"q{i}" for i in range(startindex,startindex+DOF))))
        qd = Matrix(symbols(" ".join(f"dq{i}" for i in range(startindex,startindex+DOF))))
        q2d = Matrix(symbols(" ".join(f"ddq{i}" for i in range(startindex,startindex+DOF))))
        q3d = Matrix(symbols(" ".join(f"dddq{i}" for i in range(startindex,startindex+DOF))))
        q4d = Matrix(symbols(" ".join(f"ddddq{i}" for i in range(startindex,startindex+DOF))))
    else:
        q = Matrix([symbols(" ".join(f"q{i}" for i in range(startindex,startindex+DOF)))])
        qd = Matrix([symbols(" ".join(f"dq{i}" for i in range(startindex,startindex+DOF)))])
        q2d = Matrix([symbols(" ".join(f"ddq{i}" for i in range(startindex,startindex+DOF)))])
        q3d = Matrix([symbols(" ".join(f"dddq{i}" for i in range(startindex,startindex+DOF)))])
        q4d = Matrix([symbols(" ".join(f"ddddq{i}" for i in range(startindex,startindex+DOF)))])
    if derivatives:
        return q, qd, q2d, q3d, q4d
    else: 
        return q, qd, q2d

def joint_screw(axis: list, vec: list=[0,0,0], revolute: bool=True) -> MutableDenseMatrix:
    """Get joint screw coordinates from joint axis and vector to joint.

    Args:
        axis (list): 
            Joint axis w.r.t. inertial frame.
        vec (list, optional): 
            Vector to joint axis from inertial frame for revolute joint. 
            Defaults to [0,0,0].
        revolute (bool, optional): 
            Revolute (True) or prismatic (False) joint. 
            Defaults to True.

    Returns:
        sympy.Matrix: joint screw coordinates.
    """
    e = Matrix(axis)
        
    if revolute:
        y = Matrix(vec)
        return Matrix([e, y.cross(e)])
    else:
        return Matrix([0,0,0,*e])
        

def symbolic_inertia_matrix(
    index: int | str="", pointmass: bool=False) -> MutableDenseMatrix:
    """Create 3 x 3 symbolic inertia matrix with auto generated variable names.

    Args:
        index (int or str): 
            postfix for variable name. Defaults to "".
        pointmass (bool): 
            Inertial matrix = I * Identity. Default to False.
    
    Returns:
        sympy.Matrix: Inertia matrix (3,3)
    """
    if pointmass:
        I = symbols(f"I{index}", real=1, constant=1)
        return I*Identity(3)
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = symbols(
        f"Ixx{index} Iyy{index} Izz{index} Ixy{index} Ixz{index} Iyz{index}")
    
    I = Matrix([[Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]])
    return I

def SE3AdjInvMatrix(C: MutableDenseMatrix) -> MutableDenseMatrix:
    """Compute Inverse of (6x6) Adjoint matrix for SE(3)

    Args:
        C (sympy.Matrix): SE(3) Pose.

    Returns:
        sympy.Matrix: Inverse of (6x6) Adjoint matrix
    """
    AdInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], 0, 0, 0],
                    [C[0, 1], C[1, 1], C[2, 1], 0, 0, 0],
                    [C[0, 2], C[1, 2], C[2, 2], 0, 0, 0],
                    [-C[2, 3]*C[1, 0]+C[1, 3]*C[2, 0], 
                        C[2, 3]*C[0, 0]-C[0, 3]*C[2, 0],
                        (-C[1, 3])*C[0, 0]+C[0, 3]*C[1, 0], 
                        C[0, 0], C[1, 0], C[2, 0]],
                    [-C[2, 3]*C[1, 1]+C[1, 3]*C[2, 1], 
                        C[2, 3]*C[0, 1]-C[0, 3]*C[2, 1],
                        (-C[1, 3])*C[0, 1]+C[0, 3]*C[1, 1], 
                        C[0, 1], C[1, 1], C[2, 1]],
                    [-C[2, 3]*C[1, 2]+C[1, 3]*C[2, 2], 
                        C[2, 3]*C[0, 2]-C[0, 3]*C[2, 2],
                        (-C[1, 3])*C[0, 2]+C[0, 3]*C[1, 2], 
                        C[0, 2], C[1, 2], C[2, 2]]])
    return AdInv

def SE3AdjMatrix(C: MutableDenseMatrix) -> MutableDenseMatrix:
    """Compute (6x6) Adjoint matrix for SE(3)

    Args:
        C (sympy.Matrix): SE(3) Pose.

    Returns:
    sympy.Matrix: (6x6) Adjoint matrix
    """
    Ad = Matrix([[C[0, 0], C[0, 1], C[0, 2], 0, 0, 0],
                    [C[1, 0], C[1, 1], C[1, 2], 0, 0, 0],
                    [C[2, 0], C[2, 1], C[2, 2], 0, 0, 0],
                    [-C[2, 3]*C[1, 0]+C[1, 3]*C[2, 0], 
                    -C[2, 3]*C[1, 1]+C[1, 3]*C[2, 1],
                    -C[2, 3]*C[1, 2]+C[1, 3]*C[2, 2], 
                    C[0, 0], C[0, 1], C[0, 2]],
                    [C[2, 3]*C[0, 0]-C[0, 3]*C[2, 0],  
                    C[2, 3]*C[0, 1]-C[0, 3]*C[2, 1],
                    C[2, 3]*C[0, 2]-C[0, 3]*C[2, 2], 
                    C[1, 0], C[1, 1], C[1, 2]],
                    [-C[1, 3]*C[0, 0]+C[0, 3]*C[1, 0], 
                    -C[1, 3]*C[0, 1]+C[0, 3]*C[1, 1],
                    -C[1, 3]*C[0, 2]+C[0, 3]*C[1, 2], 
                    C[2, 0], C[2, 1], C[2, 2]]])
    return Ad

def SE3adMatrix(X: MutableDenseMatrix) -> MutableDenseMatrix:
    """Compute (6x6) adjoint matrix for SE(3) 
        - also known as spatial cross product in the literature.

    Args:
        X (sympy.Matrix): (6x1) spatial vector.

    Returns:
        sympy.Matrix: (6x6) adjoint matrix
    """
    ad = Matrix([[0, -X[2, 0], X[1, 0], 0, 0, 0],
                 [X[2, 0], 0, -X[0, 0], 0, 0, 0],
                 [-X[1, 0], X[0, 0], 0, 0, 0, 0],
                 [0, -X[5, 0], X[4, 0], 0, -X[2, 0], X[1, 0]],
                 [X[5, 0], 0, -X[3, 0], X[2, 0], 0, -X[0, 0]],
                 [-X[4, 0], X[3, 0], 0, -X[1, 0], X[0, 0], 0]])
    return ad

def SE3Exp(XX: MutableDenseMatrix, t: float | Expr) -> MutableDenseMatrix:
    """compute exponential mapping for SE(3).

    Args:
        XX (sympy.Matrix): (6,1) spatial vector.
        t (sympy.Expr): rotation angle.

    Returns:
        sympy.Matrix: (4,4) SE(3) Pose.
    """
    X = XX.T
    xi = Matrix(X[0:3])
    eta = Matrix(X[3:6])
    xihat = Matrix([[0, -X[2], X[1]],
                    [X[2], 0, -X[0]],
                    [-X[1], X[0], 0]])
    R = Matrix(Identity(3)) + sin(t)*xihat + (1-cos(t))*(xihat*xihat)
    if xi == zeros(3, 1):
        p = eta * t
    else:
        p = (Matrix(Identity(3))-R)*(xihat*eta) + xi*(xi.T*eta)*t
    C = R.row_join(p).col_join(Matrix([0, 0, 0, 1]).T)
    return C

def SE3Inv(C: MutableDenseMatrix) -> MutableDenseMatrix:
    """Compute analytical inverse of exponential mapping for SE(3).

    Args:
        C (sympy.Matrix): (4,4) SE(3) Pose.

    Returns:
        sympy.Matrix: (4,4) Inverse of SE(3) Pose.
    """
    CInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], 
                    -C[0, 0]*C[0, 3]-C[1, 0]*C[1, 3]-C[2, 0]*C[2, 3]],
                    [C[0, 1], C[1, 1], C[2, 1], 
                    -C[0, 1]*C[0, 3]-C[1, 1]*C[1, 3]-C[2, 1]*C[2, 3]],
                    [C[0, 2], C[1, 2], C[2, 2], -C[0, 2] *
                        C[0, 3]-C[1, 2]*C[1, 3]-C[2, 2]*C[2, 3]],
                    [0, 0, 0, 1]])
    return CInv

def SO3Exp(axis: MutableDenseMatrix, angle: float | Expr) -> MutableDenseMatrix:
    """Compute exponential mapping for SO(3).

    Args:
        axis (sympy.Matrix): Rotation axis
        angle (double): Rotation angle

    Returns:
        sympy.Matrix: Rotation matrix
    """
    axis = Matrix(axis)
    xhat = Matrix([[0, -axis[2, 0], axis[1, 0]],
                    [axis[2, 0], 0, -axis[0, 0]],
                    [-axis[1, 0], axis[0, 0], 0]])
    R = Matrix(Identity(3)) + sin(angle) * xhat + (1-cos(angle))*(xhat*xhat)
    return R

def inertia_matrix(Ixx: float | Expr=0, Ixy: float | Expr=0, 
                   Ixz: float | Expr=0, Iyy: float | Expr=0, 
                   Iyz: float | Expr=0, Izz: float | Expr=0) -> MutableDenseMatrix:
    """Create 3 x 3 inertia matrix from independent inertia values.

    Args:
        Ixx (float or sympy.Expr): Inertia value I11. Defaults to 0.
        Ixy (float or sympy.Expr): Inertia value I12. Defaults to 0.
        Ixz (float or sympy.Expr): Inertia value I13. Defaults to 0.
        Iyy (float or sympy.Expr): Inertia value I22. Defaults to 0.
        Iyz (float or sympy.Expr): Inertia value I23. Defaults to 0.
        Izz (float or sympy.Expr): Inertia value I33. Defaults to 0.

    Returns:
        sympy.Matrix: Inertia matrix (3,3)
    """
    I = Matrix([[Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]])
    return I

def transformation_matrix(r: MutableDenseMatrix=Matrix(Identity(3)), 
                         t: MutableDenseMatrix | list=zeros(3, 1)) -> MutableDenseMatrix:
    """Build transformation matrix from rotation and translation.

    Args:
        r (sympy.Matrix): 
            SO(3) Rotation matrix (3,3). 
            Defaults to sympy.Matrix(Identity(3))
        t (sympy.Matrix | list): 
            Translation vector (3,1). Defaults to sympy.zeros(3,1)

    Returns:
        sympy.Matrix: Transformation matrix (4,4)
    """
    if type(t) is list:
        t = Matrix(t)
    T = Matrix(Identity(4))
    T[:3,:3] = r
    T[:3,3] = t
    return T

def mass_matrix_mixed_data(m: float | Expr, Theta: MutableDenseMatrix, 
                        COM: MutableDenseMatrix) -> MutableDenseMatrix:
    """Build mass-inertia matrix in SE(3) from mass, inertia and 
    center of mass information w.r.t. the body fixed frame.
    If the inertia is w.r.t. the COM use `mass_matrix_URDF_data` instead.

    Args:
        m (float | sympy.Expr): Mass.
        Theta (array_like): (3,3) Inertia tensor with respect to body-fixed frame.
        COM (array_like): Center of mass (3,1).

    Returns:
        sympy.Matrix: Mass-inertia matrix (6,6).
    """
    M = Matrix([[Theta[0, 0], Theta[0, 1], Theta[0, 2], 0, (-COM[2])*m, COM[1]*m],
                [Theta[0, 1], Theta[1, 1], Theta[1, 2], COM[2]*m, 0, (-COM[0])*m],
                [Theta[0, 2], Theta[1, 2], Theta[2, 2], (-COM[1])*m, COM[0]*m, 0],
                [0, COM[2]*m, (-COM[1])*m, m, 0, 0],
                [(-COM[2])*m, 0, COM[0]*m, 0, m, 0],
                [COM[1]*m, (-COM[0])*m, 0, 0, 0, m]])
    return M

def mass_matrix_URDF_data(m: float | Expr, Inertia: MutableDenseMatrix, 
                        COM: MutableDenseMatrix, R: MutableDenseMatrix = Matrix(Identity(3))) -> MutableDenseMatrix:
    """Build mass-inertia matrix in SE(3) from mass, inertia and 
    center of mass information, as written in URDF. This means, that 
    the Inertia tensor is with respect to the COM.
    If your inertia tensor is w.r.t. the body fixed frame, use
    `mass_matrix_mixed_data` instead.

    Args:
        m (float | sympy.Expr): Mass.
        Inertia (array_like): (3,3) Inertia tensor with respect to COM.
        COM (array_like): Center of mass (3,1).
        R (array_like): (3,3) SO3 rotation matrix of the COM frame to
            the body frame. Defaults to Identity.

    Returns:
        sympy.Matrix: Mass-inertia matrix (6,6).
    """
    Inertia = R * Inertia * R.T
    Inertia -= Matrix([
            [-m * (COM[1]**2 + COM[2]**2), m * COM[0] * COM[1], m * COM[0] * COM[2]],
            [m * COM[0] * COM[1], -m * (COM[0]**2 + COM[2]**2), m * COM[1] * COM[2]],
            [m * COM[0] * COM[2], m * COM[1] * COM[2], -m * (COM[0]**2 + COM[1]**2)]
        ])
    M = Matrix([[Inertia[0, 0], Inertia[0, 1], Inertia[0, 2], 0, (-COM[2])*m, COM[1]*m],
                [Inertia[0, 1], Inertia[1, 1], Inertia[1, 2], COM[2]*m, 0, (-COM[0])*m],
                [Inertia[0, 2], Inertia[1, 2], Inertia[2, 2], (-COM[1])*m, COM[0]*m, 0],
                [0, COM[2]*m, (-COM[1])*m, m, 0, 0],
                [(-COM[2])*m, 0, COM[0]*m, 0, m, 0],
                [COM[1]*m, (-COM[0])*m, 0, 0, 0, m]])
    return M

def mass_matrix_mixed_data_identification(m: float | Expr, Theta: MutableDenseMatrix, 
                        h: MutableDenseMatrix) -> MutableDenseMatrix:
    """Build mass-inertia matrix in SE(3) from mass, inertia and 
    first moment of mass, i.e. m*com_vector.

    Args:
        m (float): Mass.
        Theta (array_like): Inertia (3,3).
        h (array_like): First moment of mass, i.e. m*com_vector. (3,1).

    Returns:
        sympy.Matrix: Mass-inertia matrix (6,6).
    """
    M = Matrix([[Theta[0, 0], Theta[0, 1], Theta[0, 2], 0, 
                    (-h[2]), h[1]],
                [Theta[0, 1], Theta[1, 1], Theta[1, 2],
                    h[2], 0, (-h[0])],
                [Theta[0, 2], Theta[1, 2], Theta[2, 2],
                    (-h[1]), h[0], 0],
                [0, h[2], (-h[1]), m, 0, 0],
                [(-h[2]), 0, h[0], 0, m, 0],
                [h[1], (-h[0]), 0, 0, 0, m]])
    return M

def mass_matrix_to_parameter_vector(
    M: MutableDenseMatrix
    ) -> tuple[Expr,Expr,Expr,Expr,Expr,Expr,Expr,Expr,Expr,Expr]:
    """Get parameter vector for regressor matrix from mass-inertia matrix.
        M -> m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz
    
    Args:
        M (MutableDenseMatrix): (6,6) Mass-inertia matrix.
        
    Returns:
        tuple[Expr]: 
            (m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz).
    """
    m = M[3,3]
    I = M[:3,:3] 
    cxyz = Matrix([M[2,4], M[0,5], M[1,3]])
    return m, *cxyz, I[0,0], I[0,1], I[0,2], I[1,1], I[1,2], I[2,2]

def rpy_to_matrix(coords: list | MutableDenseMatrix) -> MutableDenseMatrix:
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Args:
        coords (MutableDenseMatrix | list): 
            The (3,) roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns:
        MutableDenseMatrix: 
            The corresponding homogenous 3x3 rotation matrix.        
    """
    c3 = cos(coords[0])
    c2 = cos(coords[1])
    c1 = cos(coords[2])
    s3 = sin(coords[0])
    s2 = sin(coords[1])
    s1 = sin(coords[2])
    return Matrix([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ])

def xyz_rpy_to_matrix(xyz_rpy: list | MutableDenseMatrix) -> MutableDenseMatrix:
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

    Args:
        xyz_rpy (MutableDenseMatrix | list): 
            The (6,) xyz_rpy vector.

    Returns:
        MutableDenseMatrix: 
            The (4,4) homogenous transform matrix.
    """
    matrix = Matrix(Identity(4))
    matrix[:3, 3] = xyz_rpy[:3]
    matrix[:3, :3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix

def matrix_to_rpy(R: ArrayLike, solution: int=1) -> np.ndarray:
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Args:
        R (ArrayLike): 
            A 3x3 homogenous rotation matrix.
        solution (int, optional): 
            Either 1 or 2, indicating which solution to return. 
            Defaults to 1.

    Returns:
        ndarray:
            The (3,) roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2,0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2,0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0,1], R[0,2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0,1], -R[0,2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2,0])
        else:
            p = np.pi + np.arcsin(R[2,0])
        r = np.arctan2(R[2,1] / np.cos(p), R[2,2] / np.cos(p))
        y = np.arctan2(R[1,0] / np.cos(p), R[0,0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


def matrix_to_xyz_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.
    
    Args:
        matrix (ndarray): 
            The (4, 4) homogenous transform matrix.

    Returns:
        ndarray: The (6,) xyz_rpy vector.
    """
    xyz = matrix[:3,3]
    rpy = matrix_to_rpy(matrix[:3,:3])
    return np.hstack((xyz, rpy))

def quaternion_to_matrix(Q: list | MutableDenseMatrix) -> MutableDenseMatrix:
    """Convert a quaternion into SO(3) rotation matrix.

    Args:
        Q (list | sympy.MutableDenseMatrix): Quaternion in order [w,x,y,z].
        
    Returns:
        sympy.Matrix: (3,3) Rotation matrix.
    """
    # ensure symbolic values
    Q = Matrix(Q)  
    
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = Matrix([[r00, r01, r02],
                         [r10, r11, r12],
                         [r20, r21, r22]])

    return rot_matrix