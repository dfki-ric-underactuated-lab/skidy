from sympy import Matrix, Identity, sin, cos, zeros

def SE3AdjInvMatrix(C):
    """Compute Inverse of (6x6) Adjoint matrix for SE(3)

    Args:
        C ([type]): [description] TODO

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

def SE3AdjMatrix(C):
    """Compute (6x6) Adjoint matrix for SE(3)

    Args:
        C ([type]): [description] TODO

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

def SE3adMatrix(X):
    """Compute (6x6) adjoint matrix for SE(3) 
        - also known as spatial cross product in the literature.

    Args:
        X ([type]): [description] TODO

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

def SE3Exp(XX, t):
    """compute exponential mapping for SE(3).

    Args:
        XX ([type]): [description] TODO
        t ([type]): [description]

    Returns:
        [type]: [description]
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

def SE3Inv(C):
    """Compute analytical inverse of exponential mapping for SE(3).

    Args:
        C ([type]): [description] TODO

    Returns:
        [type]: [description]
    """
    CInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], 
                    -C[0, 0]*C[0, 3]-C[1, 0]*C[1, 3]-C[2, 0]*C[2, 3]],
                    [C[0, 1], C[1, 1], C[2, 1], 
                    -C[0, 1]*C[0, 3]-C[1, 1]*C[1, 3]-C[2, 1]*C[2, 3]],
                    [C[0, 2], C[1, 2], C[2, 2], -C[0, 2] *
                        C[0, 3]-C[1, 2]*C[1, 3]-C[2, 2]*C[2, 3]],
                    [0, 0, 0, 1]])
    return CInv

def SO3Exp(x, t):
    """Compute exponential mapping for SO(3).

    Args:
        x (sympy.Matrix): Rotation axis
        t (double): Rotation angle

    Returns:
        sympy.Matrix: Rotation matrix
    """
    xhat = Matrix([[0, -x[2, 0], x[1, 0]],
                    [x[2, 0], 0, -x[0, 0]],
                    [-x[1, 0], x[0, 0], 0]])
    R = Matrix(Identity(3)) + sin(t) * xhat + (1-cos(t))*(xhat*xhat)
    return R

def InertiaMatrix(Ixx, Ixy, Ixz, Iyy, Iyz, Izz):
    """Create 3 x 3 inertia matrix from independent inertia values.

    Args:
        Ixx: Inertia value I11
        Ixy: Inertia value I12SE3adMatrix
        Ixz: Inertia value I13
        Iyy: Inertia value I22
        Iyz: Inertia value I23
        Izz: Inertia value I33

    Returns:
        sympy.Matrix: Inertia matrix (3,3)
    """
    I = Matrix([[Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]])
    return I

def TransformationMatrix(r=Matrix(Identity(3)), t=zeros(3, 1)):
    """Build transformation matrix from rotation and translation.

    Args:
        r (sympy.Matrix): 
            SO(3) Rotation matrix (3,3). 
            Defaults to sympy.Matrix(Identity(3))
        t (sympy.Matrix): 
            Translation vector (3,1). Defaults to sympy.zeros(3,1)

    Returns:
        sympy.Matrix: Transformation matrix (4,4)
    """
    T = r.row_join(t).col_join(Matrix([[0, 0, 0, 1]]))
    return T

def MassMatrixMixedData(m, Theta, COM):
    """Build mass-inertia matrix in SE(3) from mass, inertia and 
    center of mass information.

    Args:
        m (float): Mass.
        Theta (array_like): Inertia (3,3).
        COM (array_like): Center of mass (3,1).

    Returns:
        sympy.Matrix: Mass-inertia matrix (6,6).
    """
    M = Matrix([[Theta[0, 0], Theta[0, 1], Theta[0, 2], 0, 
                    (-COM[2])*m, COM[1]*m],
                [Theta[0, 1], Theta[1, 1], Theta[1, 2],
                    COM[2]*m, 0, (-COM[0]*m)],
                [Theta[0, 2], Theta[1, 2], Theta[2, 2],
                    (-COM[1])*m, COM[0]*m, 0],
                [0, COM[2]*m, (-COM[1]*m), m, 0, 0],
                [(-COM[2])*m, 0, COM[0]*m, 0, m, 0],
                [COM[1]*m, (-COM[0])*m, 0, 0, 0, m]])
    return M

def rpy_to_matrix(coords):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    Adapted from urdfpy 

    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Parameters
    ----------
    coords : (3,) floatxyz_rpy_to_matrix
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns
    -------
    R : (3,3) float
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

def xyz_rpy_to_matrix(xyz_rpy):
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

    Adapted from urdfpy

    Parameters
    ----------
    xyz_rpy : (6,) float
        The xyz_rpy vector.

    Returns
    -------
    matrix : (4,4) float
        The homogenous transform matrix.
    """
    matrix = Matrix(Identity(4))
    matrix[:3, 3] = xyz_rpy[:3]
    matrix[:3, :3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix

