from kinematics_generator import SymbolicKinDyn
from sympy import symbols, Matrix, Identity, init_printing


if __name__ == "__main__":
    init_printing()
    
    s = SymbolicKinDyn()

    # Declaration of symbolic variables
    q1, q2 = symbols("q1 q2")  # joint positions
    dq1, dq2 = symbols("dq1 dq2")  # joint velocities
    ddq1, ddq2 = symbols("ddq1 ddq2")  # joint accelerations

    # mass and inertia values
    m1, m2, I1, I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
    # center of gravitiy and gravity
    cg1, cg2, g = symbols("cg1 cg2 g", real=1, constant=1)
    L1, L2 = symbols("L1 L2", real=1, constant=1)  # link lenghts
    pi = symbols("pi", real=1, constant=1)  # pi

    s.gravity_vector = Matrix([0, g, 0])  # define gravity vector

    # Joint screw coordinates in spatial representation

    s.Y = []
    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    s.Y.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    s.Y.append(Matrix([e2, y2.cross(e2)]))

    # Reference configurations of bodies (i.e. of body-fixed reference frames)

    r1 = Matrix([0, 0, 0])
    r2 = Matrix([L1, 0, 0])

    s.A = []
    s.A.append(Matrix(Identity(3)).row_join(
        r1).col_join(Matrix([0, 0, 0, 1]).T))
    s.A.append(Matrix(Identity(3)).row_join(
        r2).col_join(Matrix([0, 0, 0, 1]).T))

    # s.B = []
    # s.B.append(Matrix(Identity(3)).row_join(r1).col_join(Matrix([0,0,0,1]).T))
    # s.B.append(Matrix(Identity(3)).row_join(r2).col_join(Matrix([0,0,0,1]).T))

    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([L2, 0, 0])
    s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    # is calculated internally from Y
    # s.X = []
    # s.X.append(s.SE3AdjInvMatrix(s.A[0])*s.Y[0])
    # s.X.append(s.SE3AdjInvMatrix(s.A[1])*s.Y[1])

    # Joint screw coordinates in body-fixed representation

    # s.X = []
    # s.X.append(Matrix([0,0,1,0,0,0]).T)
    # s.X.append(Matrix([0,0,1,0,0,0]).T)

    # Mass-Inertia parameters
    cg1 = Matrix([L1, 0, 0]).T
    cg2 = Matrix([L2, 0, 0]).T
    I1 = m1*L1*L1
    I2 = m2*L2*L2

    s.Mb = []
    s.Mb.append(s.MassMatrixMixedData(m1, I1*Identity(3), cg1))
    s.Mb.append(s.MassMatrixMixedData(m2, I2*Identity(3), cg2))

    # Declaring generalized vectors
    q = Matrix([q1, q2])
    qd = Matrix([dq1, dq2])
    q2d = Matrix([ddq1, ddq2])

    # Kinematics
    F = s.closed_form_kinematics_body_fixed(q, qd, q2d, True)
    Q = s.closed_form_inv_dyn_body_fixed(
        q, qd, q2d, simplify_expressions=True)
    
    s.generateCode(python=True, C=True, Matlab=True,
                   use_global_vars=True, name="R2_plant", project="Project")
