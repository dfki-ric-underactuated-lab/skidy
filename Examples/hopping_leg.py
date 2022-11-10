from sympy import symbols, Matrix, Identity, init_printing
from kinematics_generator import SymbolicKinDyn, MassMatrixMixedData, TransformationMatrix


if __name__ == "__main__":
    init_printing()

    s = SymbolicKinDyn()

    # Declaration of symbolic variables
    q0, q1, q2 = symbols("q0 q1 q2")
    dq0, dq1, dq2 = symbols("dq0 dq1 dq2")
    ddq0, ddq1, ddq2 = symbols("ddq0 ddq1 ddq2")
    g, L1, L2, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1, Ixx2, Ixy2, Ixz2, Iyy2, Iyz2, Izz2, m0, m1, m2, cx1, cy1, cz1, cx2, cy2, cz2 = symbols(
        "g L1 L2 Ixx1 Ixy1 Ixz1 Iyy1 Iyz1 Izz1 Ixx2 Ixy2 Ixz2 Iyy2 Iyz2 Izz2 m0 m1 m2 cx1 cy2 cz1 cx2 cy2 cz2", constant=True, real=True)

    s.gravity_vector = Matrix([-g, 0, 0])

    # Joint screw coordinates in spatial representation
    # define config_representation to be spacial
    s.config_representation = s.SPACIAL

    s.joint_screw_coord = []
    s.joint_screw_coord.append(Matrix([0, 0, 0, -1, 0, 0])) # prismatic joint

    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    s.joint_screw_coord.append(Matrix([e1, y1.cross(e1)])) # revolute joint 1

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    s.joint_screw_coord.append(Matrix([e2, y2.cross(e2)])) # revolute joint 2

    # Reference configurations of bodies (i.e. of spacial frames)
    r0 = Matrix([0, 0, 0])
    r1 = Matrix([0, 0, 0])
    r2 = Matrix([L1, 0, 0])

    s.body_ref_config = []
    s.body_ref_config.append(TransformationMatrix(t=r0))
    s.body_ref_config.append(TransformationMatrix(t=r1))
    s.body_ref_config.append(TransformationMatrix(t=r2))
    
    # Reference configurations of bodies (i.e. of body-fixed reference frames)
    # define config_representation to be body fixed
    # s.config_representation = s.BODY_FIXED
    # r0 = Matrix([0, 0, 0]) 
    # r1 = Matrix([0, 0, 0]) 
    # r2 = Matrix([L1, 0, 0])

    # s.body_ref_config = []
    # s.body_ref_config.append(TransformationMatrix(t=r0))
    # s.body_ref_config.append(TransformationMatrix(t=r1))
    # s.body_ref_config.append(TransformationMatrix(t=r2))
    
    
    
    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([L2, 0, 0])
    s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    # s.joint_screw_coord = []
    # s.joint_screw_coord.append(s.SE3AdjInvMatrix(s.A[0])*s.Y[0])
    # s.joint_screw_coord.append(s.SE3AdjInvMatrix(s.A[1])*s.Y[1])
    # s.joint_screw_coord.append(s.SE3AdjInvMatrix(s.A[2])*s.Y[2])
    
    
    
    # Joint screw coordinates in body-fixed representation
    # s.joint_screw_coord = []
    # s.joint_screw_coord.append(Matrix([0,0,0,-1,0,0]))
    # s.joint_screw_coord.append(Matrix([0,0,1,0,0,0]))
    # s.joint_screw_coord.append(Matrix([0,0,1,0,0,0]))

    # Mass-Inertia parameters

    s.Mb = []
    s.Mb.append(MassMatrixMixedData(
        m0, Matrix(Identity(3)), Matrix([0, 0, 0])))
    s.Mb.append(MassMatrixMixedData(m1, s.InertiaMatrix(
        Ixx1, Ixy1, Ixz1, Iyy1, Ixz1, Izz1), Matrix([cx1, cy1, cz1])))
    s.Mb.append(MassMatrixMixedData(m2, s.InertiaMatrix(
        Ixx2, Ixy2, Ixz2, Iyy2, Ixz2, Izz2), Matrix([cx2, cy2, cz2])))

    # Declaring generalized vectors
    q = Matrix([q0, q1, q2])
    qd = Matrix([dq0, dq1, dq2])
    q2d = Matrix([ddq0, ddq1, ddq2])

    # Kinematics
    F = s.closed_form_kinematics_body_fixed(q, qd, q2d)
    Q = s.closed_form_inv_dyn_body_fixed(q, qd, q2d)
    s.generateCode(python=True, C=True, Matlab=True,
                   use_global_vars=True, name="hopperPlant", project="HoppingLeg")
