from skidy import SymbolicKinDyn, mass_matrix_mixed_data
from sympy import symbols, Matrix, Identity, init_printing

# Declaration of symbolic variables
q1, q2 = symbols("q1 q2")  # joint positions
dq1, dq2 = symbols("dq1 dq2")  # joint velocities
ddq1, ddq2 = symbols("ddq1 ddq2")  # joint accelerations

# mass and inertia values
m1, m2, I1, I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
# center of gravitiy and gravity
cg1, cg2, g = symbols("cg1 cg2 g", real=1, constant=1)
L1, L2 = symbols("L1 L2", real=1, constant=1)  # link lengths
pi = symbols("pi", real=1, constant=1)  # pi

# define gravity vector
gravity_vector = Matrix([0, g, 0])  

# Joint screw coordinates in spatial representation
joint_screw_coord = []
e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
joint_screw_coord.append(Matrix([e1, y1.cross(e1)]))

e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
y2 = Matrix([L1, 0, 0])  # Vector to joint axis from inertial Frame
joint_screw_coord.append(Matrix([e2, y2.cross(e2)]))

# # Reference configurations of bodies (i.e. of body-fixed reference frames)
# r1 = Matrix([0, 0, 0])
# r2 = Matrix([L1, 0, 0])

# body_ref_config = []
# body_ref_config.append(Matrix(Identity(3)).row_join(
#     r1).col_join(Matrix([0, 0, 0, 1]).T))
# body_ref_config.append(Matrix(Identity(3)).row_join(
#     r2).col_join(Matrix([0, 0, 0, 1]).T))

# spatial reference frame (in this case same as body fixed)
r1 = Matrix([0, 0, 0])
r2 = Matrix([L1, 0, 0])

body_ref_config = []
body_ref_config.append(Matrix(Identity(3)).row_join(
    r1).col_join(Matrix([0, 0, 0, 1]).T))
body_ref_config.append(Matrix(Identity(3)).row_join(
    r2).col_join(Matrix([0, 0, 0, 1]).T))

# End-effector configuration wrt last link body fixed frame in the chain
re = Matrix([L2, 0, 0])
ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

# Joint screw coordinates in body-fixed representation
# joint_screw_coord = []
# joint_screw_coord.append(Matrix([0,0,1,0,0,0]).T)
# joint_screw_coord.append(Matrix([0,0,1,0,0,0]).T)

# Mass-Inertia parameters
cg1 = Matrix([L1, 0, 0])
cg2 = Matrix([L2, 0, 0])
I1 = m1*L1*L1
I2 = m2*L2*L2

Mb = []
Mb.append(mass_matrix_mixed_data(m1, I1*Identity(3), cg1))
Mb.append(mass_matrix_mixed_data(m2, I2*Identity(3), cg2))

# Declaring generalized vectors
q = Matrix([q1, q2])
qd = Matrix([dq1, dq2])
q2d = Matrix([ddq1, ddq2])

s = SymbolicKinDyn(gravity_vector=gravity_vector, 
                    ee=ee, 
                    body_ref_config=body_ref_config, 
                    joint_screw_coord=joint_screw_coord, 
                    config_representation= "spatial", 
                    Mb=Mb)


# Kinematics
F = s.closed_form_kinematics_body_fixed(q, qd, q2d, True)
Q = s.closed_form_inv_dyn_body_fixed(
    q, qd, q2d, simplify_expressions=True)

s.generate_code(python=True, C=True, Matlab=True,
                use_global_vars=True, name="R2_plant", project="Project")
