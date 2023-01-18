from kinematics_generator.kinematics_generator import SymbolicKinDyn, mass_matrix_mixed_data, transformation_matrix
from sympy import symbols, Matrix, Identity, init_printing
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))


init_printing()

# Declaration of symbolic variables
q1, q2 = symbols("q1 q2")  # joint positions
dq1, dq2 = symbols("dq1 dq2")  # joint velocities
ddq1, ddq2 = symbols("ddq1 ddq2")  # joint accelerations

# mass and inertia values
m1, m2 = symbols("m1 m2", real=1, constant=1)

# gravity
g = symbols("g", real=1, constant=1)
L1, L2 = symbols("L1 L2", real=1, constant=1)  # link lengths
pi = symbols("pi", real=1, constant=1)  # pi

gravity_vector = Matrix([0, -g, 0])  # define gravity vector

# Joint screw coordinates in spatial representation

joint_screw_coord = []
e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
# Joint screw coordinates in spacial representation (6,1)
joint_screw_coord.append(Matrix([e1, y1.cross(e1)]))

e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
y2 = Matrix([0, -L1, 0])  # Vector to joint axis from inertial Frame
# Joint screw coordinates in spacial representation (6,1)
joint_screw_coord.append(Matrix([e2, y2.cross(e2)]))



# Reference configurations of bodies (i.e. of absolute reference frames)
r1 = Matrix([0, 0, 0])
r2 = Matrix([0, -L1, 0])

body_ref_config = []
# no rotation, just translation
body_ref_config.append(transformation_matrix(t=r1))
body_ref_config.append(transformation_matrix(t=r2))

# End-effector configuration wrt last link body fixed frame in the chain
re = Matrix([0, -L2, 0])
ee = transformation_matrix(t=re)

# Mass-Inertia parameters
cg1 = Matrix([0, -L1, 0]).T
cg2 = Matrix([0, -L2, 0]).T
I1 = m1*L1**2
I2 = m2*L2**2

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
                    config_representation="spacial", 
                    Mb=Mb)
s.closed_form_kinematics_body_fixed(q,qd,q2d)
s.closed_form_inv_dyn_body_fixed(q,qd,q2d)
s.generate_code(python=True, C=False, Matlab=False, latex=False,name="DoublePendulumPlant",project="DoublePendulum")