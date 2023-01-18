from sympy import symbols, Matrix, Identity
from KinematicsGenerator.kinematics_generator import SymbolicKinDyn, mass_matrix_mixed_data, transformation_matrix


# Declaration of symbolic variables
q1 = symbols("q1")  # joint positions
dq1 = symbols("dq1")  # joint velocities
ddq1 = symbols("ddq1")  # joint accelerations

# mass and inertia values
m1 = symbols("m1", real=1, constant=1)

# gravity
g = symbols("g", real=1, constant=1)
L1 = symbols("L1", real=1, constant=1)  # link lengths
pi = symbols("pi", real=1, constant=1)  # pi


gravity_vector = Matrix([0, -g, 0])  # define gravity vector

# Joint screw coordinates in spatial representation

joint_screw_coord = []
e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
# Joint screw coordinates in spacial representation (6,1)
joint_screw_coord.append(Matrix([e1, y1.cross(e1)]))


# Reference configurations of bodies (i.e. of body-fixed reference frames)
r1 = Matrix([0, 0, 0])

body_ref_config = []
body_ref_config.append(transformation_matrix(t=r1)) # no rotation, just translation

# End-effector configuration wrt last link body fixed frame in the chain
re = Matrix([0, -L1, 0])
ee = transformation_matrix(t=re)


# Mass-Inertia parameters
cg1 = Matrix([0, -L1, 0]).T
I1 = m1*L1**2

Mb = []
Mb.append(mass_matrix_mixed_data(m1, I1*Identity(3), cg1))

# Declaring generalized vectors
q = Matrix([q1])
qd = Matrix([dq1])
q2d = Matrix([ddq1])

s = SymbolicKinDyn(gravity_vector=gravity_vector, 
                    ee=ee,
                    body_ref_config=body_ref_config,
                    joint_screw_coord=joint_screw_coord,
                    config_representation="body_fixed", 
                    Mb=Mb)
s.closed_form_kinematics_body_fixed(q,qd,q2d)
s.closed_form_inv_dyn_body_fixed(q,qd,q2d)
s.generateCode(python=True, C=False, Matlab=False, latex=False,name="SinglePendulumPlant",project="SinglePendulum")