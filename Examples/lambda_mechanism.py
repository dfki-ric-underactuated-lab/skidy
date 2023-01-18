from sympy import symbols, Matrix, Identity
from KinematicsGenerator.kinematics_generator import SymbolicKinDyn as skd
from KinematicsGenerator.kinematics_generator import transformation_matrix, mass_matrix_mixed_data

# Declaration of symbolic variables
q1, q2, q3 = symbols("q1 q2 q3")
dq1, dq2, dq3 = symbols("dq1 dq2 dq3")
ddq1, ddq2, ddq3 = symbols("ddq1 ddq2 ddq3")

m1, m2, m3, I1, I2, I3 = symbols("m1, m2, m3, I1, I2, I3",real=True)
Lc1, Lc2, Lc3, g = symbols("Lc1, Lc2, Lc3, g", real=True)
L1, L2 = symbols("L1, L2", real=True)

gravity_vector = Matrix([0,g,0])

# Graph description
parent = [0,
          0,
          2]
child = [[],
         [3],
         []]
support = [[1],
           [2],
           [2,3]]

# joint screw coordinates
Y = []

# For revolute joint
e1 = Matrix([0,0,1])
y1 = Matrix([0,0,0])
Y.append(Matrix([e1,y1.cross(e1)]))

# For revolute joint
e2 = Matrix([0,0,1])
y2 = Matrix([L1,0,0])
Y.append(Matrix([e2,y2.cross(e2)]))

# For prismatic joint
e3 = Matrix([1,0,0])
Y.append(Matrix([Matrix([0,0,0]),e3]))

# Reference configurations of body (i.e. w.r.t. inertia frame)
A = []

A.append(Matrix(Identity(4)))

r2 = Matrix([L1,0,0])
A.append(transformation_matrix(t=r2))

r3 = Matrix([L1,0,0])
A.append(transformation_matrix(t=r3))

# End-effector configuration wrt last link body fixed frame in the chain
re = Matrix([L2, 0, 0])
ee = transformation_matrix(t=re)

# Mass Inertia Parameters
Mb = []
Mb.append(mass_matrix_mixed_data(m1,I1*Identity(3),Matrix([Lc1, 0, 0])))
Mb.append(mass_matrix_mixed_data(m2,I2*Identity(3),Matrix([Lc2, 0, 0])))
Mb.append(mass_matrix_mixed_data(m3,I3*Identity(3),Matrix([Lc3, 0, 0])))

# EOM for the subsystem II i.e. RP Leg
q = Matrix([q1,q2,q3])
qd = Matrix([dq1,dq2,dq3])
q2d = Matrix([ddq1,ddq2,ddq3])

s = skd(gravity_vector=gravity_vector,ee=ee,Mb=Mb,parent=parent,support=support,child=child,A=A,Y=Y)
print(s.closed_form_kinematics_body_fixed(q,qd,q2d, simplify_expressions=True))
print(s.closed_form_inv_dyn_body_fixed(q,qd,q2d,simplify_expressions=True))
s.generateCode(python=True, C=True, Matlab=True,
                use_global_vars=True, name="lambdaMechanism", project="LambdaMechanism")