from skidy import SymbolicKinDyn
import os

from sympy import symbols, Matrix, Identity

q0, q1, q2 = symbols("q0 q1 q2")
dq0, dq1, dq2 = symbols("dq0 dq1 dq2")
ddq0, ddq1, ddq2 = symbols("ddq0 ddq1 ddq2")
g, L1,L2, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1, Ixx2, Ixy2, Ixz2, Iyy2, Iyz2, Izz2, m0, m1, m2, cx1,cy1,cz1, cx2, cy2, cz2 = symbols("g L1 L2 Ixx1 Ixy1 Ixz1 Iyy1 Iyz1 Izz1 Ixx2 Ixy2 Ixz2 Iyy2 Iyz2 Izz2 m0 m1 m2 cx1 cy2 cz1 cx2 cy2 cz2", constant=True, real = True)
    

s = SymbolicKinDyn()


s.gravity_vector = Matrix([0,0,g])

# end effector
re = Matrix([0, 0, L2])
s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

s.load_from_urdf(os.path.join(os.path.dirname(__file__),"urdf/hopping_leg.urdf"),symbolic=1, simplify_numbers=1, cse=1)


q = Matrix([q0,q1, q2])
qd = Matrix([dq0, dq1, dq2])
q2d = Matrix([ddq0, ddq1, ddq2])

# Kinematics
F = s.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=0,cse=1)
# Q = s.closed_form_inv_dyn_body_fixed(q,qd,q2d, simplify=0)

s.generate_code(python=1,C=0,Matlab=0,name="HoppingLeg")
