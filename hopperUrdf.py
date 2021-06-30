from kinematics_generator import SymbolicKinDyn

from sympy import *

q0, q1, q2 = symbols("q0 q1 q2")
dq0, dq1, dq2 = symbols("dq0 dq1 dq2")
ddq0, ddq1, ddq2 = symbols("ddq0 ddq1 ddq2")
g, L1,L2, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1, Ixx2, Ixy2, Ixz2, Iyy2, Iyz2, Izz2, m0, m1, m2, cx1,cy1,cz1, cx2, cy2, cz2 = symbols("g L1 L2 Ixx1 Ixy1 Ixz1 Iyy1 Iyz1 Izz1 Ixx2 Ixy2 Ixz2 Iyy2 Iyz2 Izz2 m0 m1 m2 cx1 cy2 cz1 cx2 cy2 cz2", constant=True, real = True)
    

o1xx,o1xy,o1xz,o1yy,o1yz,o1zz,o1tx,o1ty,o1tz,a1x,a1y,a1z, t1x,t1y,t1z = symbols("o1xx,o1xy,o1xz,o1yy,o1yz,o1zz,o1tx,o1ty,o1tz,a1x,a1y,a1z, t1x,t1y,t1z")
o2xx,o2xy,o2xz,o2yy,o2yz,o2zz,o2tx,o2ty,o2tz,a2x,a2y,a2z, t2x,t2y,t2z = symbols("o2xx,o2xy,o2xz,o2yy,o2yz,o2zz,o2tx,o2ty,o2tz,a2x,a2y,a2z, t2x,t2y,t2z")
o3xx,o3xy,o3xz,o3yy,o3yz,o3zz,o3tx,o3ty,o3tz,a3x,a3y,a3z, t3x,t3y,t3z = symbols("o3xx,o3xy,o3xz,o3yy,o3yz,o3zz,o3tx,o3ty,o3tz,a3x,a3y,a3z, t3x,t3y,t3z")

s = SymbolicKinDyn()


s.B=[]
# s.B.append(Matrix([[o1xx,o1xy,o1xz,o1tx],
#                    [-o1xy,o1yy,o1yz,o1ty],
#                    [-o1xz,-o1yz,o1zz,o1tz],
#                    [0,0,0,1]]))

# s.B.append(Matrix([[o2xx,o2xy,o2xz,o2tx],
#                    [-o2xy,o2yy,o2yz,o2ty],
#                    [-o2xz,-o2yz,o2zz,o2tz],
#                    [0,0,0,1]]))

s.B.append(Matrix([[1,0,0,o1tx],
                   [0,1,0,o1ty],
                   [0,0,1,o1tz],
                   [0,0,0,1]]))

# s.B.append(Matrix([[1,0,0,o2tx],
#                    [0,1,0,o2ty],
#                    [0,0,1,o2tz],
#                    [0,0,0,1]]))

# s.B.append(Matrix([[o3xx,o3xy,o3xz,o3tx],
#                    [-o3xy,o3yy,o3yz,o3ty],
#                    [-o3xz,-o3yz,o3zz,o3tz],
#                    [0,0,0,1]]))

s.X = []
# s.X.append(Matrix([a1x,a1y,a1z,t1x,t1y,t1z]))
# s.X.append(Matrix([a2x,a2y,a2z,t2x,t2y,t2z]))
# s.X.append(Matrix([a3x,a3y,a3z,t3x,t3y,t3z]))

# s.X.append(Matrix([a1x,a1y,a1z,t1x,t1y,t1z]))
s.X.append(Matrix([a1x,0,0,t1x,0,0]))
# s.X.append(Matrix([0,0,0,t2x,t2y,t2z]))

# s.load_from_urdf()
s.gravity_vector = Matrix([0,0,g])


re = Matrix([0, 0, L2])
s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)


# q = Matrix([q0,q1, q2])
# qd = Matrix([dq0, dq1, dq2])
# q2d = Matrix([ddq0, ddq1, ddq2])
# q = Matrix([q0,q1])
# qd = Matrix([dq0, dq1])
# q2d = Matrix([ddq0, ddq1])

q = Matrix([q0])
qd = Matrix([dq0])
q2d = Matrix([ddq0])
# s.n = len(q)
# s.load_from_urdf()
# Kinematics
F = s.closed_form_kinematics_body_fixed_parallel(q, qd, q2d)
    
print(F)
