from KinematicsGenerator.kinematics_generator import SymbolicKinDyn, mass_matrix_mixed_data
from sympy import symbols, Matrix, Identity

s = SymbolicKinDyn()
q1, q2, q3, q4 = symbols("q1 q2 q3 q4", real = True)
dq1, dq2, dq3, dq4 = symbols("dq1 dq2 dq3 dq4", real=True)
ddq1, ddq2, ddq3, ddq4 = symbols("ddq1 ddq2 ddq3 ddq4", real = True)

m1, c1x, c1y,c1z, I1, I1xx, I1xy, I1xz, I1yy, I1yz, I1zz = symbols("m1, c1x, c1y,c1z, I1, I1xx, I1xy, I1xz, I1yy, I1yz, I1zz", real=1, constant=1)
m2, c2x, c2y,c2z, I2, I2xx, I2xy, I2xz, I2yy, I2yz, I2zz = symbols("m2, c2x, c2y,c2z, I2, I2xx, I2xy, I2xz, I2yy, I2yz, I2zz", real=1, constant=1)
m3, c3x, c3y,c3z, I3, I3xx, I3xy, I3xz, I3yy, I3yz, I3zz = symbols("m3, c3x, c3y,c3z, I3, I3xx, I3xy, I3xz, I3yy, I3yz, I3zz", real=1, constant=1)
m4, c4x, c4y,c4z, I4, I4xx, I4xy, I4xz, I4yy, I4yz, I4zz = symbols("m4, c4x, c4y,c4z, I4, I4xx, I4xy, I4xz, I4yy, I4yz, I4zz", real=1, constant=1)

pi, g = symbols("pi g", real=1, constant=1)

base_offset, thigh_v_offset, thigh_h_offset, thigh, shank = symbols("base_offset thigh_v_offset thigh_h_offset thigh shank")

s.gravity_vector = Matrix([0, 0, g])

dh_table = Matrix([[0,0,0,0,base_offset],
                   [0, pi/2, 0, 0, thigh_v_offset],
                   [0,-pi/2, 0, -pi/2, thigh_h_offset],
                   [0, 0, thigh, pi, 0]])

s.dh_to_screw_coord(dh_table)
re = Matrix([shank,0,0])
s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([[0,0,0,1]]))

s.Mb = []
s.Mb.append(mass_matrix_mixed_data(m1, s.InertiaMatrix(I1xx, I1xy, I1xz, I1yy, I1yz, I1zz),Matrix([c1x,c1y,c1z])))
s.Mb.append(mass_matrix_mixed_data(m2, s.InertiaMatrix(I2xx, I2xy, I2xz, I2yy, I2yz, I2zz),Matrix([c2x,c2y,c2z])))
s.Mb.append(mass_matrix_mixed_data(m3, s.InertiaMatrix(I3xx, I3xy, I3xz, I3yy, I3yz, I3zz),Matrix([c3x,c3y,c3z])))
s.Mb.append(mass_matrix_mixed_data(m4, s.InertiaMatrix(I4xx, I4xy, I4xz, I4yy, I4yz, I4zz),Matrix([c4x,c4y,c4z])))

q = Matrix([q1,q2,q3,q4])
qd = Matrix([dq1,dq2,dq3,dq4])
q2d = Matrix([ddq1,ddq2,ddq3,ddq4])

T = s.closed_form_kinematics_body_fixed(q,qd,q2d,simplify_expressions=0)

Q = s.closed_form_inv_dyn_body_fixed(q,qd,q2d,simplify_expressions=0)

# s.generateCode(folder="./crex_code")
