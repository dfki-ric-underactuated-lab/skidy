from sympy.abc import *

q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = symbols("q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10", real=1, constant = 0)
dq, dq0, dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9, dq10 = symbols("dq, dq0, dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9, dq10", real=1, constant = 0)
ddq, ddq0, ddq1, ddq2, ddq3, ddq4, ddq5, ddq6, ddq7, ddq8, ddq9, ddq10 = symbols("ddq, ddq0, ddq1, ddq2, ddq3, ddq4, ddq5, ddq6, ddq7, ddq8, ddq9, ddq10", real=1, constant = 0)

Fx, Fy, Fz, Mx, My, Mz, fx, fy, fz, mx, my, mz = symbols("Fx, Fy, Fz, Mx, My, Mz, fx, fy, fz, mx, my, mz", real=1)

m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = symbols("m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10", real=1, constant = 1)

l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = symbols("l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10", real=1, constant = 1)
L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10 = symbols("L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10", real=1, constant = 1)

x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols("x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10", real=1, constant = 1)
y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = symbols("y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10", real=1, constant = 1)
z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10 = symbols("z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10", real=1, constant = 1)

I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = symbols("I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10", real=1, constant = 1)
i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10 = symbols("i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10", real=1, constant = 1)

Ixx, Ixx0, Ixx1, Ixx2, Ixx3, Ixx4, Ixx5, Ixx6, Ixx7, Ixx8, Ixx9, Ixx10 = symbols("Ixx, Ixx0, Ixx1, Ixx2, Ixx3, Ixx4, Ixx5, Ixx6, Ixx7, Ixx8, Ixx9, Ixx10", real=1, constant = 1)
Iyy, Iyy0, Iyy1, Iyy2, Iyy3, Iyy4, Iyy5, Iyy6, Iyy7, Iyy8, Iyy9, Iyy10 = symbols("Iyy, Iyy0, Iyy1, Iyy2, Iyy3, Iyy4, Iyy5, Iyy6, Iyy7, Iyy8, Iyy9, Iyy10", real=1, constant = 1)
Izz, Izz0, Izz1, Izz2, Izz3, Izz4, Izz5, Izz6, Izz7, Izz8, Izz9, Izz10 = symbols("Izz, Izz0, Izz1, Izz2, Izz3, Izz4, Izz5, Izz6, Izz7, Izz8, Izz9, Izz10", real=1, constant = 1)
Ixy, Ixy0, Ixy1, Ixy2, Ixy3, Ixy4, Ixy5, Ixy6, Ixy7, Ixy8, Ixy9, Ixy10 = symbols("Ixy, Ixy0, Ixy1, Ixy2, Ixy3, Ixy4, Ixy5, Ixy6, Ixy7, Ixy8, Ixy9, Ixy10", real=1, constant = 1)
Ixz, Ixz0, Ixz1, Ixz2, Ixz3, Ixz4, Ixz5, Ixz6, Ixz7, Ixz8, Ixz9, Ixz10 = symbols("Ixz, Ixz0, Ixz1, Ixz2, Ixz3, Ixz4, Ixz5, Ixz6, Ixz7, Ixz8, Ixz9, Ixz10", real=1, constant = 1)
Iyz, Iyz0, Iyz1, Iyz2, Iyz3, Iyz4, Iyz5, Iyz6, Iyz7, Iyz8, Iyz9, Iyz10 = symbols("Iyz, Iyz0, Iyz1, Iyz2, Iyz3, Iyz4, Iyz5, Iyz6, Iyz7, Iyz8, Iyz9, Iyz10", real=1, constant = 1)

ixx, ixx0, ixx1, ixx2, ixx3, ixx4, ixx5, ixx6, ixx7, ixx8, ixx9, ixx10 = symbols("ixx, ixx0, ixx1, ixx2, ixx3, ixx4, ixx5, ixx6, ixx7, ixx8, ixx9, ixx10", real=1, constant = 1)
iyy, iyy0, iyy1, iyy2, iyy3, iyy4, iyy5, iyy6, iyy7, iyy8, iyy9, iyy10 = symbols("iyy, iyy0, iyy1, iyy2, iyy3, iyy4, iyy5, iyy6, iyy7, iyy8, iyy9, iyy10", real=1, constant = 1)
izz, izz0, izz1, izz2, izz3, izz4, izz5, izz6, izz7, izz8, izz9, izz10 = symbols("izz, izz0, izz1, izz2, izz3, izz4, izz5, izz6, izz7, izz8, izz9, izz10", real=1, constant = 1)
ixy, ixy0, ixy1, ixy2, ixy3, ixy4, ixy5, ixy6, ixy7, ixy8, ixy9, ixy10 = symbols("ixy, ixy0, ixy1, ixy2, ixy3, ixy4, ixy5, ixy6, ixy7, ixy8, ixy9, ixy10", real=1, constant = 1)
ixz, ixz0, ixz1, ixz2, ixz3, ixz4, ixz5, ixz6, ixz7, ixz8, ixz9, ixz10 = symbols("ixz, ixz0, ixz1, ixz2, ixz3, ixz4, ixz5, ixz6, ixz7, ixz8, ixz9, ixz10", real=1, constant = 1)
iyz, iyz0, iyz1, iyz2, iyz3, iyz4, iyz5, iyz6, iyz7, iyz8, iyz9, iyz10 = symbols("iyz, iyz0, iyz1, iyz2, iyz3, iyz4, iyz5, iyz6, iyz7, iyz8, iyz9, iyz10", real=1, constant = 1)

c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = symbols("c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10", real=1, constant = 1)

cx, cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9, cx10 = symbols("cx, cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9, cx10", real=1, constant = 1)
cy, cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8, cy9, cy10 = symbols("cy, cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8, cy9, cy10", real=1, constant = 1)
cz, cz0, cz1, cz2, cz3, cz4, cz5, cz6, cz7, cz8, cz9, cz10 = symbols("cz, cz0, cz1, cz2, cz3, cz4, cz5, cz6, cz7, cz8, cz9, cz10", real=1, constant = 1)

cg, cg0, cg1, cg2, cg3, cg4, cg5, cg6, cg7, cg8, cg9, cg10 = symbols("cg, cg0, cg1, cg2, cg3, cg4, cg5, cg6, cg7, cg8, cg9, cg10", real=1, constant = 1)

cgx, cgx0, cgx1, cgx2, cgx3, cgx4, cgx5, cgx6, cgx7, cgx8, cgx9, cgx10 = symbols("cgx, cgx0, cgx1, cgx2, cgx3, cgx4, cgx5, cgx6, cgx7, cgx8, cgx9, cgx10", real=1, constant = 1)
cgy, cgy0, cgy1, cgy2, cgy3, cgy4, cgy5, cgy6, cgy7, cgy8, cgy9, cgy10 = symbols("cgy, cgy0, cgy1, cgy2, cgy3, cgy4, cgy5, cgy6, cgy7, cgy8, cgy9, cgy10", real=1, constant = 1)
cgz, cgz0, cgz1, cgz2, cgz3, cgz4, cgz5, cgz6, cgz7, cgz8, cgz9, cgz10 = symbols("cgz, cgz0, cgz1, cgz2, cgz3, cgz4, cgz5, cgz6, cgz7, cgz8, cgz9, cgz10", real=1, constant = 1)

Lc, Lc0, Lc1, Lc2, Lc3, Lc4, Lc5, Lc6, Lc7, Lc8, Lc9, Lc10 = symbols("Lc, Lc0, Lc1, Lc2, Lc3, Lc4, Lc5, Lc6, Lc7, Lc8, Lc9, Lc10", real=1, constant = 1)
lc, lc0, lc1, lc2, lc3, lc4, lc5, lc6, lc7, lc8, lc9, lc10 = symbols("lc, lc0, lc1, lc2, lc3, lc4, lc5, lc6, lc7, lc8, lc9, lc10", real=1, constant = 1)
