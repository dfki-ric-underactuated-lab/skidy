from sympy import symbols, Matrix, Identity
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from kinematics_generator import SymbolicKinDyn


if __name__ == "__main__":
    # Declaration of symbolic variables
    q1, q2 = symbols("q1 q2")  # joint positions
    dq1, dq2 = symbols("dq1 dq2")  # joint velocities
    ddq1, ddq2 = symbols("ddq1 ddq2")  # joint accelerations

    # mass and inertia values
    m1, m2, I1, I2 = symbols("m1 m2 I1 I2", real=1, constant=1)

    # gravity
    g = symbols("g", real=1, constant=1)
    L1, L2 = symbols("L1 L2", real=1, constant=1)  # link lengths
    pi = symbols("pi", real=1, constant=1)  # pi

    
    gravity_vector = Matrix([0, 0, -g])  # define gravity vector
    
    # Joint screw coordinates in spatial representation

    Y = []
    e1 = Matrix([1, 0, 0])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation (6,1)
    Y.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([1, 0, 0])  # joint axis of revolute joint
    y2 = Matrix([0, 0, -L1])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation (6,1)
    Y.append(Matrix([e2, y2.cross(e2)]))

    
    
    # Reference configurations of bodies (i.e. of body-fixed reference frames)
    r1 = Matrix([0, 0, 0])
    r2 = Matrix([0, 0, -L1])
    
    A = []
    A.append(SymbolicKinDyn.TransformationMatrix(t=r1)) # no rotation, just translation
    A.append(SymbolicKinDyn.TransformationMatrix(t=r2))
    
    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([0, 0, -L2])
    ee = SymbolicKinDyn.TransformationMatrix(t=re)

    
    # Mass-Inertia parameters
    cg1 = Matrix([0, 0, -L1]).T
    cg2 = Matrix([0, 0, -L2]).T
    I1 = m1*L1**2
    I2 = m2*L2**2

    Mb = []
    Mb.append(SymbolicKinDyn.MassMatrixMixedData(m1, I1*Identity(3), cg1))
    Mb.append(SymbolicKinDyn.MassMatrixMixedData(m2, I2*Identity(3), cg2))

    # Declaring generalized vectors
    q = Matrix([q1, q2])
    qd = Matrix([dq1, dq2])
    q2d = Matrix([ddq1, ddq2])

    s = SymbolicKinDyn(gravity_vector=gravity_vector, ee=ee,A = A, Y = Y, Mb=Mb)
    s.closed_form_kinematics_body_fixed(q,qd,q2d)
    s.closed_form_inv_dyn_body_fixed(q,qd,q2d)
    s.generateCode(python=True, C=True, Matlab=False,name="DoublePendulumPlant",project="DoublePendulum")