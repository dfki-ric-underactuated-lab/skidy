from skidy import (SymbolicKinDyn,
                   transformation_matrix,
                   mass_matrix_mixed_data,
                   joint_screw,
                   inertia_matrix,
                   SO3Exp,
                   generalized_vectors)
from skidy.symbols import g, L1, m1
import sympy

# Define connectivity graph
parent = [0]

child = [[]]

support = [[1]]

# gravity vector
gravity = sympy.Matrix([0,-g,0])

# representation of joint screw coordinates and body reference configurations
representation = 'body_fixed'

# joint screw coordinates (6x1 sympy.Matrix per joint)
joint_screw_coord = []
joint_screw_coord.append(joint_screw(axis=[0,0,1], vec=[0,0,0], revolute=True))

# body reference configurations (4x4 SE3 Pose (sympy.Matrix) per link)
body_ref_config = []
body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))

# end-effector configuration w.r.t. last link body fixed frame in the chain (4x4 SE3 Pose (sympy.Matrix))
ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,-L1,0])

# mass_inertia parameters (6x6 sympy.Matrix per link)
I1 = m1*L1**2

Mb = []
Mb.append(mass_matrix_mixed_data(m1, I1*sympy.Identity(3) , sympy.Matrix([0,-L1,0])))

q, qd, q2d = generalized_vectors(len(body_ref_config), startindex=1)
WEE = sympy.zeros(6,1)

skd = SymbolicKinDyn(gravity_vector=gravity,
                     ee=ee,
                     body_ref_config=body_ref_config,
                     joint_screw_coord=joint_screw_coord,
                     config_representation=representation,
                     Mb=Mb,
                     parent=parent,
                     child=child,
                     support=support,
                     )

# run Calculations
skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE, simplify=True)

# Generate Code
skd.generate_code(python=True, C=False, Matlab=False, latex=True,
                 folder="./generated_code", use_global_vars=True,
                 name="SinglePendulumPlant", project="SinglePendulum")
