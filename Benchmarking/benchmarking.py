from KinematicsGenerator.parser import robot_from_yaml
import time
import os
os.chdir(os.path.dirname(__file__))

r1 = robot_from_yaml("./r.yaml")
r2 = robot_from_yaml("./rr.yaml")
r3 = robot_from_yaml("./rrr.yaml")
r4 = robot_from_yaml("./rrrr.yaml")


startr1 = time.time()
r1.closed_form_kinematics_body_fixed()
kinr1 = time.time()
r1.closed_form_inv_dyn_body_fixed()
dynr1 =  time.time()

print("r:")
print("kin:", kinr1-startr1)
print("dyn:", dynr1-kinr1)
print("sum:", dynr1-startr1)

startr2 = time.time()
r2.closed_form_kinematics_body_fixed()
kinr2 = time.time()
r2.closed_form_inv_dyn_body_fixed()
dynr2 =  time.time()

print("rr:")
print("kin:", kinr2-startr2)
print("dyn:", dynr2-kinr2)
print("sum:", dynr2-startr2)


startr3 = time.time()
r3.closed_form_kinematics_body_fixed()
kinr3 = time.time()
r3.closed_form_inv_dyn_body_fixed()
dynr3 =  time.time()

print("rrr:")
print("kin:", kinr3-startr3)
print("dyn:", dynr3-kinr3)
print("sum:", dynr3-startr3)

startr4 = time.time()
r4.closed_form_kinematics_body_fixed()
kinr4 = time.time()
r4.closed_form_inv_dyn_body_fixed()
dynr4 =  time.time()

print("rrrr:")
print("kin:", kinr4-startr4)
print("dyn:", dynr4-kinr4)
print("sum:", dynr4-startr4)

