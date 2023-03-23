import unittest
import unittest.mock
import sys
import os
from textwrap import dedent
import shutil
import warnings
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
import skidy
from skidy import (SE3AdjInvMatrix, SE3AdjMatrix, 
                                  SE3adMatrix, SE3Exp, SE3Inv, SO3Exp, 
                                  inertia_matrix, transformation_matrix, 
                                  mass_matrix_mixed_data, rpy_to_matrix, 
                                  xyz_rpy_to_matrix)
from skidy.parser import robot_from_yaml, robot_from_json, robot_from_urdf
from sympy import Matrix, cos, sin, symbols, Identity, simplify, zeros, parse_expr
import random
import numpy as np

try:
    from oct2py import octave
except (ImportError,ModuleNotFoundError):
    octave = None # skip octave tests
    warnings.warn("Cannot import oct2py. Skip all tests for the generated matlab/octave code.")
try:
    import cython
except (ImportError,ModuleNotFoundError):
    cython = None  # skip cython tests
    warnings.warn("Cannot import cython. Skip all tests for the generated cython code.")
    
delete_generated_code = True # False deactivates cleanup functions

def prepare(cls):
    cls.skd = skidy.SymbolicKinDyn()
    cls.q1, cls.q2 = symbols("q1 q2")
    cls.dq1, cls.dq2 = symbols("dq1 dq2")
    cls.ddq1, cls.ddq2 = symbols("ddq1 ddq2")
    cls.dddq1, cls.dddq2 = symbols("dddq1 dddq2")
    cls.ddddq1, cls.ddddq2 = symbols("ddddq1 ddddq2")

    
    cls.m1, cls.m2, cls.I1, cls.I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
    cls.cg1, cls.cg2, cls.g = symbols("cg1 cg2 g", real=1, constant=1)
    cls.L1, cls.L2 = symbols("L1 L2", real=1, constant=1)
    
    cls.skd.config_representation = "spatial"
    
    cls.skd.gravity_vector = Matrix([0, cls.g, 0])

    # Joint screw coordinates in spatial representation

    cls.skd.joint_screw_coord = []
    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spatial representation
    cls.skd.joint_screw_coord.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([cls.L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spatial representation
    cls.skd.joint_screw_coord.append(Matrix([e2, y2.cross(e2)]))

    # Reference configurations of bodies (i.e. of spatial reference frames)
    r1 = Matrix([0, 0, 0])
    r2 = Matrix([cls.L1, 0, 0])

    cls.skd.body_ref_config = []
    cls.skd.body_ref_config.append(Matrix(Identity(3)).row_join(
        r1).col_join(Matrix([0, 0, 0, 1]).T))
    cls.skd.body_ref_config.append(Matrix(Identity(3)).row_join(
        r2).col_join(Matrix([0, 0, 0, 1]).T))

    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([cls.L2, 0, 0])
    cls.skd.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    # cls.skd.joint_screw_coord = []
    # cls.skd.joint_screw_coord.append(SE3AdjInvMatrix(cls.skd.A[0])*cls.skd.Y[0])
    # cls.skd.joint_screw_coord.append(SE3AdjInvMatrix(cls.skd.A[1])*cls.skd.Y[1])

    # Mass-Inertia parameters
    cg1 = Matrix([cls.L1, 0, 0]).T
    cg2 = Matrix([cls.L2, 0, 0]).T
    I1 = cls.m1*cls.L1**2
    I2 = cls.m2*cls.L2**2

    cls.skd.Mb = []
    cls.skd.Mb.append(mass_matrix_mixed_data(cls.m1, I1*Identity(3), cg1))
    cls.skd.Mb.append(mass_matrix_mixed_data(cls.m2, I2*Identity(3), cg2))

    # Declaring generalised vectors
    cls.q = Matrix([cls.q1, cls.q2])
    cls.qd = Matrix([cls.dq1, cls.dq2])
    cls.q2d = Matrix([cls.ddq1, cls.ddq2])
    cls.q3d = Matrix([cls.dddq1, cls.dddq2])
    cls.q4d = Matrix([cls.ddddq1, cls.ddddq2])
        

class abstractFKinTest():
    def testfkin(self):
        self.assertEqual(
            simplify(
                self.skd.fkin
                - Matrix([[cos(self.q1+self.q2), 
                        -sin(self.q1+self.q2),
                        0,
                        self.L2*cos(self.q1+self.q2)+self.L1*cos(self.q1)],
                        [sin(self.q1+self.q2), 
                        cos(self.q1+self.q2),
                        0, 
                        self.L2*sin(self.q1+self.q2)+self.L1*sin(self.q1)],
                        [0,0,1,0],
                        [0,0,0,1]])
                ),
            zeros(4,4)
        )
        
    def testJ(self):
        self.assertEqual(
            simplify(
                self.skd.J
                - Matrix([[0,0],
                        [0,0],
                        [1,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [1,1],
                        [self.L1*sin(self.q2),0],
                        [self.L1*cos(self.q2),0],
                        [0,0]])
            ),
            zeros(12,2)
        )
        
    def testJb_ee(self):
        self.assertEqual(
            simplify(
                self.skd.Jb_ee
                -Matrix([[0,0],
                        [0,0],
                        [1,1],
                        [self.L1*sin(self.q2),0],
                        [self.L2+self.L1*cos(self.q2),self.L2],
                        [0,0]])
            ),
            zeros(6,2)
        )
        
    def testJh_ee(self):
        self.assertEqual(
            simplify(
                self.skd.Jh_ee
                - Matrix([[0,0],
                        [0,0],
                        [1,1],
                        [-self.L2*sin(self.q1+self.q2)-self.L1*sin(self.q1),
                        -self.L2*sin(self.q1+self.q2)],
                        [self.L2*cos(self.q1+self.q2)+self.L1*cos(self.q1),
                        self.L2*cos(self.q1+self.q2)],
                        [0,0]])
            ),
            zeros(6,2)
        )
        
    def testJb(self):
        self.assertEqual(
            simplify(
                self.skd.Jb
                -Matrix([[0,0],
                        [0,0],
                        [1,1],
                        [self.L1*sin(self.q2),0],
                        [self.L1*cos(self.q2),0],
                        [0,0]])
            ),
            zeros(6,2)
        )
        
    def testJh(self):
        self.assertEqual(
            simplify(
                self.skd.Jh
                - Matrix([[0,0],
                        [0,0],
                        [1,1],
                        [-self.L1*sin(self.q1),0],
                        [self.L1*cos(self.q1),0],
                        [0,0]])
            ),
            zeros(6,2)
        )
    
    def testJdot(self):
        self.assertEqual(
            simplify(
                self.skd.Jdot
                - Matrix([[0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [self.L1*self.dq2*cos(self.q2),0],
                        [-self.L1*self.dq2*sin(self.q2),0],
                        [0,0]])
            ),
            zeros(12,2)
        )
    
    def testVbd_BFn(self):
        self.assertEqual(
            simplify(
                self.skd.Vbd_BFn 
                -Matrix([[0],
                        [0],
                        [self.ddq1+self.ddq2],
                        [self.L1*(self.ddq1*sin(self.q2) + self.dq1*self.dq2*cos(self.q2))],
                        [self.L1*(self.ddq1*cos(self.q2) - self.dq1*self.dq2*sin(self.q2))],
                        [0]])
            ), 
            zeros(6,1)
        )
    
    
    def testVhd_BFn(self):
        self.assertEqual(
            simplify(self.skd.Vhd_BFn -
            Matrix([[0],
                    [0],
                    [self.ddq1+self.ddq2],
                    [-self.L1*(cos(self.q1)*self.dq1**2 + self.ddq1*sin(self.q1))],
                    [-self.L1*(sin(self.q1)*self.dq1**2 - self.ddq1*cos(self.q1))],
                    [0]])),
            zeros(6,1)
        )
    
    def testVb_ee(self):
        self.assertEqual(
            simplify(self.skd.Vb_ee -
            Matrix([[0],
                    [0],
                    [self.dq1+self.dq2],
                    [self.L1*self.dq1*sin(self.q2)],
                    [self.L2*(self.dq1+self.dq2) + self.L1*self.dq1*cos(self.q2)],
                    [0]])),
            zeros(6,1)
        )
    
    
    def testVh_ee(self):
        self.assertEqual(
            simplify(self.skd.Vh_ee -
            Matrix([[0],
                    [0],
                    [self.dq1+self.dq2],
                    [- self.L2*self.dq1*sin(self.q1 + self.q2) 
                     - self.L2*self.dq2*sin(self.q1 + self.q2) 
                     - self.L1*self.dq1*sin(self.q1)],
                    [self.L2*self.dq1*cos(self.q1 + self.q2) 
                     + self.L2*self.dq2*cos(self.q1 + self.q2) 
                     + self.L1*self.dq1*cos(self.q1)],
                    [0]])),
            zeros(6,1)
        )
    
    
    def testVbd_ee(self):
        self.assertEqual(
            simplify(self.skd.Vbd_ee-
            Matrix([[0],
                    [0],
                    [self.ddq1+self.ddq2],
                    [self.L1*(self.ddq1*sin(self.q2) + self.dq1*self.dq2*cos(self.q2))],
                    [self.L1*(self.ddq1*cos(self.q2) - self.dq1*self.dq2*sin(self.q2))
                     +self.L2*(self.ddq1+self.ddq2)],
                    [0]])),
            zeros(6,1)
        )
        
        
    def testVhd_ee(self):
        self.assertEqual(
            simplify(simplify(self.skd.Vhd_ee)-
            Matrix([[0],
                    [0],
                    [self.ddq1+self.ddq2],
                    [- self.L2*self.dq1**2*cos(self.q1 + self.q2) 
                     - self.L2*self.dq2**2*cos(self.q1 + self.q2) 
                     - self.L1*self.dq1**2*cos(self.q1) 
                     - self.L2*self.ddq1*sin(self.q1 + self.q2) 
                     - self.L2*self.ddq2*sin(self.q1 + self.q2) 
                     - self.L1*self.ddq1*sin(self.q1) 
                     - 2*self.L2*self.dq1*self.dq2*cos(self.q1 + self.q2)],
                    [self.L2*self.ddq1*cos(self.q1 + self.q2) 
                     - self.L2*self.dq2**2*sin(self.q1 + self.q2) 
                     - self.L1*self.dq1**2*sin(self.q1) 
                     - self.L2*self.dq1**2*sin(self.q1 + self.q2) 
                     + self.L2*self.ddq2*cos(self.q1 + self.q2) 
                     + self.L1*self.ddq1*cos(self.q1) 
                     - 2*self.L2*self.dq1*self.dq2*sin(self.q1 + self.q2)],
                    [0]])),
            zeros(6,1)
        )
        
    
    
    def testJb_ee_dot(self):
        self.assertEqual(
            simplify(
                self.skd.Jb_ee_dot
                -Matrix([[0,0],
                        [0,0],
                        [0,0],
                        [self.L1*self.dq2*cos(self.q2),0],
                        [-self.L1*self.dq2*sin(self.q2),0],[0,0]])
            ),
            zeros(6,2)
        )
        
    def testJh_ee_dot(self):
        self.assertEqual(
            simplify(self.skd.Jh_ee_dot-
            Matrix([[0,0],
                    [0,0],
                    [0,0],
                    [-self.L2*self.dq1*cos(self.q1 + self.q2) 
                     - self.L2*self.dq2*cos(self.q1 + self.q2) 
                     - self.L1*self.dq1*cos(self.q1), 
                     -self.L2*cos(self.q1 + self.q2)*(self.dq1 + self.dq2)],
                    [-self.L2*self.dq1*sin(self.q1 + self.q2) 
                     - self.L2*self.dq2*sin(self.q1 + self.q2) 
                     - self.L1*self.dq1*sin(self.q1), 
                     -self.L2*sin(self.q1 + self.q2)*(self.dq1 + self.dq2)],
                    [0,0]])),
            zeros(6,2)
        )
    
        
    def testJh_dot(self):
        self.assertEqual(
            simplify(self.skd.Jh_dot-
            Matrix([[0,0],
                    [0,0],
                    [0,0],
                    [-self.L1*self.dq1*cos(self.q1),0],
                    [-self.L1*self.dq1*sin(self.q1),0],
                    [0,0]])),
            zeros(6,2)
        )
    
    def testJb_dot(self):
        self.assertEqual(
            simplify(
                self.skd.Jb_dot
                -Matrix([[0,0],
                        [0,0],
                        [0,0],
                        [self.L1*self.dq2*cos(self.q2),0],
                        [-self.L1*self.dq2*sin(self.q2),0],
                        [0,0]])
            ),
            zeros(6,2)
        )
    
class TestFKin(abstractFKinTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        T = cls.skd.closed_form_kinematics_body_fixed(cls.q,cls.qd,cls.q2d,True,False,False)


class TestFKin_parallel(abstractFKinTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        T = cls.skd.closed_form_kinematics_body_fixed(cls.q,cls.qd,cls.q2d,True,False,True)


class AbstractInvDyn():
    
    def testM(self):
        self.assertEqual(
            simplify(self.skd.M-
            Matrix([[self.L1**2*self.m1
                     + self.L1**2*self.m2
                     + self.L2**2*self.m2
                     + 2*self.L1*self.L2*self.m2*cos(self.q2),
                     self.L2*self.m2*(self.L2+self.L1*cos(self.q2))],
                    [self.L2*self.m2*(self.L2+self.L1*cos(self.q2)),
                     self.L2**2*self.m2]])),
            zeros(2,2)
        )      
        
    def testJ(self):
        self.assertEqual(
            simplify(
                self.skd.J
                - Matrix([[0,0],
                         [0,0],
                         [1,0],
                         [0,0],
                         [0,0],
                         [0,0],
                         [0,0],
                         [0,0],
                         [1,1],
                         [self.L1*sin(self.q2),0],
                         [self.L1*cos(self.q2),0],
                         [0,0]])
            ),
            zeros(12,2)
        )
    
    def testC(self):
        self.assertEqual(
            self.skd.simplify(self.skd.C-
            Matrix([[-2*self.L1*self.L2*self.dq2*self.m2*sin(self.q2),
                     -self.L1*self.L2*self.dq2*self.m2*sin(self.q2)],
                    [self.L1*self.L2*self.m2*sin(self.q2)*(self.dq1-self.dq2),
                     self.L1*self.L2*self.dq1*self.m2*sin(self.q2)]])),
            zeros(2,2)
        )
        
    def testQgrav(self):
        self.assertEqual(
            self.skd.simplify(self.skd.Qgrav-
            Matrix([[self.g*(self.L2*self.m2*cos(self.q1+self.q2)
                             +self.L1*self.m1*cos(self.q1)
                             +self.L1*self.m2*cos(self.q1))],
                    [self.L2*self.g*self.m2*cos(self.q1+self.q2)]])),
            zeros(2,1)
        )
        
    def testQ(self):
        Q1 = (self.L1**2*self.ddq1*self.m1+self.L1**2*self.ddq1*self.m2 
              + self.L2**2*self.ddq1*self.m2+self.L2**2*self.ddq2*self.m2
              + self.L2*self.g*self.m2*cos(self.q1+self.q2)
              + self.L1*self.g*self.m1*cos(self.q1)
              + self.L1*self.g*self.m2*cos(self.q1)
              - self.L1*self.L2*self.dq2**2*self.m2*sin(self.q2)
              + 2*self.L1*self.L2*self.ddq1*self.m2*cos(self.q2)
              + self.L1*self.L2*self.ddq2*self.m2*cos(self.q2)
              - 2*self.L1*self.L2*self.dq1*self.dq2*self.m2*sin(self.q2))
        
        Q2 = self.L2*self.m2*(self.L1*sin(self.q2)*self.dq1**2
                              + self.L2*self.ddq1
                              + self.L2*self.ddq2
                              + self.g*cos(self.q1+self.q2)
                              + self.L1*self.ddq1*cos(self.q2))       
        self.assertEqual(
            simplify(self.skd.Q - Matrix([Q1,Q2])),
            zeros(2,1)
        )
        
    def testMd(self):
        self.assertEqual(
            simplify(
                self.skd.Md
                - self.skd._time_derivative(self.skd.M, level=1)
            ),
            zeros(2,2)
        )

    def testCd(self):
        self.assertEqual(
            simplify(
                self.skd.Cd
                - self.skd._time_derivative(self.skd.C, level=1)
            ),
            zeros(2,2)
        )

    def testQdgrav(self):
        self.assertEqual(
            simplify(
                self.skd.Qdgrav
                - self.skd._time_derivative(self.skd.Qgrav, level=1)
            ),
            zeros(2,1)
        )

    def testQd(self):
        self.assertEqual(
            simplify(
                self.skd.Qd
                - self.skd._time_derivative(self.skd.Q, level=1)
            ),
            zeros(2,1)
        )

    def testM2d(self):
        self.assertEqual(
            simplify(
                self.skd.M2d
                - self.skd._time_derivative(self.skd.M, level=2)
            ),
            zeros(2,2)
        )

    def testC2d(self):
        self.assertEqual(
            simplify(
                self.skd.C2d
                - self.skd._time_derivative(self.skd.C, level=2)
            ),
            zeros(2,2)
        )

    def testQ2dgrav(self):
        self.assertEqual(
            simplify(
                self.skd.Q2dgrav
                - self.skd._time_derivative(self.skd.Qgrav, level=2)
            ),
            zeros(2,1)
        )

    def testQ2d(self):
        self.assertEqual(
            simplify(
                self.skd.Q2d
                - self.skd._time_derivative(self.skd.Q, level=2)
            ),
            zeros(2,1)
        )

        
class TestInvDyn(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        Q = cls.skd.closed_form_inv_dyn_body_fixed(cls.q,cls.qd,cls.q2d,cls.q3d,cls.q4d,WEE=zeros(6,1),
                                                 simplify=True,cse=False,parallel=False)


class TestInvDynParallel(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        Q = cls.skd.closed_form_inv_dyn_body_fixed(cls.q,cls.qd,cls.q2d,cls.q3d,cls.q4d,WEE=zeros(6,1),
                                                  simplify=True,cse=False,parallel=True)


class TestKinGen(unittest.TestCase):
    
    # def setUp(self):
        # self.skd = skidy.SymbolicKinDyn()
    
    def testInertiaMatrix(self):
        self.assertEqual(
            inertia_matrix(1,2,3,4,5,6), 
            Matrix([[1,2,3],[2,4,5],[3,5,6]])
        )
        
    def testTransformationMatrix(self):
        self.assertEqual(
            transformation_matrix(Matrix([[1,2,3],[4,5,6],[7,8,9]]),
                                        Matrix([10,11,12])),
            Matrix([[1,2,3,10],[4,5,6,11],[7,8,9,12],[0,0,0,1]])
        )

    def testSO3Exp(self):
        t = random.randint(0,100)
        self.assertEqual(
            SO3Exp(Matrix([0,0,1]), t),
            Matrix([[cos(t),-sin(t),0],[sin(t),cos(t),0],[0,0,1]])
        )   
    
    def testSE3Exp(self):
        t = random.randint(0,100)
        self.assertEqual(
            SE3Exp(Matrix([0,0,0,0,0,1]),t = t),
            Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]])
        )
        self.assertEqual(
            SE3Exp(Matrix([0,0,1,0,0,0]),t = t),
            Matrix([[cos(t),-sin(t),0,0],[sin(t),cos(t),0,0],[0,0,1,0],[0,0,0,1]])
        )
        
    # def testLoadFromURDF(self):
        # pass
    
    # def testxyz_rpy_to_matrix(self):
    #     pass
    
    # def testrpy_to_matrix(self):
        # pass
        
    # def testDhToScrewCoord(self):
    #     pass
    
    def testMassMatrixMixedData(self):
        m = random.randint(0,100)
        Ixx = random.randint(0,100)
        Ixy = random.randint(0,100)
        Ixz = random.randint(0,100)
        Iyy = random.randint(0,100)
        Iyz = random.randint(0,100)
        Izz = random.randint(0,100)
        I = Matrix([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
        c1=random.randint(0,100)
        c2=random.randint(0,100)
        c3=random.randint(0,100)
        com = [c1,c2,c3]
        self.assertEqual(
            mass_matrix_mixed_data(m,I,com),
            Matrix([[Ixx,Ixy,Ixz, 0,-m*c3,m*c2],
                    [Ixy,Iyy,Iyz, m*c3,0,-m*c1],
                    [Ixz,Iyz,Izz,-m*c2,m*c1,0],
                    [0,m*c3,-m*c2,m,0,0],
                    [-m*c3,0,m*c1,0,m,0],
                    [m*c2,-m*c1,0,0,0,m]])
        )
    
    def testSE3Inv(self):
        a,x,y,z = symbols("a,x,y,z")
        m = Matrix([[cos(a),-sin(a),0,x],
                    [sin(a), cos(a),0,y],
                    [0,0,1,z],
                    [0,0,0,1]])
        self.assertEqual(simplify(SE3Inv(m) - m.inv()), zeros(4,4))
    
    # def testSE3adMatrix(self):
    #     pass
    
    # def testSE3AdjMatrix(self):
    #     pass
    
    # def testSE3AdjInvMatrix(self):
    #     pass
    
    
class AbstractGeneratedCodeTests:    
    def testfkin(self):
        q1 = random.random()
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.forward_kinematics(q1,q2),
                np.array([[np.cos(q1+q2), 
                        -np.sin(q1+q2),
                        0,
                        self.L2*np.cos(q1+q2)+self.L1*np.cos(q1)],
                        [np.sin(q1+q2), 
                        np.cos(q1+q2),
                        0, 
                        self.L2*np.sin(q1+q2)+self.L1*np.sin(q1)],
                        [0,0,1,0],
                        [0,0,0,1]])
            ))
        
    def testJ(self):
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.system_jacobian_matrix(q2),
                np.array([[0,0],
                        [0,0],
                        [1,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [1,1],
                        [self.L1*np.sin(q2),0],
                        [self.L1*np.cos(q2),0],
                        [0,0]])
            ))
        
    def testJb_ee(self):
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_jacobian_matrix_ee(q2),
                np.array([[0,0],
                          [0,0],
                          [1,1],
                          [self.L1*np.sin(q2),0],
                          [self.L2+self.L1*np.cos(q2),self.L2],
                          [0,0]])
            ))
        
    def testJh_ee(self):
        q1 = random.random()
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_jacobian_matrix_ee(q1,q2),
                np.array([[0,0],
                          [0,0],
                          [1,1],
                          [-self.L2*np.sin(q1+q2)-self.L1*np.sin(q1),
                          -self.L2*np.sin(q1+q2)],
                          [self.L2*np.cos(q1+q2)+self.L1*np.cos(q1),
                          self.L2*np.cos(q1+q2)],
                          [0,0]])
            ))
        
    def testJb(self):
        q1 = random.random()
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_jacobian_matrix(q2),
                np.array([[0,0],
                          [0,0],
                          [1,1],
                          [self.L1*np.sin(q2),0],
                          [self.L1*np.cos(q2),0],
                          [0,0]])
            ))
        
    def testJh(self):
        q1 = random.random()
        q2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_jacobian_matrix(q1),
                np.array([[0,0],
                        [0,0],
                        [1,1],
                        [-self.L1*np.sin(q1),0],
                        [self.L1*np.cos(q1),0],
                        [0,0]])
            ))
    
    def testJdot(self):
        q1 = random.random()
        q2 = random.random()
        dq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.system_jacobian_dot(q2,dq2),
                np.array([[0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [self.L1*dq2*np.cos(q2),0],
                        [-self.L1*dq2*np.sin(q2),0],
                        [0,0]])
            ))
    
    def testVbd_BFn(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_acceleration(q2,dq1,dq2,ddq1,ddq2),
                np.array([[0],
                        [0],
                        [ddq1+ddq2],
                        [self.L1*(ddq1*np.sin(q2) + dq1*dq2*np.cos(q2))],
                        [self.L1*(ddq1*np.cos(q2) - dq1*dq2*np.sin(q2))],
                        [0]])
            ))
    
    
    def testVhd_BFn(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_acceleration(q1,dq1,ddq1,ddq2),
                np.array([[0],
                          [0],
                          [ddq1+ddq2],
                          [-self.L1*(np.cos(q1)*dq1**2 + ddq1*np.sin(q1))],
                          [-self.L1*(np.sin(q1)*dq1**2 - ddq1*np.cos(q1))],
                          [0]])
            ))
    
    def testVb_ee(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_twist_ee(q2,dq1,dq2),
                np.array([[0],
                          [0],
                          [dq1+dq2],
                          [self.L1*dq1*np.sin(q2)],
                          [self.L2*(dq1+dq2) + self.L1*dq1*np.cos(q2)],
                          [0]])
                ))
    
    
    def testVh_ee(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_twist_ee(q1,q2,dq1,dq2),
                np.array([[0],
                          [0],
                          [dq1+dq2],
                          [- self.L2*dq1*np.sin(q1 + q2) 
                           - self.L2*dq2*np.sin(q1 + q2) 
                           - self.L1*dq1*np.sin(q1)],
                          [self.L2*dq1*np.cos(q1 + q2) 
                           + self.L2*dq2*np.cos(q1 + q2) 
                           + self.L1*dq1*np.cos(q1)],
                          [0]])
                ))
    
    
    def testVbd_ee(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_acceleration_ee(q2,dq1,dq2,ddq1,ddq2),
                np.array([[0],
                          [0],
                          [ddq1+ddq2],
                          [self.L1*(ddq1*np.sin(q2) + dq1*dq2*np.cos(q2))],
                          [self.L1*(ddq1*np.cos(q2) - dq1*dq2*np.sin(q2))
                           +self.L2*(ddq1+ddq2)],
                          [0]])
                ))
        
        
    def testVhd_ee(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_acceleration_ee(q1,q2,dq1,dq2,ddq1,ddq2),
                np.array([[0],
                          [0],
                          [ddq1+ddq2],
                          [- self.L2*dq1**2*np.cos(q1 + q2) 
                           - self.L2*dq2**2*np.cos(q1 + q2) 
                           - self.L1*dq1**2*np.cos(q1) 
                           - self.L2*ddq1*np.sin(q1 + q2) 
                           - self.L2*ddq2*np.sin(q1 + q2) 
                           - self.L1*ddq1*np.sin(q1) 
                           - 2*self.L2*dq1*dq2*np.cos(q1 + q2)],
                          [self.L2*ddq1*np.cos(q1 + q2) 
                           - self.L2*dq2**2*np.sin(q1 + q2) 
                           - self.L1*dq1**2*np.sin(q1) 
                           - self.L2*dq1**2*np.sin(q1 + q2) 
                           + self.L2*ddq2*np.cos(q1 + q2) 
                           + self.L1*ddq1*np.cos(q1) 
                           - 2*self.L2*dq1*dq2*np.sin(q1 + q2)],
                          [0]])
                ))
    
    
    def testJb_ee_dot(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_jacobian_matrix_ee_dot(q2,dq2),
                np.array([[0,0],
                          [0,0],
                          [0,0],
                          [self.L1*dq2*np.cos(q2),0],
                          [-self.L1*dq2*np.sin(q2),0],[0,0]])
            ))
        
    def testJh_ee_dot(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_jacobian_matrix_ee_dot(q1,q2,dq1,dq2),
                np.array([[0,0],
                          [0,0],
                          [0,0],
                          [-self.L2*dq1*np.cos(q1 + q2) 
                           - self.L2*dq2*np.cos(q1 + q2) 
                           - self.L1*dq1*np.cos(q1), 
                           -self.L2*np.cos(q1 + q2)*(dq1 + dq2)],
                          [-self.L2*dq1*np.sin(q1 + q2) 
                           - self.L2*dq2*np.sin(q1 + q2) 
                           - self.L1*dq1*np.sin(q1), 
                           -self.L2*np.sin(q1 + q2)*(dq1 + dq2)],
                          [0,0]])
                ))
    
        
    def testJh_dot(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.hybrid_jacobian_matrix_dot(q1,dq1),
                np.array([[0,0],
                          [0,0],
                          [0,0],
                          [-self.L1*dq1*np.cos(q1),0],
                          [-self.L1*dq1*np.sin(q1),0],
                          [0,0]])
                ))
    
    def testJb_dot(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.body_jacobian_matrix_dot(q2,dq2),
                np.array([[0,0],
                          [0,0],
                          [0,0],
                          [self.L1*dq2*np.cos(q2),0],
                          [-self.L1*dq2*np.sin(q2),0],
                          [0,0]])
            ))
    
    
    # dynamics
    def testM(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.generalized_mass_inertia_matrix(q2),
                np.array([[self.L1**2*self.m1
                           + self.L1**2*self.m2
                           + self.L2**2*self.m2
                           + 2*self.L1*self.L2*self.m2*np.cos(q2),
                           self.L2*self.m2*(self.L2+self.L1*np.cos(q2))],
                          [self.L2*self.m2*(self.L2+self.L1*np.cos(q2)),
                           self.L2**2*self.m2]])
                ))
        
    
    def testC(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.coriolis_centrifugal_matrix(q2,dq1,dq2),
                np.array([[-2*self.L1*self.L2*dq2*self.m2*np.sin(q2),
                           -self.L1*self.L2*dq2*self.m2*np.sin(q2)],
                          [self.L1*self.L2*self.m2*np.sin(q2)*(dq1-dq2),
                           self.L1*self.L2*dq1*self.m2*np.sin(q2)]])
                ))
                              
        
    def testQgrav(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.gravity_vector(q1,q2),
                np.array([[self.g*(self.L2*self.m2*np.cos(q1+q2)
                           +self.L1*self.m1*np.cos(q1)
                           +self.L1*self.m2*np.cos(q1))],
                          [self.L2*self.g*self.m2*np.cos(q1+q2)]])
                ))
        
    def testQ(self):
        q1 = random.random()
        q2 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        Q1 = (self.L1**2*ddq1*self.m1+self.L1**2*ddq1*self.m2 
              + self.L2**2*ddq1*self.m2+self.L2**2*ddq2*self.m2
              + self.L2*self.g*self.m2*np.cos(q1+q2)
              + self.L1*self.g*self.m1*np.cos(q1)
              + self.L1*self.g*self.m2*np.cos(q1)
              - self.L1*self.L2*dq2**2*self.m2*np.sin(q2)
              + 2*self.L1*self.L2*ddq1*self.m2*np.cos(q2)
              + self.L1*self.L2*ddq2*self.m2*np.cos(q2)
              - 2*self.L1*self.L2*dq1*dq2*self.m2*np.sin(q2))
        
        Q2 = self.L2*self.m2*(self.L1*np.sin(q2)*dq1**2
                              + self.L2*ddq1
                              + self.L2*ddq2
                              + self.g*np.cos(q1+q2)
                              + self.L1*ddq1*np.cos(q2))       
        for plant in self.plants:
            self.assertIsNone(np.testing.assert_allclose(
                plant.inverse_dynamics(q1,q2,dq1,dq2,ddq1,ddq2),
                np.array([[Q1],[Q2]])
            ))
    
class TestGeneratedPythonCode(AbstractGeneratedCodeTests,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        
        simplify_ex=True # necessary due to attributes in function calls
        cse=random.choice([True, False])
        parallel=random.choice([True, False])
                
        
        cls.skd.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.skd.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.skd.generate_code(python=True,C=False,Matlab=False,cython=False,latex=False,
                            folder=folder, use_global_vars=True, name="testplant")
        
        cls.L1 = random.random()
        cls.L2 = random.random()
        cls.g = 9.81
        cls.m1 = random.random()
        cls.m2 = random.random()
        
        cls.plants = []
        
        # import generated python code
        from generated_code.python.testplant import Testplant as Pyplant
        cls.pyplant = Pyplant(L1=cls.L1,L2=cls.L2,g=cls.g,m1=cls.m1,m2=cls.m2)
        cls.plants.append(cls.pyplant)
    
    @unittest.skipIf(not delete_generated_code,"selected to keep generated code")    
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.folder)
        except:
            pass

@unittest.skipIf(cython is None, "Skip cython test as I cannot import cython")
class TestGeneratedCythonCode(AbstractGeneratedCodeTests,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        
        simplify_ex=True # necessary due to attributes in function calls
        cse=random.choice([True, False])
        parallel=random.choice([True, False])
                
        
        cls.skd.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.skd.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.skd.generate_code(python=False,C=False,Matlab=False,cython=True,latex=False,
                            folder=folder, use_global_vars=True, name="testplant")
        
        cls.L1 = random.random()
        cls.L2 = random.random()
        cls.g = 9.81
        cls.m1 = random.random()
        cls.m2 = random.random()
        
        cls.plants = []
        
        
        # build and import generated cython code
        import pyximport; pyximport.install(build_dir=os.path.join(folder,"cython","build"))
        from generated_code.cython.testplant import Testplant as Cyplant
        cls.cyplant = Cyplant(L1=cls.L1,L2=cls.L2,g=cls.g,m1=cls.m1,m2=cls.m2)
        cls.plants.append(cls.cyplant)
                
    @unittest.skipIf(not delete_generated_code,"selected to keep generated code")    
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.folder)
        except:
            pass
    
@unittest.skipIf(octave is None, "Skip matlab test as I cannot import oct2py")
class TestGeneratedMatlabCode(AbstractGeneratedCodeTests,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        
        simplify_ex=True # necessary due to attributes in function calls
        cse=random.choice([True, False])
        parallel=random.choice([True, False])
                
        
        cls.skd.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.skd.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.skd.generate_code(python=False,C=False,Matlab=True,cython=False,latex=False,
                            folder=folder, use_global_vars=True, name="testplant")
        
        cls.L1 = random.random()
        cls.L2 = random.random()
        cls.g = 9.81
        cls.m1 = random.random()
        cls.m2 = random.random()
        
        cls.plants = []
                
        octave.addpath(os.path.join(folder,"matlab"))
        cls.mplant = MatlabClass(f"testplant({cls.L1},{cls.L2},{cls.g},{cls.m1},{cls.m2})")
        cls.plants.append(cls.mplant)
                
    @unittest.skipIf(not delete_generated_code,"selected to keep generated code")    
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.folder)
        except:
            pass

class TestYamlParser(unittest.TestCase):
    start = "---\n"
    parent = dedent(
        """
        parent:
          - {}
        """)
    child = dedent(
        """
        child:
          - {}
        """)
    support = dedent(
        """
        support:
          - {}
        """
    )
    gravity = "gravity: {}\n"
    representation = "representation: {}"
    joint_screw_coord = [
        dedent(# [3] axis [3] vec
            """
            joint_screw_coord:
              - type: revolute
                axis: {}
                vec: {}
            """
        ),
        dedent(# [3] prismatic axis
            """
            joint_screw_coord:
              - type: prismatic
                axis: {} 
            """
        ),
        dedent(# [6] list of 6 (whole vector)
            """
            joint_screw_coord:
              - {}
            """
        )        
    ]
    body_ref_config = [
        dedent( # [[4x4]] transfomation matrix
            """
            body_ref_config:
              - {}
            """
        ),
        dedent( # [3] axis, float angle, [3] translation
            """
            body_ref_config:
              - rotation:
                  axis: {}
                  angle: {}
                translation: {}
            """    
        ),
        dedent( # [[3,3]] SO3 pose, [3] translation
            """
            body_ref_config:
              - rotation:
                  {}
                translation: {}
            """    
        ),
        dedent( # [3] rpy, [3] translation
            """
            body_ref_config:
              - rotation:
                  rpy: {}
                translation: {}
            """    
        ),
        dedent( # [6] xyzrpy
            """
            body_ref_config:
              - xyzrpy: {}
            """    
        ),
        dedent( # [4] quaternion, [3] translation
            """
            body_ref_config:
              - rotation:
                  Q: {}
                translation: {}
            """    
        )
    ]
    ee = [
        dedent( # [3] axis, float angle, [3] translation
            """
            ee:
              rotation:
                axis: {}
                angle: {}
              translation: {}
            """
        ),
        dedent( # [3] axis, float angle, [3] translation
            """
            ee:
              - rotation:
                  axis: {}
                  angle: {}
                translation: {}
            """
        ),
        dedent( # [[4x4]] transfomation matrix
            """
            ee:
              - {}
            """
        ),
        dedent( # [[3,3]] SO3 pose, [3] translation
            """
            ee:
              - rotation:
                  {}
                translation: {}
            """    
        ),
        dedent( # [3] rpy, [3] translation
            """
            ee:
              - rotation:
                  rpy: {}
                translation: {}
            """    
        ),
        dedent( # [6] xyzrpy
            """
            ee:
              - xyzrpy: {}
            """    
        ),
        dedent( # [4] quaternion, [3] translation
            """
            ee:
              - rotation:
                  Q: {}
                translation: {}
            """    
        )
    ]
    mass_inertia = [
        dedent( # [[6x6]] matrix
            """
            mass_inertia:
              - {}
            """
        ),
        # float, [[3x3]] inertia matrix, [3] com
        # or: float, [6] inertia values, [3] com
        # or: float, float, [3] com
        dedent(
            """
            mass_inertia:
              - mass: {}
                inertia: {}
                com: {}
            """
        ),
        dedent( # float, 6 x float, [3] com
            """
            mass_inertia:
              - mass: {}
                inertia:
                  Ixx: {}
                  Ixy: {}
                  Ixz: {}
                  Iyy: {}
                  Iyz: {}
                  Izz: {}
                com: {}
            """
        ),
        dedent( # float, int index, bool, [3] com
            """
            mass_inertia:
              - mass: {}
                inertia:
                  index: {}
                  pointmass: {}
                com: {}
            """
        ),
    ]
    q = "q: [q1]\n"
    qd = "qd: [dq1]\n"
    q2d = "q2d: [ddq1]\n"
    q3d = "q3d: [dddq1]\n"
    q4d = "q4d: [ddddq1]\n"
    WEE = "WEE: [Mx,My,Mz,Fx,Fy,Fz]\n"
    WDEE = "WDEE: [dMx,dMy,dMz,dFx,dFy,dFz]\n"
    W2DEE = "W2DEE: [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz]\n"

    def testYamlParser(self):
        joint_screw = [
            self.joint_screw_coord[0].format([0,0,1],[0,0,0]),
            # self.joint_screw_coord[1].format([0,0,1]),
            self.joint_screw_coord[2].format([0,0,1,0,0,0]),
        ]
        
        body_ref = [
            self.body_ref_config[0].format("[[cos(a),-sin(a),0,x],"
                                           " [sin(a),cos(a),0,y],"
                                           " [0,0,1,z],"
                                           " [0,0,0,1]]"), # SE3
            self.body_ref_config[1].format([0,0,1],"a","[x,y,z]"), # axis, angle, translation
            self.body_ref_config[2].format("[[cos(a),-sin(a),0],"
                                           " [sin(a),cos(a),0],"
                                           " [0,0,1]]",
                                           "[x,y,z]"), # SO3, translation
            self.body_ref_config[3].format("[0,0,a]","[x,y,z]"), # rpy, translation
            self.body_ref_config[4].format("[x,y,z,0,0,a]"), # xyzrpy
            self.body_ref_config[5].format("[cos(a/2),0,0,sin(a/2)]","[x,y,z]"), # Q, translation
            ]
        ee = [
            self.ee[0].format([0,0,1],"a","[x,y,z]"), # axis, angle, translation
            self.ee[1].format([0,0,1],"a","[x,y,z]"), # axis, angle, translation
            self.ee[2].format("[[cos(a),-sin(a),0,x],"
                                           " [sin(a),cos(a),0,y],"
                                           " [0,0,1,z],"
                                           " [0,0,0,1]]"), # SE3
            self.ee[3].format("[[cos(a),-sin(a),0],"
                                           " [sin(a),cos(a),0],"
                                           " [0,0,1]]",
                                           "[x,y,z]"), # SO3, translation
            self.ee[4].format("[0,0,a]","[x,y,z]"), # rpy, translation
            self.ee[5].format("[x,y,z,0,0,a]"), # xyzrpy
            self.ee[6].format("[cos(a/2),0,0,sin(a/2)]","[x,y,z]"), # Q, translation
            
        ]
        mb = [
            self.mass_inertia[0].format(dedent(
                """\
                [[   Ixx1,    Ixy1,    Ixz1,       0, -cz1*m1,  cy1*m1],
                 [   Ixy1,    Iyy1,    Iyz1,  cz1*m1,       0, -cx1*m1],
                 [   Ixz1,    Iyz1,    Izz1, -cy1*m1,  cx1*m1,       0],
                 [      0,  cz1*m1, -cy1*m1,      m1,       0,       0],
                 [-cz1*m1,       0,  cx1*m1,       0,      m1,       0],
                 [ cy1*m1, -cx1*m1,       0,       0,       0,      m1]]""")), #(6,6) matrix
            self.mass_inertia[1].format("m1","[[Ixx1,Ixy1,Ixz1],"
                                             " [Ixy1,Iyy1,Iyz1],"
                                             " [Ixz1,Iyz1,Izz1]]",
                                        "[cx1,cy1,cz1]"), # m, I, com
            self.mass_inertia[1].format("m1","[Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1]", "[cx1,cy1,cz1]"), # m, [Ixyz], com
            # self.mass_inertia[1].format("m1","I1", "[cx1,cy1,cz1]"), # m, x * 1, com
            self.mass_inertia[2].format("m1","Ixx1","Ixy1","Ixz1","Iyy1","Iyz1","Izz1", "[cx1,cy1,cz1]"), # m, Ixx, Ixy, ..., Izz, com
            self.mass_inertia[3].format("m1",1,False,"[cx1,cy1,cz1]"), # m, index, pointmass, com
        ]
        for i in range(max(len(joint_screw),len(body_ref),len(ee),len(mb))):    
            file = "\n".join([
                self.start,
                self.parent.format(0),
                self.child.format([]),
                self.support.format([1]),
                self.gravity.format("[0,0,g]"),
                self.representation.format("spatial"),
                joint_screw[i%len(joint_screw)],
                body_ref[i%len(body_ref)],
                ee[i%len(ee)],
                mb[i%len(mb)],
                self.q,
                self.qd,
                self.q2d,
                self.q3d,
                self.q4d,
                self.WEE,
                self.WDEE,
                self.W2DEE
            ])
            with unittest.mock.patch(
                'builtins.open',
                new=unittest.mock.mock_open(read_data=file),
                create=True
            ) as mock_file:
                skd = robot_from_yaml("robot.yaml")
            with self.subTest("parent"):
                self.assertEqual(skd.parent, [0])
            with self.subTest("child"):
                self.assertEqual(skd.child, [[]])
            with self.subTest("support"):
                self.assertEqual(skd.support, [[1]])
            with self.subTest("gravity_vector"):
                self.assertEqual(skd.gravity_vector,[0,0,symbols("g")])
            with self.subTest("config_representation"):
                self.assertEqual(skd.config_representation, "spatial")
            with self.subTest("joint_screw_coord"):
                self.assertEqual(skd.joint_screw_coord, [Matrix([0,0,1,0,0,0])])
            with self.subTest("body_ref_config"):
                self.assertEqual(skd.simplify(skd.body_ref_config[0]), 
                                  transformation_matrix(SO3Exp([0,0,1],symbols("a")),Matrix(["x","y","z"])))
            with self.subTest("ee"):
                self.assertEqual(skd.simplify(skd.ee), transformation_matrix(SO3Exp([0,0,1],symbols("a")),Matrix(["x","y","z"])))
            with self.subTest("mass_inertia"):
                self.assertEqual(skd.simplify(skd.Mb[0]-Matrix(parse_expr(dedent("""\
                    [[   Ixx1,    Ixy1,    Ixz1,       0, -cz1*m1,  cy1*m1],
                    [   Ixy1,    Iyy1,    Iyz1,  cz1*m1,       0, -cx1*m1],
                    [   Ixz1,    Iyz1,    Izz1, -cy1*m1,  cx1*m1,       0],
                    [      0,  cz1*m1, -cy1*m1,      m1,       0,       0],
                    [-cz1*m1,       0,  cx1*m1,       0,      m1,       0],
                    [ cy1*m1, -cx1*m1,       0,       0,       0,      m1]]"""
                    )))),
                    zeros(6,6)
                )
            with self.subTest("q"):
                self.assertEqual(skd.q, Matrix([["q1"]]))
            with self.subTest("qd"):
                self.assertEqual(skd.qd, Matrix([["dq1"]]))
            with self.subTest("q2d"):
                self.assertEqual(skd.q2d, Matrix([["ddq1"]]))
            with self.subTest("q3d"):
                self.assertEqual(skd.q3d, Matrix([["dddq1"]]))
            with self.subTest("q4d"):
                self.assertEqual(skd.q4d, Matrix([["ddddq1"]]))
            with self.subTest("WEE"):
                self.assertEqual(skd.WEE, Matrix([*symbols("Mx,My,Mz,Fx,Fy,Fz")]))
            with self.subTest("WDEE"):
                self.assertEqual(skd.WDEE, Matrix([*symbols("dMx,dMy,dMz,dFx,dFy,dFz")]))
            with self.subTest("W2DEE"):
                self.assertEqual(skd.W2DEE, Matrix([*symbols("ddMx,ddMy,ddMz,ddFx,ddFy,ddFz")]))
            



class MatlabClass():
    _counter = 0
    def __init__(self, objdef = "") -> None:
        """Use matlab object as python class.

        Args:
            objdef (str, optional): Class initialization as string. Defaults to "".
        """
        MatlabClass._counter += 1
        self.name = f"object_for_python{MatlabClass._counter}"
        octave.eval(f"{self.name} = {objdef};")
    
    def __getattr__(self, item):
        """Maps values to attributes.
        Only called if there *isn't* an attribute with this name
        """
        def f(*args):
            call = f"{self.name}.{item}({','.join([str(arg) for arg in args])});"
            return octave.eval(call)
        return f    

if __name__ == "__main__":
    unittest.main()