import unittest
import sys
import os
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
from sympy import Matrix, cos, sin, symbols, Identity, simplify, zeros
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
    cls.s = skidy.SymbolicKinDyn()
    cls.q1, cls.q2 = symbols("q1 q2")
    cls.dq1, cls.dq2 = symbols("dq1 dq2")
    cls.ddq1, cls.ddq2 = symbols("ddq1 ddq2")

    
    cls.m1, cls.m2, cls.I1, cls.I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
    cls.cg1, cls.cg2, cls.g = symbols("cg1 cg2 g", real=1, constant=1)
    cls.L1, cls.L2 = symbols("L1 L2", real=1, constant=1)
    
    cls.s.config_representation = "spatial"
    
    cls.s.gravity_vector = Matrix([0, cls.g, 0])

    # Joint screw coordinates in spatial representation

    cls.s.joint_screw_coord = []
    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spatial representation
    cls.s.joint_screw_coord.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([cls.L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spatial representation
    cls.s.joint_screw_coord.append(Matrix([e2, y2.cross(e2)]))

    # Reference configurations of bodies (i.e. of spatial reference frames)
    r1 = Matrix([0, 0, 0])
    r2 = Matrix([cls.L1, 0, 0])

    cls.s.body_ref_config = []
    cls.s.body_ref_config.append(Matrix(Identity(3)).row_join(
        r1).col_join(Matrix([0, 0, 0, 1]).T))
    cls.s.body_ref_config.append(Matrix(Identity(3)).row_join(
        r2).col_join(Matrix([0, 0, 0, 1]).T))

    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([cls.L2, 0, 0])
    cls.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    # cls.s.joint_screw_coord = []
    # cls.s.joint_screw_coord.append(SE3AdjInvMatrix(cls.s.A[0])*cls.s.Y[0])
    # cls.s.joint_screw_coord.append(SE3AdjInvMatrix(cls.s.A[1])*cls.s.Y[1])

    # Mass-Inertia parameters
    cg1 = Matrix([cls.L1, 0, 0]).T
    cg2 = Matrix([cls.L2, 0, 0]).T
    I1 = cls.m1*cls.L1**2
    I2 = cls.m2*cls.L2**2

    cls.s.Mb = []
    cls.s.Mb.append(mass_matrix_mixed_data(cls.m1, I1*Identity(3), cg1))
    cls.s.Mb.append(mass_matrix_mixed_data(cls.m2, I2*Identity(3), cg2))

    # Declaring generalised vectors
    cls.q = Matrix([cls.q1, cls.q2])
    cls.qd = Matrix([cls.dq1, cls.dq2])
    cls.q2d = Matrix([cls.ddq1, cls.ddq2])
        

class abstractFKinTest():
    def testfkin(self):
        self.assertEqual(
            simplify(
                self.s.fkin
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
                self.s.J
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
                self.s.Jb_ee
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
                self.s.Jh_ee
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
                self.s.Jb
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
                self.s.Jh
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
                self.s.Jdot
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
                self.s.Vbd_BFn 
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
            simplify(self.s.Vhd_BFn -
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
            simplify(self.s.Vb_ee -
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
            simplify(self.s.Vh_ee -
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
            simplify(self.s.Vbd_ee-
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
            simplify(simplify(self.s.Vhd_ee)-
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
                self.s.Jb_ee_dot
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
            simplify(self.s.Jh_ee_dot-
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
            simplify(self.s.Jh_dot-
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
                self.s.Jb_dot
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
        T = cls.s.closed_form_kinematics_body_fixed(cls.q,cls.qd,cls.q2d,True,False,False)


class TestFKin_parallel(abstractFKinTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        T = cls.s.closed_form_kinematics_body_fixed(cls.q,cls.qd,cls.q2d,True,False,True)


class AbstractInvDyn():
    
    def testM(self):
        self.assertEqual(
            simplify(self.s.M-
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
                self.s.J
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
            self.s.simplify(self.s.C-
            Matrix([[-2*self.L1*self.L2*self.dq2*self.m2*sin(self.q2),
                     -self.L1*self.L2*self.dq2*self.m2*sin(self.q2)],
                    [self.L1*self.L2*self.m2*sin(self.q2)*(self.dq1-self.dq2),
                     self.L1*self.L2*self.dq1*self.m2*sin(self.q2)]])),
            zeros(2,2)
        )
        
    def testQgrav(self):
        self.assertEqual(
            self.s.simplify(self.s.Qgrav-
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
            simplify(self.s.Q - Matrix([Q1,Q2])),
            zeros(2,1)
        )

        
class TestInvDyn(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare(cls)
        Q = cls.s.closed_form_inv_dyn_body_fixed(cls.q,cls.qd,cls.q2d,WEE=zeros(6,1),
                                                 simplify=True,cse=False,parallel=False)


class TestInvDynParallel(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(self):
        prepare(self)
        Q = self.s.closed_form_inv_dyn_body_fixed(self.q,self.qd,self.q2d,WEE=zeros(6,1),
                                                  simplify=True,cse=False,parallel=True)


class TestKinGen(unittest.TestCase):
    
    # def setUp(self):
        # self.s = skidy.SymbolicKinDyn()
    
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
                
        
        cls.s.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.s.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.s.generate_code(python=True,C=False,Matlab=False,cython=False,latex=False,
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
                
        
        cls.s.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.s.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.s.generate_code(python=False,C=False,Matlab=False,cython=True,latex=False,
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
                
        
        cls.s.closed_form_kinematics_body_fixed(
            q = cls.q, qd=cls.qd,q2d=cls.q2d,
            simplify=simplify_ex,cse=cse,parallel=parallel)
        cls.s.closed_form_inv_dyn_body_fixed(
            q=cls.q, qd=cls.qd, q2d=cls.q2d, WEE=zeros(6,1),
            simplify=simplify_ex,cse=cse,parallel=parallel)
        
        folder = os.path.join(dirname(__file__),"generated_code")
        cls.folder = folder
        
        cls.s.generate_code(python=False,C=False,Matlab=True,cython=False,latex=False,
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