import unittest
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
import kinematics_generator
from sympy import Matrix, cos, sin, symbols, Identity, simplify, zeros
import random




def prepare(cls):
    cls.s = kinematics_generator.SymbolicKinDyn()
    cls.q1, cls.q2 = symbols("q1 q2")
    cls.dq1, cls.dq2 = symbols("dq1 dq2")
    cls.ddq1, cls.ddq2 = symbols("ddq1 ddq2")

    
    cls.m1, cls.m2, cls.I1, cls.I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
    cls.cg1, cls.cg2, cls.g = symbols("cg1 cg2 g", real=1, constant=1)
    cls.L1, cls.L2 = symbols("L1 L2", real=1, constant=1)
    
    cls.s.gravity_vector = Matrix([0, cls.g, 0])

    # Joint screw coordinates in spatial representation

    cls.s.Y = []
    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    cls.s.Y.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([cls.L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    cls.s.Y.append(Matrix([e2, y2.cross(e2)]))

    # Reference configurations of bodies (i.e. of body-fixed reference frames)

    r1 = Matrix([0, 0, 0])
    r2 = Matrix([cls.L1, 0, 0])

    cls.s.A = []
    cls.s.A.append(Matrix(Identity(3)).row_join(
        r1).col_join(Matrix([0, 0, 0, 1]).T))
    cls.s.A.append(Matrix(Identity(3)).row_join(
        r2).col_join(Matrix([0, 0, 0, 1]).T))

    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([cls.L2, 0, 0])
    cls.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    cls.s.X = []
    cls.s.X.append(cls.s.SE3AdjInvMatrix(cls.s.A[0])*cls.s.Y[0])
    cls.s.X.append(cls.s.SE3AdjInvMatrix(cls.s.A[1])*cls.s.Y[1])

    # Mass-Inertia parameters
    cg1 = Matrix([cls.L1, 0, 0]).T
    cg2 = Matrix([cls.L2, 0, 0]).T
    I1 = cls.m1*cls.L1**2
    I2 = cls.m2*cls.L2**2

    cls.s.Mb = []
    cls.s.Mb.append(cls.s.MassMatrixMixedData(cls.m1, I1*Identity(3), cg1))
    cls.s.Mb.append(cls.s.MassMatrixMixedData(cls.m2, I2*Identity(3), cg2))

    # Declaring generalised vectors
    cls.q = Matrix([cls.q1, cls.q2])
    cls.qd = Matrix([cls.dq1, cls.dq2])
    cls.q2d = Matrix([cls.ddq1, cls.ddq2])
        

class abstractFKinTest():
    def testfkin(self):
        self.assertEqual(
            self.s.fkin,
            Matrix([[cos(self.q1+self.q2), -sin(self.q1+self.q2),0,self.L2*cos(self.q1+self.q2)+self.L1*cos(self.q1)],
                    [sin(self.q1+self.q2), cos(self.q1+self.q2),0, self.L2*sin(self.q1+self.q2)+self.L1*sin(self.q1)],
                    [0,0,1,0],[0,0,0,1]])
        )
        
    def testJ(self):
        self.assertEqual(
            self.s.J,
            Matrix([[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,1],[self.L1*sin(self.q2),0],[self.L1*cos(self.q2),0],[0,0]])
        )
        
    def testJb_ee(self):
        self.assertEqual(
            self.s.Jb_ee,
            Matrix([[0,0],[0,0],[1,1],[self.L1*sin(self.q2),0],[self.L2+self.L1*cos(self.q2),self.L2],[0,0]])
        )
        
    def testJh_ee(self):
        self.assertEqual(
            self.s.Jh_ee,
            Matrix([[0,0],[0,0],[1,1],[-self.L2*sin(self.q1+self.q2)-self.L1*sin(self.q1),-self.L2*sin(self.q1+self.q2)],[self.L2*cos(self.q1+self.q2)+self.L1*cos(self.q1),self.L2*cos(self.q1+self.q2)],[0,0]])
        )
        
    def testJb(self):
        self.assertEqual(
            self.s.Jb,
            Matrix([[0,0],[0,0],[1,1],[self.L1*sin(self.q2),0],[self.L1*cos(self.q2),0],[0,0]])
        )
        
    def testJh(self):
        self.assertEqual(
            self.s.Jh,
            Matrix([[0,0],[0,0],[1,1],[-self.L1*sin(self.q1),0],[self.L1*cos(self.q1),0],[0,0]])
        )
    
    def testJdot(self):
        self.assertEqual(
            self.s.Jdot,
            Matrix([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[self.L1*self.dq2*cos(self.q2),0],[-self.L1*self.dq2*sin(self.q2),0],[0,0]])
        )
    
    def testVbd_BFn(self):
        self.assertEqual(
            simplify(self.s.Vbd_BFn),
            Matrix([[0],[0],[self.ddq1+self.ddq2],[self.L1*(self.ddq1*sin(self.q2) + self.dq1*self.dq2*cos(self.q2))],[self.L1*(self.ddq1*cos(self.q2) - self.dq1*self.dq2*sin(self.q2))],[0]])
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
                    [- self.L2*self.dq1*sin(self.q1 + self.q2) - self.L2*self.dq2*sin(self.q1 + self.q2) - self.L1*self.dq1*sin(self.q1)],
                    [self.L2*self.dq1*cos(self.q1 + self.q2) + self.L2*self.dq2*cos(self.q1 + self.q2) + self.L1*self.dq1*cos(self.q1)],
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
                    [self.L1*(self.ddq1*cos(self.q2) - self.dq1*self.dq2*sin(self.q2))+self.L2*(self.ddq1+self.ddq2)],
                    [0]])),
            zeros(6,1)
        )
        
        
    def testVhd_ee(self):
        self.assertEqual(
            simplify(simplify(self.s.Vhd_ee)-
            Matrix([[0],
                    [0],
                    [self.ddq1+self.ddq2],
                    [- self.L2*self.dq1**2*cos(self.q1 + self.q2) - self.L2*self.dq2**2*cos(self.q1 + self.q2) - self.L1*self.dq1**2*cos(self.q1) - self.L2*self.ddq1*sin(self.q1 + self.q2) - self.L2*self.ddq2*sin(self.q1 + self.q2) - self.L1*self.ddq1*sin(self.q1) - 2*self.L2*self.dq1*self.dq2*cos(self.q1 + self.q2)],
                    [ self.L2*self.ddq1*cos(self.q1 + self.q2) - self.L2*self.dq2**2*sin(self.q1 + self.q2) - self.L1*self.dq1**2*sin(self.q1) - self.L2*self.dq1**2*sin(self.q1 + self.q2) + self.L2*self.ddq2*cos(self.q1 + self.q2) + self.L1*self.ddq1*cos(self.q1) - 2*self.L2*self.dq1*self.dq2*sin(self.q1 + self.q2)],
                    [0]])),
            zeros(6,1)
        )
        
    
    
    def testJb_ee_dot(self):
        self.assertEqual(
            self.s.Jb_ee_dot,
            Matrix([[0,0],[0,0],[0,0],[self.L1*self.dq2*cos(self.q2),0],[-self.L1*self.dq2*sin(self.q2),0],[0,0]])
        )
        
    def testJh_ee_dot(self):
        self.assertEqual(
            simplify(self.s.Jh_ee_dot-
            Matrix([[0,0],
                    [0,0],
                    [0,0],
                    [- self.L2*self.dq1*cos(self.q1 + self.q2) - self.L2*self.dq2*cos(self.q1 + self.q2) - self.L1*self.dq1*cos(self.q1), -self.L2*cos(self.q1 + self.q2)*(self.dq1 + self.dq2)],
                    [- self.L2*self.dq1*sin(self.q1 + self.q2) - self.L2*self.dq2*sin(self.q1 + self.q2) - self.L1*self.dq1*sin(self.q1), -self.L2*sin(self.q1 + self.q2)*(self.dq1 + self.dq2)],
                    [0,0]])),
            zeros(6,2)
        )
    
        
    def testJh_dot(self):
        self.assertEqual(
            simplify(self.s.Jh_dot-
            Matrix([[0,0],[0,0],[0,0],[-self.L1*self.dq1*cos(self.q1),0],[-self.L1*self.dq1*sin(self.q1),0],[0,0]])),
            zeros(6,2)
        )
    
    def testJb_dot(self):
        self.assertEqual(
            self.s.Jb_dot,
            Matrix([[0,0],[0,0],[0,0],[self.L1*self.dq2*cos(self.q2),0],[-self.L1*self.dq2*sin(self.q2),0],[0,0]])
        )
    
class TestFKin(abstractFKinTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        prepare(self)
        T = self.s.closed_form_kinematics_body_fixed(self.q,self.qd,self.q2d,True,False,False)


class TestFKin_parallel(abstractFKinTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        prepare(self)
        T = self.s.closed_form_kinematics_body_fixed(self.q,self.qd,self.q2d,True,False,True)


class AbstractInvDyn():
    
    def testM(self):
        self.assertEqual(
            simplify(self.s.M-
            Matrix([[self.L1**2*self.m1+ self.L1**2*self.m2+self.L2**2*self.m2+2*self.L1*self.L2*self.m2*cos(self.q2),
                     self.L2*self.m2*(self.L2+self.L1*cos(self.q2))],
                    [self.L2*self.m2*(self.L2+self.L1*cos(self.q2)),
                     self.L2**2*self.m2]])),
            zeros(2,2)
        )      
        
    def testJ(self):
        self.assertEqual(
            self.s.J,
            Matrix([[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,1],[self.L1*sin(self.q2),0],[self.L1*cos(self.q2),0],[0,0]])
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
            Matrix([[self.g*(self.L2*self.m2*cos(self.q1+self.q2)+self.L1*self.m1*cos(self.q1)+self.L1*self.m2*cos(self.q1))],
                    [self.L2*self.g*self.m2*cos(self.q1+self.q2)]])),
            zeros(2,1)
        )
        
    def testQ(self):
        Q1 = (self.L1**2*self.ddq1*self.m1+self.L1**2*self.ddq1*self.m2 + self.L2**2*self.ddq1*self.m2+self.L2**2*self.ddq2*self.m2+self.L2*self.g*self.m2*cos(self.q1+self.q2)
              +self.L1*self.g*self.m1*cos(self.q1)+self.L1*self.g*self.m2*cos(self.q1)
              -self.L1*self.L2*self.dq2**2*self.m2*sin(self.q2)+2*self.L1*self.L2*self.ddq1*self.m2*cos(self.q2)
              +self.L1*self.L2*self.ddq2*self.m2*cos(self.q2)-2*self.L1*self.L2*self.dq1*self.dq2*self.m2*sin(self.q2))
        
        Q2 = self.L2*self.m2*(self.L1*sin(self.q2)*self.dq1**2+self.L2*self.ddq1+self.L2*self.ddq2+self.g*cos(self.q1+self.q2)+self.L1*self.ddq1*cos(self.q2))       
        self.assertEqual(
            simplify(self.s.Q - Matrix([Q1,Q2])),
            zeros(2,1)
        )

        
class TestInvDyn(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(self):
        prepare(self)
        Q = self.s.closed_form_inv_dyn_body_fixed(self.q,self.qd,self.q2d,zeros(6,1),True,False,False)


class TestInvDynParallel(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(self):
        prepare(self)
        Q = self.s.closed_form_inv_dyn_body_fixed(self.q,self.qd,self.q2d,zeros(6,1),True,False,True)


class TestKinGen(unittest.TestCase):
    
    def setUp(self):
        self.s = kinematics_generator.SymbolicKinDyn()
    
    def testInertiaMatrix(self):
        self.assertEqual(
            self.s.InertiaMatrix(1,2,3,4,5,6), 
            Matrix([[1,2,3],[2,4,5],[3,5,6]])
        )
        
    def testTransformationMatrix(self):
        self.assertEqual(
            self.s.TransformationMatrix(Matrix([[1,2,3],[4,5,6],[7,8,9]]),Matrix([10,11,12])),
            Matrix([[1,2,3,10],[4,5,6,11],[7,8,9,12],[0,0,0,1]])
        )

    def testSO3Exp(self):
        t = random.randint(0,100)
        self.assertEqual(
            self.s.SO3Exp(Matrix([0,0,1]), t),
            Matrix([[cos(t),-sin(t),0],[sin(t),cos(t),0],[0,0,1]])
        )   
    
    def testSE3Exp(self):
        t = random.randint(0,100)
        self.assertEqual(
            self.s.SE3Exp(Matrix([0,0,0,0,0,1]),t = t),
            Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]])
        )
        self.assertEqual(
            self.s.SE3Exp(Matrix([0,0,1,0,0,0]),t = t),
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
            self.s.MassMatrixMixedData(m,I,com),
            Matrix([[Ixx,Ixy,Ixz, 0,-m*c3,m*c2],
                    [Ixy,Iyy,Iyz, m*c3,0,-m*c1],
                    [Ixz,Iyz,Izz,-m*c2,m*c1,0],
                    [0,m*c3,-m*c2,m,0,0],
                    [-m*c3,0,m*c1,0,m,0],
                    [m*c2,-m*c1,0,0,0,m]])
        )
    
    # def testSE3Inv(self):
    #     pass
    
    # def testSE3adMatrix(self):
    #     pass
    
    # def testSE3AdjMatrix(self):
    #     pass
    
    # def testSE3AdjInvMatrix(self):
    #     pass
    
    
    
    
        
if __name__ == "__main__":
    unittest.main()