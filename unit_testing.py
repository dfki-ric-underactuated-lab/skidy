import unittest
import sys
import kinematics_generator
from sympy import *
import random


    
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


class TestFKin(abstractFKinTest, unittest.TestCase):
    # @classmethod
    # def setUpTestData(cls):
    #     # Set up data for the whole TestCase
    #     cls.foo = Foo.objects.create(bar="Test")
    @classmethod
    def setUpClass(self):
        self.s = kinematics_generator.SymbolicKinDyn()
        self.q1, self.q2 = symbols("q1 q2")
        self.dq1, self.dq2 = symbols("dq1 dq2")
        self.ddq1, self.ddq2 = symbols("ddq1 ddq2")

        self.L1, self.L2 = symbols("L1 L2", real=1, constant=1)
        
        # Joint screw coordinates in spatial representation

        self.s.Y = []
        e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e1, y1.cross(e1)]))

        e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y2 = Matrix([self.L1, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e2, y2.cross(e2)]))

        # Reference configurations of bodies (i.e. of body-fixed reference frames)

        r1 = Matrix([0, 0, 0])
        r2 = Matrix([self.L1, 0, 0])

        self.s.A = []
        self.s.A.append(Matrix(Identity(3)).row_join(
            r1).col_join(Matrix([0, 0, 0, 1]).T))
        self.s.A.append(Matrix(Identity(3)).row_join(
            r2).col_join(Matrix([0, 0, 0, 1]).T))

        # End-effector configuration wrt last link body fixed frame in the chain
        re = Matrix([self.L2, 0, 0])
        self.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

        # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
        self.s.X = []
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[0])*self.s.Y[0])
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[1])*self.s.Y[1])

        # Declaring generalised vectors
        self.q = Matrix([self.q1, self.q2])
        self.qd = Matrix([self.dq1, self.dq2])
        self.q2d = Matrix([self.ddq1, self.ddq2])
        T = self.s.closed_form_kinematics_body_fixed(self.q,self.qd,self.q2d)


class TestFKin_parallel(abstractFKinTest, unittest.TestCase):
    # @classmethod
    # def setUpTestData(cls):
    #     # Set up data for the whole TestCase
    #     cls.foo = Foo.objects.create(bar="Test")
    @classmethod
    def setUpClass(self):
        self.s = kinematics_generator.SymbolicKinDyn()
        self.q1, self.q2 = symbols("q1 q2")
        self.dq1, self.dq2 = symbols("dq1 dq2")
        self.ddq1, self.ddq2 = symbols("ddq1 ddq2")

        self.L1, self.L2 = symbols("L1 L2", real=1, constant=1)
        
        # Joint screw coordinates in spatial representation

        self.s.Y = []
        e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e1, y1.cross(e1)]))

        e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y2 = Matrix([self.L1, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e2, y2.cross(e2)]))

        # Reference configurations of bodies (i.e. of body-fixed reference frames)

        r1 = Matrix([0, 0, 0])
        r2 = Matrix([self.L1, 0, 0])

        self.s.A = []
        self.s.A.append(Matrix(Identity(3)).row_join(
            r1).col_join(Matrix([0, 0, 0, 1]).T))
        self.s.A.append(Matrix(Identity(3)).row_join(
            r2).col_join(Matrix([0, 0, 0, 1]).T))

        # End-effector configuration wrt last link body fixed frame in the chain
        re = Matrix([self.L2, 0, 0])
        self.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

        # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
        self.s.X = []
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[0])*self.s.Y[0])
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[1])*self.s.Y[1])

        # Declaring generalised vectors
        self.q = Matrix([self.q1, self.q2])
        self.qd = Matrix([self.dq1, self.dq2])
        self.q2d = Matrix([self.ddq1, self.ddq2])
        T = self.s.closed_form_kinematics_body_fixed_parallel(self.q,self.qd,self.q2d)


class AbstractInvDyn():
    
    def testM(self):
        self.assertEqual(
            self.s.M,
            Matrix([[self.L1**2*self.m1+ self.L1**2*self.m2+self.L2**2*self.m2+2*self.L1*self.L2*self.m2*cos(self.q2),
                     self.L2*self.m2*(self.L2+self.L1*cos(self.q2))],
                    [self.L2*self.m2*(self.L2+self.L1*cos(self.q2)),
                     self.L2**2*self.m2]])
        )      
        
    def testJ(self):
        self.assertEqual(
            self.s.J,
            Matrix([[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,1],[self.L1*sin(self.q2),0],[self.L1*cos(self.q2),0],[0,0]])
        )
    
    def testC(self):
        self.assertEqual(
            self.s.C,
            Matrix([[-2*self.L1*self.L2*self.dq2*self.m2*sin(self.q2),
                     -self.L1*self.L2*self.dq2*self.m2*sin(self.q2)],
                    [self.L1*self.L2*self.m2*sin(self.q2)*(self.dq1-self.dq2),
                     self.L1*self.L2*self.dq1*self.m2*sin(self.q2)]])
        )
        
    def testQgrav(self):
        self.assertEqual(
            self.s.Qgrav,
            Matrix([[self.g*(self.L2*self.m2*cos(self.q1+self.q2)+self.L1*self.m1*cos(self.q1)+self.L1*self.m2*cos(self.q1))],
                    [self.L2*self.g*self.m2*cos(self.q1+self.q2)]])
        )
        
    def testQ(self):
        Q1 = (self.L1**2*self.ddq1*self.m1+self.L1**2*self.ddq1*self.m2 + self.L2**2*self.ddq1*self.m2+self.L2**2*self.ddq2*self.m2+self.L2*self.g*self.m2*cos(self.q1+self.q2)
              +self.L1*self.g*self.m1*cos(self.q1)+self.L1*self.g*self.m2*cos(self.q1)
              -self.L1*self.L2*self.dq2**2*self.m2*sin(self.q2)+2*self.L1*self.L2*self.ddq1*self.m2*cos(self.q2)
              +self.L1*self.L2*self.ddq2*self.m2*cos(self.q2)-2*self.L1*self.L2*self.dq1*self.dq2*self.m2*sin(self.q2))
        
        Q2 = self.L2*self.m2*(self.L1*sin(self.q2)*self.dq1**2+self.L2*self.ddq1+self.L2*self.ddq2+self.g*cos(self.q1+self.q2)+self.L1*self.ddq1*cos(self.q2))       
        # print(simplify(self.s.Q[0]-Q1))
        self.assertEqual(
            simplify(self.s.Q - Matrix([Q1,Q2])),
            zeros(2,1)
        )

        
class TestInvDyn(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.s = kinematics_generator.SymbolicKinDyn()
        self.q1, self.q2 = symbols("q1 q2")
        self.dq1, self.dq2 = symbols("dq1 dq2")
        self.ddq1, self.ddq2 = symbols("ddq1 ddq2")

        
        self.m1, self.m2, self.I1, self.I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
        self.cg1, self.cg2, self.g = symbols("cg1 cg2 g", real=1, constant=1)
        self.L1, self.L2 = symbols("L1 L2", real=1, constant=1)
        
        self.s.gravity_vector = Matrix([0, self.g, 0])

        # Joint screw coordinates in spatial representation

        self.s.Y = []
        e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e1, y1.cross(e1)]))

        e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y2 = Matrix([self.L1, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e2, y2.cross(e2)]))

        # Reference configurations of bodies (i.e. of body-fixed reference frames)

        r1 = Matrix([0, 0, 0])
        r2 = Matrix([self.L1, 0, 0])

        self.s.A = []
        self.s.A.append(Matrix(Identity(3)).row_join(
            r1).col_join(Matrix([0, 0, 0, 1]).T))
        self.s.A.append(Matrix(Identity(3)).row_join(
            r2).col_join(Matrix([0, 0, 0, 1]).T))

        # End-effector configuration wrt last link body fixed frame in the chain
        re = Matrix([self.L2, 0, 0])
        self.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

        # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
        self.s.X = []
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[0])*self.s.Y[0])
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[1])*self.s.Y[1])

        # Mass-Inertia parameters
        cg1 = Matrix([self.L1, 0, 0]).T
        cg2 = Matrix([self.L2, 0, 0]).T
        I1 = self.m1*self.L1**2
        I2 = self.m2*self.L2**2

        self.s.Mb = []
        self.s.Mb.append(self.s.MassMatrixMixedData(self.m1, I1*Identity(3), cg1))
        self.s.Mb.append(self.s.MassMatrixMixedData(self.m2, I2*Identity(3), cg2))

        # Declaring generalised vectors
        self.q = Matrix([self.q1, self.q2])
        self.qd = Matrix([self.dq1, self.dq2])
        self.q2d = Matrix([self.ddq1, self.ddq2])
        Q = self.s.closed_form_inv_dyn_body_fixed(self.q,self.qd,self.q2d)


class TestInvDynParallel(AbstractInvDyn,unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.s = kinematics_generator.SymbolicKinDyn()
        self.q1, self.q2 = symbols("q1 q2")
        self.dq1, self.dq2 = symbols("dq1 dq2")
        self.ddq1, self.ddq2 = symbols("ddq1 ddq2")

        
        self.m1, self.m2, self.I1, self.I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
        self.cg1, self.cg2, self.g = symbols("cg1 cg2 g", real=1, constant=1)
        self.L1, self.L2 = symbols("L1 L2", real=1, constant=1)
        
        self.s.gravity_vector = Matrix([0, self.g, 0])

        # Joint screw coordinates in spatial representation

        self.s.Y = []
        e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e1, y1.cross(e1)]))

        e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
        y2 = Matrix([self.L1, 0, 0])  # Vector to joint axis from inertial Frame
        # Joint screw coordinates in spacial representation
        self.s.Y.append(Matrix([e2, y2.cross(e2)]))

        # Reference configurations of bodies (i.e. of body-fixed reference frames)

        r1 = Matrix([0, 0, 0])
        r2 = Matrix([self.L1, 0, 0])

        self.s.A = []
        self.s.A.append(Matrix(Identity(3)).row_join(
            r1).col_join(Matrix([0, 0, 0, 1]).T))
        self.s.A.append(Matrix(Identity(3)).row_join(
            r2).col_join(Matrix([0, 0, 0, 1]).T))

        # End-effector configuration wrt last link body fixed frame in the chain
        re = Matrix([self.L2, 0, 0])
        self.s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

        # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
        self.s.X = []
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[0])*self.s.Y[0])
        self.s.X.append(self.s.SE3AdjInvMatrix(self.s.A[1])*self.s.Y[1])

        # Mass-Inertia parameters
        cg1 = Matrix([self.L1, 0, 0]).T
        cg2 = Matrix([self.L2, 0, 0]).T
        I1 = self.m1*self.L1**2
        I2 = self.m2*self.L2**2

        self.s.Mb = []
        self.s.Mb.append(self.s.MassMatrixMixedData(self.m1, I1*Identity(3), cg1))
        self.s.Mb.append(self.s.MassMatrixMixedData(self.m2, I2*Identity(3), cg2))

        # Declaring generalised vectors
        self.q = Matrix([self.q1, self.q2])
        self.qd = Matrix([self.dq1, self.dq2])
        self.q2d = Matrix([self.ddq1, self.ddq2])
        Q = self.s.closed_form_inv_dyn_body_fixed_parallel(self.q,self.qd,self.q2d)


    # def testClosed_form_inv_dyn_body_fixed(self):
    #     pass
    
    # def testClosed_form_inv_dyn_body_fixed_parallel(self):
        # pass

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