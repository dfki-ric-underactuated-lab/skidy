import numpy


class Plant():
	def __init__(self, L1, L2, g, m1, m2):
		self.L1, self.L2, self.g, self.m1, self.m2 = L1, L2, g, m1, m2

	def forward_kinematics(self, q1, q2):
		L1, L2 = self.L1, self.L2
		forward_kinematics = numpy.array([[numpy.cos(q1 + q2), -numpy.sin(q1 + q2), 0, L1*numpy.cos(q1) + L2*numpy.cos(q1 + q2)], [numpy.sin(q1 + q2), numpy.cos(q1 + q2), 0, L1*numpy.sin(q1) + L2*numpy.sin(q1 + q2)], [0, 0, 1, 0], [0, 0, 0, 1]])
		return forward_kinematics

	def system_jacobian_matrix(self, q2):
		L1 = self.L1
		system_jacobian_matrix = numpy.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [L1*numpy.sin(q2), 0], [L1*numpy.cos(q2), 0], [0, 0]])
		return system_jacobian_matrix

	def body_jacobian_matrix(self, q2):
		L1 = self.L1
		body_jacobian_matrix = numpy.array([[0, 0], [0, 0], [1, 1], [L1*numpy.sin(q2), 0], [L1*numpy.cos(q2), 0], [0, 0]])
		return body_jacobian_matrix

	def hybrid_jacobian_matrix(self, q1):
		L1 = self.L1
		hybrid_jacobian_matrix = numpy.array([[0, 0], [0, 0], [1, 1], [-L1*numpy.sin(q1), 0], [L1*numpy.cos(q1), 0], [0, 0]])
		return hybrid_jacobian_matrix

	def system_jacobian_dot(self, dq2, q2):
		L1 = self.L1
		system_jacobian_dot = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [L1*dq2*numpy.cos(q2), 0], [-L1*dq2*numpy.sin(q2), 0], [0, 0]])
		return system_jacobian_dot

	def body_twist_ee(self, dq1, dq2, q2):
		L1, L2 = self.L1, self.L2
		body_twist_ee = numpy.array([[0], [0], [dq1 + dq2], [L1*dq1*numpy.sin(q2)], [L1*dq1*numpy.cos(q2) + L2*(dq1 + dq2)], [0]])
		return body_twist_ee

	def hybrid_twist_ee(self, dq1, dq2, q1, q2):
		L1, L2 = self.L1, self.L2
		hybrid_twist_ee = numpy.array([[0], [0], [dq1 + dq2], [-L1*dq1*numpy.sin(q1) - L2*dq1*numpy.sin(q1 + q2) - L2*dq2*numpy.sin(q1 + q2)], [L1*dq1*numpy.cos(q1) + L2*dq1*numpy.cos(q1 + q2) + L2*dq2*numpy.cos(q1 + q2)], [0]])
		return hybrid_twist_ee

	def body_jacobian_matrix_ee(self, q2):
		L1, L2 = self.L1, self.L2
		body_jacobian_matrix_ee = numpy.array([[0, 0], [0, 0], [1, 1], [L1*numpy.sin(q2), 0], [L1*numpy.cos(q2) + L2, L2], [0, 0]])
		return body_jacobian_matrix_ee

	def hybrid_jacobian_matrix_ee(self, q1, q2):
		L1, L2 = self.L1, self.L2
		hybrid_jacobian_matrix_ee = numpy.array([[0, 0], [0, 0], [1, 1], [-L1*numpy.sin(q1) - L2*numpy.sin(q1 + q2), -L2*numpy.sin(q1 + q2)], [L1*numpy.cos(q1) + L2*numpy.cos(q1 + q2), L2*numpy.cos(q1 + q2)], [0, 0]])
		return hybrid_jacobian_matrix_ee

	def generalized_mass_inertia_matrix(self, q2):
		L1, L2, m1, m2 = self.L1, self.L2, self.m1, self.m2
		generalized_mass_inertia_matrix = numpy.array([[L1**2*m1 + L1**2*m2 + 2*L1*L2*m2*numpy.cos(q2) + L2**2*m2, L2*m2*(L1*numpy.cos(q2) + L2)], [L2*m2*(L1*numpy.cos(q2) + L2), L2**2*m2]])
		return generalized_mass_inertia_matrix

	def coriolis_centrifugal_matrix(self, dq1, dq2, q2):
		L1, L2, m2 = self.L1, self.L2, self.m2
		coriolis_centrifugal_matrix = numpy.array([[-2*L1*L2*dq2*m2*numpy.sin(q2), -L1*L2*dq2*m2*numpy.sin(q2)], [L1*L2*m2*(dq1 - dq2)*numpy.sin(q2), L1*L2*dq1*m2*numpy.sin(q2)]])
		return coriolis_centrifugal_matrix

	def gravity_vector(self, q1, q2):
		L1, L2, g, m1, m2 = self.L1, self.L2, self.g, self.m1, self.m2
		gravity_vector = numpy.array([[g*(L1*m1*numpy.cos(q1) + L1*m2*numpy.cos(q1) + L2*m2*numpy.cos(q1 + q2))], [L2*g*m2*numpy.cos(q1 + q2)]])
		return gravity_vector

	def inverse_dynamics(self, ddq1, ddq2, dq1, dq2, q1, q2):
		L1, L2, g, m1, m2 = self.L1, self.L2, self.g, self.m1, self.m2
		inverse_dynamics = numpy.array([[L1**2*ddq1*m1 + L1**2*ddq1*m2 + 2*L1*L2*ddq1*m2*numpy.cos(q2) + L1*L2*ddq2*m2*numpy.cos(q2) - 2*L1*L2*dq1*dq2*m2*numpy.sin(q2) - L1*L2*dq2**2*m2*numpy.sin(q2) + L1*g*m1*numpy.cos(q1) + L1*g*m2*numpy.cos(q1) + L2**2*ddq1*m2 + L2**2*ddq2*m2 + L2*g*m2*numpy.cos(q1 + q2)], [L2*m2*(L1*ddq1*numpy.cos(q2) + L1*dq1**2*numpy.sin(q2) + L2*ddq1 + L2*ddq2 + g*numpy.cos(q1 + q2))]])
		return inverse_dynamics

	def hybrid_acceleration(self, ddq1, ddq2, dq1, q1):
		L1 = self.L1
		hybrid_acceleration = numpy.array([[0], [0], [ddq1 + ddq2], [-L1*(ddq1*numpy.sin(q1) + dq1**2*numpy.cos(q1))], [L1*(ddq1*numpy.cos(q1) - dq1**2*numpy.sin(q1))], [0]])
		return hybrid_acceleration

	def body_acceleration(self, ddq1, ddq2, dq1, dq2, q2):
		L1 = self.L1
		body_acceleration = numpy.array([[0], [0], [ddq1 + ddq2], [L1*(ddq1*numpy.sin(q2) + dq1*dq2*numpy.cos(q2))], [L1*(ddq1*numpy.cos(q2) - dq1*dq2*numpy.sin(q2))], [0]])
		return body_acceleration

	def hybrid_acceleration_ee(self, ddq1, ddq2, dq1, dq2, q1, q2):
		L1, L2 = self.L1, self.L2
		hybrid_acceleration_ee = numpy.array([[0], [0], [ddq1 + ddq2], [-L1*ddq1*numpy.sin(q1) - L1*dq1**2*numpy.cos(q1) - L2*ddq1*numpy.sin(q1 + q2) - L2*ddq2*numpy.sin(q1 + q2) - L2*dq1**2*numpy.cos(q1 + q2) - 2*L2*dq1*dq2*numpy.cos(q1 + q2) - L2*dq2**2*numpy.cos(q1 + q2)], [L1*ddq1*numpy.cos(q1) - L1*dq1**2*numpy.sin(q1) + L2*ddq1*numpy.cos(q1 + q2) + L2*ddq2*numpy.cos(q1 + q2) - L2*dq1**2*numpy.sin(q1 + q2) - 2*L2*dq1*dq2*numpy.sin(q1 + q2) - L2*dq2**2*numpy.sin(q1 + q2)], [0]])
		return hybrid_acceleration_ee

	def body_acceleration_ee(self, ddq1, ddq2, dq1, dq2, q2):
		L1, L2 = self.L1, self.L2
		body_acceleration_ee = numpy.array([[0], [0], [ddq1 + ddq2], [L1*(ddq1*numpy.sin(q2) + dq1*dq2*numpy.cos(q2))], [L1*(ddq1*numpy.cos(q2) - dq1*dq2*numpy.sin(q2)) + L2*(ddq1 + ddq2)], [0]])
		return body_acceleration_ee

	def hybrid_jacobian_matrix_dot(self, dq1, q1):
		L1 = self.L1
		hybrid_jacobian_matrix_dot = numpy.array([[0, 0], [0, 0], [0, 0], [-L1*dq1*numpy.cos(q1), 0], [-L1*dq1*numpy.sin(q1), 0], [0, 0]])
		return hybrid_jacobian_matrix_dot

	def body_jacobian_matrix_dot(self, dq2, q2):
		L1 = self.L1
		body_jacobian_matrix_dot = numpy.array([[0, 0], [0, 0], [0, 0], [L1*dq2*numpy.cos(q2), 0], [-L1*dq2*numpy.sin(q2), 0], [0, 0]])
		return body_jacobian_matrix_dot

	def hybrid_jacobian_matrix_ee_dot(self, dq1, dq2, q1, q2):
		L1, L2 = self.L1, self.L2
		hybrid_jacobian_matrix_ee_dot = numpy.array([[0, 0], [0, 0], [0, 0], [-L1*dq1*numpy.cos(q1) - L2*dq1*numpy.cos(q1 + q2) - L2*dq2*numpy.cos(q1 + q2), -L2*(dq1 + dq2)*numpy.cos(q1 + q2)], [-L1*dq1*numpy.sin(q1) - L2*dq1*numpy.sin(q1 + q2) - L2*dq2*numpy.sin(q1 + q2), -L2*(dq1 + dq2)*numpy.sin(q1 + q2)], [0, 0]])
		return hybrid_jacobian_matrix_ee_dot

	def body_jacobian_matrix_ee_dot(self, dq2, q2):
		L1 = self.L1
		body_jacobian_matrix_ee_dot = numpy.array([[0, 0], [0, 0], [0, 0], [L1*dq2*numpy.cos(q2), 0], [-L1*dq2*numpy.sin(q2), 0], [0, 0]])
		return body_jacobian_matrix_ee_dot