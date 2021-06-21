
#ifndef PROJECT__PLANT__H
#define PROJECT__PLANT__H

void forward_kinematics(double q1, double q2, double *out);
void system_jacobian_matrix(double q2, double *out);
void body_jacobian_matrix(double q2, double *out);
void hybrid_jacobian_matrix(double q1, double *out);
void system_jacobian_dot(double dq2, double q2, double *out);
void body_twist_ee(double dq1, double dq2, double q2, double *out);
void hybrid_twist_ee(double dq1, double dq2, double q1, double q2, double *out);
void body_jacobian_matrix_ee(double q2, double *out);
void hybrid_jacobian_matrix_ee(double q1, double q2, double *out);
void generalized_mass_inertia_matrix(double q2, double *out);
void coriolis_centrifugal_matrix(double dq1, double dq2, double q2, double *out);
void gravity_vector(double q1, double q2, double *out);
void inverse_dynamics(double ddq1, double ddq2, double dq1, double dq2, double q1, double q2, double *out);
void hybrid_acceleration(double ddq1, double ddq2, double dq1, double q1, double *out);
void body_acceleration(double ddq1, double ddq2, double dq1, double dq2, double q2, double *out);
void hybrid_acceleration_ee(double ddq1, double ddq2, double dq1, double dq2, double q1, double q2, double *out);
void body_acceleration_ee(double ddq1, double ddq2, double dq1, double dq2, double q2, double *out);
void hybrid_jacobian_matrix_dot(double dq1, double q1, double *out);
void body_jacobian_matrix_dot(double dq2, double q2, double *out);
void hybrid_jacobian_matrix_ee_dot(double dq1, double dq2, double q1, double q2, double *out);
void body_jacobian_matrix_ee_dot(double dq2, double q2, double *out);

#endif

