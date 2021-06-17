
#ifndef PROJECT__PLANT__H
#define PROJECT__PLANT__H

void forward_kinematics(double q1, double q2, double *out_533331568737891010);
void system_jacobian_matrix(double q2, double *out_105279131531531221);
void body_jacobian_matrix(double q2, double *out_7158770792675460266);
void hybrid_jacobian_matrix(double q1, double *out_7505495277599162950);
void system_jacobian_dot(double dq2, double q2, double *out_4561059558003435982);
void body_twist_ee(double dq1, double dq2, double q2, double *out_2202487785058670561);
void hybrid_twist_ee(double dq1, double dq2, double q1, double q2, double *out_8673364558687002157);
void body_jacobian_matrix_ee(double q2, double *out_6279707935581118024);
void hybrid_jacobian_matrix_ee(double q1, double q2, double *out_4641531331680948144);
void generalized_mass_inertia_matrix(double q2, double *out_2142058865511428381);
void coriolis_centrifugal_matrix(double dq1, double dq2, double q2, double *out_8731067158543827054);
void gravity_vector(double q1, double q2, double *out_5408515624771725552);
void inverse_dynamics(double ddq1, double ddq2, double dq1, double dq2, double q1, double q2, double *out_7140826462901308056);

#endif

