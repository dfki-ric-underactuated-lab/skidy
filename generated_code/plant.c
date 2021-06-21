#include "plant.h"
#include <math.h>

void forward_kinematics(double q1, double q2, double *out) {

   out[0] = cos(q1 + q2);
   out[1] = -sin(q1 + q2);
   out[2] = 0;
   out[3] = L1*cos(q1) + L2*cos(q1 + q2);
   out[4] = sin(q1 + q2);
   out[5] = cos(q1 + q2);
   out[6] = 0;
   out[7] = L1*sin(q1) + L2*sin(q1 + q2);
   out[8] = 0;
   out[9] = 0;
   out[10] = 1;
   out[11] = 0;
   out[12] = 0;
   out[13] = 0;
   out[14] = 0;
   out[15] = 1;

}

void system_jacobian_matrix(double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 1;
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
   out[8] = 0;
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;
   out[12] = 0;
   out[13] = 0;
   out[14] = 0;
   out[15] = 0;
   out[16] = 1;
   out[17] = 1;
   out[18] = L1*sin(q2);
   out[19] = 0;
   out[20] = L1*cos(q2);
   out[21] = 0;
   out[22] = 0;
   out[23] = 0;

}

void body_jacobian_matrix(double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 1;
   out[5] = 1;
   out[6] = L1*sin(q2);
   out[7] = 0;
   out[8] = L1*cos(q2);
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;

}

void hybrid_jacobian_matrix(double q1, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 1;
   out[5] = 1;
   out[6] = -L1*sin(q1);
   out[7] = 0;
   out[8] = L1*cos(q1);
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;

}

void system_jacobian_dot(double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 0;
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
   out[8] = 0;
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;
   out[12] = 0;
   out[13] = 0;
   out[14] = 0;
   out[15] = 0;
   out[16] = 0;
   out[17] = 0;
   out[18] = L1*dq2*cos(q2);
   out[19] = 0;
   out[20] = -L1*dq2*sin(q2);
   out[21] = 0;
   out[22] = 0;
   out[23] = 0;

}

void body_twist_ee(double dq1, double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = dq1 + dq2;
   out[3] = L1*dq1*sin(q2);
   out[4] = L1*dq1*cos(q2) + L2*(dq1 + dq2);
   out[5] = 0;

}

void hybrid_twist_ee(double dq1, double dq2, double q1, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = dq1 + dq2;
   out[3] = -L1*dq1*sin(q1) - L2*dq1*sin(q1 + q2) - L2*dq2*sin(q1 + q2);
   out[4] = L1*dq1*cos(q1) + L2*dq1*cos(q1 + q2) + L2*dq2*cos(q1 + q2);
   out[5] = 0;

}

void body_jacobian_matrix_ee(double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 1;
   out[5] = 1;
   out[6] = L1*sin(q2);
   out[7] = 0;
   out[8] = L1*cos(q2) + L2;
   out[9] = L2;
   out[10] = 0;
   out[11] = 0;

}

void hybrid_jacobian_matrix_ee(double q1, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 1;
   out[5] = 1;
   out[6] = -L1*sin(q1) - L2*sin(q1 + q2);
   out[7] = -L2*sin(q1 + q2);
   out[8] = L1*cos(q1) + L2*cos(q1 + q2);
   out[9] = L2*cos(q1 + q2);
   out[10] = 0;
   out[11] = 0;

}

void generalized_mass_inertia_matrix(double q2, double *out) {

   out[0] = pow(L1, 2)*m1 + pow(L1, 2)*m2 + 2*L1*L2*m2*cos(q2) + pow(L2, 2)*m2;
   out[1] = L2*m2*(L1*cos(q2) + L2);
   out[2] = L2*m2*(L1*cos(q2) + L2);
   out[3] = pow(L2, 2)*m2;

}

void coriolis_centrifugal_matrix(double dq1, double dq2, double q2, double *out) {

   out[0] = -2*L1*L2*dq2*m2*sin(q2);
   out[1] = -L1*L2*dq2*m2*sin(q2);
   out[2] = L1*L2*m2*(dq1 - dq2)*sin(q2);
   out[3] = L1*L2*dq1*m2*sin(q2);

}

void gravity_vector(double q1, double q2, double *out) {

   out[0] = g*(L1*m1*cos(q1) + L1*m2*cos(q1) + L2*m2*cos(q1 + q2));
   out[1] = L2*g*m2*cos(q1 + q2);

}

void inverse_dynamics(double ddq1, double ddq2, double dq1, double dq2, double q1, double q2, double *out) {

   out[0] = pow(L1, 2)*ddq1*m1 + pow(L1, 2)*ddq1*m2 + 2*L1*L2*ddq1*m2*cos(q2) + L1*L2*ddq2*m2*cos(q2) - 2*L1*L2*dq1*dq2*m2*sin(q2) - L1*L2*pow(dq2, 2)*m2*sin(q2) + L1*g*m1*cos(q1) + L1*g*m2*cos(q1) + pow(L2, 2)*ddq1*m2 + pow(L2, 2)*ddq2*m2 + L2*g*m2*cos(q1 + q2);
   out[1] = L2*m2*(L1*ddq1*cos(q2) + L1*pow(dq1, 2)*sin(q2) + L2*ddq1 + L2*ddq2 + g*cos(q1 + q2));

}

void hybrid_acceleration(double ddq1, double ddq2, double dq1, double q1, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = ddq1 + ddq2;
   out[3] = -L1*(ddq1*sin(q1) + pow(dq1, 2)*cos(q1));
   out[4] = L1*(ddq1*cos(q1) - pow(dq1, 2)*sin(q1));
   out[5] = 0;

}

void body_acceleration(double ddq1, double ddq2, double dq1, double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = ddq1 + ddq2;
   out[3] = L1*(ddq1*sin(q2) + dq1*dq2*cos(q2));
   out[4] = L1*(ddq1*cos(q2) - dq1*dq2*sin(q2));
   out[5] = 0;

}

void hybrid_acceleration_ee(double ddq1, double ddq2, double dq1, double dq2, double q1, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = ddq1 + ddq2;
   out[3] = -L1*ddq1*sin(q1) - L1*pow(dq1, 2)*cos(q1) - L2*ddq1*sin(q1 + q2) - L2*ddq2*sin(q1 + q2) - L2*pow(dq1, 2)*cos(q1 + q2) - 2*L2*dq1*dq2*cos(q1 + q2) - L2*pow(dq2, 2)*cos(q1 + q2);
   out[4] = L1*ddq1*cos(q1) - L1*pow(dq1, 2)*sin(q1) + L2*ddq1*cos(q1 + q2) + L2*ddq2*cos(q1 + q2) - L2*pow(dq1, 2)*sin(q1 + q2) - 2*L2*dq1*dq2*sin(q1 + q2) - L2*pow(dq2, 2)*sin(q1 + q2);
   out[5] = 0;

}

void body_acceleration_ee(double ddq1, double ddq2, double dq1, double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = ddq1 + ddq2;
   out[3] = L1*(ddq1*sin(q2) + dq1*dq2*cos(q2));
   out[4] = L1*(ddq1*cos(q2) - dq1*dq2*sin(q2)) + L2*(ddq1 + ddq2);
   out[5] = 0;

}

void hybrid_jacobian_matrix_dot(double dq1, double q1, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 0;
   out[5] = 0;
   out[6] = -L1*dq1*cos(q1);
   out[7] = 0;
   out[8] = -L1*dq1*sin(q1);
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;

}

void body_jacobian_matrix_dot(double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 0;
   out[5] = 0;
   out[6] = L1*dq2*cos(q2);
   out[7] = 0;
   out[8] = -L1*dq2*sin(q2);
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;

}

void hybrid_jacobian_matrix_ee_dot(double dq1, double dq2, double q1, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 0;
   out[5] = 0;
   out[6] = -L1*dq1*cos(q1) - L2*dq1*cos(q1 + q2) - L2*dq2*cos(q1 + q2);
   out[7] = -L2*(dq1 + dq2)*cos(q1 + q2);
   out[8] = -L1*dq1*sin(q1) - L2*dq1*sin(q1 + q2) - L2*dq2*sin(q1 + q2);
   out[9] = -L2*(dq1 + dq2)*sin(q1 + q2);
   out[10] = 0;
   out[11] = 0;

}

void body_jacobian_matrix_ee_dot(double dq2, double q2, double *out) {

   out[0] = 0;
   out[1] = 0;
   out[2] = 0;
   out[3] = 0;
   out[4] = 0;
   out[5] = 0;
   out[6] = L1*dq2*cos(q2);
   out[7] = 0;
   out[8] = -L1*dq2*sin(q2);
   out[9] = 0;
   out[10] = 0;
   out[11] = 0;

}
