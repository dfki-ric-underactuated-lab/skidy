function out1 = hybrid_jacobian_matrix_ee_dot(dq1, dq2, q1, q2)
  global L1 L2 g m1 m2

  out1 = [0 0; 0 0; 0 0; -L1.*dq1.*cos(q1) - L2.*dq1.*cos(q1 + q2) - L2.*dq2.*cos(q1 + q2) -L2.*(dq1 + dq2).*cos(q1 + q2); -L1.*dq1.*sin(q1) - L2.*dq1.*sin(q1 + q2) - L2.*dq2.*sin(q1 + q2) -L2.*(dq1 + dq2).*sin(q1 + q2); 0 0];

end
