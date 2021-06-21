function out1 = hybrid_jacobian_matrix_dot(dq1, q1)
  global L1 L2 g m1 m2

  out1 = [0 0; 0 0; 0 0; -L1.*dq1.*cos(q1) 0; -L1.*dq1.*sin(q1) 0; 0 0];

end
