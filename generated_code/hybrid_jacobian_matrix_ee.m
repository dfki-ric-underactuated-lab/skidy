function out1 = hybrid_jacobian_matrix_ee(q1, q2)
  global L1 L2 g m1 m2

  out1 = [0 0; 0 0; 1 1; -L1.*sin(q1) - L2.*sin(q1 + q2) -L2.*sin(q1 + q2); L1.*cos(q1) + L2.*cos(q1 + q2) L2.*cos(q1 + q2); 0 0];

end
