function out1 = body_jacobian_matrix_ee_dot(dq2, q2)
  global L1 L2 g m1 m2

  out1 = [0 0; 0 0; 0 0; L1.*dq2.*cos(q2) 0; -L1.*dq2.*sin(q2) 0; 0 0];

end
