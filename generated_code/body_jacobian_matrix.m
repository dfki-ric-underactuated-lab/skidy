function out1 = body_jacobian_matrix(q2)
  global L1 L2 g m1 m2

  out1 = [0 0; 0 0; 1 1; L1.*sin(q2) 0; L1.*cos(q2) 0; 0 0];

end
