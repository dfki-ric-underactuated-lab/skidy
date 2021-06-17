function out1 = gravity_vector(q1, q2)
  global L1 L2 g m1 m2

  out1 = [g.*(L1.*m1.*cos(q1) + L1.*m2.*cos(q1) + L2.*m2.*cos(q1 + q2)); L2.*g.*m2.*cos(q1 + q2)];

end
