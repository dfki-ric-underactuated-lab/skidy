function out1 = body_acceleration(ddq1, ddq2, dq1, dq2, q2)
  global L1 L2 g m1 m2

  out1 = [0; 0; ddq1 + ddq2; L1.*(ddq1.*sin(q2) + dq1.*dq2.*cos(q2)); L1.*(ddq1.*cos(q2) - dq1.*dq2.*sin(q2)); 0];

end
