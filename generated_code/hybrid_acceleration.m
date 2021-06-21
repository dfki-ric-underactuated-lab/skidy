function out1 = hybrid_acceleration(ddq1, ddq2, dq1, q1)
  global L1 L2 g m1 m2

  out1 = [0; 0; ddq1 + ddq2; -L1.*(ddq1.*sin(q1) + dq1.^2.*cos(q1)); L1.*(ddq1.*cos(q1) - dq1.^2.*sin(q1)); 0];

end
