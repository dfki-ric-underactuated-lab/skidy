function out1 = hybrid_acceleration_ee(ddq1, ddq2, dq1, dq2, q1, q2)
  global L1 L2 g m1 m2

  out1 = [0; 0; ddq1 + ddq2; -L1.*ddq1.*sin(q1) - L1.*dq1.^2.*cos(q1) - L2.*ddq1.*sin(q1 + q2) - L2.*ddq2.*sin(q1 + q2) - L2.*dq1.^2.*cos(q1 + q2) - 2*L2.*dq1.*dq2.*cos(q1 + q2) - L2.*dq2.^2.*cos(q1 + q2); L1.*ddq1.*cos(q1) - L1.*dq1.^2.*sin(q1) + L2.*ddq1.*cos(q1 + q2) + L2.*ddq2.*cos(q1 + q2) - L2.*dq1.^2.*sin(q1 + q2) - 2*L2.*dq1.*dq2.*sin(q1 + q2) - L2.*dq2.^2.*sin(q1 + q2); 0];

end
