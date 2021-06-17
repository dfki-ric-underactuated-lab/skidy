function out1 = inverse_dynamics(ddq1, ddq2, dq1, dq2, q1, q2)
  global L1 L2 g m1 m2

  out1 = [L1.^2.*ddq1.*m1 + L1.^2.*ddq1.*m2 + 2*L1.*L2.*ddq1.*m2.*cos(q2) + L1.*L2.*ddq2.*m2.*cos(q2) - 2*L1.*L2.*dq1.*dq2.*m2.*sin(q2) - L1.*L2.*dq2.^2.*m2.*sin(q2) + L1.*g.*m1.*cos(q1) + L1.*g.*m2.*cos(q1) + L2.^2.*ddq1.*m2 + L2.^2.*ddq2.*m2 + L2.*g.*m2.*cos(q1 + q2); L2.*m2.*(L1.*ddq1.*cos(q2) + L1.*dq1.^2.*sin(q2) + L2.*ddq1 + L2.*ddq2 + g.*cos(q1 + q2))];

end
