function out1 = body_twist_ee(dq1, dq2, q2)
  global L1 L2 g m1 m2

  out1 = [0; 0; dq1 + dq2; L1.*dq1.*sin(q2); L1.*dq1.*cos(q2) + L2.*(dq1 + dq2); 0];

end
