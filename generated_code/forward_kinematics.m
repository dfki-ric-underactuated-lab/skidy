function out1 = forward_kinematics(q1, q2)
  global L1 L2 g m1 m2

  out1 = [cos(q1 + q2) -sin(q1 + q2) 0 L1.*cos(q1) + L2.*cos(q1 + q2); sin(q1 + q2) cos(q1 + q2) 0 L1.*sin(q1) + L2.*sin(q1 + q2); 0 0 1 0; 0 0 0 1];

end
