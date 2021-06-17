function out1 = coriolis_centrifugal_matrix(dq1, dq2, q2)
  global L1 L2 g m1 m2

  out1 = [-2*L1.*L2.*dq2.*m2.*sin(q2) -L1.*L2.*dq2.*m2.*sin(q2); L1.*L2.*m2.*(dq1 - dq2).*sin(q2) L1.*L2.*dq1.*m2.*sin(q2)];

end
