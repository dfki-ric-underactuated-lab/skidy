function out1 = generalized_mass_inertia_matrix(q2)
  global L1 L2 g m1 m2

  out1 = [L1.^2.*m1 + L1.^2.*m2 + 2*L1.*L2.*m2.*cos(q2) + L2.^2.*m2 L2.*m2.*(L1.*cos(q2) + L2); L2.*m2.*(L1.*cos(q2) + L2) L2.^2.*m2];

end
