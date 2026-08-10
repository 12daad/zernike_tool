function phi = rect_zernike_recon(X, Y, coef)

M = length(coef);

Z = rect_zernike(X, Y, 1:M);
phi = sum(Z .* reshape(coef, 1, 1, []), 3);

end
