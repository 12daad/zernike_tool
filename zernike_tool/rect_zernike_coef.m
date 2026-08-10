function coef = rect_zernike_coef(X, Y, phi, i_noll)

coef = zeros(size(i_noll));
Z = rect_zernike(X, Y, i_noll);

for i=1:length(i_noll)
    coef(i) = rect_zernike_product(Z(:,:,i), phi);
end

end
