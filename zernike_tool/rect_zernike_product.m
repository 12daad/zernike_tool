function G = rect_zernike_product(Zi, Zj)
[ROW, COL, M] = size(Zi);

G = zeros(M);
for i=1:M
    for j=1:M
        G(i,j) = sum(Zi(:,:,i) .* Zj(:,:,j), "all") / (ROW*COL);
    end
end
end
