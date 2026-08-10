N = 512;

x = linspace(-1.5, 1.5, N);
y = linspace(-1, 1, N);

[X, Y] = meshgrid(x, y);

i_noll = 1:20;

[Z, dZdX, dZdY] = rect_zernike(X, Y, i_noll);

J = numel(i_noll);

mask = isfinite(X) & isfinite(Y);

Zmat = reshape(Z, [], J);
Zmat = Zmat(mask(:), :);

G = rect_zernike_product(Z, Z);

figure;
imagesc(G);
axis image;
colormap gray
colorbar;
title('Inner product matrix');
set(gca, "YDir", "normal")

fprintf('Orthogonality error = %.3e\n', ...
    norm(G - eye(J), 'fro'));
