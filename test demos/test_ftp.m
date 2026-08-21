clear

dx = 4.5;
Nx = 256;
Ny = 256;
x = (-Nx/2:Nx/2-1) * dx;
y = (-Ny/2:Ny/2-1) * dx;
[X, Y] = meshgrid(x, y);

coef0 = zeros(1, 15);
% Use multiple Zernike modes
coef0(4)  = 0.30;
coef0(5)  = -0.20;
coef0(6)  = 0.50;
coef0(7)  = 0.15;
coef0(8)  = -0.10;
coef0(9)  = 0.20;
coef0(10) = -0.15;
coef0(11) = 1.00;
coef0(12) = 0.25;
coef0(13) = -0.20;
coef0(14) = 0.10;
coef0(15) = 0.15;

phi0 = rect_zernike_recon(X, Y, coef0);

fx_carry = 1/.45*sind(1);
I = abs(exp(1j*phi0) + exp(-1j*2*pi*fx_carry*X)).^2;
% I = awgn(I, 3);

phi_recon = ftp(X, Y, I, [fx_carry, 0], 0.02, 20);
coef = rect_zernike_coef(X, Y, phi_recon, 1:15);
%%
figure
imagesc(x, y, I)
colormap("gray")
colorbar
grid off
axis image

figure
stem(1:length(coef0), coef0, ...
    'Marker', 'o', ...
    'LineWidth', 1.2)
hold on
stem(1:length(coef), coef, ...
    'Marker', 'x', ...
    'LineWidth', 1.2)
hold off
grid on


figure
subplot(311)
imagesc(x, y, phi0)
colormap("hot")
colorbar
axis image
title("Phase Ground")
subplot(312)
imagesc(x, y, phi_recon)
colorbar
axis image
title("Phase Recovered")
subplot(313)
imagesc(x, y, abs(phi_recon-phi0))
colorbar
axis image
title("Phase Error")

error_in_wavelength = sqrt(mean((phi_recon-phi0).^2, "all")) / (2*pi);
sprintf("RMSE Error: %.1E wavelength", error_in_wavelength)




