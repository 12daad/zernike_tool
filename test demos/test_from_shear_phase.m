clear
close all

n_xshear = 20;
n_yshear = 20;

dx = 4.5;
Nx = 256;
Ny = 256;
x = (-Nx/2:Nx/2-1) * dx;
y = (-Ny/2:Ny/2-1) * dx;
[X, Y] = meshgrid(x, y);

coef0 = zeros(1, 12);
coef0(11) = 1;
phi0 = rect_zernike_recon(X, Y, coef0);

% phi_xshear = circshift(phi0, -n_xshear, 2);
phi_xshear = [phi0(:, n_xshear+1:end), ones(Ny, n_xshear).*phi0(:, end)];
f_carry = 1/.45*sind(1);
I_xshear = abs(exp(1j*phi0) + exp(1j*phi_xshear) .* exp(1j*2*pi*f_carry*X)).^2;
I_xshear = awgn(I_xshear, 3);

% phi_yshear = circshift(phi0, -n_yshear, 1);
phi_yshear = [phi0(n_yshear+1:end, :); ones(n_yshear, Nx).*phi0(end, :)];
fy_carry = 1/.45*sind(1);
I_yshear = abs(exp(1j*phi0) + exp(1j*phi_yshear) .* exp(1j*2*pi*f_carry*Y)).^2;
I_yshear = awgn(I_yshear, 3);

phi_xgrad = ftp(X, Y, I_xshear, [f_carry, 0], 0.02, 15);
phi_ygrad = ftp(X, Y, I_yshear, [0, f_carry], 0.02, 15);
phi_grad(:,:,1) = phi_xgrad;
phi_grad(:,:,2) = phi_ygrad;
% phi_grad(:,:,1) = phi_xshear - phi0;
% phi_grad(:,:,2) = phi_yshear - phi0;
phi = from_shear_phase(X, Y, phi_grad, [n_xshear, n_yshear]);

coef = rect_zernike_coef(X,Y,phi,1:15);
coef(2:3) = 0;
phi = rect_zernike_recon(X,Y,coef);

%%
%%
figure
stem(coef0, 'Marker', 'o')
hold on
stem(coef, 'Marker', 'x')
hold off

figure
subplot(211)
imagesc(x, y, phi_xshear - phi0)
colormap("gray")
colorbar
grid on
axis image
subplot(212)
imagesc(x, y, phi_xgrad)
colormap("gray")
colorbar
grid on
axis image

figure
subplot(211)
imagesc(x, y, I_xshear)
colormap("gray")
colorbar
grid on
axis image
subplot(212)
imagesc(x, y, I_yshear)
colormap("gray")
colorbar
grid on
axis image

figure
subplot(311)
imagesc(x, y, phi0)
colormap("hot")
colorbar
axis image
title("Phase Ground")
subplot(312)
imagesc(x, y, phi)
colorbar
axis image
title("Phase Recovered")
subplot(313)
imagesc(x, y, abs(phi-phi0))
colorbar
axis image
title("Phase Error")

error_in_wavelength = sqrt(mean((phi-phi0).^2, "all")) / (2*pi);
sprintf("RMSE Error: %.1E wavelength", error_in_wavelength)



