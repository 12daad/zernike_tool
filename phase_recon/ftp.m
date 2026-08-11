function phi = ftp(X, Y, I, fc, win_size, n_noll)
[Ny, Nx] = size(I);
dx = Y(2)-Y(1);
I_fft = fftshift(fft2(I));
fx = (-Nx/2:Nx/2-1) * (1/Nx/dx);
fy = (-Ny/2:Ny/2-1) * (1/Ny/dx);
[FX, FY] = meshgrid(fx, fy);
mask = exp(-sqrt((FX-fc(1)).^2+(FY-fc(2)).^2)/win_size).^2;
I_fft_filtered = I_fft .* mask;
phi = angle(ifft2(ifftshift(I_fft_filtered)));
phi = unwrap2D(phi);
coef = rect_zernike_coef(X,Y,phi,1:n_noll);
coef_c = rect_zernike_coef(X,Y,2*pi*fc(1)*X+2*pi*fc(2)*Y,1:n_noll);
coef_c([1,4:n_noll]) = 0;
coef = coef - coef_c;
phi = rect_zernike_recon(X, Y, coef);

figure
subplot(211)
imagesc(fx, fy, log(1+abs(I_fft)))
axis image
subplot(212)
imagesc(fx, fy, log(1+abs(I_fft_filtered)))
axis image
end
