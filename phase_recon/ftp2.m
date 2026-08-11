function phi = ftp2(X, Y, I, fc, win_size, n_noll)
[Ny, Nx] = size(I);
dx = Y(2)-Y(1);
fx = (-Nx/2:Nx/2-1) * (1/Nx/dx);
fy = (-Ny/2:Ny/2-1) * (1/Ny/dx);
[FX, FY] = meshgrid(fx, fy);

% mask = exp(-sqrt((FX-fc(1)).^2+(FY-fc(2)).^2)/win_size).^2;
% I_fft_filtered = I_fft .* mask;
% phi = angle(ifft2(ifftshift(I_fft_filtered)));
% phi = unwrap2D(phi);

phi = unwrap2D(angle(ifft2(ifftshift(fftshift(fft2(I .* exp(-1j*2*pi*fc(1)*X-1j*2*pi*fc(2)*Y))) .* exp(-sqrt(FX.^2+FY.^2)/win_size).^2))));
coef = rect_zernike_coef(X,Y,phi,1:n_noll);
coef(2:3) = 0;
phi = rect_zernike_recon(X, Y, coef);

end
