function phi = ftp(X, Y, I, fc, win_size, n_noll, varargin)
%%
p = inputParser();
p.addRequired("X", @(x) ismatrix(X) && isreal(X))
p.addRequired("Y", @(x) ismatrix(X) && isreal(X))
p.addRequired("I", @(x) ismatrix(X) && isreal(X))
p.addRequired("fc", @(x) isvector(x) && length(x) == 2)
p.addRequired("win_size", @(x) isscalar(x) && x > 0)
p.addRequired("n_noll", @(x) isscalar(x))
p.addOptional("refrection", 1, @(x) ismatrix(x) && all(x(:)>0))
p.addParameter("DispLog", false, @(x) isscalar(x) && islogical(x));

p.parse(X, Y, I, fc, win_size, n_noll, varargin{:})
p_res = p.Results;
refrection = p_res.refrection;
%%

[Ny, Nx] = size(I);
dx = Y(2)-Y(1);
I_fft = fftshift(fft2(I ./ refrection));
fx = (-Nx/2:Nx/2-1) * (1/Nx/dx);
fy = (-Ny/2:Ny/2-1) * (1/Ny/dx);
[FX, FY] = meshgrid(fx, fy);
mask = exp(-sqrt((FX-fc(1)).^2+(FY-fc(2)).^2)/win_size).^2;
I_fft_filtered = I_fft .* mask;
phi = angle(ifft2(ifftshift(I_fft_filtered)));
phi = unwrap2D(phi);
coef = rect_zernike_coef(X,Y,phi,1:n_noll);
coef(2:3) = 0; % Remove carrier
phi = rect_zernike_recon(X, Y, coef);

if p_res.DispLog
    figure
    subplot(211)
    imagesc(fx, fy, log(1+abs(I_fft)))
    axis image
    subplot(212)
    imagesc(fx, fy, log(1+abs(I_fft_filtered)))
    axis image
end
end
