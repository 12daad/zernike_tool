function phi = ftp(X, Y, I, varargin)
% $\phi_0 = \phi_0(x, y)$
% $\phi_{carrier} = k_x x$, $k_x > 0$ if the carrier wave propagates 
% toward +x axis
% 
%     ----> x axis
%   E_carrier E_0
%     /        |
%    /         |
%   /          |
% 
% The interference field is
% $I ~ \cos(\phi_0 - k_x x) = \exp(-jk_x x) \exp(\phi_0) + \exp(jk_x x) \exp(-\phi_0)$
% $\exp(-jk_x x) \exp(\phi_0)$ donets the -1 order ($k_x>0$)
% in this case, the -1 order should be remained after a BPF $H = H^-$ process, and
% finally $\phi_0 ~ angle(FT[I] H)$
%
% In short, the sign of carrier is defined as the above diagram, if the
% carrier is positive, use a BPF $H = H^-$, oherwise $H = H^+$
%
%%
p = inputParser();
p.addRequired("X", @(x) ismatrix(x) && isreal(x))
p.addRequired("Y", @(x) ismatrix(x) && isreal(x))
p.addRequired("I", @(x) ismatrix(x) && isreal(x))
p.addOptional("fc", [nan, nan], @(x) isvector(x) && length(x) == 2)
p.addOptional("win_size", nan, @(x) isscalar(x) && x > 0)
p.addOptional("n_noll", nan, @(x) isscalar(x))
p.addOptional("refrection", 1, @(x) ismatrix(x) && all(x(:)>0))
p.addParameter("DispLog", false, @(x) isscalar(x) && islogical(x));

p.parse(X, Y, I, varargin{:})
p_res = p.Results;
is_preview = all(isnan([p_res.fc p_res.win_size, p_res.n_noll]));
fc = p_res.fc;
win_size = p_res.win_size;
n_noll = p_res.n_noll;
refrection = p_res.refrection;

%%

[Ny, Nx] = size(I);
dx = Y(2)-Y(1);
I_fft = fftshift(fft2(I ./ refrection));
fx = (-Nx/2:Nx/2-1) * (1/Nx/dx);
fy = (-Ny/2:Ny/2-1) * (1/Ny/dx);
if ~is_preview
[FX, FY] = meshgrid(fx, fy);
mask = exp(-sqrt((FX-fc(1)).^2+(FY-fc(2)).^2)/win_size).^2;
% mask = 
I_fft_filtered = I_fft .* mask;
phi = angle(ifft2(ifftshift(I_fft_filtered)));
amp = abs(ifft2(ifftshift(I_fft_filtered)));
phi = unwrap2D(phi);
coef = rect_zernike_coef(X,Y,phi,1:n_noll);
coef(2:3) = 0;
phi = rect_zernike_recon(X,Y,coef);
else
I_fft_filtered = I_fft;
phi = [];
end

if p_res.DispLog || is_preview
    figure
    Imin = min(log(1+abs(I_fft).^2), [], "all");
    Imax = max(log(1+abs(I_fft).^2), [], "all");
    subplot(131)
    imagesc(fx, fy, log(1+abs(I_fft).^2))
    colorbar
    clim([Imin Imax]); 
    axis image
    colormap hot
    subplot(132)
    imagesc(fx, fy, log(1+abs(I_fft_filtered).^2))    
    colorbar
    clim([Imin Imax]); 
    axis image
    colormap hot
    title("Phase Recovered")
end
end
