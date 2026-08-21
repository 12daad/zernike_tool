clear
close all
clc

%% ============================================================
% Parameters
% ============================================================

dx = 4.5;
dy = dx;

Nx = 512;
Ny = 512;

x = (-Nx/2:Nx/2-1) * dx;
y = (-Ny/2:Ny/2-1) * dy;

[X, Y] = meshgrid(x, y);

%% ============================================================
% Shear parameters
%
% size(n_shear) = [1, 2, n_shear_count]
% ============================================================

n_shear = zeros(1, 2, 3);

% x shear
n_shear(:,:,1) = [20, 0];

% y shear
n_shear(:,:,2) = [0, 20];

% diagonal shear
n_shear(:,:,3) = [20, 20];

n_shear_count = size(n_shear, 3);

%% ============================================================
% Ground-truth phase
% ============================================================

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

%% ============================================================
% Allocate
% ============================================================

phi_shear_true = zeros(Ny, Nx, n_shear_count);
phi_shear      = zeros(Ny, Nx, n_shear_count);
I_shear        = zeros(Ny, Nx, n_shear_count);

%% ============================================================
% Carrier
%
% Use the same carrier direction for all shear measurements.
% ============================================================

f_carry = 1 / 0.45 * sind(1);

%% ============================================================
% Generate shear interferograms
% ============================================================

for k = 1:n_shear_count

    nx = n_shear(1,1,k);
    ny = n_shear(1,2,k);

    %% ---------------------------------------------------------
    % Generate shifted phase
    % ---------------------------------------------------------

    phi_shift = NaN(size(phi0));

    % Valid region
    %
    % Positive shear:
    %
    % phi_shift(y,x)
    %     = phi0(y+ny, x+nx)

    x_valid = 1:(Nx - nx);
    y_valid = 1:(Ny - ny);

    phi_shift(y_valid, x_valid) = ...
        phi0(y_valid + ny, x_valid + nx);

    % ----------------------------------------------------------
    % Boundary padding
    %
    % These pixels will be rejected by from_shear_phase().
    % ----------------------------------------------------------

    if nx > 0
        phi_shift(:, Nx-nx+1:Nx) = ...
            repmat(phi0(:, Nx), 1, nx);
    end

    if ny > 0
        phi_shift(Ny-ny+1:Ny, :) = ...
            repmat(phi0(Ny, :), ny, 1);
    end

    %% ---------------------------------------------------------
    % True shear phase
    % ---------------------------------------------------------

    phi_shear_true(:,:,k) = phi_shift - phi0;

    %% ---------------------------------------------------------
    % Generate interferogram
    %
    % I = | exp(i*phi0)
    %       +
    %       exp(i*phi_shift)*exp(i*carrier) |^2
    % ---------------------------------------------------------

    carrier = 2*pi*f_carry*X;

    I_shear(:,:,k) = abs( ...
        exp(1j*phi0) + ...
        exp(1j*phi_shift) .* exp(1j*carrier) ...
        ).^2;

    %% ---------------------------------------------------------
    % Add noise
    % ---------------------------------------------------------

    I_shear(:,:,k) = awgn(I_shear(:,:,k), 3);

    %% ---------------------------------------------------------
    % Extract shear phase
    % ---------------------------------------------------------

    phi_shear(:,:,k) = ftp( ...
        X, ...
        Y, ...
        I_shear(:,:,k), ...
        [f_carry, 0], ...
        0.02, ...
        15);

end

%% ============================================================
% Reconstruct phase from all shear measurements simultaneously
% ============================================================

phi = from_shear_phase( ...
    X, ...
    Y, ...
    phi_shear, ...
    n_shear, ...
    1:15);

%% ============================================================
% Zernike fitting of recovered phase
% ============================================================

coef = rect_zernike_coef(X, Y, phi, 1:15);

%% ============================================================
% Reconstruct fitted phase
% ============================================================

phi_fit = rect_zernike_recon(X, Y, coef);

%% ============================================================
% Compare Zernike coefficients
% ============================================================

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

xlabel('Noll Index')
ylabel('Coefficient')

legend( ...
    'Ground Truth', ...
    'Recovered', ...
    'Location', 'best')

title('Zernike Coefficients')

%% ============================================================
% Display shear phases
% ============================================================

figure

for k = 1:n_shear_count

    subplot(n_shear_count, 2, 2*k-1)

    imagesc(x, y, phi_shear_true(:,:,k))

    axis image
    colorbar
    colormap("gray")

    xlabel('x')
    ylabel('y')

    title(sprintf( ...
        'True Shear [%d, %d]', ...
        n_shear(1,1,k), ...
        n_shear(1,2,k)))


    subplot(n_shear_count, 2, 2*k)

    imagesc(x, y, phi_shear(:,:,k))

    axis image
    colorbar
    colormap("gray")

    xlabel('x')
    ylabel('y')

    title(sprintf( ...
        'Extracted Shear [%d, %d]', ...
        n_shear(1,1,k), ...
        n_shear(1,2,k)))

end

%% ============================================================
% Display interferograms
% ============================================================

figure

for k = 1:n_shear_count

    subplot(n_shear_count, 1, k)

    imagesc(x, y, I_shear(:,:,k))

    axis image
    colorbar
    colormap("gray")

    xlabel('x')
    ylabel('y')

    title(sprintf( ...
        'Interferogram, Shear [%d, %d]', ...
        n_shear(1,1,k), ...
        n_shear(1,2,k)))

end

%% ============================================================
% Phase comparison
% ============================================================

figure

subplot(311)

imagesc(x, y, phi0)

axis image
colorbar
colormap("hot")

xlabel('x')
ylabel('y')

title('Phase Ground Truth')


subplot(312)

imagesc(x, y, phi_fit)

axis image
colorbar

xlabel('x')
ylabel('y')

title('Phase Recovered')


subplot(313)

imagesc(x, y, abs(phi_fit - phi0))

axis image
colorbar

xlabel('x')
ylabel('y')

title('Absolute Phase Error')

%% ============================================================
% RMSE
% ============================================================

rmse = sqrt(mean((phi_fit - phi0).^2, "all"));

error_in_wavelength = rmse / (2*pi);

fprintf('\n');
fprintf('========================================\n');
fprintf('Phase Reconstruction Result\n');
fprintf('========================================\n');
fprintf('RMSE              : %.6e wavelength\n', error_in_wavelength);
fprintf('========================================\n');
