%% ============================================================
% Test rect_zernike gradient
%
% Check:
%   1. Analytical gradient vs numerical gradient
%   2. dH/dX = dZ/dX * U
%   3. dH/dY = dZ/dY * U
%
% ============================================================

clear;
clc;
close all;


%% ============================================================
% Parameters
% ============================================================

Nx = 256;
Ny = 256;

% Rectangular aperture
x = linspace(-1.5, 1.5, Nx);
y = linspace(-1.0, 1.0, Ny);

[X, Y] = meshgrid(x, y);

% Zernike modes
i_noll = 1:20;


%% ============================================================
% Generate rectangular Zernike
% ============================================================

[Z, dZdX, dZdY, U] = ...
    rect_zernike(X, Y, i_noll);


%% ============================================================
% Aperture mask
% ============================================================

mask = isfinite(X) & isfinite(Y);


%% ============================================================
% Grid spacing
% ============================================================

dx = x(2) - x(1);
dy = y(2) - y(1);


%% ============================================================
% 1. Check analytical gradient using numerical differentiation
% ============================================================

num_modes = numel(i_noll);

errX_abs = zeros(1, num_modes);
errY_abs = zeros(1, num_modes);

errX_rel = zeros(1, num_modes);
errY_rel = zeros(1, num_modes);

for k = 1:num_modes

    Zk = Z(:,:,k);

    %% Numerical gradient

    % MATLAB gradient:
    %
    % dZ/dY -> first output
    % dZ/dX -> second output
    %
    [dZdx_num, dZdy_num] = gradient(Zk, dx, dy);


    %% Only compare inside a safe interior region
    %
    % Avoid the outermost pixels because gradient() uses
    % one-sided differences there.

    valid = mask;

    valid(1,:)   = false;
    valid(end,:) = false;
    valid(:,1)   = false;
    valid(:,end) = false;


    %% Analytical gradient

    dZdx_ana = dZdX(:,:,k);
    dZdy_ana = dZdY(:,:,k);


    %% Difference

    diffX = dZdx_ana - dZdx_num;
    diffY = dZdy_ana - dZdy_num;

    diffX = diffX(valid);
    diffY = diffY(valid);

    anaX = dZdx_ana(valid);
    anaY = dZdy_ana(valid);


    %% Absolute RMS error

    errX_abs(k) = sqrt(mean(diffX.^2));
    errY_abs(k) = sqrt(mean(diffY.^2));


    %% Relative RMS error

    errX_rel(k) = ...
        sqrt(mean(diffX.^2)) / ...
        (eps + sqrt(mean(anaX.^2)));

    errY_rel(k) = ...
        sqrt(mean(diffY.^2)) / ...
        (eps + sqrt(mean(anaY.^2)));

end


%% ============================================================
% Display gradient errors
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('Gradient verification\n');
fprintf('============================================================\n');

fprintf('\n');
fprintf('%8s %16s %16s %16s %16s\n', ...
    'Noll', ...
    'X RMS', ...
    'X Relative', ...
    'Y RMS', ...
    'Y Relative');

fprintf('------------------------------------------------------------\n');

for k = 1:num_modes

    fprintf('%8d %16.4e %16.4e %16.4e %16.4e\n', ...
        i_noll(k), ...
        errX_abs(k), ...
        errX_rel(k), ...
        errY_abs(k), ...
        errY_rel(k));

end


%% ============================================================
% Plot gradient error
% ============================================================

figure;

semilogy(i_noll, errX_rel, 'o-', ...
    'LineWidth', 1.5);
hold on;

semilogy(i_noll, errY_rel, 's-', ...
    'LineWidth', 1.5);

grid on;

xlabel('Noll index');
ylabel('Relative RMS error');

legend( ...
    'dZ/dX', ...
    'dZ/dY', ...
    'Location', 'northwest');

title('Analytical gradient vs numerical gradient');


%% ============================================================
% 2. Visual comparison for a selected mode
% ============================================================

mode = 10;

Zk = Z(:,:,mode);

[dZdx_num, dZdy_num] = gradient(Zk, dx, dy);

dZdx_ana = dZdX(:,:,mode);
dZdy_ana = dZdY(:,:,mode);


%% X gradient

figure;

subplot(1,3,1);

imagesc(x, y, dZdx_ana);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title(sprintf('Analytical dZ/dX, Noll %d', i_noll(mode)));


subplot(1,3,2);

imagesc(x, y, dZdx_num);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title('Numerical dZ/dX');


subplot(1,3,3);

imagesc(x, y, dZdx_ana - dZdx_num);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title('Difference');


%% Y gradient

figure;

subplot(1,3,1);

imagesc(x, y, dZdy_ana);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title(sprintf('Analytical dZ/dY, Noll %d', i_noll(mode)));


subplot(1,3,2);

imagesc(x, y, dZdy_num);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title('Numerical dZ/dY');


subplot(1,3,3);

imagesc(x, y, dZdy_ana - dZdy_num);
axis image;
colorbar;

xlabel('X');
ylabel('Y');

title('Difference');


%% ============================================================
% 3. Check the transformation relationship
%
% Original Zernike:
%
%       H = Z0 * U
%
% Therefore:
%
%       dH/dX = dZ0/dX * U
%       dH/dY = dZ0/dY * U
%
% This is already used internally by rect_zernike.
%
% We verify the final result by reconstructing the gradients
% from the returned transformation matrix.
% ============================================================

% Generate the original Zernike basis explicitly.

J = max(i_noll);

x0 = X(mask);
y0 = Y(mask);

% Reproduce the same normalization used by rect_zernike

xc = mean(X(mask));
yc = mean(Y(mask));

Xc = X - xc;
Yc = Y - yc;

rho_max = max( ...
    sqrt(Xc(mask).^2 + Yc(mask).^2));

xn = Xc / rho_max;
yn = Yc / rho_max;


Z0  = zeros(nnz(mask), J);
dX0 = zeros(nnz(mask), J);
dY0 = zeros(nnz(mask), J);

for j = 1:J

    % Noll -> (n,m)
    [n,m] = noll_index_test(j);

    [Zj,dZjdx,dZjdy] = ...
        zernike_cartesian_test( ...
            n, m, xn(mask), yn(mask));

    Z0(:,j)  = Zj;
    dX0(:,j) = dZjdx;
    dY0(:,j) = dZjdy;

end


%% Reconstruct all modes

H0 = Z0 * U;

dHdx0 = dX0 * U;
dHdy0 = dY0 * U;


%% Compare with rect_zernike output

Z_test = zeros(nnz(mask), num_modes);
dX_test = zeros(nnz(mask), num_modes);
dY_test = zeros(nnz(mask), num_modes);

for k = 1:num_modes

    Zk = Z(:,:,k);
    dXk = dZdX(:,:,k);
    dYk = dZdY(:,:,k);

    Z_test(:,k)  = Zk(mask);
    dX_test(:,k) = dXk(mask);
    dY_test(:,k) = dYk(mask);

end


%% Important:
% U contains all modes up to max(i_noll), so select requested columns.

U_req = U(:, i_noll);

dHdx_expected = dX0 * U_req / rho_max;
dHdy_expected = dY0 * U_req / rho_max;


%% Error

transform_error_X = ...
    norm(dX_test - dHdx_expected, 'fro') / ...
    norm(dX_test, 'fro');

transform_error_Y = ...
    norm(dY_test - dHdy_expected, 'fro') / ...
    norm(dY_test, 'fro');


fprintf('\n');
fprintf('============================================================\n');
fprintf('Transformation verification\n');
fprintf('============================================================\n');

fprintf('dH/dX relative error = %.4e\n', ...
    transform_error_X);

fprintf('dH/dY relative error = %.4e\n', ...
    transform_error_Y);


%% ============================================================
% 4. Orthogonality check
% ============================================================

Zmat = zeros(nnz(mask), num_modes);

for k = 1:num_modes
    tmp = Z(:,:,k);
    Zmat(:,k) = tmp(mask);
end

G = Zmat' * Zmat / nnz(mask);

orth_error = norm( ...
    G - eye(num_modes), ...
    'fro');

fprintf('\n');
fprintf('============================================================\n');
fprintf('Orthogonality\n');
fprintf('============================================================\n');

fprintf('||G-I||_F = %.4e\n', orth_error);


%% ============================================================
% 5. Helper functions
% ============================================================

function [n,m] = noll_index_test(j)

n = 0;
count = 0;

while j > count + n + 1

    count = count + n + 1;
    n = n + 1;

end

k = j - count;

if mod(n,2) == 0

    if k == 1

        m = 0;

    else

        kk = k - 1;

        if mod(kk,2) == 1
            m = -2*ceil(kk/2);
        else
            m = 2*(kk/2);
        end

    end

else

    m = -n + 2*(k-1);

end

end


function [Z,dZdx,dZdy] = ...
    zernike_cartesian_test(n,m,x,y)

ma = abs(m);

Z    = zeros(size(x));
dZdx = zeros(size(x));
dZdy = zeros(size(x));

r2 = x.^2 + y.^2;

zc = x + 1i*y;

for k = 0:(n-ma)/2

    coeff = (-1)^k * ...
        factorial(n-k) / ...
        (factorial(k) * ...
        factorial((n+ma)/2-k) * ...
        factorial((n-ma)/2-k));

    p = n - 2*k;

    h = (p-ma)/2;


    %% Angular part

    if ma == 0

        A = ones(size(x));

        dAx = zeros(size(x));
        dAy = zeros(size(x));

    else

        zm = zc.^ma;

        if ma == 1
            zm1 = ones(size(x));
        else
            zm1 = zc.^(ma-1);
        end

        if m > 0

            A = real(zm);

            dAx = ma * real(zm1);
            dAy = -ma * imag(zm1);

        else

            A = imag(zm);

            dAx = ma * imag(zm1);
            dAy = ma * real(zm1);

        end

    end


    %% Radial part

    if h == 0

        B = ones(size(x));

        dBx = zeros(size(x));
        dBy = zeros(size(x));

    else

        B = r2.^h;

        Bm1 = r2.^(h-1);

        dBx = 2*h*x .* Bm1;
        dBy = 2*h*y .* Bm1;

    end


    %% Accumulate

    Z = Z + coeff * A .* B;

    dZdx = dZdx + coeff * ...
        (dAx .* B + A .* dBx);

    dZdy = dZdy + coeff * ...
        (dAy .* B + A .* dBy);

end

end
