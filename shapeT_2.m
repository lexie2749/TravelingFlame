%% T-Junction PINN-CFD Hybrid for Flame Propagation with GIF Output
% This script simulates flame propagation in a T-junction using a hybrid
% PINN-CFD approach and creates an animated GIF of the results.
clear; close all; clc;

%% 1. Physical and Grid Parameters
% =========================================================================
% Geometry parameters
main_channel_length = 0.2; % meters
channel_width = 0.04;    % meters
branch_length = 0.1;     % meters
branch_pos_x_start = 0.08; % meters
branch_pos_x_end = 0.12; % meters

% Physical parameters
Le = 0.7;          % Lewis number
beta = 8.0;        % Zeldovich number
alpha = 0.85;      % Heat release parameter
D_th = 1e-5;       % Thermal diffusivity (m²/s)
D_mass = D_th / Le;% Mass diffusivity
U_flow = 0.05;     % Convective velocity (m/s) - reduced for better visualization

% Grid and time parameters
Nx = 100;           % X points
Ny = 80;            % Y points
Nt = 1000;          % Time steps - reduced for faster computation
dt = 0.005;         % Time step (s)
t_final = Nt * dt;

% Neural network parameters
hiddenSize = 30;       % Reduced for faster training
numHiddenLayers = 3;   % Reduced for faster training
learningRate = 0.001;
numEpochs = 300;       % Reduced for faster training
numTrainPoints = 1500;

%% 2. Create Grid and Initial Condition
% =========================================================================
x_vec = linspace(0, main_channel_length, Nx);
y_vec = linspace(-channel_width/2, channel_width/2 + branch_length, Ny);
[X, Y] = meshgrid(x_vec, y_vec);

% Create a mask to define the T-junction geometry
is_in_main = (X >= 0) & (X <= main_channel_length) & (Y >= -channel_width/2) & (Y <= channel_width/2);
is_in_branch = (X >= branch_pos_x_start) & (X <= branch_pos_x_end) & (Y >= channel_width/2) & (Y <= channel_width/2 + branch_length);
mask = is_in_main | is_in_branch;

% Initialize fields
T = zeros(Ny, Nx, Nt);
Y_fuel = ones(Ny, Nx, Nt);

% Initial condition - hot spot at the left entrance
x_ignition = 0.01;
y_ignition = 0;
for i = 1:Ny
    for j = 1:Nx
        if mask(i,j)
            % Stronger initial perturbation for better propagation
            dist_sq = ((X(i,j) - x_ignition)^2 / 0.003^2 + (Y(i,j) - y_ignition)^2 / 0.003^2);
            T(i,j,1) = 0.05 + 0.95 * exp(-dist_sq);
            Y_fuel(i,j,1) = 1 - 0.8 * exp(-dist_sq);
        end
    end
end

%% 3. Create Neural Network and Training Data
% =========================================================================
layers = [
    featureInputLayer(3, 'Name', 'input') % (x, y, t)
    fullyConnectedLayer(hiddenSize, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
];
for l = 2:numHiddenLayers
    layers = [layers
        fullyConnectedLayer(hiddenSize, 'Name', ['fc' num2str(l)])
        tanhLayer('Name', ['tanh' num2str(l)])
    ];
end
layers = [layers
    fullyConnectedLayer(2, 'Name', 'output') % (T, Y_fuel)
    sigmoidLayer('Name', 'sigmoid')
];
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

% Generate training data points within T-junction
fprintf('Generating training data points...\n');
num_initial_points = numTrainPoints * 10;
x_train_all = main_channel_length * rand(num_initial_points, 1);
y_train_all = (channel_width/2 + branch_length - (-channel_width/2)) * rand(num_initial_points, 1) - channel_width/2;

% Filter points within T-junction geometry
is_valid_point = is_in_tjunction(x_train_all, y_train_all, main_channel_length, channel_width, branch_pos_x_start, branch_pos_x_end, branch_length);
x_train_valid = x_train_all(is_valid_point);
y_train_valid = y_train_all(is_valid_point);

% Ensure we have enough points
num_actual_train_points = min(length(x_train_valid), numTrainPoints);
x_train = x_train_valid(1:num_actual_train_points);
y_train = y_train_valid(1:num_actual_train_points);
t_train = t_final * rand(num_actual_train_points, 1);
X_train = dlarray([x_train, y_train, t_train]', 'CB');

fprintf('Using %d training points\n', num_actual_train_points);

%% 4. Train Network (Purely Physics-Informed)
% =========================================================================
fprintf('Training PINN for T-Junction...\n');
lossHistory = zeros(numEpochs, 1);
averageGrad = [];
averageSqGrad = [];
iteration = 0;
tic;

for epoch = 1:numEpochs
    iteration = iteration + 1;
    [loss, gradients] = dlfeval(@physicsLossT, net, X_train, D_th, D_mass, Le, beta, alpha, U_flow);
    
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
        averageGrad, averageSqGrad, iteration, learningRate);
    lossHistory(epoch) = extractdata(loss);
    if mod(epoch, 50) == 0
        fprintf('Epoch %d/%d, Loss: %.6f\n', epoch, numEpochs, lossHistory(epoch));
    end
end
trainingTime = toc;
fprintf('Training completed in %.2f seconds\n', trainingTime);

%% 5. Hybrid CFD-PINN Simulation
% =========================================================================
fprintf('Running hybrid simulation...\n');
tic;

for n = 2:Nt
    current_t = (n-1) * dt;
    
    % Identify flame front region
    [T_x, T_y] = gradient(T(:,:,n-1));
    dx = x_vec(2) - x_vec(1);
    dy = y_vec(2) - y_vec(1);
    T_x = T_x / dx;
    T_y = T_y / dy;
    gradT_mag = sqrt(T_x.^2 + T_y.^2);
    max_grad = max(gradT_mag(mask));
    if max_grad > 0
        flame_front = gradT_mag > 0.2 * max_grad;
    else
        flame_front = false(size(mask));
    end
    
    % Update temperature and fuel fields
    for i = 2:Ny-1
        for j = 2:Nx-1
            if mask(i,j)
                if flame_front(i,j) && rand() < 0.3  % Use PINN for 30% of flame front points
                    % Use PINN for flame front
                    X_point = dlarray([X(i,j); Y(i,j); current_t], 'CB');
                    Y_pred = predict(net, X_point);
                    Y_pred = extractdata(Y_pred);
                    T(i,j,n) = max(0, min(1, Y_pred(1)));
                    Y_fuel(i,j,n) = max(0, min(1, Y_pred(2)));
                else
                    % Use finite difference
                    dx = x_vec(2) - x_vec(1);
                    dy = y_vec(2) - y_vec(1);
                    
                    % Compute Laplacians with boundary handling
                    if j > 1 && j < Nx && i > 1 && i < Ny
                        laplacian_T = (T(i,j+1,n-1) - 2*T(i,j,n-1) + T(i,j-1,n-1))/dx^2 + ...
                                    (T(i+1,j,n-1) - 2*T(i,j,n-1) + T(i-1,j,n-1))/dy^2;
                        laplacian_Y = (Y_fuel(i,j+1,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i,j-1,n-1))/dx^2 + ...
                                    (Y_fuel(i+1,j,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i-1,j,n-1))/dy^2;
                        
                        % Convection term
                        if j > 1
                            conv_T = U_flow * (T(i,j,n-1) - T(i,j-1,n-1))/dx;
                            conv_Y = U_flow * (Y_fuel(i,j,n-1) - Y_fuel(i,j-1,n-1))/dx;
                        else
                            conv_T = 0;
                            conv_Y = 0;
                        end
                    else
                        laplacian_T = 0;
                        laplacian_Y = 0;
                        conv_T = 0;
                        conv_Y = 0;
                    end
                    
                    % Reaction rate
                    T_local = max(0, min(1, T(i,j,n-1)));
                    Y_local = max(0, min(1, Y_fuel(i,j,n-1)));
                    if T_local > 0.1  % Only react above threshold
                        omega = (beta^2/(2*Le)) * Y_local * exp(-beta*(1-T_local)/(1-alpha*(1-T_local)));
                    else
                        omega = 0;
                    end
                    
                    % Update equations
                    T(i,j,n) = T(i,j,n-1) + dt * (D_th*laplacian_T - conv_T + omega);
                    Y_fuel(i,j,n) = Y_fuel(i,j,n-1) + dt * (D_mass*laplacian_Y - conv_Y - omega);
                    
                    % Ensure bounds
                    T(i,j,n) = max(0, min(1, T(i,j,n)));
                    Y_fuel(i,j,n) = max(0, min(1, Y_fuel(i,j,n)));
                end
            end
        end
    end
    
    % Apply boundary conditions
    T = applyTjuncBC_simple(T, n, mask);
    Y_fuel = applyTjuncBC_simple(Y_fuel, n, mask);
    
    if mod(n, 100) == 0
        fprintf('  Progress: %.0f%%\n', 100 * n / Nt);
    end
end
simulationTime = toc;
fprintf('Simulation completed in %.2f seconds\n', simulationTime);

%% 6. Create Visualization and GIF
% =========================================================================
% Create output folder
output_folder = 'T_junction_flame_frames';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Plot training loss
figure('Position', [100 100 600 400]);
semilogy(lossHistory, 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('PINN Training Loss');
grid on;
saveas(gcf, fullfile(output_folder, 'training_loss.png'));

% Create figure for animation
fprintf('Creating animation frames...\n');
h = figure('Position', [100 100 800 600]);

% Prepare for GIF
gif_filename = fullfile(output_folder, 'T_junction_flame.gif');
frame_interval = 15;  % Save every 15th frame for complete propagation
frame_count = 0;

% Create colormap
cmap = hot(256);

for n = 1:frame_interval:Nt
    clf(h);
    
    % Create T-junction visualization
    T_display = T(:,:,n);
    T_display(~mask) = NaN;  % Set outside geometry to NaN
    
    % Plot temperature field
    imagesc(x_vec, y_vec, T_display);
    set(gca, 'YDir', 'normal');
    colormap(cmap);
    colorbar;
    caxis([0 1]);
    
    % Add geometry outline
    hold on;
    % Main channel outline
    plot([0, main_channel_length], [-channel_width/2, -channel_width/2], 'w-', 'LineWidth', 2);
    plot([0, main_channel_length], [channel_width/2, channel_width/2], 'w-', 'LineWidth', 2);
    plot([0, 0], [-channel_width/2, channel_width/2], 'w-', 'LineWidth', 2);
    plot([main_channel_length, main_channel_length], [-channel_width/2, channel_width/2], 'w-', 'LineWidth', 2);
    
    % Branch outline
    plot([branch_pos_x_start, branch_pos_x_start], [channel_width/2, channel_width/2 + branch_length], 'w-', 'LineWidth', 2);
    plot([branch_pos_x_end, branch_pos_x_end], [channel_width/2, channel_width/2 + branch_length], 'w-', 'LineWidth', 2);
    plot([branch_pos_x_start, branch_pos_x_end], [channel_width/2 + branch_length, channel_width/2 + branch_length], 'w-', 'LineWidth', 2);
    hold off;
    
    % Labels and title
    xlabel('x (m)', 'FontSize', 12);
    ylabel('y (m)', 'FontSize', 12);
    title(sprintf('T-Junction Flame Propagation | Time = %.3f s', (n-1)*dt), 'FontSize', 14);
    axis equal;
    axis([0 main_channel_length -channel_width/2-0.01 channel_width/2+branch_length+0.01]);
    
    drawnow;
    
    % Save individual frame
    frame_count = frame_count + 1;
    frame_filename = fullfile(output_folder, sprintf('frame_%04d.png', frame_count));
    print(h, frame_filename, '-dpng', '-r100');
    
    % Create/append to GIF
    frame = getframe(h);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    if n == 1
        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
    
    fprintf('  Saved frame %d/%d\n', frame_count, ceil(Nt/frame_interval));
end

close(h);

% Create a final summary figure
figure('Position', [100 100 1200 800]);

% Plot snapshots at different times
times_to_plot = [1, round(Nt/4), round(Nt/2), round(3*Nt/4), Nt];
for idx = 1:length(times_to_plot)
    subplot(2, 3, idx);
    n = times_to_plot(idx);
    T_display = T(:,:,n);
    T_display(~mask) = NaN;
    
    imagesc(x_vec, y_vec, T_display);
    set(gca, 'YDir', 'normal');
    colormap(hot);
    colorbar;
    caxis([0 1]);
    title(sprintf('t = %.2f s', (n-1)*dt));
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal;
    axis([0 main_channel_length -channel_width/2-0.01 channel_width/2+branch_length+0.01]);
end

% Add overall title
sgtitle('T-Junction Flame Propagation Evolution', 'FontSize', 16);
saveas(gcf, fullfile(output_folder, 'flame_evolution_summary.png'));

fprintf('\n=== Simulation Complete ===\n');
fprintf('GIF saved as: %s\n', gif_filename);
fprintf('Individual frames saved in: %s\n', output_folder);
fprintf('Total frames saved: %d\n', frame_count);

%% --- Helper Functions ---
% =========================================================================
function in_junc = is_in_tjunction(x, y, main_length, width, branch_x1, branch_x2, branch_y_length)
    in_main = (x >= 0) & (x <= main_length) & (y >= -width/2) & (y <= width/2);
    in_branch = (x >= branch_x1) & (x <= branch_x2) & (y >= width/2) & (y <= width/2 + branch_y_length);
    in_junc = in_main | in_branch;
end

function [loss, gradients] = physicsLossT(net, X, D_th, D_mass, Le, beta, alpha, U_flow)
    % X is a dlarray with dimensions (C, B), where C=3 (x,y,t)
    
    % Predict T and Y_fuel
    Y_pred = forward(net, X);
    T_pred = Y_pred(1,:);
    Y_fuel_pred = Y_pred(2,:);
    
    % Extract coordinates
    x = X(1,:);
    y = X(2,:);
    t = X(3,:);
    
    % Compute gradients separately for each variable
    % Temperature gradients
    T_t = dlgradient(sum(T_pred, 'all'), t, 'EnableHigherDerivatives', true);
    T_x = dlgradient(sum(T_pred, 'all'), x, 'EnableHigherDerivatives', true);
    T_y = dlgradient(sum(T_pred, 'all'), y, 'EnableHigherDerivatives', true);
    
    % Second derivatives for temperature
    T_xx = dlgradient(sum(T_x, 'all'), x, 'EnableHigherDerivatives', false);
    T_yy = dlgradient(sum(T_y, 'all'), y, 'EnableHigherDerivatives', false);
    
    % Fuel gradients
    Y_t = dlgradient(sum(Y_fuel_pred, 'all'), t, 'EnableHigherDerivatives', true);
    Y_x = dlgradient(sum(Y_fuel_pred, 'all'), x, 'EnableHigherDerivatives', true);
    Y_y = dlgradient(sum(Y_fuel_pred, 'all'), y, 'EnableHigherDerivatives', true);
    
    % Second derivatives for fuel
    Y_xx = dlgradient(sum(Y_x, 'all'), x, 'EnableHigherDerivatives', false);
    Y_yy = dlgradient(sum(Y_y, 'all'), y, 'EnableHigherDerivatives', false);

    % Reaction rate
    omega = (beta^2/(2*Le)) * Y_fuel_pred .* exp(-beta*(1-T_pred)./(1-alpha*(1-T_pred)));

    % Governing equations as loss terms
    f_T = T_t + U_flow*T_x - D_th*(T_xx + T_yy) - omega;
    f_Y = Y_t + U_flow*Y_x - D_mass*(Y_xx + Y_yy) + omega;

    % Total loss
    loss = mean(f_T.^2, 'all') + mean(f_Y.^2, 'all');
    
    gradients = dlgradient(loss, net.Learnables);
end

function field = applyTjuncBC_simple(field, n, mask)
    % Simple boundary condition: zero gradient at boundaries
    [Ny, Nx, ~] = size(field);
    
    % Apply zero-gradient BC at domain boundaries
    field(1, :, n) = field(2, :, n);
    field(Ny, :, n) = field(Ny-1, :, n);
    field(:, 1, n) = field(:, 2, n);
    field(:, Nx, n) = field(:, Nx-1, n);
    
    % Ensure field is zero outside the T-junction
    field(:, :, n) = field(:, :, n) .* mask;
end