%% Enhanced PINN for Ring Flame Propagation
% Based on methods from "Using Physics-Informed neural networks for solving 
% Navier-Stokes equations in fluid dynamic complex scenarios"

clear; close all; clc;

%% 1. Physical Parameters (Section 3 of paper)
% =========================================================================

% Geometry (normalized as per paper)
R_inner = 0.04;      % Inner radius (m)
R_outer = 0.05;      % Outer radius (m)

% Physical parameters
Le = 0.7;            % Lewis number
beta = 8.0;          % Zeldovich number  
alpha = 0.85;        % Heat release parameter
Re = 100;            % Reynolds number (as per paper)
D_th = 1e-5;         % Thermal diffusivity (m²/s)
D_mass = D_th/Le;    % Mass diffusivity

% Grid parameters (Section 6.6 - optimal point distribution)
Nr = 30;             % Radial points
Ntheta = 100;        % Angular points

% Multiple time windows (Section 5.2 and Appendix B)
numTimeWindows = 3;  % As recommended in paper
timePerWindow = 1.0; % seconds
dt = 0.005;
stepsPerWindow = round(timePerWindow/dt);
totalSteps = numTimeWindows * stepsPerWindow;

% Training parameters (from paper's best practices)
hiddenLayers = 6;    % Optimal from Table 5
neuronsPerLayer = 256; % Optimal from Table 4
learningRateStart = 0.001;
decayRate = 0.999947; % Exponential decay as per paper
numEpochs = 100000;  % Per time window

%% 2. Create Computational Grid
% =========================================================================

r = linspace(R_inner, R_outer, Nr);
theta = linspace(0, 2*pi, Ntheta);
[Theta, R] = meshgrid(theta, r);
X = R .* cos(Theta);
Y = R .* sin(Theta);

% Initialize fields
T = zeros(Nr, Ntheta, totalSteps);
Y_fuel = ones(Nr, Ntheta, totalSteps);

% Initial condition - hot spot
theta_ignition = pi/4;
for i = 1:Nr
    for j = 1:Ntheta
        T(i,j,1) = 0.1 + 0.9*exp(-((theta(j)-theta_ignition)/0.3)^2);
        Y_fuel(i,j,1) = 1 - 0.5*exp(-((theta(j)-theta_ignition)/0.3)^2);
    end
end

%% 3. Modified Fourier Network (MFN) Architecture (Appendix A)
% =========================================================================

% Build MFN-inspired network with standard MATLAB layers
layers = [
    featureInputLayer(3, 'Name', 'input')
    % Fourier feature encoding using standard layers
    fullyConnectedLayer(128, 'Name', 'fourier_encoding1')
    tanhLayer('Name', 'fourier_activation1')  % For sinusoidal-like behavior
    fullyConnectedLayer(128, 'Name', 'fourier_encoding2')
    tanhLayer('Name', 'fourier_activation2')
];

% Add transformation layers as per MFN architecture
layers = [layers
    fullyConnectedLayer(neuronsPerLayer, 'Name', 'transform1')
    tanhLayer('Name', 'tanh_transform1')
    fullyConnectedLayer(neuronsPerLayer, 'Name', 'transform2')
    tanhLayer('Name', 'tanh_transform2')
];

% Main hidden layers with tanh activation (MATLAB compatible)
for i = 1:hiddenLayers
    layers = [layers
        fullyConnectedLayer(neuronsPerLayer, 'Name', ['fc' num2str(i)])
        tanhLayer('Name', ['tanh' num2str(i)])  % Using tanh for compatibility
    ];
end

% Output layer
layers = [layers
    fullyConnectedLayer(3, 'Name', 'output')  % [T, Y_fuel, p]
    sigmoidLayer('Name', 'output_activation')  % Ensure bounded outputs
];

% Create network
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

%% 4. Simplified Loss Function (Compatible with MATLAB)
% =========================================================================

function [loss, gradients] = simplifiedLoss(net, X, Y_target, Le, beta, alpha)
    % Forward pass
    Y_pred = forward(net, X);
    
    % Extract predictions
    T_pred = Y_pred(1,:);
    Y_fuel_pred = Y_pred(2,:);
    p_pred = Y_pred(3,:);
    
    % Simplified reaction rate
    omega = (beta^2/(2*Le)) * Y_fuel_pred .* exp(-beta*(1-T_pred));
    
    % Physics-based regularization (simplified)
    L_physics = 0.1 * mean(abs(omega));
    
    % Conservation constraint
    L_conservation = mean((T_pred + (1-alpha)*Y_fuel_pred - 1).^2);
    
    % Data loss
    L_data = mean((Y_pred - Y_target).^2, 'all');
    
    % Total loss
    loss = L_data + 0.1*L_physics + 0.1*L_conservation;
    
    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end

%% 5. Training with Multiple Time Windows (Section 5.2 & 7.1)
% =========================================================================

fprintf('Training PINN with multiple time windows...\n');
models = cell(numTimeWindows, 1);

for window = 1:numTimeWindows
    fprintf('\n=== Time Window %d/%d ===\n', window, numTimeWindows);
    
    % Time range for this window
    t_start = (window-1) * timePerWindow;
    t_end = window * timePerWindow;
    
    % Generate training points (stratified sampling as per paper)
    numTrainPoints = 4850;  % Mid-level from Table 7
    r_train = R_inner + (R_outer - R_inner) * rand(numTrainPoints, 1);
    theta_train = 2*pi * rand(numTrainPoints, 1);
    t_train = t_start + (t_end - t_start) * rand(numTrainPoints, 1);
    
    X_train = dlarray([r_train'; theta_train'; t_train'], 'CB');
    
    % Generate target data (from previous window or initial conditions)
    if window == 1
        % Use initial conditions
        T_target = 0.1 + 0.9*exp(-((theta_train-theta_ignition)/0.3).^2);
        Y_target = 1 - 0.5*exp(-((theta_train-theta_ignition)/0.3).^2);
    else
        % Fine-tuning: use previous window's final state (Section 7.1)
        net = models{window-1};  % Start from previous model
        X_prev = dlarray([r_train'; theta_train'; repmat(t_start, 1, numTrainPoints)], 'CB');
        Y_prev = predict(net, X_prev);
        T_target = extractdata(Y_prev(1,:))';
        Y_target = extractdata(Y_prev(2,:))';
    end
    
    p_target = zeros(numTrainPoints, 1);  % Pressure target
    Y_target = dlarray([T_target'; Y_target'; p_target'], 'CB');
    
    % Adam optimizer with exponential decay
    averageGrad = [];
    averageSqGrad = [];
    
    % Training loop
    tic;
    lossHistory = zeros(numEpochs, 1);
    for epoch = 1:numEpochs
        iteration = (window-1)*numEpochs + epoch;
        learningRate = learningRateStart * (decayRate^iteration);
        
        % Compute loss and gradients with simplified function
        [loss, gradients] = dlfeval(@simplifiedLoss, net, X_train, ...
            Y_target, Le, beta, alpha);
        
        % Update parameters using Adam
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
            averageGrad, averageSqGrad, iteration, learningRate);
        
        lossHistory(epoch) = extractdata(loss);
        
        if mod(epoch, 10000) == 0
            fprintf('  Epoch %d/%d, Loss: %.6f, LR: %.6f\n', ...
                epoch, numEpochs, lossHistory(epoch), learningRate);
        end
    end
    windowTime = toc;
    fprintf('  Window %d training time: %.2f minutes\n', window, windowTime/60);
    
    % Store trained model
    models{window} = net;
end

%% 6. Hybrid Simulation with Enhanced PINN
% =========================================================================

fprintf('\nRunning enhanced hybrid simulation...\n');
tic;

for n = 2:totalSteps
    current_t = (n-1) * dt;
    window_idx = min(ceil(current_t / timePerWindow), numTimeWindows);
    net = models{window_idx};
    
    % Compute gradient for flame front detection
    [dT_dr, dT_dtheta] = gradient(T(:,:,n-1));
    gradT_mag = sqrt(dT_dr.^2 + dT_dtheta.^2);
    flame_front = gradT_mag > 0.3*max(gradT_mag(:));  % Adjusted threshold
    
    for i = 2:Nr-1
        for j = 1:Ntheta
            if flame_front(i,j)
                % Use PINN for flame front
                X_point = dlarray([r(i); theta(j); current_t], 'CB');
                Y_pred = predict(net, X_point);
                Y_pred = extractdata(Y_pred);
                T(i,j,n) = Y_pred(1);
                Y_fuel(i,j,n) = Y_pred(2);
            else
                % Finite difference for bulk regions (optimized)
                j_m = mod(j-2, Ntheta) + 1;
                j_p = mod(j, Ntheta) + 1;
                
                dr = r(2) - r(1);
                dtheta = theta(2) - theta(1);
                
                % Enhanced finite difference with upwind scheme
                if j > 1 && j < Ntheta
                    % Central difference
                    laplacian_T = (T(i+1,j,n-1) - 2*T(i,j,n-1) + T(i-1,j,n-1))/dr^2 + ...
                                 (1/r(i))*(T(i+1,j,n-1) - T(i-1,j,n-1))/(2*dr) + ...
                                 (1/r(i)^2)*(T(i,j_p,n-1) - 2*T(i,j,n-1) + T(i,j_m,n-1))/dtheta^2;
                else
                    % Upwind for boundaries
                    laplacian_T = (T(i+1,j,n-1) - 2*T(i,j,n-1) + T(i-1,j,n-1))/dr^2;
                end
                
                % Similar for Y_fuel
                laplacian_Y = (Y_fuel(i+1,j,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i-1,j,n-1))/dr^2 + ...
                             (1/r(i))*(Y_fuel(i+1,j,n-1) - Y_fuel(i-1,j,n-1))/(2*dr) + ...
                             (1/r(i)^2)*(Y_fuel(i,j_p,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i,j_m,n-1))/dtheta^2;
                
                % Enhanced reaction rate with stability check
                T_local = max(0, min(1, T(i,j,n-1)));
                Y_local = max(0, min(1, Y_fuel(i,j,n-1)));
                
                if T_local > 0.01 && Y_local > 0.01
                    denominator = 1 - alpha*(1-T_local);
                    if abs(denominator) > 1e-6
                        omega = (beta^2/(2*Le)) * Y_local * exp(-beta*(1-T_local)/denominator);
                        omega = min(omega, 100);  % Limit for stability
                    else
                        omega = 0;
                    end
                else
                    omega = 0;
                end
                
                % Time integration with CFL condition
                CFL = 0.5;  % Courant number
                dt_stable = CFL * min(dr^2, (r(i)*dtheta)^2) / D_th;
                dt_use = min(dt, dt_stable);
                
                T(i,j,n) = T(i,j,n-1) + dt_use * (D_th * laplacian_T + omega);
                Y_fuel(i,j,n) = Y_fuel(i,j,n-1) + dt_use * (D_mass * laplacian_Y - omega);
                
                % Bound values
                T(i,j,n) = max(0, min(1, T(i,j,n)));
                Y_fuel(i,j,n) = max(0, min(1, Y_fuel(i,j,n)));
            end
        end
    end
    
    % Enhanced boundary conditions
    % Inner and outer walls (no-slip, adiabatic)
    T(1,:,n) = T(2,:,n);
    T(Nr,:,n) = T(Nr-1,:,n);
    Y_fuel(1,:,n) = Y_fuel(2,:,n);
    Y_fuel(Nr,:,n) = Y_fuel(Nr-1,:,n);
    
    % Periodic boundary in theta
    T(:,1,n) = T(:,end-1,n);
    T(:,end,n) = T(:,2,n);
    Y_fuel(:,1,n) = Y_fuel(:,end-1,n);
    Y_fuel(:,end,n) = Y_fuel(:,2,n);
    
    if mod(n, 100) == 0
        fprintf('  Progress: %.0f%%\n', 100*n/totalSteps);
    end
end
simulationTime = toc;

%% 7. Enhanced Visualization with Paper's Metrics
% =========================================================================

figure('Position', [100 100 1600 900]);

% Use the custom colormap as before
custom_colors = [
    0.0, 0.0, 0.5;   % Dark blue (cold)
    0.0, 0.3, 0.8;   % Blue
    0.0, 0.6, 0.9;   % Cyan
    0.0, 0.8, 0.5;   % Green-cyan
    0.3, 0.9, 0.3;   % Green
    0.7, 0.9, 0.0;   % Yellow-green
    1.0, 0.9, 0.0;   % Yellow
    1.0, 0.6, 0.0;   % Orange
    1.0, 0.3, 0.0;   % Dark orange
    1.0, 0.0, 0.0;   % Red (hot)
];
custom_cmap = interp1(linspace(0,1,size(custom_colors,1)), custom_colors, linspace(0,1,256));

% Temperature field
subplot(2,3,1);
pcolor(X, Y, T(:,:,end));
shading interp;
colormap(gca, custom_cmap);
colorbar;
hold on;
contour(X, Y, T(:,:,end), [0.3, 0.5, 0.7], 'k-', 'LineWidth', 1.5);
hold off;
title('Temperature (Final)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

% Fuel concentration
subplot(2,3,2);
fuel_cmap = [
    0.0, 0.0, 0.8;   % Blue (consumed)
    0.3, 0.3, 0.9;
    0.6, 0.6, 1.0;
    1.0, 1.0, 1.0;   % White (partial)
    1.0, 0.8, 0.8;
    1.0, 0.5, 0.5;
    1.0, 0.0, 0.0;   % Red (unburned)
];
fuel_map = interp1(linspace(0,1,size(fuel_cmap,1)), fuel_cmap, linspace(0,1,256));
pcolor(X, Y, Y_fuel(:,:,end));
shading interp;
colormap(gca, fuel_map);
colorbar;
hold on;
contour(X, Y, Y_fuel(:,:,end), [0.2, 0.5, 0.8], 'k--', 'LineWidth', 1);
hold off;
title('Fuel Mass Fraction (Final)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

% Reaction rate
subplot(2,3,3);
omega_vis = zeros(Nr, Ntheta);
for i = 1:Nr
    for j = 1:Ntheta
        T_local = T(i,j,end);
        Y_local = Y_fuel(i,j,end);
        if T_local > 0.01 && Y_local > 0.01
            omega_vis(i,j) = (beta^2/(2*Le)) * Y_local * exp(-beta*(1-T_local)/(1-alpha*(1-T_local)));
        end
    end
end
pcolor(X, Y, omega_vis);
shading interp;
colormap(gca, 'turbo');
colorbar;
title('Reaction Rate (ω)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

% Flame propagation angle
subplot(2,3,4);
flame_angle = zeros(totalSteps, 1);
for n = 1:totalSteps
    [~, idx] = max(T(:,:,n), [], 'all');
    [~, j_max] = ind2sub([Nr, Ntheta], idx);
    flame_angle(n) = theta(j_max);
end
time_vec = (0:totalSteps-1) * dt;
plot(time_vec, flame_angle*180/pi, 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2]);
xlabel('Time (s)'); ylabel('Flame Angle (°)');
title('Flame Propagation', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'GridAlpha', 0.3);

% Mean Absolute Error (MAE) tracking
subplot(2,3,5);
% Calculate theoretical flame speed
S_L = sqrt(D_th * mean(omega_vis(:)));  % Simplified ZFK relation
theoretical_angle = flame_angle(1) + S_L * time_vec';
mae_angle = abs(flame_angle - theoretical_angle);
semilogy(time_vec, mae_angle, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
xlabel('Time (s)'); ylabel('MAE (degrees)');
title('Propagation Error vs Theory', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 3D visualization
subplot(2,3,6);
surf(X, Y, T(:,:,end));
shading interp;
colormap(gca, custom_cmap);
colorbar;
view(45, 30);
lighting gouraud;
light('Position', [1 1 1]);
title('3D Temperature Field', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x (m)'); ylabel('y (m)'); zlabel('Temperature');
axis tight;

%% 8. Performance Metrics (as per paper Section 6.1)
% =========================================================================

fprintf('\n=== Performance Summary (Paper Metrics) ===\n');
fprintf('Simulation Time: %.2f s\n', simulationTime);
fprintf('Time per Window: %.2f s (avg)\n', simulationTime/numTimeWindows);

% Calculate MAE metrics as per paper
MAE_avg = mean(mae_angle);
MAE_st = mean(mae_angle(round(0.8*end):end));  % Steady state

fprintf('\nMean Absolute Error (MAE):\n');
fprintf('  Average: %.4f degrees\n', MAE_avg);
fprintf('  Steady State: %.4f degrees\n', MAE_st);

% Calculate flame speed
flame_speed = mean(diff(unwrap(flame_angle))) / dt;
fprintf('\nFlame Propagation:\n');
fprintf('  Average Speed: %.3f rad/s\n', flame_speed);
fprintf('  Laminar Flame Speed (S_L): %.4f m/s\n', S_L);

% Energy conservation check
total_energy = squeeze(sum(sum(T + (1-alpha)*Y_fuel, 1), 2));
energy_drift = abs(total_energy(end) - total_energy(1)) / abs(total_energy(1));
fprintf('\nConservation Metrics:\n');
fprintf('  Energy Drift: %.2f%%\n', energy_drift*100);

% Reynolds and Lewis number effects
fprintf('\nDimensionless Parameters:\n');
fprintf('  Reynolds Number: %.0f\n', Re);
fprintf('  Lewis Number: %.2f\n', Le);
fprintf('  Zeldovich Number: %.1f\n', beta);

fprintf('\n=== Enhanced PINN Simulation Complete ===\n');