%% Simplified PINN-CFD Hybrid for Ring Flame Propagation
% This version uses simplified gradient computation for better compatibility
clear; close all; clc;

% Set up the radius range and pre-allocate storage
R_inner_start = 0.4; % 25 mm inner radius (50 mm diameter)
R_inner_end = 0.5;   % 75 mm inner radius (150 mm diameter)
R_thickness = 0.1;   % 10 mm ring thickness
numRadii = 11;        % 25 to 75 mm with 5 mm intervals
radii_vec = linspace(R_inner_start, R_inner_end, numRadii);
flame_speeds = zeros(1, numRadii);
t_final = 2000 * 0.005; % Keep this constant for all runs

for i = 1:numRadii
    R_inner = radii_vec(i);
    R_outer = R_inner + R_thickness;
    
    fprintf('Starting simulation for R_inner = %.3f m...\n', R_inner);
    
    %% 1. Physical Parameters
    % =========================================================================
    % Geometry
    % R_inner and R_outer are now set in the loop
    
    % Physical parameters (unchanged)
    Le = 0.7; % Lewis number
    beta = 8.0; % Zeldovich number
    alpha = 0.85; % Heat release parameter
    D_th = 1e-5; % Thermal diffusivity (m²/s)
    D_mass = D_th/Le; % Mass diffusivity
    
    % Grid parameters
    Nr = 30; % Radial points
    Ntheta = 100; % Angular points
    Nt = 2000; % Time steps
    dt = 0.005; % Time step (s)
    
    %% 2. Create Grid
    % =========================================================================
    r = linspace(R_inner, R_outer, Nr);
    theta = linspace(0, 2*pi, Ntheta);
    [Theta, R] = meshgrid(theta, r);
    X = R .* cos(Theta);
    Y = R .* sin(Theta);
    
    % Initialize fields
    T = zeros(Nr, Ntheta, Nt);
    Y_fuel = ones(Nr, Ntheta, Nt);
    
    % Initial condition - hot spot
    theta_ignition = pi/4;
    for i_r = 1:Nr
        for j_th = 1:Ntheta
            T(i_r,j_th,1) = 0.1 + 0.9*exp(-((theta(j_th)-theta_ignition)/0.3)^2);
            Y_fuel(i_r,j_th,1) = 1 - 0.5*exp(-((theta(j_th)-theta_ignition)/0.3)^2);
        end
    end
    
    %% 3. Create Neural Network
    % =========================================================================
    % Build network layers
    hiddenSize = 50;
    numHiddenLayers = 4;
    layers = [
        featureInputLayer(3, 'Name', 'input')
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
        fullyConnectedLayer(2, 'Name', 'output')
        sigmoidLayer('Name', 'sigmoid') % Ensure outputs are in [0,1]
    ];
    
    % Create network
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
    
    %% 4. Generate Training Data
    % =========================================================================
    % Create training points
    numTrainPoints = 2000;
    r_train = R_inner + (R_outer - R_inner) * rand(numTrainPoints, 1);
    theta_train = 2*pi * rand(numTrainPoints, 1);
    t_train = t_final * rand(numTrainPoints, 1);
    
    % Create training data matrix
    X_train = [r_train, theta_train, t_train]';
    X_train = dlarray(X_train, 'CB');
    
    % Generate synthetic target data based on traveling wave solution
    T_target = zeros(numTrainPoints, 1);
    Y_target = ones(numTrainPoints, 1);
    wave_speed = 0.5; % rad/s
    for l = 1:numTrainPoints
        % Traveling wave solution
        phase = theta_train(l) - wave_speed * t_train(l);
        T_target(l) = 0.5 * (1 + tanh(5*(cos(phase) - 0.5)));
        Y_target(l) = 1 - 0.5 * T_target(l);
    end
    Y_target = [T_target'; Y_target'];
    Y_target = dlarray(Y_target, 'CB');
    
    %% 6. Train Network
    % =========================================================================
    % Neural network parameters (continued)
    learningRate = 0.001;
    numEpochs = 500;
    
    fprintf('Training PINN for R_inner = %.3f m...\n', R_inner);
    lossHistory = zeros(numEpochs, 1);
    % Adam optimizer state
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;
    tic;
    for epoch = 1:numEpochs
        iteration = iteration + 1;
        % Compute loss and gradients
        [loss, gradients] = dlfeval(@simpleLoss, net, X_train, Y_target, Le, beta, alpha);
        % Update parameters
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
            averageGrad, averageSqGrad, iteration, learningRate);
        lossHistory(epoch) = extractdata(loss);
        if mod(epoch, 50) == 0
            fprintf('Epoch %d/%d, Loss: %.6f\n', epoch, numEpochs, lossHistory(epoch));
        end
    end
    
    %% 7. Hybrid CFD-PINN Simulation
    % =========================================================================
    fprintf('Running hybrid simulation...\n');
    tic;
    for n = 2:Nt
        current_t = (n-1) * dt;
        % Compute temperature gradient
        [dT_dr, dT_dtheta] = gradient(T(:,:,n-1));
        gradT_mag = sqrt(dT_dr.^2 + dT_dtheta.^2);
        
        % Identify flame front (high gradient regions)
        flame_front = gradT_mag > 0.5*max(gradT_mag(:));
        
        for i_r = 2:Nr-1
            for j_th = 1:Ntheta
                if flame_front(i_r,j_th)
                    % Use PINN for flame front
                    X_point = dlarray([r(i_r); theta(j_th); current_t], 'CB');
                    Y_pred = predict(net, X_point);
                    Y_pred = extractdata(Y_pred);
                    T(i_r,j_th,n) = Y_pred(1);
                    Y_fuel(i_r,j_th,n) = Y_pred(2);
                else
                    % Use finite difference for other regions
                    j_m = mod(j_th-2, Ntheta) + 1;
                    j_p = mod(j_th, Ntheta) + 1;
                    % Laplacian in cylindrical coordinates
                    dr = r(2) - r(1);
                    dtheta = theta(2) - theta(1);
                    % Temperature
                    laplacian_T = (T(i_r+1,j_th,n-1) - 2*T(i_r,j_th,n-1) + T(i_r-1,j_th,n-1))/dr^2 + ...
                        (1/r(i_r))*(T(i_r+1,j_th,n-1) - T(i_r-1,j_th,n-1))/(2*dr) + ...
                        (1/r(i_r)^2)*(T(i_r,j_p,n-1) - 2*T(i_r,j_th,n-1) + T(i_r,j_m,n-1))/dtheta^2;
                    % Species
                    laplacian_Y = (Y_fuel(i_r+1,j_th,n-1) - 2*Y_fuel(i_r,j_th,n-1) + Y_fuel(i_r-1,j_th,n-1))/dr^2 + ...
                        (1/r(i_r))*(Y_fuel(i_r+1,j_th,n-1) - Y_fuel(i_r-1,j_th,n-1))/(2*dr) + ...
                        (1/r(i_r)^2)*(Y_fuel(i_r,j_p,n-1) - 2*Y_fuel(i_r,j_th,n-1) + Y_fuel(i_r,j_m,n-1))/dtheta^2;
                    % Reaction rate
                    T_local = max(0, min(1, T(i_r,j_th,n-1)));
                    Y_local = max(0, min(1, Y_fuel(i_r,j_th,n-1)));
                    if T_local > 0.01 && Y_local > 0.01
                        omega = (beta^2/(2*Le)) * Y_local * exp(-beta*(1-T_local)/(1-alpha*(1-T_local)));
                    else
                        omega = 0;
                    end
                    % Time integration
                    T(i_r,j_th,n) = T(i_r,j_th,n-1) + dt * (D_th * laplacian_T + omega);
                    Y_fuel(i_r,j_th,n) = Y_fuel(i_r,j_th,n-1) + dt * (D_mass * laplacian_Y - omega);
                    % Bound values
                    T(i_r,j_th,n) = max(0, min(1, T(i_r,j_th,n)));
                    Y_fuel(i_r,j_th,n) = max(0, min(1, Y_fuel(i_r,j_th,n)));
                end
            end
        end
        % Boundary conditions
        T(1,:,n) = T(2,:,n);
        T(Nr,:,n) = T(Nr-1,:,n);
        Y_fuel(1,:,n) = Y_fuel(2,:,n);
        Y_fuel(Nr,:,n) = Y_fuel(Nr-1,:,n);
        % Periodic boundary
        T(:,1,n) = T(:,end-1,n);
        T(:,end,n) = T(:,2,n);
        Y_fuel(:,1,n) = Y_fuel(:,end-1,n);
        Y_fuel(:,end,n) = Y_fuel(:,2,n);
        if mod(n, 200) == 0
            fprintf(' Progress: %.0f%%\n', 100*n/Nt);
        end
    end
    
    % Calculate flame speed and store it
    flame_angle = zeros(Nt, 1);
    for n = 1:Nt
        [~, idx] = max(T(:,:,n), [], 'all');
        [~, j_max] = ind2sub([Nr, Ntheta], idx);
        flame_angle(n) = theta(j_max);
    end
    flame_speed = mean(diff(unwrap(flame_angle))) / dt;
    flame_speeds(i) = flame_speed;
    
    fprintf('Simulation for R_inner = %.2f m complete. Flame speed: %.3f rad/s\n', R_inner, flame_speed);
    
end % End of the main loop

%% 8. Final Plot
% =========================================================================
figure;
plot(radii_vec * 1000, flame_speeds, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Ring Inner Radius (mm)');
ylabel('Average Flame Speed (rad/s)');
title('Flame Speed vs. Ring Radius (10 mm thickness)');
grid on;

fprintf('\nAll simulations complete. Final plot generated.\n');

%% 9. Simple Loss Function (Function definition must be at the end)
% =========================================================================
function [loss, gradients] = simpleLoss(net, X, Y_target, Le, beta, alpha)
    % Forward pass
    Y_pred = forward(net, X);
    % Data loss
    L_data = mean((Y_pred - Y_target).^2, 'all');
    % Extract predictions
    T_pred = Y_pred(1,:);
    Y_fuel_pred = Y_pred(2,:);
    % Physics-based regularization (simplified)
    omega = (beta^2/(2*Le)) * mean(Y_fuel_pred .* exp(-beta*(1-T_pred)), 'all');
    L_physics = 0.1 * abs(omega);
    % Conservation constraint
    L_conservation = mean((T_pred + (1-alpha)*Y_fuel_pred - 1).^2, 'all');
    % Total loss
    loss = L_data + 0.1*L_physics + 0.1*L_conservation;
    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end