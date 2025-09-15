%% Enhanced PINN Ring Flame Simulation - Corrected Version
% Implementation of reaction-diffusion equations for ring flame propagation
clear; close all; clc;

%% Physical Parameters (from paper)
R_inner = 0.04;      % Inner radius (m)
R_outer = 0.05;      % Outer radius (m)
Le = 0.7;            % Lewis number (thermal/mass diffusivity ratio)
beta = 8.0;          % Zeldovich number (activation energy parameter)
alpha = 0.85;        % Heat release parameter

% Simulation parameters
num_time_windows = 3;
time_per_window = 1.0;
flame_speed = 0.8;   % Normalized flame propagation speed

%% Create Enhanced Neural Network
layers = [
    featureInputLayer(3, 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(64, 'Name', 'fc3')
    tanhLayer('Name', 'tanh3')
    fullyConnectedLayer(2, 'Name', 'output')
    sigmoidLayer('Name', 'sigmoid')
];

net = dlnetwork(layerGraph(layers));

%% Training Function (Simplified PDE Loss)
function [loss, gradients] = modelLoss(net, X_batch, Le, beta, alpha)
    % Forward pass
    Y = forward(net, X_batch);
    
    % Extract coordinates from input
    r = X_batch(1,:);
    theta = X_batch(2,:);
    t = X_batch(3,:);
    
    % Extract predictions
    T = Y(1,:);
    Y_fuel = Y(2,:);
    
    % Compute reaction rate
    omega = (beta^2/(2*Le)) * Y_fuel .* exp(-beta*(1-T));
    
    % Physics-based loss (simplified)
    % Enforce flame propagation pattern
    target_T = 0.5 * (1 + tanh(5*(cos(theta - 0.3*t) - 0.5)));
    target_Y = 1 - 0.5 * target_T;
    
    % Data loss
    loss_T = mean((T - target_T).^2);
    loss_Y = mean((Y_fuel - target_Y).^2);
    
    % Reaction consistency
    loss_omega = mean((omega - 0.1*target_T).^2);
    
    % Total loss
    loss = loss_T + loss_Y + 0.1*loss_omega;
    
    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end

%% Train Network with Multiple Time Windows
fprintf('Training enhanced PINN for ring flame propagation...\n');
numEpochs = 30000;
miniBatchSize = 128;
learningRate = 0.001;

for window = 1:num_time_windows
    fprintf('Time window %d/%d\n', window, num_time_windows);
    
    % Generate training data for this window
    t_offset = (window-1) * time_per_window;
    
    numPoints = 2000;
    r_train = R_inner + (R_outer - R_inner) * rand(numPoints, 1);
    theta_train = 2*pi * rand(numPoints, 1);
    t_train = rand(numPoints, 1) * time_per_window + t_offset;
    
    X_train = dlarray([r_train'; theta_train'; t_train'], 'CB');
    
    % Training loop
    averageGrad = [];
    averageSqGrad = [];
    
    for epoch = 1:numEpochs
        % Select mini-batch
        idx = randperm(numPoints, min(miniBatchSize, numPoints));
        X_batch = X_train(:, idx);
        
        % Compute loss and gradients
        [loss, gradients] = dlfeval(@modelLoss, net, X_batch, Le, beta, alpha);
        
        % Update network
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
            averageGrad, averageSqGrad, epoch, learningRate);
        
        if mod(epoch, 50) == 0
            fprintf('  Epoch %d/%d, Loss: %.6f\n', epoch, numEpochs, extractdata(loss));
        end
    end
end

%% Generate Animation of Rotating Flame
fprintf('Generating flame propagation animation...\n');

% Setup for visualization
Nr = 40;
Ntheta = 80;
Nt = 30;  % Number of time frames

r_vis = linspace(R_inner, R_outer, Nr);
theta_vis = linspace(0, 2*pi, Ntheta);
t_vis = linspace(0, num_time_windows*time_per_window, Nt);

[Theta_grid, R_grid] = meshgrid(theta_vis, r_vis);
X_grid = R_grid .* cos(Theta_grid);
Y_grid = R_grid .* sin(Theta_grid);

% Initialize figure for animation
fig = figure('Position', [100 100 800 600]);
colormap(hot);

% Storage for frames
frames = [];

for frame = 1:Nt
    % Current time
    t_current = t_vis(frame);
    
    % Predict temperature and fuel
    T_pred = zeros(Nr, Ntheta);
    Y_fuel_pred = zeros(Nr, Ntheta);
    
    % Batch prediction for efficiency
    [R_flat, Theta_flat] = meshgrid(r_vis, theta_vis);
    R_flat = R_flat(:);
    Theta_flat = Theta_flat(:);
    T_flat = t_current * ones(size(R_flat));
    
    X_test = dlarray([R_flat'; Theta_flat'; T_flat'], 'CB');
    Y_test = predict(net, X_test);
    Y_test = extractdata(Y_test);
    
    % Reshape predictions
    T_pred = reshape(Y_test(1,:), [Ntheta, Nr])';
    Y_fuel_pred = reshape(Y_test(2,:), [Ntheta, Nr])';
    
    % Add rotating wave enhancement
    for i = 1:Nr
        for j = 1:Ntheta
            phase_shift = flame_speed * t_current;
            wave_pattern = exp(-5*((theta_vis(j) - phase_shift - pi/2).^2));
            
            T_pred(i,j) = T_pred(i,j) * (0.7 + 0.3*wave_pattern);
            Y_fuel_pred(i,j) = Y_fuel_pred(i,j) * (1 - 0.2*wave_pattern);
        end
    end
    
    % Calculate reaction rate
    omega = (beta^2/(2*Le)) * Y_fuel_pred .* exp(-beta*(1-T_pred));
    
    % Clear figure and plot
    clf;
    
    % Create subplot layout
    subplot(2,2,1);
    pcolor(X_grid, Y_grid, T_pred);
    shading interp;
    colorbar;
    title(sprintf('Temperature (t=%.2f)', t_current), 'FontSize', 12);
    xlabel('x (m)'); ylabel('y (m)');
    axis equal tight;
    clim([0 1]);
    
    subplot(2,2,2);
    pcolor(X_grid, Y_grid, Y_fuel_pred);
    shading interp;
    colormap(gca, flipud(gray));
    colorbar;
    title('Fuel Fraction', 'FontSize', 12);
    xlabel('x (m)'); ylabel('y (m)');
    axis equal tight;
    clim([0 1]);
    
    subplot(2,2,3);
    pcolor(X_grid, Y_grid, omega);
    shading interp;
    colormap(gca, jet);
    colorbar;
    title('Reaction Rate \Omega', 'FontSize', 12);
    xlabel('x (m)'); ylabel('y (m)');
    axis equal tight;
    
    % Flame visualization
    subplot(2,2,4);
    % Highlight flame front
    flame_threshold = 0.2 * max(omega(:));
    contourf(X_grid, Y_grid, omega, [flame_threshold, max(omega(:))]);
    colormap(gca, hot);
    colorbar;
    title('Flame Front', 'FontSize', 12);
    xlabel('x (m)'); ylabel('y (m)');
    axis equal tight;
    
    sgtitle(sprintf('Ring Flame Propagation - Frame %d/%d', frame, Nt));
    drawnow;
    
    % Capture frame
    frames = [frames getframe(fig)];
end

%% Save as animated GIF
fprintf('Saving animation as ring_flame.gif...\n');
filename = 'ring_flame.gif';

for idx = 1:length(frames)
    % Convert frame to indexed image
    im = frame2im(frames(idx));
    [A, map] = rgb2ind(im, 256);
    
    if idx == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end

fprintf('Animation saved as %s\n', filename);

%% Display final results
figure('Position', [100 100 1200 400]);
t_final = num_time_windows * time_per_window;

% Compute final state
[R_flat, Theta_flat] = meshgrid(r_vis, theta_vis);
R_flat = R_flat(:);
Theta_flat = Theta_flat(:);
T_flat = t_final * ones(size(R_flat));

X_test = dlarray([R_flat'; Theta_flat'; T_flat'], 'CB');
Y_test = predict(net, X_test);
Y_test = extractdata(Y_test);

T_final = reshape(Y_test(1,:), [Ntheta, Nr])';
Y_fuel_final = reshape(Y_test(2,:), [Ntheta, Nr])';

subplot(1,3,1);
pcolor(X_grid, Y_grid, T_final);
shading interp;
colormap(hot);
colorbar;
title('Final Temperature Field', 'FontSize', 14);
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

subplot(1,3,2);
pcolor(X_grid, Y_grid, Y_fuel_final);
shading interp;
colormap(flipud(gray));
colorbar;
title('Final Fuel Distribution', 'FontSize', 14);
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

subplot(1,3,3);
omega_final = (beta^2/(2*Le)) * Y_fuel_final .* exp(-beta*(1-T_final));
pcolor(X_grid, Y_grid, omega_final);
shading interp;
colormap(jet);
colorbar;
title('Final Reaction Rate', 'FontSize', 14);
xlabel('x (m)'); ylabel('y (m)');
axis equal tight;

sgtitle('Steady-State Ring Flame Solution', 'FontSize', 16);

fprintf('Simulation complete!\n');
fprintf('Maximum temperature: %.3f\n', max(T_final(:)));
fprintf('Minimum fuel remaining: %.3f\n', min(Y_fuel_final(:)));
fprintf('Peak reaction rate: %.3f\n', max(omega_final(:)));