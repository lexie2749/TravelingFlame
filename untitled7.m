%% Simplified PINN-CFD Hybrid for Ring Flame Propagation
% This version uses simplified gradient computation for better compatibility
clear; close all; clc;
%% 1. Physical Parameters
% =========================================================================
% Geometry
R_inner = 0.04;      % Inner radius (m)
R_outer = 0.05;      % Outer radius (m)
% Physical parameters
Le = 0.7;            % Lewis number
beta = 8.0;          % Zeldovich number
alpha = 0.85;        % Heat release parameter
D_th = 1e-5;         % Thermal diffusivity (m²/s)
D_mass = D_th/Le;    % Mass diffusivity
% Grid parameters
Nr = 30;             % Radial points
Ntheta = 100;        % Angular points
Nt = 2000;           % Time steps
dt = 0.005;          % Time step (s)
t_final = Nt * dt;
% Neural network parameters
hiddenSize = 50;
numHiddenLayers = 4;
learningRate = 0.001;
numEpochs = 500;
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
for i = 1:Nr
    for j = 1:Ntheta
        T(i,j,1) = 0.1 + 0.9*exp(-((theta(j)-theta_ignition)/0.3)^2);
        Y_fuel(i,j,1) = 1 - 0.5*exp(-((theta(j)-theta_ignition)/0.3)^2);
    end
end
%% 3. Create Neural Network
% =========================================================================
% Build network layers
layers = [
    featureInputLayer(3, 'Name', 'input')
    fullyConnectedLayer(hiddenSize, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
];
for i = 2:numHiddenLayers
    layers = [layers
        fullyConnectedLayer(hiddenSize, 'Name', ['fc' num2str(i)])
        tanhLayer('Name', ['tanh' num2str(i)])
    ];
end
layers = [layers
    fullyConnectedLayer(2, 'Name', 'output')
    sigmoidLayer('Name', 'sigmoid')  % Ensure outputs are in [0,1]
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
wave_speed = 0.5;  % rad/s
for i = 1:numTrainPoints
    % Traveling wave solution
    phase = theta_train(i) - wave_speed * t_train(i);
    T_target(i) = 0.5 * (1 + tanh(5*(cos(phase) - 0.5)));
    Y_target(i) = 1 - 0.5 * T_target(i);
end
Y_target = [T_target'; Y_target'];
Y_target = dlarray(Y_target, 'CB');
%% 5. Simple Loss Function
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
%% 6. Train Network
% =========================================================================
fprintf('Training PINN...\n');
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
trainingTime = toc;
fprintf('Training completed in %.2f seconds\n', trainingTime);
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
                % Use finite difference for other regions
                j_m = mod(j-2, Ntheta) + 1;
                j_p = mod(j, Ntheta) + 1;
                
                % Laplacian in cylindrical coordinates
                dr = r(2) - r(1);
                dtheta = theta(2) - theta(1);
                
                % Temperature
                laplacian_T = (T(i+1,j,n-1) - 2*T(i,j,n-1) + T(i-1,j,n-1))/dr^2 + ...
                             (1/r(i))*(T(i+1,j,n-1) - T(i-1,j,n-1))/(2*dr) + ...
                             (1/r(i)^2)*(T(i,j_p,n-1) - 2*T(i,j,n-1) + T(i,j_m,n-1))/dtheta^2;
                
                % Species
                laplacian_Y = (Y_fuel(i+1,j,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i-1,j,n-1))/dr^2 + ...
                             (1/r(i))*(Y_fuel(i+1,j,n-1) - Y_fuel(i-1,j,n-1))/(2*dr) + ...
                             (1/r(i)^2)*(Y_fuel(i,j_p,n-1) - 2*Y_fuel(i,j,n-1) + Y_fuel(i,j_m,n-1))/dtheta^2;
                
                % Reaction rate
                T_local = max(0, min(1, T(i,j,n-1)));
                Y_local = max(0, min(1, Y_fuel(i,j,n-1)));
                
                if T_local > 0.01 && Y_local > 0.01
                    omega = (beta^2/(2*Le)) * Y_local * exp(-beta*(1-T_local)/(1-alpha*(1-T_local)));
                else
                    omega = 0;
                end
                
                % Time integration
                T(i,j,n) = T(i,j,n-1) + dt * (D_th * laplacian_T + omega);
                Y_fuel(i,j,n) = Y_fuel(i,j,n-1) + dt * (D_mass * laplacian_Y - omega);
                
                % Bound values
                T(i,j,n) = max(0, min(1, T(i,j,n)));
                Y_fuel(i,j,n) = max(0, min(1, Y_fuel(i,j,n)));
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
        fprintf('  Progress: %.0f%%\n', 100*n/Nt);
    end
end
simulationTime = toc;
%% 8. Visualization
% =========================================================================
% --- Separate plot for Training Loss ---
figure('Position', [100 100 800 600]);
semilogy(lossHistory, 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('Training Loss');
grid on;
saveas(gcf, 'training_loss.png'); % Save the training loss plot as a PNG
% --- Separate plot for Flame Propagation Angle ---
figure('Position', [100 100 800 600]);
flame_angle = zeros(Nt, 1);
for n = 1:Nt
[~, idx] = max(T(:,:,n), [], 'all');
[~, j_max] = ind2sub([Nr, Ntheta], idx);
flame_angle(n) = theta(j_max);
end
time_vec = (0:Nt-1) * dt;
plot(time_vec, flame_angle*180/pi, 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Flame Angle (°)');
title('Flame Propagation');
grid on;
saveas(gcf, 'flame_propagation_angle.png'); % Save the flame propagation plot
% --- Save Individual Animation Frames as PNG files ---
fprintf('Saving individual animation frames...\n');
output_folder = 'flame_frames'; % Define a subfolder to save the frames
if ~exist(output_folder, 'dir')
    mkdir(output_folder); % Create the folder if it doesn't exist
end
h = figure('Position', [100 100 800 600]);
axis tight manual;

for n = 1:50:Nt
    % 确保后续绘图命令作用于 h
    figure(h); 
    
    pcolor(X, Y, T(:,:,n));
    shading interp;
    colormap(jet); % 或者 colormap(turbo);
    caxis([0 1]);
    title(sprintf('Time = %.2f s', (n-1)*dt));
    xlabel('x (m)'); ylabel('y (m)');
    axis equal tight;
    drawnow;
    
    % Capture the frame
    frame = getframe(h);
    im = frame2im(frame);
    % Create a unique filename for each frame
    filename = fullfile(output_folder, sprintf('frame_%04d.png', n));
    % Save the frame as a PNG file
    imwrite(im, filename);
end
close(h); % Close the figure after saving all frames
fprintf('Individual frames saved in the "%s" folder.\n', output_folder);