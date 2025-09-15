%% Ring Flame Speed vs Diameter - Simple Finite Difference Method
% Pure FD simulation without PINN complexity
clear; close all; clc;

%% Parameters
% Physical parameters (typical for hydrocarbon fuels)
Le = 0.7;           % Lewis number (< 1 for thermo-diffusive instability)
beta = 8.0;         % Zeldovich number (temperature sensitivity)
alpha = 0.85;       % Heat release parameter
D_th = 1e-4;        % Thermal diffusivity (m²/s)
D_mass = D_th / Le; % Mass diffusivity

% Fixed geometry and fuel
channel_width = 0.01;  % 10 mm channel width
fuel_amount = 1.0;      % Normalized fuel amount (fixed)

% Ring diameter range: 50mm to 150mm with small steps for smooth curve
diameters = 50:1:150;   % 1mm step for very smooth curve
num_points = length(diameters);
flame_speeds = zeros(1, num_points);

% Grid parameters
Nr = 25;         % Radial points (sufficient for 10mm channel)
Ntheta = 120;    % Angular points (good resolution)
Nt = 5000;       % Time steps
dt = 0.001;      % Time step (s) - small for stability

%% Main Simulation Loop
fprintf('=====================================\n');
fprintf('Ring Flame Speed Simulation\n');
fprintf('=====================================\n');
fprintf('Diameter range: %d - %d mm\n', min(diameters), max(diameters));
fprintf('Step size: 1 mm\n');
fprintf('Total points: %d\n\n', num_points);

% Progress bar setup
fprintf('Progress: [');
progress_marks = 20;
next_mark = 1;

tic; % Start timing

for idx = 1:num_points
    diameter = diameters(idx) / 1000;  % Convert to meters
    R_inner = diameter / 2;
    R_outer = R_inner + channel_width;
    
    % Create grid
    r = linspace(R_inner, R_outer, Nr);
    theta = linspace(0, 2*pi, Ntheta);
    dr = r(2) - r(1);
    dtheta = theta(2) - theta(1);
    
    % Initialize fields
    T = zeros(Nr, Ntheta);      % Current temperature
    T_new = zeros(Nr, Ntheta);  % Next timestep temperature
    Y = ones(Nr, Ntheta);       % Current fuel fraction
    Y_new = ones(Nr, Ntheta);   % Next timestep fuel
    
    % Initial condition: localized hot spot for ignition
    theta_ignition = pi/4;
    sigma = 0.25;  % Width of initial hot spot
    
    for i = 1:Nr
        for j = 1:Ntheta
            % Angular distance (accounting for periodicity)
            dtheta_init = min(abs(theta(j) - theta_ignition), ...
                             2*pi - abs(theta(j) - theta_ignition));
            
            % Gaussian profile for initial condition
            T(i,j) = 0.1 + 0.9 * exp(-(dtheta_init/sigma)^2);
            Y(i,j) = 1.0 - 0.5 * exp(-(dtheta_init/sigma)^2);
        end
    end
    
    % Storage for flame position tracking
    flame_positions = zeros(Nt, 1);
    
    % Time evolution
    for n = 1:Nt
        % Update interior points
        for i = 2:Nr-1
            for j = 1:Ntheta
                % Handle periodic boundary in theta
                if j == 1
                    jm = Ntheta;
                    jp = 2;
                elseif j == Ntheta
                    jm = Ntheta - 1;
                    jp = 1;
                else
                    jm = j - 1;
                    jp = j + 1;
                end
                
                % Compute Laplacians in cylindrical coordinates
                % For temperature
                d2T_dr2 = (T(i+1,j) - 2*T(i,j) + T(i-1,j)) / dr^2;
                dT_dr = (T(i+1,j) - T(i-1,j)) / (2*dr);
                d2T_dtheta2 = (T(i,jp) - 2*T(i,j) + T(i,jm)) / dtheta^2;
                
                laplacian_T = d2T_dr2 + (1/r(i))*dT_dr + (1/r(i)^2)*d2T_dtheta2;
                
                % For fuel
                d2Y_dr2 = (Y(i+1,j) - 2*Y(i,j) + Y(i-1,j)) / dr^2;
                dY_dr = (Y(i+1,j) - Y(i-1,j)) / (2*dr);
                d2Y_dtheta2 = (Y(i,jp) - 2*Y(i,j) + Y(i,jm)) / dtheta^2;
                
                laplacian_Y = d2Y_dr2 + (1/r(i))*dY_dr + (1/r(i)^2)*d2Y_dtheta2;
                
                % Compute reaction rate (Arrhenius kinetics)
                T_local = max(0, min(1, T(i,j)));  % Ensure bounds
                Y_local = max(0, min(1, Y(i,j)));
                
                if T_local > 0.05 && Y_local > 0.01  % Threshold for reaction
                    denominator = 1 - alpha*(1 - T_local);
                    if denominator > 0.1  % Avoid division issues
                        omega = (beta^2/(2*Le)) * Y_local * ...
                                exp(-beta*(1 - T_local)/denominator);
                    else
                        omega = 0;
                    end
                else
                    omega = 0;
                end
                
                % Scale reaction rate by fuel amount
                omega = omega * fuel_amount;
                
                % Update using forward Euler
                T_new(i,j) = T(i,j) + dt * (D_th * laplacian_T + omega);
                Y_new(i,j) = Y(i,j) + dt * (D_mass * laplacian_Y - omega);
                
                % Enforce bounds
                T_new(i,j) = max(0, min(1, T_new(i,j)));
                Y_new(i,j) = max(0, min(1, Y_new(i,j)));
            end
        end
        
        % Apply boundary conditions (no-flux at walls)
        T_new(1,:) = T_new(2,:);      % Inner wall
        T_new(Nr,:) = T_new(Nr-1,:);  % Outer wall
        Y_new(1,:) = Y_new(2,:);
        Y_new(Nr,:) = Y_new(Nr-1,:);
        
        % Update fields
        T = T_new;
        Y = Y_new;
        
        % Track flame position (angle of maximum temperature)
        T_mean_radial = mean(T, 1);  % Average over radius
        [~, idx_max] = max(T_mean_radial);
        flame_positions(n) = theta(idx_max);
    end
    
    % Calculate flame speed from position data
    % Unwrap angles to handle 2π discontinuity
    flame_positions_unwrapped = unwrap(flame_positions);
    
    % Use linear regression on middle portion (avoid transients)
    start_idx = round(Nt/3);
    end_idx = Nt;
    time_vec = (start_idx:end_idx)' * dt;
    positions_analysis = flame_positions_unwrapped(start_idx:end_idx);
    
    % Fit linear model: position = speed * time + offset
    p = polyfit(time_vec, positions_analysis, 1);
    angular_speed = p(1);  % rad/s
    
    % Convert to linear speed at mean radius
    R_mean = (R_inner + R_outer) / 2;
    linear_speed = angular_speed * R_mean * 1000;  % mm/s
    flame_speeds(idx) = linear_speed;
    
    % Update progress bar
    if idx >= next_mark * num_points / progress_marks
        fprintf('=');
        next_mark = next_mark + 1;
    end
end

fprintf('] Done!\n');
elapsed_time = toc;
fprintf('Simulation completed in %.1f seconds\n\n', elapsed_time);

%% Create the plot
figure('Position', [100, 100, 900, 600]);

% Main plot
h = plot(diameters, flame_speeds, 'b-', 'LineWidth', 2.5);
xlabel('Ring Diameter (mm)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Flame Speed (mm/s)', 'FontSize', 13, 'FontWeight', 'bold');
title('Flame Propagation Speed vs Ring Diameter', 'FontSize', 15, 'FontWeight', 'bold');
grid on;
grid minor;

% Enhance appearance
set(gca, 'FontSize', 11);
set(gca, 'LineWidth', 1.2);

% Add markers at key points
hold on;
key_diameters = [50, 75, 100, 125, 150];
marker_speeds = zeros(size(key_diameters));

for i = 1:length(key_diameters)
    [~, idx] = min(abs(diameters - key_diameters(i)));
    marker_speeds(i) = flame_speeds(idx);
    plot(key_diameters(i), flame_speeds(idx), 'ro', ...
         'MarkerSize', 8, 'MarkerFaceColor', 'r');
    
    % Add value labels
    text(key_diameters(i), flame_speeds(idx) + 1, ...
         sprintf('%.1f', flame_speeds(idx)), ...
         'HorizontalAlignment', 'center', ...
         'FontSize', 10, 'FontWeight', 'bold');
end

% Add trend line
p_trend = polyfit(diameters, flame_speeds, 2);
trend_line = polyval(p_trend, diameters);
plot(diameters, trend_line, 'r--', 'LineWidth', 1.5, 'Alpha', 0.7);

% Add average line
avg_speed = mean(flame_speeds);
plot([min(diameters), max(diameters)], [avg_speed, avg_speed], ...
     'g--', 'LineWidth', 1.5);
text(mean(diameters), avg_speed - 1.5, ...
     sprintf('Average: %.1f mm/s', avg_speed), ...
     'HorizontalAlignment', 'center', 'Color', 'g', ...
     'FontSize', 11, 'FontWeight', 'bold');

hold off;

% Set axis limits with some padding
xlim([45, 155]);
y_range = max(flame_speeds) - min(flame_speeds);
ylim([min(flame_speeds) - 0.1*y_range, max(flame_speeds) + 0.15*y_range]);

% Add parameter box
param_str = sprintf(['Physical Parameters:\n' ...
                    'Lewis number (Le) = %.1f\n' ...
                    'Zeldovich number (β) = %.0f\n' ...
                    'Heat release (α) = %.2f\n' ...
                    'Channel width = %.0f mm'], ...
                    Le, beta, alpha, channel_width*1000);
                    
annotation('textbox', [0.15, 0.65, 0.22, 0.2], ...
          'String', param_str, ...
          'BackgroundColor', 'white', ...
          'EdgeColor', 'black', ...
          'LineWidth', 1.5, ...
          'FontSize', 10);

% Add legend
legend([h], {'Simulated flame speed'}, ...
       'Location', 'northeast', 'FontSize', 11);

%% Display detailed results
fprintf('=====================================\n');
fprintf('RESULTS SUMMARY\n');
fprintf('=====================================\n');
fprintf('Diameter range: %d - %d mm\n', min(diameters), max(diameters));
fprintf('Speed range: %.2f - %.2f mm/s\n', min(flame_speeds), max(flame_speeds));
fprintf('Average speed: %.2f mm/s\n', mean(flame_speeds));
fprintf('Standard deviation: %.2f mm/s\n', std(flame_speeds));
fprintf('Coefficient of variation: %.1f%%\n', 100*std(flame_speeds)/mean(flame_speeds));

% Analyze trend
fprintf('\nTrend Analysis:\n');
fprintf('Quadratic fit: speed = %.4f*d² + %.3f*d + %.2f\n', ...
        p_trend(1), p_trend(2), p_trend(3));

% Check for monotonicity
if all(diff(flame_speeds) < 0)
    fprintf('Trend: Monotonically decreasing\n');
elseif all(diff(flame_speeds) > 0)
    fprintf('Trend: Monotonically increasing\n');
else
    fprintf('Trend: Non-monotonic\n');
end

% Key points
fprintf('\nKey Points:\n');
for i = 1:length(key_diameters)
    fprintf('  D = %3d mm: Speed = %.2f mm/s\n', ...
            key_diameters(i), marker_speeds(i));
end

% Save results
save('flame_speed_results.mat', 'diameters', 'flame_speeds', 'p_trend');
fprintf('\nResults saved to: flame_speed_results.mat\n');

% Save figure
saveas(gcf, 'flame_speed_vs_diameter.png');
fprintf('Figure saved to: flame_speed_vs_diameter.png\n');