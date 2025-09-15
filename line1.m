%% Ring-Shaped Trough Flame Propagation with Separate Figure Export
% Enhanced CFD simulation with independent image output
% Each plot is saved individually as PNG and GIF
clear; close all; clc;
%% Initialize Simulation Parameters
fprintf('\n🔥 Ring-Shaped Trough Flame Propagation Simulation (Independent Image Output)\n');
fprintf('====================================\n\n');
% Create simulation data structure
sim = struct();
% Domain parameters - rectangular domain representing the unfolded trough
sim.Lx = 1.2;      % Domain length (m)
sim.Ly = 0.5;      % Domain height (trough width)
sim.nx = 300;      % Number of grid points in x direction
sim.ny = 60;       % Number of grid points in y direction
% Time parameters
sim.dt = 0.0001;   % Time step size
sim.t_final = 2.0; % Final simulation time
% Physical parameters - adjusted for faster flame propagation
sim.D_T = 0.025;   % Thermal diffusivity (increased)
sim.D_Y = 0.015;   % Fuel diffusivity (increased)
sim.nu = 0.01;     % Kinematic viscosity
sim.v_base = 1.2;  % Base flow velocity (increased)
sim.Da = 25.0;     % Damköhler number (increased reaction rate)
sim.beta = 12.0;   % Heat release parameter (increased)
sim.E_a = 3.5;     % Activation energy parameter (reduced to speed up reaction)
sim.rho = 1.0;     % Density
% Animation parameters
sim.save_interval = 100; % Save a frame every 100 time steps
%% Create Output Folders
% Use the current working directory to create the output folders
current_dir = pwd;
folders = struct();
folders.main = fullfile(current_dir, 'flame_simulation_output');
folders.temperature = fullfile(folders.main, '01_temperature_field');
folders.velocity = fullfile(folders.main, '02_velocity_field');
folders.fuel = fullfile(folders.main, '03_fuel_concentration');
folders.centerline = fullfile(folders.main, '04_centerline_profile');
folders.propagation = fullfile(folders.main, '05_flame_propagation');
folders.reaction = fullfile(folders.main, '06_reaction_rate');
% Create all folders
folderList = fieldnames(folders);
for i = 1:length(folderList)
    folderPath = folders.(folderList{i});
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end
%% Initialize Computational Domain
sim.dx = sim.Lx / (sim.nx - 1);
sim.dy = sim.Ly / (sim.ny - 1);
sim.x = linspace(0, sim.Lx, sim.nx);
sim.y = linspace(0, sim.Ly, sim.ny);
[sim.X, sim.Y] = meshgrid(sim.x, sim.y);
%% Initialize Flow Field Variables
% Velocity field - Poiseuille flow profile
y_norm = (sim.Y - sim.Ly/2) / (sim.Ly/2);
sim.u = sim.v_base * (1 - y_norm.^2);  % Parabolic velocity profile
sim.v = zeros(sim.ny, sim.nx);         % Initial vertical velocity is zero
sim.p = zeros(sim.ny, sim.nx);         % Pressure field
% Initial temperature field - ignite from the left side
sim.T = ones(sim.ny, sim.nx);  % Base temperature
ignition_zone = sim.X < 0.4;
sim.T(ignition_zone) = 3.0;  % High ignition temperature
% Initial fuel distribution
sim.Y_fuel = ones(sim.ny, sim.nx);     % Uniform fuel distribution
sim.Y_fuel(ignition_zone) = 0.1;       % Fuel already consumed in the ignition zone
% History logs
sim.T_history = {};
sim.Y_history = {};
sim.u_history = {};
sim.flame_position = [];
sim.max_temp = [];
%% Prepare GIF files
gif_files = struct();
gif_files.temperature = fullfile(folders.main, 'temperature_field.gif');
gif_files.velocity = fullfile(folders.main, 'velocity_field.gif');
gif_files.fuel = fullfile(folders.main, 'fuel_concentration.gif');
gif_files.centerline = fullfile(folders.main, 'centerline_profile.gif');
gif_files.propagation = fullfile(folders.main, 'flame_propagation.gif');
gif_files.reaction = fullfile(folders.main, 'reaction_rate.gif');
frame_rate = 20;
delay_time = 1/frame_rate;
%% Create independent figure windows
figures = struct();
figures.temperature = figure('Name', 'Temperature Field', 'Position', [50, 50, 800, 600], 'Visible', 'off');
figures.velocity = figure('Name', 'Velocity Field', 'Position', [100, 100, 800, 600], 'Visible', 'off');
figures.fuel = figure('Name', 'Fuel Concentration', 'Position', [150, 150, 800, 600], 'Visible', 'off');
figures.centerline = figure('Name', 'Centerline Profile', 'Position', [200, 200, 800, 600], 'Visible', 'off');
figures.propagation = figure('Name', 'Flame Propagation', 'Position', [250, 250, 800, 600], 'Visible', 'off');
figures.reaction = figure('Name', 'Reaction Rate', 'Position', [300, 300, 800, 600], 'Visible', 'off');
%% Main Simulation Loop
fprintf('🔥 Starting fast flame propagation simulation...\n');
fprintf('----------------------------------------\n');
t = 0;
step = 0;
frame_count = 0;
first_frame = true;
tic;
while t < sim.t_final
    % Advance CFD time step
    sim = advance_cfd_timestep(sim);
    
    % Update time
    t = t + sim.dt;
    step = step + 1;
    
    % Save and visualize
    if mod(step, sim.save_interval) == 0
        frame_count = frame_count + 1;
        
        % Save historical data
        sim.T_history{end+1} = sim.T;
        sim.Y_history{end+1} = sim.Y_fuel;
        sim.u_history{end+1} = sim.u;
        
        % Calculate flame front position
        T_mean = mean(sim.T, 1);
        flame_threshold = 1.5;
        flame_indices = find(T_mean > flame_threshold, 1, 'last');
        if ~isempty(flame_indices)
            sim.flame_position(end+1) = sim.x(flame_indices);
        else
            sim.flame_position(end+1) = sim.x(1);
        end
        sim.max_temp(end+1) = max(sim.T(:));
        
        % Calculate flame speed
        if length(sim.flame_position) > 1
            current_speed = (sim.flame_position(end) - sim.flame_position(end-1))/(sim.dt*sim.save_interval);
        else
            current_speed = 0;
        end
        
        % Print progress
        fprintf('Time: %.3f s, Max Temp: %.2f K, Flame Pos: %.2f m, Speed: %.3f m/s\n', ...
            t, sim.max_temp(end), sim.flame_position(end), current_speed);
        
        % Plot and save each independent figure
        plot_and_save_figures(sim, t, frame_count, first_frame, delay_time, figures, folders, gif_files);
        
        if first_frame
            first_frame = false;
        end
    end
end
elapsed_time = toc;
% Close all figure windows
figNames = fieldnames(figures);  % Fixed typo: was 'fieldname'
for i = 1:length(figNames)
    close(figures.(figNames{i}));
end
fprintf('\n✅ Simulation complete!\n');
fprintf('📊 Generated %d animation frames\n', frame_count);
fprintf('⏱️ Total simulation time: %.2f seconds\n', elapsed_time);
fprintf('🔥 Average flame speed: %.3f m/s\n', mean(diff(sim.flame_position)/(sim.dt*sim.save_interval)));
fprintf('📁 All files saved to folder: %s\n', folders.main);
%% Generate final analysis plots (saved independently)
generate_final_analysis(sim, folders.main);
fprintf('\n🎉 Flame propagation simulation finished!\n');
fprintf('====================================\n\n');
%% ==================== Plotting and Saving Functions ====================
function plot_and_save_figures(sim, t, frame_count, first_frame, delay_time, figures, folders, gif_files)
    
    % 1. Temperature field
    figure(figures.temperature);
    clf;
    contourf(sim.X, sim.Y, sim.T, 30, 'LineStyle', 'none');
    colormap(jet);
    caxis([0.8, 3.5]);
    hcb = colorbar;
    ylabel(hcb, 'Temperature (K)', 'FontSize', 10);
    title(sprintf('Temperature Field (t = %.2f s)', t), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Axial Position x (m)', 'FontSize', 11);
    ylabel('Radial Position y (m)', 'FontSize', 11);
    axis equal tight;
    
    % Add flame front marker
    hold on;
    if sim.flame_position(end) > 0
        plot([sim.flame_position(end), sim.flame_position(end)], [0, sim.Ly], ...
             'w--', 'LineWidth', 2);
        text(sim.flame_position(end), sim.Ly*0.95, 'Flame Front', ...
             'Color', 'white', 'FontSize', 10, 'HorizontalAlignment', 'center');
    end
    hold off;
    
    save_figure(figures.temperature, folders.temperature, gif_files.temperature, ...
                'temperature', frame_count, first_frame, delay_time);
    
    % 2. Velocity field with streamlines
    figure(figures.velocity);
    clf;
    contourf(sim.X, sim.Y, sqrt(sim.u.^2 + sim.v.^2), 20, 'LineStyle', 'none');
    colormap(parula);
    colorbar;
    hold on;
    % Add streamlines
    [startx, starty] = meshgrid(0:0.5:sim.Lx, linspace(0.05, sim.Ly-0.05, 5));
    streamline(sim.X, sim.Y, sim.u, sim.v, startx(:), starty(:));
    hold off;
    title(sprintf('Velocity Field and Streamlines (t = %.2f s)', t), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal tight;
    
    save_figure(figures.velocity, folders.velocity, gif_files.velocity, ...
                'velocity', frame_count, first_frame, delay_time);
    
    % 3. Fuel concentration
    figure(figures.fuel);
    clf;
    contourf(sim.X, sim.Y, sim.Y_fuel, 20, 'LineStyle', 'none');
    colormap(flipud(gray));
    caxis([0, 1]);
    colorbar;
    title(sprintf('Fuel Mass Fraction (t = %.2f s)', t), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal tight;
    
    save_figure(figures.fuel, folders.fuel, gif_files.fuel, ...
                'fuel', frame_count, first_frame, delay_time);
    
    % 4. Centerline profile
    figure(figures.centerline);
    clf;
    y_mid = round(sim.ny/2);
    yyaxis left;
    plot(sim.x, sim.T(y_mid, :), 'r-', 'LineWidth', 2);
    ylabel('Temperature (K)', 'Color', 'r');
    ylim([0.8, 3.5]);
    
    yyaxis right;
    plot(sim.x, sim.Y_fuel(y_mid, :), 'b-', 'LineWidth', 2);
    ylabel('Fuel Fraction', 'Color', 'b');
    ylim([0, 1.1]);
    
    xlabel('Axial Position x (m)');
    title(sprintf('Centerline Profile (t = %.2f s)', t), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    save_figure(figures.centerline, folders.centerline, gif_files.centerline, ...
                'centerline', frame_count, first_frame, delay_time);
    
    % 5. Flame propagation
    figure(figures.propagation);
    clf;
    if length(sim.flame_position) > 1
        times = (1:length(sim.flame_position)) * sim.dt * sim.save_interval;
        plot(times, sim.flame_position, 'b-', 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('Flame Position (m)');
        title(sprintf('Flame Front Propagation (t = %.2f s)', t), 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % Add speed information
        if length(times) > 5
            p = polyfit(times(end-min(10,length(times)-1):end), ...
                       sim.flame_position(end-min(10,length(times)-1):end), 1);
            text(0.5, 0.9, sprintf('Instantaneous Speed: %.2f m/s', p(1)), ...
                 'Units', 'normalized', 'FontSize', 10, 'Color', 'red');
        end
    else
        title(sprintf('Flame Front Propagation (t = %.2f s) - Awaiting Data', t), 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
    end
    
    save_figure(figures.propagation, folders.propagation, gif_files.propagation, ...
                'propagation', frame_count, first_frame, delay_time);
    
    % 6. Reaction rate distribution
    figure(figures.reaction);
    clf;
    omega = compute_reaction_rate(sim.T, sim.Y_fuel, sim.Da, sim.E_a);
    contourf(sim.X, sim.Y, omega, 20, 'LineStyle', 'none');
    colormap(hot);
    colorbar;
    title(sprintf('Reaction Rate Distribution (t = %.2f s)', t), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal tight;
    
    save_figure(figures.reaction, folders.reaction, gif_files.reaction, ...
                'reaction', frame_count, first_frame, delay_time);
end
function save_figure(fig, folder, gif_file, name, frame_count, first_frame, delay_time)
    % Save PNG
    png_filename = fullfile(folder, sprintf('%s_frame_%05d.png', name, frame_count));
    try
        print(fig, png_filename, '-dpng', '-r150');
    catch ME
        fprintf('Warning: Could not save PNG %s - %s\n', png_filename, ME.message);
    end
    
    % Save GIF
    try
        frame = getframe(fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        if first_frame
            imwrite(imind, cm, gif_file, 'gif', 'Loopcount', inf, 'DelayTime', delay_time);
        else
            imwrite(imind, cm, gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', delay_time);
        end
    catch ME
        fprintf('Warning: Could not save GIF frame %s - %s\n', gif_file, ME.message);
    end
end
%% ==================== Core CFD Functions ====================
function sim = advance_cfd_timestep(sim)
    % Complete CFD time step advancement
    
    % 1. Compute reaction rate
    omega = compute_reaction_rate(sim.T, sim.Y_fuel, sim.Da, sim.E_a);
    
    % 2. Update velocity field
    [dudx, dudy] = gradient(sim.u, sim.dx, sim.dy);
    [dvdx, dvdy] = gradient(sim.v, sim.dx, sim.dy);
    
    % Buoyancy term
    buoyancy = sim.beta * (sim.T - 1.0) * 9.81 * 0.001;
    
    % Viscous diffusion
    d2udx2 = del2(sim.u, sim.dx, sim.dy);
    d2vdx2 = del2(sim.v, sim.dx, sim.dy);
    
    % Velocity update
    sim.u = sim.u + sim.dt * (-sim.u .* dudx - sim.v .* dudy + sim.nu * d2udx2);
    sim.v = sim.v + sim.dt * (-sim.u .* dvdx - sim.v .* dvdy + sim.nu * d2vdx2 + buoyancy);
    
    % 3. Temperature transport equation
    [dTdx, dTdy] = gradient(sim.T, sim.dx, sim.dy);
    D2T = del2(sim.T, sim.dx, sim.dy);
    conv_T = sim.u .* dTdx + sim.v .* dTdy;
    
    sim.T = sim.T + sim.dt * (-conv_T + sim.D_T * D2T + sim.beta * omega);
    
    % 4. Fuel transport equation
    [dYdx, dYdy] = gradient(sim.Y_fuel, sim.dx, sim.dy);
    D2Y = del2(sim.Y_fuel, sim.dx, sim.dy);
    conv_Y = sim.u .* dYdx + sim.v .* dYdy;
    
    sim.Y_fuel = sim.Y_fuel + sim.dt * (-conv_Y + sim.D_Y * D2Y - omega);
    
    % 5. Apply boundary conditions
    sim = apply_boundary_conditions(sim);
    
    % 6. Physical constraints
    sim.T = max(1.0, min(4.0, sim.T));
    sim.Y_fuel = max(0.0, min(1.0, sim.Y_fuel));
end
function omega = compute_reaction_rate(T, Y, Da, E_a)
    % Enhanced Arrhenius reaction rate
    
    % Activation energy term
    activation = exp(-E_a ./ max(T, 0.1));
    
    % Temperature threshold function
    T_threshold = 1.3;
    ignition_func = 0.5 * (1 + tanh(5*(T - T_threshold)));
    
    % Reaction rate
    omega = Da * Y.^0.8 .* activation .* ignition_func;
end
function sim = apply_boundary_conditions(sim)
    % Apply boundary conditions
    
    % Left boundary (inlet)
    y_norm = (sim.y - sim.Ly/2) / (sim.Ly/2);
    sim.u(:, 1) = sim.v_base * (1 - y_norm'.^2);
    sim.v(:, 1) = 0;
    sim.T(:, 1) = 1.0;
    sim.Y_fuel(:, 1) = 1.0;
    
    % Right boundary (outlet)
    sim.u(:, end) = sim.u(:, end-1);
    sim.v(:, end) = sim.v(:, end-1);
    sim.T(:, end) = sim.T(:, end-1);
    sim.Y_fuel(:, end) = sim.Y_fuel(:, end-1);
    
    % Top and bottom walls
    sim.u([1, end], :) = 0;
    sim.v([1, end], :) = 0;
    sim.T(1, :) = sim.T(2, :);
    sim.T(end, :) = sim.T(end-1, :);
    sim.Y_fuel(1, :) = sim.Y_fuel(2, :);
    sim.Y_fuel(end, :) = sim.Y_fuel(end-1, :);
end
function generate_final_analysis(sim, output_folder)
    % Generate final analysis plots (saved independently)
    
    if isempty(sim.T_history)
        fprintf('⚠️ No data available for analysis\n');
        return;
    end
    
    fprintf('\n📊 Generating final analysis plots...\n');
    
    % Analysis plot 1: 3D temperature field
    fig1 = figure('Name', 'Final 3D Temperature Field', 'Position', [100, 100, 800, 600], 'Visible', 'off');
    surf(sim.X, sim.Y, sim.T, 'EdgeColor', 'none');
    colormap(jet);
    caxis([1, 3.5]);
    colorbar;
    title('Final Temperature Field (3D)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('Temperature (K)');
    view(45, 30);
    lighting gouraud;
    saveas(fig1, fullfile(output_folder, 'analysis_3d_temperature.png'));
    close(fig1);
    
    % Analysis plot 2: Flame propagation speed analysis
    if length(sim.flame_position) > 2
        fig2 = figure('Name', 'Flame Speed Analysis', 'Position', [200, 200, 800, 600], 'Visible', 'off');
        times = (1:length(sim.flame_position)) * sim.dt * sim.save_interval;
        flame_speed = diff(sim.flame_position) / (sim.dt * sim.save_interval);
        plot(times(2:end), flame_speed, 'b-', 'LineWidth', 2);
        hold on;
        mean_speed = mean(flame_speed(flame_speed > 0));
        plot(times(2:end), ones(size(times(2:end)))*mean_speed, 'r--', 'LineWidth', 1.5);
        xlabel('Time (s)', 'FontSize', 12);
        ylabel('Flame Speed (m/s)', 'FontSize', 12);
        title(sprintf('Flame Propagation Speed Analysis (Average: %.3f m/s)', mean_speed), ...
              'FontSize', 14, 'FontWeight', 'bold');
        legend('Instantaneous Speed', 'Average Speed', 'Location', 'best');
        grid on;
        saveas(fig2, fullfile(output_folder, 'analysis_flame_speed.png'));
        close(fig2);
    end
    
    % Analysis plot 3: Temperature evolution
    fig3 = figure('Name', 'Temperature Evolution', 'Position', [300, 300, 800, 600], 'Visible', 'off');
    n_frames = min(8, length(sim.T_history));
    indices = round(linspace(1, length(sim.T_history), n_frames));
    colors = jet(n_frames);
    hold on;
    
    for i = 1:n_frames
        idx = indices(i);
        T_centerline = sim.T_history{idx}(round(sim.ny/2), :);
        plot(sim.x, T_centerline, 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    
    xlabel('x Position (m)', 'FontSize', 12);
    ylabel('Temperature (K)', 'FontSize', 12);
    title('Temperature Profile Evolution', 'FontSize', 14, 'FontWeight', 'bold');
    colormap(jet);
    hcb = colorbar;
    ylabel(hcb, 'Time Progress');
    grid on;
    saveas(fig3, fullfile(output_folder, 'analysis_temperature_evolution.png'));
    close(fig3);
    
    % Analysis plot 4: Fuel consumption
    fig4 = figure('Name', 'Fuel Consumption', 'Position', [400, 400, 800, 600], 'Visible', 'off');
    times = (1:length(sim.max_temp)) * sim.dt * sim.save_interval;
    yyaxis left;
    plot(times, sim.max_temp, 'r-', 'LineWidth', 2);
    ylabel('Maximum Temperature (K)', 'Color', 'r', 'FontSize', 12);
    
    yyaxis right;
    total_fuel = cellfun(@(Y) sum(Y(:))*sim.dx*sim.dy, sim.Y_history);
    plot(times, total_fuel/total_fuel(1), 'b-', 'LineWidth', 2);
    ylabel('Remaining Fuel Ratio', 'Color', 'b', 'FontSize', 12);
    
    xlabel('Time (s)', 'FontSize', 12);
    title('Temperature and Fuel Consumption', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    saveas(fig4, fullfile(output_folder, 'analysis_fuel_consumption.png'));
    close(fig4);
    
    fprintf('✅ Final analysis plots saved to: %s\n', output_folder);
end