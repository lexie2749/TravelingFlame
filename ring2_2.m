%% PINN-CFD混合方法：双火焰碰撞仿真
% 使用物理信息神经网络(PINN)和CFD模拟两个火焰的传播与碰撞
% 展示火焰相向运动、碰撞湮灭的完整物理过程
clear; close all; clc;

%% 1. 系统初始化
% =========================================================================
fprintf('╔══════════════════════════════════════════╗\n');
fprintf('║   PINN-CFD 双火焰碰撞仿真系统 v2.0      ║\n');
fprintf('╚══════════════════════════════════════════╝\n\n');

% 物理域参数
R_inner = 0.03;     % 内半径 3cm
R_outer = 0.06;     % 外半径 6cm  
R_mean = (R_inner + R_outer) / 2;

% 燃烧参数
params = struct();
params.Le = 0.6;           % Lewis数 (<1 促进不稳定性)
params.beta = 12.0;        % Zeldovich数
params.alpha = 0.88;       % 热释放参数
params.D_th = 4e-5;        % 热扩散系数 (m²/s)
params.D_mass = params.D_th / params.Le;  % 质量扩散系数

% 计算网格
Nr = 40;            % 径向分辨率
Ntheta = 200;       % 角向分辨率
Nt = 2000;          % 时间步数
dt = 0.0015;        % 时间步长
t_final = Nt * dt;

% PINN配置
pinn_config = struct();
pinn_config.hiddenSize = 50;        % 隐层大小
pinn_config.numHiddenLayers = 4;    % 隐层数量
pinn_config.learningRate = 0.003;   % 学习率
pinn_config.numEpochs = 300;        % 初始训练轮数
pinn_config.updateInterval = 100;   % PINN更新间隔

% 输出配置
outputDir = 'PINN_CFD_collision';
if exist(outputDir, 'dir')
    rmdir(outputDir, 's');
end
mkdir(outputDir);

%% 2. 网格生成
% =========================================================================
fprintf('【1/6】生成计算网格...\n');

r = linspace(R_inner, R_outer, Nr);
theta = linspace(0, 2*pi, Ntheta);
[Theta, R] = meshgrid(theta, r);
X = R .* cos(Theta);
Y = R .* sin(Theta);

dr = r(2) - r(1);
dtheta = theta(2) - theta(1);

% 场变量初始化
T = zeros(Nr, Ntheta, Nt);          % 温度场
Y_fuel = ones(Nr, Ntheta, Nt);      % 燃料浓度
burned = zeros(Nr, Ntheta, Nt);     % 燃烧历史
reaction_rate = zeros(Nr, Ntheta, Nt); % 反应速率

%% 3. 双火焰初始条件
% =========================================================================
fprintf('【2/6】设置双火焰初始条件...\n');

% 两个对称的初始火焰位置
theta_flame1 = pi/6;      % 第一个火焰 (30°)
theta_flame2 = pi + pi/6; % 第二个火焰 (210°)
flame_width = 0.25;       % 初始火焰宽度

for i = 1:Nr
    for j = 1:Ntheta
        % 第一个火焰
        dist1 = min(abs(theta(j) - theta_flame1), 2*pi - abs(theta(j) - theta_flame1));
        if dist1 < flame_width
            intensity1 = exp(-2*(dist1/flame_width)^2);
            T(i, j, 1) = max(T(i, j, 1), 0.9*intensity1);
            Y_fuel(i, j, 1) = min(Y_fuel(i, j, 1), 1 - 0.8*intensity1);
            if intensity1 > 0.5
                burned(i, j, 1) = intensity1;
            end
        end
        
        % 第二个火焰
        dist2 = min(abs(theta(j) - theta_flame2), 2*pi - abs(theta(j) - theta_flame2));
        if dist2 < flame_width
            intensity2 = exp(-2*(dist2/flame_width)^2);
            T(i, j, 1) = max(T(i, j, 1), 0.9*intensity2);
            Y_fuel(i, j, 1) = min(Y_fuel(i, j, 1), 1 - 0.8*intensity2);
            if intensity2 > 0.5
                burned(i, j, 1) = max(burned(i, j, 1), intensity2);
            end
        end
        
        % 背景温度
        if T(i, j, 1) < 0.05
            T(i, j, 1) = 0.05;
        end
    end
end

%% 4. 构建PINN网络
% =========================================================================
fprintf('【3/6】构建PINN神经网络...\n');

% 网络架构
layers = [
    featureInputLayer(3, 'Name', 'input')  % [r, theta, t]
];

% 隐藏层
for k = 1:pinn_config.numHiddenLayers
    layers = [layers
        fullyConnectedLayer(pinn_config.hiddenSize, 'Name', ['fc' num2str(k)])
        tanhLayer('Name', ['tanh' num2str(k)])
        dropoutLayer(0.05, 'Name', ['dropout' num2str(k)])  % 防止过拟合
    ];
end

% 输出层
layers = [layers
    fullyConnectedLayer(2, 'Name', 'output')
    sigmoidLayer('Name', 'sigmoid')
];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
   
%% 5. 生成PINN训练数据（双火焰碰撞）
% =========================================================================
fprintf('【4/6】生成双火焰碰撞训练数据...\n');

numTrainPoints = 2000;
r_train = R_inner + (R_outer - R_inner) * rand(numTrainPoints, 1);
theta_train = 2*pi * rand(numTrainPoints, 1);
t_train = t_final * rand(numTrainPoints, 1);

% 生成双火焰碰撞的目标数据
T_target = zeros(numTrainPoints, 1);
Y_target_fuel = ones(numTrainPoints, 1);

flame_speed = 3.0;  % 火焰传播速度 (rad/s)
collision_time = pi / (2 * flame_speed);  % 预计碰撞时间

for k = 1:numTrainPoints
    t = t_train(k);
    theta_k = theta_train(k);
    
    if t < collision_time
        % 碰撞前：两个火焰相向传播
        % 火焰1: 从30°顺时针
        flame1_pos = theta_flame1 + flame_speed * t;
        dist1 = min(abs(theta_k - flame1_pos), 2*pi - abs(theta_k - flame1_pos));
        
        % 火焰2: 从210°逆时针
        flame2_pos = theta_flame2 - flame_speed * t;
        dist2 = min(abs(theta_k - flame2_pos), 2*pi - abs(theta_k - flame2_pos));
        
        % 火焰前锋温度分布
        flame_thickness = 0.3;
        if dist1 < flame_thickness
            T1 = 0.5 + 0.5*cos(pi*dist1/flame_thickness);
            Y1 = 0.5 - 0.4*cos(pi*dist1/flame_thickness);
        else
            T1 = 0.05;
            Y1 = 1.0;
        end
        
        if dist2 < flame_thickness
            T2 = 0.5 + 0.5*cos(pi*dist2/flame_thickness);
            Y2 = 0.5 - 0.4*cos(pi*dist2/flame_thickness);
        else
            T2 = 0.05;
            Y2 = 1.0;
        end
        
        % 合并两个火焰
        T_target(k) = max(T1, T2);
        Y_target_fuel(k) = min(Y1, Y2);
        
        % 已燃区域
        if (dist1 > flame_thickness && mod(theta_k - theta_flame1 + 2*pi, 2*pi) < flame_speed * t) || ...
           (dist2 > flame_thickness && mod(theta_flame2 - theta_k + 2*pi, 2*pi) < flame_speed * t)
            T_target(k) = 0.2 * exp(-t/0.5);  % 冷却
            Y_target_fuel(k) = 0.05;  % 燃料耗尽
        end
    else
        % 碰撞后：快速熄灭
        decay_rate = 5.0;
        T_target(k) = 0.3 * exp(-decay_rate * (t - collision_time));
        Y_target_fuel(k) = 0.05;
    end
    
    % 确保物理范围
    T_target(k) = max(0.05, min(1, T_target(k)));
    Y_target_fuel(k) = max(0, min(1, Y_target_fuel(k)));
end

%% 6. 训练PINN
% =========================================================================
fprintf('【5/6】训练PINN网络学习碰撞动力学...\n');

X_train = dlarray([r_train, theta_train, t_train]', 'CB');
Y_target = dlarray([T_target'; Y_target_fuel'], 'CB');

averageGrad = [];
averageSqGrad = [];
iteration = 0;
lossHistory = zeros(pinn_config.numEpochs, 1);

tic;
for epoch = 1:pinn_config.numEpochs
    iteration = iteration + 1;
    
    % 计算物理约束损失
    [loss, gradients] = dlfeval(@computeCollisionLoss, net, X_train, Y_target, ...
                                params, R_mean);
    
    % Adam优化
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
        averageGrad, averageSqGrad, iteration, pinn_config.learningRate);
    
    lossHistory(epoch) = extractdata(loss);
    
    % 进度显示
    if mod(epoch, 50) == 0
        fprintf('  训练进度: %d/%d | 损失: %.6f | 用时: %.1fs\n', ...
                epoch, pinn_config.numEpochs, lossHistory(epoch), toc);
    end
end
fprintf('  PINN训练完成！最终损失: %.6f\n', lossHistory(end));

%% 7. PINN-CFD混合仿真主循环
% =========================================================================
fprintf('【6/6】开始双火焰碰撞仿真...\n\n');

% 创建可视化窗口
fig = figure('Position', [50, 50, 1200, 900]);
set(fig, 'Color', 'white');

gifFile = fullfile(outputDir, 'dual_flame_collision.gif');
frameCount = 0;
saveInterval = 8;

% 性能监控
pinn_usage = zeros(Nt, 1);
flame_positions = zeros(2, Nt);  % 追踪两个火焰位置
collision_detected = false;
collision_time_actual = 0;

% 主仿真循环
for n = 2:Nt
    current_time = (n-1) * dt;
    
    % 检测火焰前锋（基于温度梯度）
    [dT_dr, dT_dtheta] = gradient(T(:, :, n-1), dr, dtheta);
    gradT_mag = sqrt(dT_dr.^2 + dT_dtheta.^2);
    threshold = prctile(gradT_mag(:), 80);
    flame_front = gradT_mag > threshold & T(:, :, n-1) > 0.2;
    
    pinn_usage(n) = sum(flame_front(:)) / (Nr * Ntheta);
    
    % PINN-CFD混合求解
    T_new = zeros(Nr, Ntheta);
    Y_new = zeros(Nr, Ntheta);
    burned_new = burned(:, :, n-1);
    
    for i = 2:Nr-1
        for j = 1:Ntheta
            if flame_front(i, j)
                % ===== PINN求解火焰前锋 =====
                X_point = dlarray([r(i); theta(j); current_time], 'CB');
                Y_pred = predict(net, X_point);
                Y_pred = extractdata(Y_pred);
                
                % 软混合策略
                blend_factor = min(1, 2*gradT_mag(i, j)/max(gradT_mag(:)));
                T_new(i, j) = blend_factor * Y_pred(1) + (1-blend_factor) * T(i, j, n-1);
                Y_new(i, j) = blend_factor * Y_pred(2) + (1-blend_factor) * Y_fuel(i, j, n-1);
                
            else
                % ===== CFD求解稳定区域 =====
                j_prev = mod(j-2, Ntheta) + 1;
                j_next = mod(j, Ntheta) + 1;
                
                % Laplacian (极坐标)
                laplacian_T = (T(i+1, j, n-1) - 2*T(i, j, n-1) + T(i-1, j, n-1))/dr^2 + ...
                            (1/r(i))*(T(i+1, j, n-1) - T(i-1, j, n-1))/(2*dr) + ...
                            (1/r(i)^2)*(T(i, j_next, n-1) - 2*T(i, j, n-1) + T(i, j_prev, n-1))/dtheta^2;
                
                laplacian_Y = (Y_fuel(i+1, j, n-1) - 2*Y_fuel(i, j, n-1) + Y_fuel(i-1, j, n-1))/dr^2 + ...
                            (1/r(i))*(Y_fuel(i+1, j, n-1) - Y_fuel(i-1, j, n-1))/(2*dr) + ...
                            (1/r(i)^2)*(Y_fuel(i, j_next, n-1) - 2*Y_fuel(i, j, n-1) + Y_fuel(i, j_prev, n-1))/dtheta^2;
                
                % 反应速率（考虑燃烧历史）
                T_local = T(i, j, n-1);
                Y_local = Y_fuel(i, j, n-1);
                
                if T_local > 0.05 && Y_local > 0.05 && burned(i, j, n-1) < 0.8
                    omega = (params.beta^2/(2*params.Le)) * Y_local * ...
                           exp(-params.beta*(1-T_local)/(1 - params.alpha*(1-T_local)));
                    omega = min(omega, 1000);
                else
                    omega = 0;
                end
                
                reaction_rate(i, j, n) = omega;
                
                % 时间推进
                T_new(i, j) = T(i, j, n-1) + dt * (params.D_th * laplacian_T + omega);
                Y_new(i, j) = Y_fuel(i, j, n-1) + dt * (params.D_mass * laplacian_Y - omega);
            end
            
            % 更新燃烧历史
            if T_new(i, j) > 0.6
                burned_new(i, j) = min(1, burned_new(i, j) + dt*5);
            end
            
            % 燃料耗尽区域的冷却
            if Y_new(i, j) < 0.1 && T_new(i, j) > 0.2
                T_new(i, j) = T_new(i, j) * 0.95;  % 指数衰减
            end
            
            % 限制物理范围
            T_new(i, j) = max(0, min(1, T_new(i, j)));
            Y_new(i, j) = max(0, min(1, Y_new(i, j)));
        end
    end
    
    % 更新场变量
    T(:, :, n) = T_new;
    Y_fuel(:, :, n) = Y_new;
    burned(:, :, n) = burned_new;
    
    % 边界条件
    T(1, :, n) = T(2, :, n);
    T(Nr, :, n) = T(Nr-1, :, n);
    Y_fuel(1, :, n) = Y_fuel(2, :, n);
    Y_fuel(Nr, :, n) = Y_fuel(Nr-1, :, n);
    
    % 周期边界
    T(:, 1, n) = T(:, Ntheta, n);
    Y_fuel(:, 1, n) = Y_fuel(:, Ntheta, n);
    
    % 检测火焰位置和碰撞
    [T_max_r, ~] = max(T(:, :, n), [], 1);
    [peaks, locs] = findpeaks(T_max_r, 'MinPeakHeight', 0.4);
    
    if length(locs) >= 2
        flame_positions(1, n) = theta(locs(1));
        flame_positions(2, n) = theta(locs(2));
        
        % 检测碰撞
        flame_distance = min(abs(diff(locs)), Ntheta - max(locs) + min(locs));
        if flame_distance < 10 && ~collision_detected
            collision_detected = true;
            collision_time_actual = current_time;
            fprintf('  ⚡ 火焰碰撞检测！时间: %.2f s\n', current_time);
        end
    end
    
    % 可视化
    if mod(n-1, saveInterval) == 0
        clf;
        
        % 主图：温度场
        subplot(2, 3, [1, 2, 4, 5]);
        pcolor(X, Y, T(:, :, n));
        shading interp;
        colormap(jet);
        caxis([0 1]);
        axis equal;
        axis([-0.08 0.08 -0.08 0.08]);
        
        % 标题（包含碰撞状态）
        if collision_detected
            title_str = sprintf('🔥 火焰碰撞！| 时间: %.2f s', current_time);
        else
            title_str = sprintf('双火焰传播 | 时间: %.2f s', current_time);
        end
        title(title_str, 'FontSize', 16, 'FontWeight', 'bold');
        
        % 添加环形边界
        hold on;
        theta_circle = linspace(0, 2*pi, 200);
        plot(R_inner*cos(theta_circle), R_inner*sin(theta_circle), 'w-', 'LineWidth', 2);
        plot(R_outer*cos(theta_circle), R_outer*sin(theta_circle), 'w-', 'LineWidth', 2);
        
        % 标记PINN区域
        contour(X, Y, double(flame_front), [0.5 0.5], 'g-', 'LineWidth', 1.5);
        
        % 标记火焰峰值位置
        if length(locs) >= 2
            for p = 1:min(2, length(locs))
                r_peak = R_mean;
                x_peak = r_peak * cos(theta(locs(p)));
                y_peak = r_peak * sin(theta(locs(p)));
                plot(x_peak, y_peak, 'wo', 'MarkerSize', 8, 'LineWidth', 2);
            end
        end
        hold off;
        
        colorbar('Location', 'eastoutside');
        
        % 燃料浓度
        subplot(2, 3, 3);
        pcolor(X, Y, Y_fuel(:, :, n));
        shading interp;
        colormap(gca, flipud(hot));
        caxis([0 1]);
        axis equal;
        axis([-0.08 0.08 -0.08 0.08]);
        title('燃料浓度', 'FontSize', 12);
        colorbar;
        
        % 反应速率
        subplot(2, 3, 6);
        pcolor(X, Y, reaction_rate(:, :, n));
        shading interp;
        colormap(gca, 'hot');
        axis equal;
        axis([-0.08 0.08 -0.08 0.08]);
        title('反应速率', 'FontSize', 12);
        colorbar;
        
        drawnow;
        
        % 保存帧
        frameCount = frameCount + 1;
        frame = getframe(fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        if frameCount == 1
            imwrite(imind, cm, gifFile, 'gif', 'Loopcount', inf, 'DelayTime', 0.04);
        else
            imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.04);
        end
        
        % 保存关键帧PNG
        if collision_detected && frameCount < 10
            frameName = fullfile(outputDir, sprintf('collision_frame_%02d.png', frameCount));
            print(fig, frameName, '-dpng', '-r150');
        end
    end
    
    % 进度报告
    if mod(n, 200) == 0
        fprintf('  仿真进度: %.1f%% | PINN使用: %.1f%% | ', ...
                100*n/Nt, pinn_usage(n)*100);
        if collision_detected
            fprintf('碰撞后时间: %.2fs\n', current_time - collision_time_actual);
        else
            fprintf('等待碰撞...\n');
        end
    end
    
    % 自适应PINN更新
    if mod(n, pinn_config.updateInterval) == 0 && n < Nt-100
        % 收集最近数据进行在线学习
        [ii, jj] = find(flame_front);
        if length(ii) > 50
            idx = randperm(length(ii), min(200, length(ii)));
            X_update = [];
            Y_update = [];
            for s = 1:length(idx)
                X_update = [X_update, [r(ii(idx(s))); theta(jj(idx(s))); current_time]];
                Y_update = [Y_update, [T(ii(idx(s)), jj(idx(s)), n); ...
                                      Y_fuel(ii(idx(s)), jj(idx(s)), n)]];
            end
            
            X_update = dlarray(X_update, 'CB');
            Y_update = dlarray(Y_update, 'CB');
            
            % 快速微调
            for update_epoch = 1:10
                iteration = iteration + 1;
                [loss, gradients] = dlfeval(@computeCollisionLoss, net, X_update, Y_update, ...
                                          params, R_mean);
                [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
                    averageGrad, averageSqGrad, iteration, pinn_config.learningRate*0.1);
            end
        end
    end
end

%% 8. 后处理与分析
% =========================================================================
fprintf('\n═══════════════════════════════════════\n');
fprintf('仿真完成！生成分析报告...\n');

% 综合分析图
figure('Position', [50, 50, 1600, 900]);

% 时空演化图
subplot(2, 3, 1);
T_spacetime = squeeze(mean(T, 1))';
imagesc((1:Nt)*dt, theta*180/pi, T_spacetime');
colormap(jet);
colorbar;
xlabel('时间 (s)');
ylabel('角度 (°)');
title('温度时空演化', 'FontSize', 12, 'FontWeight', 'bold');
hold on;
if collision_detected
    plot([collision_time_actual, collision_time_actual], [0, 360], 'w--', 'LineWidth', 2);
    text(collision_time_actual, 180, '  碰撞', 'Color', 'white', 'FontSize', 10);
end
hold off;

% 火焰轨迹
subplot(2, 3, 2);
valid_idx = find(flame_positions(1, :) > 0);
if ~isempty(valid_idx)
    plot((valid_idx-1)*dt, flame_positions(1, valid_idx)*180/pi, 'r-', 'LineWidth', 2);
    hold on;
    plot((valid_idx-1)*dt, flame_positions(2, valid_idx)*180/pi, 'b-', 'LineWidth', 2);
    if collision_detected
        plot([collision_time_actual, collision_time_actual], [0, 360], 'k--', 'LineWidth', 2);
    end
    hold off;
    xlabel('时间 (s)');
    ylabel('火焰位置 (°)');
    title('双火焰轨迹', 'FontSize', 12, 'FontWeight', 'bold');
    legend('火焰1', '火焰2', '碰撞时刻', 'Location', 'best');
    grid on;
end

% PINN损失曲线
subplot(2, 3, 3);
semilogy(lossHistory, 'g-', 'LineWidth', 2);
xlabel('训练轮数');
ylabel('损失函数');
title('PINN训练收敛', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 最大温度演化
subplot(2, 3, 4);
T_max = squeeze(max(max(T, [], 1), [], 2));
plot((1:Nt)*dt, T_max, 'm-', 'LineWidth', 2);
xlabel('时间 (s)');
ylabel('最大温度');
title('峰值温度演化', 'FontSize', 12, 'FontWeight', 'bold');
if collision_detected
    hold on;
    plot([collision_time_actual, collision_time_actual], [0, 1], 'k--', 'LineWidth', 2);
    hold off;
end
grid on;

% 燃料消耗
subplot(2, 3, 5);
total_fuel = squeeze(mean(mean(Y_fuel, 1), 2));
plot((1:Nt)*dt, total_fuel, 'b-', 'LineWidth', 2);
xlabel('时间 (s)');
ylabel('平均燃料浓度');
title('燃料消耗过程', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% PINN使用率
subplot(2, 3, 6);
area((1:Nt)*dt, pinn_usage*100, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('时间 (s)');
ylabel('PINN覆盖率 (%)');
title('PINN-CFD混合比例', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, 40]);
grid on;

sgtitle('PINN-CFD双火焰碰撞分析', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'analysis_report.png'));

% 输出统计
fprintf('═══════════════════════════════════════\n');
fprintf('📊 仿真统计:\n');
fprintf('  - 总仿真时间: %.2f s\n', t_final);
fprintf('  - 生成帧数: %d\n', frameCount);
if collision_detected
    fprintf('  - 碰撞时间: %.3f s\n', collision_time_actual);
    fprintf('  - 理论预测: %.3f s\n', pi/(2*flame_speed));
    fprintf('  - 预测误差: %.1f%%\n', abs(collision_time_actual - pi/(2*flame_speed))/(pi/(2*flame_speed))*100);
end
fprintf('  - 平均PINN使用率: %.1f%%\n', mean(pinn_usage(pinn_usage>0))*100);
fprintf('  - 最高温度: %.3f\n', max(T_max));
fprintf('\n');
fprintf('📁 输出文件:\n');
fprintf('  - GIF动画: %s\n', gifFile);
fprintf('  - 分析报告: %s\n', fullfile(outputDir, 'analysis_report.png'));
fprintf('═══════════════════════════════════════\n');

%% 函数定义
% =========================================================================

function [loss, gradients] = computeCollisionLoss(net, X, Y_target, params, R_mean)
    % 物理信息损失函数（针对火焰碰撞）
    
    % 前向传播
    Y_pred = forward(net, X);
    
    % 提取预测
    T_pred = Y_pred(1, :);
    Y_fuel_pred = Y_pred(2, :);
    
    % 1. 数据损失
    L_data = mean((Y_pred - Y_target).^2, 'all');
    
    % 2. 计算梯度
    T_grads = dlgradient(sum(T_pred, 'all'), X, 'EnableHigherDerivatives', true);
    Y_grads = dlgradient(sum(Y_fuel_pred, 'all'), X, 'EnableHigherDerivatives', true);
    
    % 提取偏导数
    T_r = T_grads(1, :);
    T_theta = T_grads(2, :);
    T_t = T_grads(3, :);
    
    Y_r = Y_grads(1, :);
    Y_theta = Y_grads(2, :);
    Y_t = Y_grads(3, :);
    
    % 3. 近似Laplacian
    laplacian_T = T_r.^2 + (1/R_mean^2) * T_theta.^2;
    laplacian_Y = Y_r.^2 + (1/R_mean^2) * Y_theta.^2;
    
    % 4. 反应项
    denominator = 1 - params.alpha*(1-T_pred) + 0.01;  % 加小量避免除零
    omega = (params.beta^2/(2*params.Le)) * Y_fuel_pred .* ...
            exp(-params.beta*(1-T_pred)./denominator);
    omega = min(omega, 100);  % 限制最大值
    
    % 5. PDE残差
    residual_T = T_t - params.D_th*laplacian_T - omega;
    residual_Y = Y_t - params.D_mass*laplacian_Y + omega;
    
    L_physics = mean(residual_T.^2) + mean(residual_Y.^2);
    
    % 6. 守恒约束
    L_conservation = mean((T_pred + (1-params.alpha)*Y_fuel_pred - 1).^2);
    
    % 7. 边界约束
    L_bounds = mean(max(0, T_pred - 1).^2) + mean(max(0, -T_pred).^2) + ...
               mean(max(0, Y_fuel_pred - 1).^2) + mean(max(0, -Y_fuel_pred).^2);
    
    % 8. 平滑性约束（减少振荡）
    L_smooth = mean((T_theta).^2) + mean((Y_theta).^2);
    
    % 总损失（自适应权重）
    w_physics = 0.01;
    w_conservation = 0.05;
    w_bounds = 0.01;
    w_smooth = 0.001;
    
    loss = L_data + w_physics*L_physics + w_conservation*L_conservation + ...
           w_bounds*L_bounds + w_smooth*L_smooth;
    
    % 计算梯度
    gradients = dlgradient(loss, net.Learnables);
end