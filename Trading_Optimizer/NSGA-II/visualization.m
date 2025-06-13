% MATLAB Script for Visualizing NSGA-II Crypto Trading Strategy Results

% Clear workspace and close figures
clear;
close all;

% Set the generation to visualize
generation = 150;  % Change this to visualize different generations

% Load data files
data_file = sprintf('gen_%d_data.csv', generation);
fronts_file = sprintf('gen_%d_fronts.csv', generation);

% Check if files exist
if ~exist(data_file, 'file') || ~exist(fronts_file, 'file')
    error('Log files for generation %d not found', generation);
end

% Load data
data = readtable(data_file);
fronts = readtable(fronts_file);

% 3D Pareto Front Visualization
figure('Name', sprintf('Pareto Fronts - Generation %d', generation), 'Position', [100, 100, 1000, 800]);

% Get unique front indices
front_indices = unique(fronts.FrontIndex);
colors = jet(length(front_indices));

% Plot each front with different color
hold on;
for i = 1:length(front_indices)
    front_idx = front_indices(i);
    front_data = fronts(fronts.FrontIndex == front_idx, :);
    
    % Plot 3D scatter
    scatter3(front_data.ROI, front_data.Risk, front_data.WinRate, 50, colors(i,:), 'filled');
end

% Add labels and title
xlabel('ROI (%)');
ylabel('Risk (negative Sharpe)');
zlabel('Win Rate (%)');
title(sprintf('Pareto Fronts - Generation %d', generation));
grid on;
view(45, 30);

% Add colorbar for front index
colormap(jet);
c = colorbar;
c.Label.String = 'Front Index';
caxis([0, max(front_indices)]);

% 2D Projections
figure('Name', sprintf('2D Projections - Generation %d', generation), 'Position', [100, 100, 1200, 400]);

% ROI vs Risk
subplot(1, 3, 1);
hold on;
for i = 1:length(front_indices)
    front_idx = front_indices(i);
    front_data = fronts(fronts.FrontIndex == front_idx, :);
    scatter(front_data.ROI, front_data.Risk, 30, colors(i,:), 'filled');
end
xlabel('ROI (%)');
ylabel('Risk (negative Sharpe)');
title('ROI vs Risk');
grid on;

% ROI vs Win Rate
subplot(1, 3, 2);
hold on;
for i = 1:length(front_indices)
    front_idx = front_indices(i);
    front_data = fronts(fronts.FrontIndex == front_idx, :);
    scatter(front_data.ROI, front_data.WinRate, 30, colors(i,:), 'filled');
end
xlabel('ROI (%)');
ylabel('Win Rate (%)');
title('ROI vs Win Rate');
grid on;

% Risk vs Win Rate
subplot(1, 3, 3);
hold on;
for i = 1:length(front_indices)
    front_idx = front_indices(i);
    front_data = fronts(fronts.FrontIndex == front_idx, :);
    scatter(front_data.Risk, front_data.WinRate, 30, colors(i,:), 'filled');
end
xlabel('Risk (negative Sharpe)');
ylabel('Win Rate (%)');
title('Risk vs Win Rate');
grid on;

% Strategy Parameter Analysis
figure('Name', sprintf('Strategy Parameters - Generation %d', generation), 'Position', [100, 100, 1200, 800]);

% Sort strategies by ROI
sorted_data = sortrows(data, 'ROI', 'descend');
top_n = min(20, height(sorted_data));  % Top 20 strategies or all if less
top_strategies = sorted_data(1:top_n, :);

% Plot parameter distributions for top strategies
subplot(2, 3, 1);
bar(top_strategies.ShortMA);
xlabel('Strategy Index');
ylabel('Short MA Period');
title('Short MA Distribution');
grid on;

subplot(2, 3, 2);
bar(top_strategies.LongMA);
xlabel('Strategy Index');
ylabel('Long MA Period');
title('Long MA Distribution');
grid on;

subplot(2, 3, 3);
bar(top_strategies.BuyThresh);
xlabel('Strategy Index');
ylabel('Buy Threshold');
title('Buy Threshold Distribution');
grid on;

subplot(2, 3, 4);
bar(top_strategies.SellThresh);
xlabel('Strategy Index');
ylabel('Sell Threshold');
title('Sell Threshold Distribution');
grid on;

subplot(2, 3, 5);
bar(top_strategies.StopLoss);
xlabel('Strategy Index');
ylabel('Stop Loss');
title('Stop Loss Distribution');
grid on;

subplot(2, 3, 6);
bar(top_strategies.TakeProfit);
xlabel('Strategy Index');
ylabel('Take Profit');
title('Take Profit Distribution');
grid on;

% Parameter correlation with ROI
figure('Name', sprintf('Parameter Correlation - Generation %d', generation), 'Position', [100, 100, 1200, 800]);

% Short MA vs ROI
subplot(2, 3, 1);
scatter(data.ShortMA, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Short MA Period');
ylabel('ROI (%)');
title('Short MA vs ROI');
grid on;
colorbar;

% Long MA vs ROI
subplot(2, 3, 2);
scatter(data.LongMA, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Long MA Period');
ylabel('ROI (%)');
title('Long MA vs ROI');
grid on;
colorbar;

% Buy Threshold vs ROI
subplot(2, 3, 3);
scatter(data.BuyThresh, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Buy Threshold');
ylabel('ROI (%)');
title('Buy Threshold vs ROI');
grid on;
colorbar;

% Sell Threshold vs ROI
subplot(2, 3, 4);
scatter(data.SellThresh, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Sell Threshold');
ylabel('ROI (%)');
title('Sell Threshold vs ROI');
grid on;
colorbar;

% Stop Loss vs ROI
subplot(2, 3, 5);
scatter(data.StopLoss, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Stop Loss');
ylabel('ROI (%)');
title('Stop Loss vs ROI');
grid on;
colorbar;

% Take Profit vs ROI
subplot(2, 3, 6);
scatter(data.TakeProfit, data.ROI, 30, data.FrontRank, 'filled');
xlabel('Take Profit');
ylabel('ROI (%)');
title('Take Profit vs ROI');
grid on;
colorbar;

% Display top 5 strategies
fprintf('Top 5 Strategies by ROI:\n');
disp(top_strategies(1:min(5, top_n), :));
