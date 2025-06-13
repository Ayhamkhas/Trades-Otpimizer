%% NSGA-II Convergence Analysis Script
% This script analyzes the convergence time across generations for different parameter sets
% It generates graphs showing how metrics evolve over generations

clear all;
close all;
clc;

%% Configuration
% Base directory containing all parameter sets
base_dir = '.'; % Change this to the directory containing your param_set folders

% Number of parameter sets to analyze
num_param_sets = 5;

% Metrics to track across generations
metrics = {'ROI', 'Risk', 'WinRate', 'Hypervolume'};

% Reference point for hypervolume calculation (for minimization objectives)
% Adjust these values based on your objectives and their ranges
ref_point = [-100, 0, 0]; % [ROI, Risk, WinRate] - assuming ROI is maximized, Risk minimized, WinRate maximized

% Colors for different parameter sets
colors = {'b', 'r', 'g', 'm', 'c'};
markers = {'o', 's', 'd', '^', 'v'};

%% Initialize data structures
convergence_data = struct();
for m = 1:length(metrics)
    convergence_data.(metrics{m}) = cell(num_param_sets, 1);
end

% Track execution time for each parameter set
execution_times = zeros(num_param_sets, 1);

%% Process each parameter set
for param_set = 1:num_param_sets
    param_set_dir = fullfile(base_dir, sprintf('param_set_%d', param_set));
    
    % Skip if directory doesn't exist
    if ~exist(param_set_dir, 'dir')
        warning('Parameter set directory not found: %s', param_set_dir);
        continue;
    end
    
    % Get all run directories
    run_dirs = dir(fullfile(param_set_dir, 'run_*'));
    run_dirs = run_dirs([run_dirs.isdir]);
    
    % Initialize arrays to store metrics for this parameter set
    num_runs = length(run_dirs);
    fprintf('Processing parameter set %d with %d runs...\n', param_set, num_runs);
    
    % Track total execution time for this parameter set
    total_exec_time = 0;
    
    % Process each run
    for run_idx = 1:min(num_runs, 5) % Limit to first 5 runs for clarity in graphs
        run_dir = fullfile(param_set_dir, run_dirs(run_idx).name);
        
        % Find all generation data files
        gen_files = dir(fullfile(run_dir, 'gen_*_data.csv'));
        
        if isempty(gen_files)
            warning('No generation files found in run directory: %s', run_dir);
            continue;
        end
        
        % Extract generation numbers and sort them
        gen_nums = zeros(length(gen_files), 1);
        for i = 1:length(gen_files)
            % Extract generation number from filename
            filename = gen_files(i).name;
            gen_str = regexp(filename, 'gen_(\d+)_data.csv', 'tokens');
            if ~isempty(gen_str)
                gen_nums(i) = str2double(gen_str{1}{1});
            end
        end
        
        % Sort generation numbers
        [gen_nums, sort_idx] = sort(gen_nums);
        gen_files = gen_files(sort_idx);
        
        % Initialize arrays to store metrics across generations for this run
        gen_metrics = struct();
        for m = 1:length(metrics)
            gen_metrics.(metrics{m}) = zeros(length(gen_nums), 1);
        end
        
        % Also track execution time per generation
        gen_times = zeros(length(gen_nums), 1);
        
        % Process each generation
        for gen_idx = 1:length(gen_nums)
            gen_num = gen_nums(gen_idx);
            
            % Load the data for this generation
            data_file = fullfile(run_dir, sprintf('gen_%d_data.csv', gen_num));
            fronts_file = fullfile(run_dir, sprintf('gen_%d_fronts.csv', gen_num));
            
            try
                % Read data file
                data = readtable(data_file);
                
                % Read fronts file for hypervolume calculation
                fronts = readtable(fronts_file);
                
                % Extract best individual (rank 0, highest ROI)
                rank0_individuals = data(data.FrontRank == 0, :);
                [~, best_roi_idx] = max(rank0_individuals.ROI);
                best_individual = rank0_individuals(best_roi_idx, :);
                
                % Calculate hypervolume for the first front
                first_front = fronts(fronts.FrontIndex == 0, :);
                if ~isempty(first_front)
                    % Extract objectives (adjust column names if needed)
                    objectives = [first_front.ROI, first_front.Risk, first_front.WinRate];
                    
                    % Calculate hypervolume
                    hv = calculate_hypervolume(objectives, ref_point);
                    
                    % Store hypervolume
                    gen_metrics.Hypervolume(gen_idx) = hv;
                else
                    gen_metrics.Hypervolume(gen_idx) = 0;
                end
                
                % Store other metrics for this generation
                for m = 1:length(metrics)-1  % Skip Hypervolume as it's already handled
                    metric_name = metrics{m};
                    gen_metrics.(metric_name)(gen_idx) = best_individual.(metric_name);
                end
                
                % Calculate execution time (based on generation number and assuming linear time)
                % This is a proxy since we don't have actual timing data
                gen_times(gen_idx) = gen_num * 0.5; % Assuming 0.5 seconds per generation as an example
                
            catch e
                warning('Error processing generation %d in run %s: %s', gen_num, run_dirs(run_idx).name, e.message);
            end
        end
        
        % Store the metrics for this run
        for m = 1:length(metrics)
            metric_name = metrics{m};
            if ~isfield(convergence_data.(metric_name){param_set}, 'generations')
                convergence_data.(metric_name){param_set}.generations = gen_nums;
                convergence_data.(metric_name){param_set}.values = cell(num_runs, 1);
                convergence_data.(metric_name){param_set}.mean_values = zeros(length(gen_nums), 1);
                convergence_data.(metric_name){param_set}.std_values = zeros(length(gen_nums), 1);
            end
            convergence_data.(metric_name){param_set}.values{run_idx} = gen_metrics.(metric_name);
        end
        
        % Add to total execution time
        total_exec_time = total_exec_time + gen_times(end);
    end
    
    % Calculate average execution time for this parameter set
    if num_runs > 0
        execution_times(param_set) = total_exec_time / min(num_runs, 5);
    end
    
    % Calculate mean and std across runs for each generation
    for m = 1:length(metrics)
        metric_name = metrics{m};
        if isfield(convergence_data.(metric_name){param_set}, 'generations')
            num_gens = length(convergence_data.(metric_name){param_set}.generations);
            
            % Initialize arrays for mean and std
            mean_values = zeros(num_gens, 1);
            std_values = zeros(num_gens, 1);
            
            % Calculate mean and std for each generation
            for gen_idx = 1:num_gens
                gen_values = zeros(num_runs, 1);
                for run_idx = 1:min(num_runs, 5)
                    if length(convergence_data.(metric_name){param_set}.values) >= run_idx && ...
                       ~isempty(convergence_data.(metric_name){param_set}.values{run_idx}) && ...
                       length(convergence_data.(metric_name){param_set}.values{run_idx}) >= gen_idx
                        gen_values(run_idx) = convergence_data.(metric_name){param_set}.values{run_idx}(gen_idx);
                    end
                end
                
                % Remove zeros (failed runs)
                gen_values = gen_values(gen_values ~= 0);
                
                if ~isempty(gen_values)
                    mean_values(gen_idx) = mean(gen_values);
                    std_values(gen_idx) = std(gen_values);
                end
            end
            
            % Store mean and std
            convergence_data.(metric_name){param_set}.mean_values = mean_values;
            convergence_data.(metric_name){param_set}.std_values = std_values;
        end
    end
end

%% Generate Convergence Graphs
% Create figures for each metric showing convergence across generations
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    % Create figure
    figure('Name', sprintf('%s Convergence', metric_name), 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Plot convergence for each parameter set
    legend_entries = {};
    for param_set = 1:num_param_sets
        if isfield(convergence_data, metric_name) && ...
           length(convergence_data.(metric_name)) >= param_set && ...
           ~isempty(convergence_data.(metric_name){param_set}) && ...
           isfield(convergence_data.(metric_name){param_set}, 'generations')
            
            generations = convergence_data.(metric_name){param_set}.generations;
            mean_values = convergence_data.(metric_name){param_set}.mean_values;
            std_values = convergence_data.(metric_name){param_set}.std_values;
            
            % Plot mean with error bars
            errorbar(generations, mean_values, std_values, ['-' markers{mod(param_set-1, length(markers))+1}], ...
                'Color', colors{mod(param_set-1, length(colors))+1}, 'LineWidth', 2, 'MarkerSize', 8, ...
                'MarkerFaceColor', colors{mod(param_set-1, length(colors))+1});
            
            legend_entries{end+1} = sprintf('Param Set %d', param_set);
        end
    end
    
    % Add labels and legend
    xlabel('Generation');
    ylabel(metric_name);
    title(sprintf('%s Convergence Across Generations', metric_name));
    legend(legend_entries, 'Location', 'best');
    grid on;
    
    % Save figure
    saveas(gcf, sprintf('%s_convergence.png', metric_name));
    saveas(gcf, sprintf('%s_convergence.fig', metric_name));
end

%% Generate Execution Time Comparison
figure('Name', 'Execution Time Comparison', 'Position', [100, 100, 800, 600]);
bar(1:num_param_sets, execution_times);
xlabel('Parameter Set');
ylabel('Execution Time (seconds)');
title('Average Execution Time per Parameter Set');
set(gca, 'XTick', 1:num_param_sets);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Set %d', x), 1:num_param_sets, 'UniformOutput', false));
grid on;

% Add value labels on top of bars
for i = 1:num_param_sets
    if execution_times(i) > 0
        text(i, execution_times(i) + max(execution_times)*0.02, sprintf('%.1f', execution_times(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end

% Save figure
saveas(gcf, 'execution_time_comparison.png');
saveas(gcf, 'execution_time_comparison.fig');

%% Generate Generation-wise Convergence Time Graph
figure('Name', 'Generation-wise Convergence Time', 'Position', [100, 100, 1000, 600]);
hold on;

% Plot convergence time for each parameter set
legend_entries = {};
for param_set = 1:num_param_sets
    if isfield(convergence_data, metrics{1}) && ...
       length(convergence_data.(metrics{1})) >= param_set && ...
       ~isempty(convergence_data.(metrics{1}){param_set}) && ...
       isfield(convergence_data.(metrics{1}){param_set}, 'generations')
        
        generations = convergence_data.(metrics{1}){param_set}.generations;
        
        % Calculate estimated time per generation (linear approximation)
        times = generations * (execution_times(param_set) / max(generations));
        
        % Plot time vs generation
        plot(generations, times, ['-' markers{mod(param_set-1, length(markers))+1}], ...
            'Color', colors{mod(param_set-1, length(colors))+1}, 'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', colors{mod(param_set-1, length(colors))+1});
        
        legend_entries{end+1} = sprintf('Param Set %d', param_set);
    end
end

% Add labels and legend
xlabel('Generation');
ylabel('Cumulative Execution Time (seconds)');
title('Generation-wise Convergence Time');
legend(legend_entries, 'Location', 'northwest');
grid on;

% Save figure
saveas(gcf, 'generation_convergence_time.png');
saveas(gcf, 'generation_convergence_time.fig');

%% Generate Hypervolume vs Time Graph
figure('Name', 'Hypervolume vs Execution Time', 'Position', [100, 100, 1000, 600]);
hold on;

% Plot hypervolume vs time for each parameter set
legend_entries = {};
for param_set = 1:num_param_sets
    if isfield(convergence_data, 'Hypervolume') && ...
       length(convergence_data.Hypervolume) >= param_set && ...
       ~isempty(convergence_data.Hypervolume{param_set}) && ...
       isfield(convergence_data.Hypervolume{param_set}, 'generations')
        
        generations = convergence_data.Hypervolume{param_set}.generations;
        hypervolume = convergence_data.Hypervolume{param_set}.mean_values;
        
        % Calculate estimated time per generation (linear approximation)
        times = generations * (execution_times(param_set) / max(generations));
        
        % Plot hypervolume vs time
        plot(times, hypervolume, ['-' markers{mod(param_set-1, length(markers))+1}], ...
            'Color', colors{mod(param_set-1, length(colors))+1}, 'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', colors{mod(param_set-1, length(colors))+1});
        
        legend_entries{end+1} = sprintf('Param Set %d', param_set);
    end
end

% Add labels and legend
xlabel('Execution Time (seconds)');
ylabel('Hypervolume');
title('Hypervolume vs Execution Time');
legend(legend_entries, 'Location', 'best');
grid on;

% Save figure
saveas(gcf, 'hypervolume_vs_time.png');
saveas(gcf, 'hypervolume_vs_time.fig');

%% Generate ROI vs Time Graph
figure('Name', 'ROI vs Execution Time', 'Position', [100, 100, 1000, 600]);
hold on;

% Plot ROI vs time for each parameter set
legend_entries = {};
for param_set = 1:num_param_sets
    if isfield(convergence_data, 'ROI') && ...
       length(convergence_data.ROI) >= param_set && ...
       ~isempty(convergence_data.ROI{param_set}) && ...
       isfield(convergence_data.ROI{param_set}, 'generations')
        
        generations = convergence_data.ROI{param_set}.generations;
        roi = convergence_data.ROI{param_set}.mean_values;
        
        % Calculate estimated time per generation (linear approximation)
        times = generations * (execution_times(param_set) / max(generations));
        
        % Plot ROI vs time
        plot(times, roi, ['-' markers{mod(param_set-1, length(markers))+1}], ...
            'Color', colors{mod(param_set-1, length(colors))+1}, 'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', colors{mod(param_set-1, length(colors))+1});
        
        legend_entries{end+1} = sprintf('Param Set %d', param_set);
    end
end

% Add labels and legend
xlabel('Execution Time (seconds)');
ylabel('ROI (%)');
title('ROI vs Execution Time');
legend(legend_entries, 'Location', 'best');
grid on;

% Save figure
saveas(gcf, 'roi_vs_time.png');
saveas(gcf, 'roi_vs_time.fig');

%% Helper function for hypervolume calculation
function hv = calculate_hypervolume(points, reference)
    % Simple hypervolume calculation for 2D and 3D objective spaces
    % points: matrix where each row is a solution and columns are objective values
    % reference: reference point (should be dominated by all points in the Pareto front)
    
    % Ensure points are properly oriented for calculation
    [n_points, n_objectives] = size(points);
    
    if n_points == 0
        hv = 0;
        return;
    end
    
    % Normalize objectives based on whether they are being maximized or minimized
    % Assuming: ROI (maximize), Risk (minimize), WinRate (maximize)
    normalized_points = points;
    normalized_reference = reference;
    
    % For maximization objectives, multiply by -1 to convert to minimization
    % (for hypervolume calculation, all objectives should be minimized)
    if n_objectives >= 1
        normalized_points(:, 1) = -normalized_points(:, 1); % ROI (maximize)
        normalized_reference(1) = -normalized_reference(1);
    end
    
    if n_objectives >= 3
        normalized_points(:, 3) = -normalized_points(:, 3); % WinRate (maximize)
        normalized_reference(3) = -normalized_reference(3);
    end
    
    % Calculate hypervolume based on dimensionality
    if n_objectives == 2
        % 2D hypervolume calculation
        hv = calculate_2d_hypervolume(normalized_points, normalized_reference);
    elseif n_objectives == 3
        % 3D hypervolume calculation
        hv = calculate_3d_hypervolume(normalized_points, normalized_reference);
    else
        % For higher dimensions, use a simple approximation
        hv = approximate_hypervolume(normalized_points, normalized_reference);
    end
end

function hv = calculate_2d_hypervolume(points, reference)
    % Calculate hypervolume for 2D objective space
    % Sort points by first objective
    [~, idx] = sort(points(:, 1));
    sorted_points = points(idx, :);
    
    % Calculate hypervolume
    hv = 0;
    prev_x = reference(1);
    
    for i = 1:size(sorted_points, 1)
        x = sorted_points(i, 1);
        y = sorted_points(i, 2);
        
        if y < reference(2)
            hv = hv + (x - prev_x) * (reference(2) - y);
            prev_x = x;
        end
    end
end

function hv = calculate_3d_hypervolume(points, reference)
    % Calculate hypervolume for 3D objective space using a simple approach
    % Sort points by first objective
    [~, idx] = sort(points(:, 1));
    sorted_points = points(idx, :);
    
    % Calculate hypervolume
    hv = 0;
    
    for i = 1:size(sorted_points, 1)
        x = sorted_points(i, 1);
        y = sorted_points(i, 2);
        z = sorted_points(i, 3);
        
        % Find dominated region for this point
        if i == 1
            prev_x = reference(1);
        else
            prev_x = sorted_points(i-1, 1);
        end
        
        if y < reference(2) && z < reference(3)
            volume = (x - prev_x) * (reference(2) - y) * (reference(3) - z);
            hv = hv + volume;
        end
    end
end

function hv = approximate_hypervolume(points, reference)
    % Simple approximation for higher dimensions
    % Calculate the dominated hypervolume as product of distances to reference
    hv = 0;
    
    for i = 1:size(points, 1)
        point_hv = 1;
        dominated = true;
        
        for j = 1:size(points, 2)
            if points(i, j) > reference(j)
                dominated = false;
                break;
            end
            point_hv = point_hv * (reference(j) - points(i, j));
        end
        
        if dominated
            hv = hv + point_hv;
        end
    end
end

% Ternary operator function
function result = ternary(condition, if_true, if_false)
    if condition
        result = if_true;
    else
        result = if_false;
    end
end
