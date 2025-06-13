%% NSGA-II Results Analysis Script
% This script analyzes the results of multiple NSGA-II runs across different parameter sets
% It extracts performance metrics and performs statistical analysis

clear all;
close all;
clc;

%% Configuration
% Base directory containing all parameter sets
base_dir = '.'; % Change this to the directory containing your param_set folders

% Number of parameter sets to analyze
num_param_sets = 5;

% Metrics to analyze
metrics = {'ROI', 'Risk', 'WinRate'};

% Generation to analyze (use -1 for the last available generation in each run)
target_gen = -1;

%% Initialize data structures
all_results = struct();
param_set_stats = struct();

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
    
    % Initialize arrays for each metric
    for m = 1:length(metrics)
        all_results.(metrics{m}){param_set} = zeros(num_runs, 1);
    end
    
    % Process each run
    for run_idx = 1:num_runs
        run_dir = fullfile(param_set_dir, run_dirs(run_idx).name);
        
        % Find all generation data files
        gen_files = dir(fullfile(run_dir, 'gen_*_data.csv'));
        
        if isempty(gen_files)
            warning('No generation files found in run directory: %s', run_dir);
            continue;
        end
        
        % Extract generation numbers
        gen_nums = zeros(length(gen_files), 1);
        for i = 1:length(gen_files)
            % Extract generation number from filename
            filename = gen_files(i).name;
            gen_str = regexp(filename, 'gen_(\d+)_data.csv', 'tokens');
            if ~isempty(gen_str)
                gen_nums(i) = str2double(gen_str{1}{1});
            end
        end
        
        % Determine which generation to analyze
        if target_gen < 0
            [~, max_idx] = max(gen_nums);
            gen_to_analyze = gen_nums(max_idx);
        else
            gen_to_analyze = target_gen;
            if ~ismember(gen_to_analyze, gen_nums)
                warning('Target generation %d not found in run %s. Using last available generation.', ...
                    target_gen, run_dirs(run_idx).name);
                [~, max_idx] = max(gen_nums);
                gen_to_analyze = gen_nums(max_idx);
            end
        end
        
        % Load the data for the selected generation
        data_file = fullfile(run_dir, sprintf('gen_%d_data.csv', gen_to_analyze));
        fronts_file = fullfile(run_dir, sprintf('gen_%d_fronts.csv', gen_to_analyze));
        
        try
            % Read data file
            data = readtable(data_file);
            
            % Extract best individual (rank 0, highest ROI)
            rank0_individuals = data(data.FrontRank == 0, :);
            [~, best_roi_idx] = max(rank0_individuals.ROI);
            best_individual = rank0_individuals(best_roi_idx, :);
            
            % Store metrics for this run
            for m = 1:length(metrics)
                metric_name = metrics{m};
                all_results.(metric_name){param_set}(run_idx) = best_individual.(metric_name);
            end
            
            fprintf('  Run %d/%d: Analyzed generation %d, Best ROI: %.2f%%\n', ...
                run_idx, num_runs, gen_to_analyze, best_individual.ROI);
            
        catch e
            warning('Error processing run %s: %s', run_dirs(run_idx).name, e.message);
        end
    end
    
    % Calculate statistics for this parameter set
    for m = 1:length(metrics)
        metric_name = metrics{m};
        metric_values = all_results.(metric_name){param_set};
        
        % Remove zeros (failed runs)
        metric_values = metric_values(metric_values ~= 0);
        
        if ~isempty(metric_values)
            param_set_stats.(metric_name).mean(param_set) = mean(metric_values);
            param_set_stats.(metric_name).median(param_set) = median(metric_values);
            param_set_stats.(metric_name).std(param_set) = std(metric_values);
            param_set_stats.(metric_name).min(param_set) = min(metric_values);
            param_set_stats.(metric_name).max(param_set) = max(metric_values);
            param_set_stats.(metric_name).values{param_set} = metric_values;
        else
            param_set_stats.(metric_name).mean(param_set) = NaN;
            param_set_stats.(metric_name).median(param_set) = NaN;
            param_set_stats.(metric_name).std(param_set) = NaN;
            param_set_stats.(metric_name).min(param_set) = NaN;
            param_set_stats.(metric_name).max(param_set) = NaN;
            param_set_stats.(metric_name).values{param_set} = [];
        end
    end
end

%% Statistical Analysis
fprintf('\n===== Statistical Analysis =====\n');

% ANOVA test for each metric
for m = 1:length(metrics)
    metric_name = metrics{m};
    fprintf('\n--- %s Analysis ---\n', metric_name);
    
    % Prepare data for ANOVA
    group_data = {};
    group_labels = {};
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'values') && ...
           length(param_set_stats.(metric_name).values) >= param_set && ...
           ~isempty(param_set_stats.(metric_name).values{param_set})
            
            group_data{end+1} = param_set_stats.(metric_name).values{param_set};
            group_labels{end+1} = sprintf('Param Set %d', param_set);
            
            fprintf('Parameter Set %d: Mean=%.2f, Median=%.2f, Std=%.2f, Min=%.2f, Max=%.2f\n', ...
                param_set, ...
                param_set_stats.(metric_name).mean(param_set), ...
                param_set_stats.(metric_name).median(param_set), ...
                param_set_stats.(metric_name).std(param_set), ...
                param_set_stats.(metric_name).min(param_set), ...
                param_set_stats.(metric_name).max(param_set));
        end
    end
    
    % Perform ANOVA if we have at least 2 groups
    if length(group_data) >= 2
        try
            [p, tbl, stats] = anova1(cell2mat(group_data'), cell2mat(group_labels'), 'off');
            fprintf('ANOVA p-value: %.4f\n', p);
            
            if p < 0.05
                fprintf('There is a statistically significant difference between parameter sets (p < 0.05)\n');
                
                % Multiple comparison test
                [c, m, h, gnames] = multcompare(stats, 'Display', 'off');
                
                % Display which groups are significantly different
                fprintf('Significantly different parameter sets:\n');
                for i = 1:size(c, 1)
                    if c(i, 6) < 0.05
                        fprintf('  Param Set %d vs Param Set %d: p=%.4f\n', ...
                            c(i, 1), c(i, 2), c(i, 6));
                    end
                end
            else
                fprintf('No statistically significant difference between parameter sets (p >= 0.05)\n');
            end
        catch e
            fprintf('Could not perform ANOVA: %s\n', e.message);
        end
    else
        fprintf('Not enough valid parameter sets for statistical comparison\n');
    end
end

%% Visualization
% Create figures for each metric
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    % Box plot
    figure('Name', sprintf('%s Comparison', metric_name), 'Position', [100, 100, 800, 600]);
    
    % Prepare data for boxplot
    boxplot_data = [];
    boxplot_groups = [];
    
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'values') && ...
           length(param_set_stats.(metric_name).values) >= param_set && ...
           ~isempty(param_set_stats.(metric_name).values{param_set})
            
            values = param_set_stats.(metric_name).values{param_set};
            boxplot_data = [boxplot_data; values];
            boxplot_groups = [boxplot_groups; repmat(param_set, length(values), 1)];
        end
    end
    
    if ~isempty(boxplot_data)
        boxplot(boxplot_data, boxplot_groups, 'Labels', arrayfun(@(x) sprintf('Set %d', x), unique(boxplot_groups), 'UniformOutput', false));
        title(sprintf('%s Comparison Across Parameter Sets', metric_name));
        ylabel(metric_name);
        grid on;
        
        % Add mean markers
        hold on;
        for param_set = 1:num_param_sets
            if isfield(param_set_stats, metric_name) && ...
               isfield(param_set_stats.(metric_name), 'mean') && ...
               length(param_set_stats.(metric_name).mean) >= param_set && ...
               ~isnan(param_set_stats.(metric_name).mean(param_set))
                
                plot(param_set, param_set_stats.(metric_name).mean(param_set), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
            end
        end
        hold off;
        
        % Save figure
        saveas(gcf, sprintf('%s_comparison.png', metric_name));
        saveas(gcf, sprintf('%s_comparison.fig', metric_name));
    else
        close(gcf);
    end
    
    % Histogram comparison
    figure('Name', sprintf('%s Distribution', metric_name), 'Position', [100, 100, 1000, 600]);
    
    valid_param_sets = 0;
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'values') && ...
           length(param_set_stats.(metric_name).values) >= param_set && ...
           ~isempty(param_set_stats.(metric_name).values{param_set})
            
            valid_param_sets = valid_param_sets + 1;
        end
    end
    
    if valid_param_sets > 0
        subplot_rows = ceil(sqrt(valid_param_sets));
        subplot_cols = ceil(valid_param_sets / subplot_rows);
        
        subplot_idx = 1;
        for param_set = 1:num_param_sets
            if isfield(param_set_stats, metric_name) && ...
               isfield(param_set_stats.(metric_name), 'values') && ...
               length(param_set_stats.(metric_name).values) >= param_set && ...
               ~isempty(param_set_stats.(metric_name).values{param_set})
                
                subplot(subplot_rows, subplot_cols, subplot_idx);
                histogram(param_set_stats.(metric_name).values{param_set}, 10);
                title(sprintf('Param Set %d', param_set));
                xlabel(metric_name);
                ylabel('Frequency');
                grid on;
                
                subplot_idx = subplot_idx + 1;
            end
        end
        
        sgtitle(sprintf('%s Distribution Across Parameter Sets', metric_name));
        
        % Save figure
        saveas(gcf, sprintf('%s_distribution.png', metric_name));
        saveas(gcf, sprintf('%s_distribution.fig', metric_name));
    else
        close(gcf);
    end
end

%% Save results
save('nsga_analysis_results.mat', 'all_results', 'param_set_stats');

% Create summary table
summary_table = table();
summary_table.ParameterSet = (1:num_param_sets)';

for m = 1:length(metrics)
    metric_name = metrics{m};
    if isfield(param_set_stats, metric_name) && isfield(param_set_stats.(metric_name), 'mean')
        summary_table.(sprintf('%s_Mean', metric_name)) = param_set_stats.(metric_name).mean';
        summary_table.(sprintf('%s_Median', metric_name)) = param_set_stats.(metric_name).median';
        summary_table.(sprintf('%s_Std', metric_name)) = param_set_stats.(metric_name).std';
        summary_table.(sprintf('%s_Min', metric_name)) = param_set_stats.(metric_name).min';
        summary_table.(sprintf('%s_Max', metric_name)) = param_set_stats.(metric_name).max';
    end
end

% Write summary to CSV
writetable(summary_table, 'nsga_analysis_summary.csv');

fprintf('\nAnalysis complete. Results saved to:\n');
fprintf('  - nsga_analysis_results.mat (MATLAB data)\n');
fprintf('  - nsga_analysis_summary.csv (CSV summary)\n');
fprintf('  - [Metric]_comparison.png/fig (Box plots)\n');
fprintf('  - [Metric]_distribution.png/fig (Histograms)\n');
