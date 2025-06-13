%% NSGA-II Results Analysis Script (Enhanced Version)
% This script analyzes the results of multiple NSGA-II runs across different parameter sets
% It extracts performance metrics and performs statistical analysis
% Enhanced with proper hypervolume calculation and robust statistical tests

clear all;
close all;
clc;

%% Configuration
% Base directory containing all parameter sets
base_dir = '.'; % Change this to the directory containing your param_set folders

% Number of parameter sets to analyze
num_param_sets = 5;

% Metrics to analyze
metrics = {'ROI', 'Risk', 'WinRate', 'Hypervolume'};

% Generation to analyze (use -1 for the last available generation in each run)
target_gen = -1;

% Reference point for hypervolume calculation (for minimization objectives)
% Adjust these values based on your objectives and their ranges
ref_point = [-100, 0, 0]; % [ROI, Risk, WinRate] - assuming ROI is maximized, Risk minimized, WinRate maximized

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
                all_results.Hypervolume{param_set}(run_idx) = hv;
            else
                all_results.Hypervolume{param_set}(run_idx) = 0;
            end
            
            % Store other metrics for this run
            for m = 1:length(metrics)-1  % Skip Hypervolume as it's already handled
                metric_name = metrics{m};
                all_results.(metric_name){param_set}(run_idx) = best_individual.(metric_name);
            end
            
            fprintf('  Run %d/%d: Analyzed generation %d, Best ROI: %.2f%%, Hypervolume: %.4f\n', ...
                run_idx, num_runs, gen_to_analyze, best_individual.ROI, all_results.Hypervolume{param_set}(run_idx));
            
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
    valid_param_sets = [];
    
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'values') && ...
           length(param_set_stats.(metric_name).values) >= param_set && ...
           ~isempty(param_set_stats.(metric_name).values{param_set})
            
            group_data{end+1} = param_set_stats.(metric_name).values{param_set};
            group_labels{end+1} = sprintf('Param Set %d', param_set);
            valid_param_sets(end+1) = param_set;
            
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
        % Prepare data for ANOVA with proper alignment
        try
            % Convert to vectors with proper group labels
            all_data = [];
            all_groups = [];
            
            for g = 1:length(group_data)
                all_data = [all_data; group_data{g}(:)];
                all_groups = [all_groups; repmat(valid_param_sets(g), length(group_data{g}), 1)];
            end
            
            % Now perform ANOVA with properly aligned data
            [p, tbl, stats] = anova1(all_data, all_groups, 'off');
            fprintf('ANOVA p-value: %.4f\n', p);
            
            if p < 0.05
                fprintf('There is a statistically significant difference between parameter sets (p < 0.05)\n');
                
                % Multiple comparison test (post-hoc)
                [c, m, h, gnames] = multcompare(stats, 'Display', 'off');
                
                % Display which groups are significantly different
                fprintf('Post-hoc analysis (Tukey-Kramer):\n');
                for i = 1:size(c, 1)
                    fprintf('  Param Set %d vs Param Set %d: diff=%.4f, CI=[%.4f, %.4f], p=%.4f %s\n', ...
                        c(i, 1), c(i, 2), c(i, 4), c(i, 3), c(i, 5), c(i, 6), ...
                        ternary(c(i, 6) < 0.05, '(significant)', ''));
                end
                
                % Create a compact table of significant differences
                fprintf('\nSignificant Differences Summary:\n');
                fprintf('Parameter Set Pairs with p < 0.05:\n');
                sig_count = 0;
                for i = 1:size(c, 1)
                    if c(i, 6) < 0.05
                        sig_count = sig_count + 1;
                        fprintf('  %d. Set %d vs Set %d (p=%.4f)\n', ...
                            sig_count, c(i, 1), c(i, 2), c(i, 6));
                    end
                end
                if sig_count == 0
                    fprintf('  No significant pairwise differences found despite overall ANOVA significance.\n');
                    fprintf('  This can happen with marginal significance or when differences are distributed across multiple comparisons.\n');
                end
            else
                fprintf('No statistically significant difference between parameter sets (p >= 0.05)\n');
            end
            
            % Save ANOVA results
            param_set_stats.(metric_name).anova_p = p;
            param_set_stats.(metric_name).anova_table = tbl;
            if p < 0.05
                param_set_stats.(metric_name).posthoc = c;
            end
            
        catch e
            fprintf('Could not perform ANOVA: %s\n', e.message);
            fprintf('Trying alternative Kruskal-Wallis test...\n');
            
            % Try non-parametric Kruskal-Wallis test instead
            try
                [p, tbl, stats] = kruskalwallis(all_data, all_groups, 'off');
                fprintf('Kruskal-Wallis p-value: %.4f\n', p);
                
                if p < 0.05
                    fprintf('There is a statistically significant difference between parameter sets (p < 0.05)\n');
                    
                    % Multiple comparison test
                    [c, m, h, gnames] = multcompare(stats, 'Display', 'off');
                    
                    % Display which groups are significantly different
                    fprintf('Post-hoc analysis (Dunn''s test with Bonferroni correction):\n');
                    for i = 1:size(c, 1)
                        fprintf('  Param Set %d vs Param Set %d: diff=%.4f, CI=[%.4f, %.4f], p=%.4f %s\n', ...
                            c(i, 1), c(i, 2), c(i, 4), c(i, 3), c(i, 5), c(i, 6), ...
                            ternary(c(i, 6) < 0.05, '(significant)', ''));
                    end
                    
                    % Create a compact table of significant differences
                    fprintf('\nSignificant Differences Summary:\n');
                    fprintf('Parameter Set Pairs with p < 0.05:\n');
                    sig_count = 0;
                    for i = 1:size(c, 1)
                        if c(i, 6) < 0.05
                            sig_count = sig_count + 1;
                            fprintf('  %d. Set %d vs Set %d (p=%.4f)\n', ...
                                sig_count, c(i, 1), c(i, 2), c(i, 6));
                        end
                    end
                    if sig_count == 0
                        fprintf('  No significant pairwise differences found despite overall test significance.\n');
                    end
                else
                    fprintf('No statistically significant difference between parameter sets (p >= 0.05)\n');
                end
                
                % Save Kruskal-Wallis results
                param_set_stats.(metric_name).kw_p = p;
                param_set_stats.(metric_name).kw_table = tbl;
                if p < 0.05
                    param_set_stats.(metric_name).posthoc = c;
                end
                
            catch e2
                fprintf('Could not perform Kruskal-Wallis test: %s\n', e2.message);
                fprintf('Statistical comparison skipped. Use visual comparison instead.\n');
            end
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
                
                plot(find(unique(boxplot_groups) == param_set), param_set_stats.(metric_name).mean(param_set), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
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

%% Convergence Analysis for Hypervolume
% Create a figure showing hypervolume convergence across generations
try
    % Find a representative run from each parameter set
    figure('Name', 'Hypervolume Convergence', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Define colors and markers for each parameter set
    colors = lines(num_param_sets);
    markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '*'};
    
    legend_entries = {};
    
    for param_set = 1:num_param_sets
        param_set_dir = fullfile(base_dir, sprintf('param_set_%d', param_set));
        
        if ~exist(param_set_dir, 'dir')
            continue;
        end
        
        % Get all run directories
        run_dirs = dir(fullfile(param_set_dir, 'run_*'));
        run_dirs = run_dirs([run_dirs.isdir]);
        
        if isempty(run_dirs)
            continue;
        end
        
        % Use the first run that has data
        for run_idx = 1:min(length(run_dirs), 3)  % Try first 3 runs
            run_dir = fullfile(param_set_dir, run_dirs(run_idx).name);
            
            % Find all generation data files
            gen_files = dir(fullfile(run_dir, 'gen_*_fronts.csv'));
            
            if isempty(gen_files)
                continue;
            end
            
            % Extract generation numbers and calculate hypervolume for each
            gen_nums = [];
            hvs = [];
            
            for i = 1:length(gen_files)
                % Extract generation number from filename
                filename = gen_files(i).name;
                gen_str = regexp(filename, 'gen_(\d+)_fronts.csv', 'tokens');
                if isempty(gen_str)
                    continue;
                end
                
                gen_num = str2double(gen_str{1}{1});
                
                % Read fronts file
                fronts_file = fullfile(run_dir, filename);
                try
                    fronts = readtable(fronts_file);
                    
                    % Extract first front
                    first_front = fronts(fronts.FrontIndex == 0, :);
                    if isempty(first_front)
                        continue;
                    end
                    
                    % Extract objectives
                    objectives = [first_front.ROI, first_front.Risk, first_front.WinRate];
                    
                    % Calculate hypervolume
                    hv = calculate_hypervolume(objectives, ref_point);
                    
                    gen_nums(end+1) = gen_num;
                    hvs(end+1) = hv;
                catch
                    % Skip problematic files
                    continue;
                end
            end
            
            % Plot if we have data
            if ~isempty(gen_nums) && ~isempty(hvs)
                % Sort by generation number
                [gen_nums, sort_idx] = sort(gen_nums);
                hvs = hvs(sort_idx);
                
                % Plot
                plot(gen_nums, hvs, ['-' markers{mod(param_set-1, length(markers))+1}], ...
                    'Color', colors(param_set,:), 'LineWidth', 1.5, ...
                    'MarkerSize', 6, 'MarkerFaceColor', colors(param_set,:));
                
                legend_entries{end+1} = sprintf('Param Set %d (Run %d)', param_set, run_idx);
                break;  % Found a good run, no need to check more
            end
        end
    end
    
    % Finalize plot
    xlabel('Generation');
    ylabel('Hypervolume');
    title('Hypervolume Convergence Across Generations');
    grid on;
    if ~isempty(legend_entries)
        legend(legend_entries, 'Location', 'southeast');
    end
    
    % Save figure
    saveas(gcf, 'hypervolume_convergence.png');
    saveas(gcf, 'hypervolume_convergence.fig');
    
catch e
    warning('Could not create hypervolume convergence plot: %s', e.message);
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

% Add statistical test results
for m = 1:length(metrics)
    metric_name = metrics{m};
    if isfield(param_set_stats, metric_name)
        if isfield(param_set_stats.(metric_name), 'anova_p')
            summary_table.(sprintf('%s_ANOVA_p', metric_name)) = repmat(param_set_stats.(metric_name).anova_p, num_param_sets, 1);
        elseif isfield(param_set_stats.(metric_name), 'kw_p')
            summary_table.(sprintf('%s_KW_p', metric_name)) = repmat(param_set_stats.(metric_name).kw_p, num_param_sets, 1);
        end
    end
end

% Write summary to CSV
writetable(summary_table, 'nsga_analysis_summary.csv');

% Create a more detailed report
fid = fopen('nsga_analysis_report.txt', 'w');
fprintf(fid, '===== NSGA-II Parameter Set Analysis Report =====\n\n');
fprintf(fid, 'Analysis Date: %s\n\n', datestr(now));

fprintf(fid, '1. SUMMARY OF RESULTS\n');
fprintf(fid, '---------------------\n\n');

% Print summary statistics
fprintf(fid, 'Parameter Set Performance Summary:\n\n');
for m = 1:length(metrics)
    metric_name = metrics{m};
    fprintf(fid, '%s:\n', metric_name);
    
    % Find best parameter set for this metric
    best_param_set = 0;
    best_value = -Inf;
    
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'mean') && ...
           length(param_set_stats.(metric_name).mean) >= param_set && ...
           ~isnan(param_set_stats.(metric_name).mean(param_set))
            
            fprintf(fid, '  Param Set %d: Mean=%.2f, Median=%.2f, Std=%.2f\n', ...
                param_set, ...
                param_set_stats.(metric_name).mean(param_set), ...
                param_set_stats.(metric_name).median(param_set), ...
                param_set_stats.(metric_name).std(param_set));
            
            % Update best if this is better (assuming higher is better for all metrics)
            if param_set_stats.(metric_name).mean(param_set) > best_value
                best_value = param_set_stats.(metric_name).mean(param_set);
                best_param_set = param_set;
            end
        end
    end
    
    if best_param_set > 0
        fprintf(fid, '  Best Parameter Set: %d (Mean=%.2f)\n', best_param_set, best_value);
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n2. STATISTICAL ANALYSIS\n');
fprintf(fid, '------------------------\n\n');

% Print statistical test results
for m = 1:length(metrics)
    metric_name = metrics{m};
    fprintf(fid, '%s:\n', metric_name);
    
    if isfield(param_set_stats, metric_name)
        if isfield(param_set_stats.(metric_name), 'anova_p')
            fprintf(fid, '  ANOVA p-value: %.4f\n', param_set_stats.(metric_name).anova_p);
            if param_set_stats.(metric_name).anova_p < 0.05
                fprintf(fid, '  Statistically significant differences found (p < 0.05)\n');
                
                if isfield(param_set_stats.(metric_name), 'posthoc')
                    fprintf(fid, '  Post-hoc analysis (Tukey-Kramer):\n');
                    c = param_set_stats.(metric_name).posthoc;
                    sig_count = 0;
                    for i = 1:size(c, 1)
                        if c(i, 6) < 0.05
                            sig_count = sig_count + 1;
                            fprintf(fid, '    Set %d vs Set %d: p=%.4f (significant)\n', ...
                                c(i, 1), c(i, 2), c(i, 6));
                        end
                    end
                    if sig_count == 0
                        fprintf(fid, '    No significant pairwise differences found despite overall ANOVA significance.\n');
                    end
                end
            else
                fprintf(fid, '  No statistically significant differences found (p >= 0.05)\n');
            end
        elseif isfield(param_set_stats.(metric_name), 'kw_p')
            fprintf(fid, '  Kruskal-Wallis p-value: %.4f\n', param_set_stats.(metric_name).kw_p);
            if param_set_stats.(metric_name).kw_p < 0.05
                fprintf(fid, '  Statistically significant differences found (p < 0.05)\n');
                
                if isfield(param_set_stats.(metric_name), 'posthoc')
                    fprintf(fid, '  Post-hoc analysis (Dunn''s test with Bonferroni correction):\n');
                    c = param_set_stats.(metric_name).posthoc;
                    sig_count = 0;
                    for i = 1:size(c, 1)
                        if c(i, 6) < 0.05
                            sig_count = sig_count + 1;
                            fprintf(fid, '    Set %d vs Set %d: p=%.4f (significant)\n', ...
                                c(i, 1), c(i, 2), c(i, 6));
                        end
                    end
                    if sig_count == 0
                        fprintf(fid, '    No significant pairwise differences found despite overall test significance.\n');
                    end
                end
            else
                fprintf(fid, '  No statistically significant differences found (p >= 0.05)\n');
            end
        else
            fprintf(fid, '  No statistical test results available\n');
        end
    else
        fprintf(fid, '  No data available for this metric\n');
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n3. RECOMMENDATIONS\n');
fprintf(fid, '-------------------\n\n');

% Determine overall best parameter set
fprintf(fid, 'Overall Recommendation:\n');

% Simple scoring: count how many times each parameter set is the best
best_counts = zeros(1, num_param_sets);
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    best_param_set = 0;
    best_value = -Inf;
    
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'mean') && ...
           length(param_set_stats.(metric_name).mean) >= param_set && ...
           ~isnan(param_set_stats.(metric_name).mean(param_set))
            
            % Update best if this is better (assuming higher is better for all metrics)
            if param_set_stats.(metric_name).mean(param_set) > best_value
                best_value = param_set_stats.(metric_name).mean(param_set);
                best_param_set = param_set;
            end
        end
    end
    
    if best_param_set > 0
        best_counts(best_param_set) = best_counts(best_param_set) + 1;
    end
end

[~, overall_best] = max(best_counts);
fprintf(fid, '  Based on performance across all metrics, Parameter Set %d is recommended.\n', overall_best);
fprintf(fid, '  This parameter set was the best performer in %d out of %d metrics.\n\n', best_counts(overall_best), length(metrics));

fprintf(fid, 'Metric-Specific Recommendations:\n');
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    best_param_set = 0;
    best_value = -Inf;
    
    for param_set = 1:num_param_sets
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'mean') && ...
           length(param_set_stats.(metric_name).mean) >= param_set && ...
           ~isnan(param_set_stats.(metric_name).mean(param_set))
            
            % Update best if this is better (assuming higher is better for all metrics)
            if param_set_stats.(metric_name).mean(param_set) > best_value
                best_value = param_set_stats.(metric_name).mean(param_set);
                best_param_set = param_set;
            end
        end
    end
    
    if best_param_set > 0
        fprintf(fid, '  For optimizing %s: Use Parameter Set %d (Mean=%.2f)\n', metric_name, best_param_set, best_value);
    end
end

fclose(fid);

fprintf('\nAnalysis complete. Results saved to:\n');
fprintf('  - nsga_analysis_results.mat (MATLAB data)\n');
fprintf('  - nsga_analysis_summary.csv (CSV summary)\n');
fprintf('  - nsga_analysis_report.txt (Detailed report)\n');
fprintf('  - [Metric]_comparison.png/fig (Box plots)\n');
fprintf('  - [Metric]_distribution.png/fig (Histograms)\n');
fprintf('  - hypervolume_convergence.png/fig (Convergence plot)\n');

%% Helper Functions

% Function to calculate hypervolume
function hv = calculate_hypervolume(points, ref_point)
    % This function calculates the hypervolume indicator for a set of points
    % Points should be a matrix where each row is a solution and each column is an objective
    % Ref_point is the reference point for hypervolume calculation
    
    % Check if points is empty
    if isempty(points)
        hv = 0;
        return;
    end
    
    % Get number of objectives
    num_obj = size(points, 2);
    
    % Normalize objectives to [0,1] range
    points_norm = zeros(size(points));
    for i = 1:num_obj
        min_val = min(points(:,i));
        max_val = max(points(:,i));
        
        % Avoid division by zero
        if max_val - min_val < 1e-10
            points_norm(:,i) = 0.5;
        else
            % Normalize based on objective direction
            if ref_point(i) < min_val  % Maximization
                points_norm(:,i) = (points(:,i) - min_val) / (max_val - min_val);
            else  % Minimization
                points_norm(:,i) = 1 - (points(:,i) - min_val) / (max_val - min_val);
            end
        end
    end
    
    % For 2D and 3D, use specialized algorithms
    if num_obj == 2
        hv = hypervolume2D(points_norm);
    elseif num_obj == 3
        hv = hypervolume3D(points_norm);
    else
        % For higher dimensions, use a simple approximation
        hv = approximate_hypervolume(points_norm);
    end
end

% Calculate hypervolume for 2D case
function hv = hypervolume2D(points)
    % Sort points by first objective (ascending)
    [~, idx] = sort(points(:,1));
    sorted_points = points(idx,:);
    
    % Calculate hypervolume
    hv = 0;
    prev_x = 0;
    prev_y = 0;
    
    for i = 1:size(sorted_points, 1)
        x = sorted_points(i,1);
        y = sorted_points(i,2);
        
        if y > prev_y
            hv = hv + (x - prev_x) * y;
            prev_x = x;
            prev_y = y;
        end
    end
    
    % Add final rectangle
    hv = hv + (1 - prev_x) * prev_y;
end

% Calculate hypervolume for 3D case
function hv = hypervolume3D(points)
    % This is a simplified 3D hypervolume calculation
    % For a more accurate implementation, consider using a dedicated library
    
    % Sort points by first objective
    [~, idx] = sort(points(:,1));
    sorted_points = points(idx,:);
    
    hv = 0;
    for i = 1:size(sorted_points, 1)
        % Calculate contribution of this point
        if i == 1
            x_contrib = sorted_points(i,1);
        else
            x_contrib = sorted_points(i,1) - sorted_points(i-1,1);
        end
        
        % Calculate 2D hypervolume in the YZ plane
        yz_points = sorted_points(i:end, 2:3);
        yz_hv = hypervolume2D(yz_points);
        
        % Add contribution
        hv = hv + x_contrib * yz_hv;
    end
end

% Simple approximation for higher dimensions
function hv = approximate_hypervolume(points)
    % Calculate dominated hypervolume using Monte Carlo sampling
    % This is a very simple approximation - for real applications, use a proper library
    
    num_samples = 10000;
    count = 0;
    
    % Generate random points in the unit hypercube
    samples = rand(num_samples, size(points, 2));
    
    % Count dominated samples
    for i = 1:num_samples
        sample = samples(i,:);
        dominated = false;
        
        for j = 1:size(points, 1)
            if all(points(j,:) >= sample)
                dominated = true;
                break;
            end
        end
        
        if dominated
            count = count + 1;
        end
    end
    
    % Hypervolume is the fraction of dominated samples
    hv = count / num_samples;
end

% Ternary operator function (condition ? if_true : if_false)
function result = ternary(condition, if_true, if_false)
    if condition
        result = if_true;
    else
        result = if_false;
    end
end
