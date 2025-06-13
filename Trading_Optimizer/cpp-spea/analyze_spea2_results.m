%% SPEA2 Results Analysis Script
% This script analyzes the results of multiple SPEA2 runs across different parameter sets
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
        gen_files = dir(fullfile(run_dir, 'gen_*_archive.csv'));
        
        % If no archive files found, try data files
        if isempty(gen_files)
            gen_files = dir(fullfile(run_dir, 'gen_*_data.csv'));
        end
        
        if isempty(gen_files)
            warning('No generation files found in run directory: %s', run_dir);
            continue;
        end
        
        % Extract generation numbers
        gen_nums = zeros(length(gen_files), 1);
        for i = 1:length(gen_files)
            % Extract generation number from filename
            filename = gen_files(i).name;
            gen_str = regexp(filename, 'gen_(\d+)_', 'tokens');
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
        archive_file = fullfile(run_dir, sprintf('gen_%d_archive.csv', gen_to_analyze));
        data_file = fullfile(run_dir, sprintf('gen_%d_data.csv', gen_to_analyze));
        
        try
            % Try to read archive file first
            if exist(archive_file, 'file')
                archive = readtable(archive_file);
                
                % Extract non-dominated solutions (SPEA2Fitness < 1)
                if ismember('SPEA2Fitness', archive.Properties.VariableNames)
                    archive = archive(archive.SPEA2Fitness < 1, :);
                end
                
                % If archive is empty, try using all solutions
                if isempty(archive)
                    archive = readtable(archive_file);
                end
            else
                % If archive file doesn't exist, use data file
                archive = readtable(data_file);
                
                % Extract non-dominated solutions (SPEA2Fitness < 1)
                if ismember('SPEA2Fitness', archive.Properties.VariableNames)
                    archive = archive(archive.SPEA2Fitness < 1, :);
                end
                
                % If still empty, use all solutions
                if isempty(archive)
                    archive = readtable(data_file);
                end
            end
            
            % Extract best individual (highest ROI)
            [~, best_roi_idx] = max(archive.ROI);
            best_individual = archive(best_roi_idx, :);
            
            % Calculate hypervolume for the non-dominated solutions
            if ~isempty(archive)
                % Extract objectives (adjust column names if needed)
                objectives = [archive.ROI, archive.Risk, archive.WinRate];
                
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

%% Save summary statistics to CSV
summary_table = table();
summary_table.Metric = metrics';

for param_set = 1:num_param_sets
    for stat_type = {'mean', 'median', 'std', 'min', 'max'}
        col_name = sprintf('Set%d_%s', param_set, stat_type{1});
        summary_table.(col_name) = zeros(length(metrics), 1);
        
        for m = 1:length(metrics)
            metric_name = metrics{m};
            if isfield(param_set_stats, metric_name) && ...
               isfield(param_set_stats.(metric_name), stat_type{1}) && ...
               length(param_set_stats.(metric_name).(stat_type{1})) >= param_set
                
                summary_table.(col_name)(m) = param_set_stats.(metric_name).(stat_type{1})(param_set);
            end
        end
    end
end

% Write summary to CSV
writetable(summary_table, 'spea2_analysis_summary.csv');
fprintf('\nSummary statistics saved to spea2_analysis_summary.csv\n');

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
        title(sprintf('SPEA2: %s Comparison Across Parameter Sets', metric_name));
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
        saveas(gcf, sprintf('SPEA2_%s_comparison.png', metric_name));
        saveas(gcf, sprintf('SPEA2_%s_comparison.fig', metric_name));
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
                histogram(param_set_stats.(metric_name).values{param_set}, 'Normalization', 'probability');
                title(sprintf('Set %d', param_set));
                xlabel(metric_name);
                ylabel('Probability');
                grid on;
                
                subplot_idx = subplot_idx + 1;
            end
        end
        
        sgtitle(sprintf('SPEA2: %s Distribution Across Parameter Sets', metric_name));
        
        % Save figure
        saveas(gcf, sprintf('SPEA2_%s_distribution.png', metric_name));
        saveas(gcf, sprintf('SPEA2_%s_distribution.fig', metric_name));
    else
        close(gcf);
    end
end

%% Generate detailed report
report_file = fopen('spea2_analysis_report.txt', 'w');
fprintf(report_file, 'SPEA2 Parameter Set Analysis Report\n');
fprintf(report_file, '==================================\n\n');

% Overall summary
fprintf(report_file, 'Overall Summary:\n');
fprintf(report_file, '--------------\n');

% Find best parameter set for each metric
best_param_sets = struct();
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    if strcmp(metric_name, 'Risk')
        % For Risk, lower (more negative) is better
        [best_val, best_idx] = min(param_set_stats.(metric_name).mean);
        comparison_op = '<';
    else
        % For other metrics, higher is better
        [best_val, best_idx] = max(param_set_stats.(metric_name).mean);
        comparison_op = '>';
    end
    
    best_param_sets.(metric_name) = best_idx;
    
    fprintf(report_file, 'Best parameter set for %s: Set %d (Mean: %.2f)\n', ...
        metric_name, best_idx, best_val);
end
fprintf(report_file, '\n');

% Statistical significance summary
fprintf(report_file, 'Statistical Significance:\n');
fprintf(report_file, '------------------------\n');
for m = 1:length(metrics)
    metric_name = metrics{m};
    
    if isfield(param_set_stats.(metric_name), 'anova_p')
        fprintf(report_file, '%s: ANOVA p-value = %.4f %s\n', ...
            metric_name, param_set_stats.(metric_name).anova_p, ...
            ternary(param_set_stats.(metric_name).anova_p < 0.05, '(significant)', '(not significant)'));
    elseif isfield(param_set_stats.(metric_name), 'kw_p')
        fprintf(report_file, '%s: Kruskal-Wallis p-value = %.4f %s\n', ...
            metric_name, param_set_stats.(metric_name).kw_p, ...
            ternary(param_set_stats.(metric_name).kw_p < 0.05, '(significant)', '(not significant)'));
    else
        fprintf(report_file, '%s: No statistical test performed\n', metric_name);
    end
end
fprintf(report_file, '\n');

% Detailed metrics for each parameter set
fprintf(report_file, 'Detailed Metrics by Parameter Set:\n');
fprintf(report_file, '--------------------------------\n');
for param_set = 1:num_param_sets
    fprintf(report_file, 'Parameter Set %d:\n', param_set);
    
    for m = 1:length(metrics)
        metric_name = metrics{m};
        
        if isfield(param_set_stats, metric_name) && ...
           isfield(param_set_stats.(metric_name), 'mean') && ...
           length(param_set_stats.(metric_name).mean) >= param_set && ...
           ~isnan(param_set_stats.(metric_name).mean(param_set))
            
            fprintf(report_file, '  %s: Mean=%.2f, Median=%.2f, Std=%.2f, Min=%.2f, Max=%.2f\n', ...
                metric_name, ...
                param_set_stats.(metric_name).mean(param_set), ...
                param_set_stats.(metric_name).median(param_set), ...
                param_set_stats.(metric_name).std(param_set), ...
                param_set_stats.(metric_name).min(param_set), ...
                param_set_stats.(metric_name).max(param_set));
        end
    end
    fprintf(report_file, '\n');
end

% Recommendations
fprintf(report_file, 'Recommendations:\n');
fprintf(report_file, '---------------\n');

% Count how many times each parameter set is the best
param_set_counts = zeros(num_param_sets, 1);
for m = 1:length(metrics)
    metric_name = metrics{m};
    if isfield(best_param_sets, metric_name)
        best_idx = best_param_sets.(metric_name);
        if best_idx > 0 && best_idx <= num_param_sets
            param_set_counts(best_idx) = param_set_counts(best_idx) + 1;
        end
    end
end

% Find the overall best parameter set
[~, overall_best] = max(param_set_counts);
fprintf(report_file, 'Overall best parameter set: Set %d\n', overall_best);
fprintf(report_file, 'This parameter set performs best in %d out of %d metrics.\n', ...
    param_set_counts(overall_best), length(metrics));
fprintf(report_file, '\n');

% Specific recommendations for each objective
fprintf(report_file, 'For maximizing ROI: Use Parameter Set %d\n', best_param_sets.ROI);
fprintf(report_file, 'For minimizing Risk: Use Parameter Set %d\n', best_param_sets.Risk);
fprintf(report_file, 'For maximizing Win Rate: Use Parameter Set %d\n', best_param_sets.WinRate);
fprintf(report_file, 'For best Pareto front approximation (Hypervolume): Use Parameter Set %d\n', best_param_sets.Hypervolume);

fclose(report_file);
fprintf('Detailed analysis report saved to spea2_analysis_report.txt\n');

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
