%% NSGA-II vs SPEA2 Comparative Analysis Script
% This script compares the performance of NSGA-II (Set 5) and SPEA2 (Set 5)
% It performs statistical analysis and generates comparative visualizations

clear all;
close all;
clc;

%% Configuration
% Base directory containing the algorithm result folders
base_dir = "."; % Change this if your folders are elsewhere

% Algorithm names and their corresponding folder names
algorithms = {
    struct("name", "NSGA-II", "folder", "param_set_5"), ...
    struct("name", "SPEA2", "folder", "param_set_nsga") % Assuming SPEA2 results are also in a folder named param_set_5
};

% Metrics to analyze
metrics = {"ROI", "Risk", "WinRate", "Hypervolume"};

% Generation to analyze (use -1 for the last available generation)
target_gen = -1;

% Reference point for hypervolume calculation (for minimization objectives)
% Adjust these values based on your objectives and their ranges
ref_point = [-100, 0, 0]; % [ROI, Risk, WinRate] - assuming ROI maximized, Risk minimized, WinRate maximized

% Colors and markers for plotting
colors = {"b", "r"}; % Blue for NSGA-II, Red for SPEA2
markers = {"o", "s"};

%% Initialize data structures
results = struct();
convergence = struct();

%% Process each algorithm
for algo_idx = 1:length(algorithms)
    algo_name = algorithms{algo_idx}.name;
    algo_folder = algorithms{algo_idx}.folder;
    param_set_dir = fullfile(base_dir, algo_folder);
    
    fprintf("Processing %s results from %s...\n", algo_name, param_set_dir);
    
    % Skip if directory doesn"t exist
    if ~exist(param_set_dir, "dir")
        warning("Algorithm directory not found: %s", param_set_dir);
        results.(algo_name) = struct(); % Initialize empty struct
        convergence.(algo_name) = struct();
        continue;
    end
    
    % Get all run directories
    run_dirs = dir(fullfile(param_set_dir, "run_*"));
    run_dirs = run_dirs([run_dirs.isdir]);
    
    % Initialize arrays to store metrics for this algorithm
    num_runs = length(run_dirs);
    fprintf("  Found %d runs for %s\n", num_runs, algo_name);
    
    % Initialize results structure for this algorithm
    for m = 1:length(metrics)
        results.(algo_name).(metrics{m}) = zeros(num_runs, 1);
    end
    results.(algo_name).ParetoFronts = cell(num_runs, 1);
    
    % Initialize convergence structure for this algorithm
    for m = 1:length(metrics)
        convergence.(algo_name).(metrics{m}).generations = [];
        convergence.(algo_name).(metrics{m}).mean_values = [];
        convergence.(algo_name).(metrics{m}).std_values = [];
    end
    convergence.(algo_name).ExecutionTime.mean = 0;
    convergence.(algo_name).ExecutionTime.std = 0;
    
    run_metrics_over_gens = struct();
    run_exec_times = zeros(num_runs, 1);
    max_generations_across_runs = 0;
    
    % Process each run
    for run_idx = 1:num_runs
        run_dir = fullfile(param_set_dir, run_dirs(run_idx).name);
        
        % Find generation files (try archive/fronts first, then data)
        archive_pattern = "gen_*_archive.csv";
        fronts_pattern = "gen_*_fronts.csv";
        data_pattern = "gen_*_data.csv";
        
        gen_files = dir(fullfile(run_dir, archive_pattern));
        file_type = "archive";
        if isempty(gen_files)
            gen_files = dir(fullfile(run_dir, fronts_pattern));
            file_type = "fronts";
        end
        if isempty(gen_files)
            gen_files = dir(fullfile(run_dir, data_pattern));
            file_type = "data";
        end
        
        if isempty(gen_files)
            warning("No generation files found in run directory: %s", run_dir);
            continue;
        end
        
        % Extract generation numbers and sort
        gen_nums = zeros(length(gen_files), 1);
        for i = 1:length(gen_files)
            filename = gen_files(i).name;
            gen_str = regexp(filename, "gen_(\d+)_", "tokens");
            if ~isempty(gen_str)
                gen_nums(i) = str2double(gen_str{1}{1});
            end
        end
        [gen_nums, sort_idx] = sort(gen_nums);
        gen_files = gen_files(sort_idx);
        max_generations_across_runs = max(max_generations_across_runs, gen_nums(end));
        
        % Initialize arrays to store metrics across generations for this run
        gen_metrics = struct();
        for m = 1:length(metrics)
            gen_metrics.(metrics{m}) = zeros(length(gen_nums), 1);
        end
        
        % Process each generation for convergence data
        for gen_idx = 1:length(gen_nums)
            gen_num = gen_nums(gen_idx);
            
            % Load the data for this generation
            if strcmp(file_type, "archive")
                file_path = fullfile(run_dir, sprintf("gen_%d_archive.csv", gen_num));
            elseif strcmp(file_type, "fronts")
                file_path = fullfile(run_dir, sprintf("gen_%d_fronts.csv", gen_num));
            else
                file_path = fullfile(run_dir, sprintf("gen_%d_data.csv", gen_num));
            end
            
            try
                gen_data = readtable(file_path);
                
                % Extract non-dominated solutions
                if strcmp(algo_name, "SPEA2") && ismember("SPEA2Fitness", gen_data.Properties.VariableNames)
                    pareto_solutions = gen_data(gen_data.SPEA2Fitness < 1, :);
                elseif strcmp(algo_name, "NSGA-II") && ismember("FrontRank", gen_data.Properties.VariableNames)
                    pareto_solutions = gen_data(gen_data.FrontRank == 0, :);
                else % Fallback if specific columns aren"t present
                    pareto_solutions = gen_data; 
                end
                
                % If no non-dominated solutions found, use all data
                if isempty(pareto_solutions)
                    pareto_solutions = gen_data;
                end
                
                % Calculate metrics for this generation
                if ~isempty(pareto_solutions)
                    % Best ROI
                    [best_roi, ~] = max(pareto_solutions.ROI);
                    gen_metrics.ROI(gen_idx) = best_roi;
                    
                    % Best Risk (minimum)
                    [best_risk, ~] = min(pareto_solutions.Risk);
                    gen_metrics.Risk(gen_idx) = best_risk;
                    
                    % Best WinRate
                    [best_winrate, ~] = max(pareto_solutions.WinRate);
                    gen_metrics.WinRate(gen_idx) = best_winrate;
                    
                    % Hypervolume
                    objectives = [pareto_solutions.ROI, pareto_solutions.Risk, pareto_solutions.WinRate];
                    hv = calculate_hypervolume(objectives, ref_point);
                    gen_metrics.Hypervolume(gen_idx) = hv;
                end
                
            catch e
                warning("Error processing generation %d in run %s: %s", gen_num, run_dirs(run_idx).name, e.message);
            end
        end
        
        % Store metrics for this run
        for m = 1:length(metrics)
            run_metrics_over_gens.(metrics{m}){run_idx} = gen_metrics.(metrics{m});
        end
        run_metrics_over_gens.generations{run_idx} = gen_nums;
        
        % Estimate execution time (simple proxy)
        run_exec_times(run_idx) = gen_nums(end) * 0.5; % Example: 0.5 sec per generation
        
        % --- Process final generation for overall results ---
        gen_to_analyze = gen_nums(end); % Use the last generation
        
        if strcmp(file_type, "archive")
            final_file_path = fullfile(run_dir, sprintf("gen_%d_archive.csv", gen_to_analyze));
        elseif strcmp(file_type, "fronts")
            final_file_path = fullfile(run_dir, sprintf("gen_%d_fronts.csv", gen_to_analyze));
        else
            final_file_path = fullfile(run_dir, sprintf("gen_%d_data.csv", gen_to_analyze));
        end
        
        try
            final_data = readtable(final_file_path);
            
            % Extract non-dominated solutions for final Pareto front
            if strcmp(algo_name, "SPEA2") && ismember("SPEA2Fitness", final_data.Properties.VariableNames)
                final_pareto = final_data(final_data.SPEA2Fitness < 1, :);
            elseif strcmp(algo_name, "NSGA-II") && ismember("FrontRank", final_data.Properties.VariableNames)
                final_pareto = final_data(final_data.FrontRank == 0, :);
            else
                final_pareto = final_data;
            end
            
            if isempty(final_pareto)
                final_pareto = final_data;
            end
            
            % Store final Pareto front objectives
            if ~isempty(final_pareto)
                results.(algo_name).ParetoFronts{run_idx} = [final_pareto.ROI, final_pareto.Risk, final_pareto.WinRate];
                
                % Store final metrics (best ROI from Pareto front)
                [best_roi_final, best_idx_final] = max(final_pareto.ROI);
                results.(algo_name).ROI(run_idx) = best_roi_final;
                results.(algo_name).Risk(run_idx) = final_pareto.Risk(best_idx_final);
                results.(algo_name).WinRate(run_idx) = final_pareto.WinRate(best_idx_final);
                
                % Store final hypervolume
                final_objectives = [final_pareto.ROI, final_pareto.Risk, final_pareto.WinRate];
                final_hv = calculate_hypervolume(final_objectives, ref_point);
                results.(algo_name).Hypervolume(run_idx) = final_hv;
            end
            
        catch e
            warning("Error processing final generation %d in run %s: %s", gen_to_analyze, run_dirs(run_idx).name, e.message);
        end
    end
    
    % --- Aggregate convergence data --- 
    common_generations = 0:max_generations_across_runs;
    convergence.(algo_name).ExecutionTime.mean = mean(run_exec_times(run_exec_times > 0));
    convergence.(algo_name).ExecutionTime.std = std(run_exec_times(run_exec_times > 0));
    
    for m = 1:length(metrics)
        metric_name = metrics{m};
        num_gens = length(common_generations);
        gen_values_matrix = NaN(num_runs, num_gens);
        
        for run_idx = 1:num_runs
            if length(run_metrics_over_gens.(metric_name)) >= run_idx && ~isempty(run_metrics_over_gens.(metric_name){run_idx})
                run_gens = run_metrics_over_gens.generations{run_idx};
                run_vals = run_metrics_over_gens.(metric_name){run_idx};
                
                % Interpolate or use NaN for missing generations
                interp_vals = interp1(run_gens, run_vals, common_generations, "linear", NaN);
                gen_values_matrix(run_idx, :) = interp_vals;
            end
        end
        
        % Calculate mean and std, ignoring NaNs
        convergence.(algo_name).(metric_name).generations = common_generations;
        convergence.(algo_name).(metric_name).mean_values = nanmean(gen_values_matrix, 1);
        convergence.(algo_name).(metric_name).std_values = nanstd(gen_values_matrix, 0, 1);
    end
end

%% Statistical Comparison (NSGA-II vs SPEA2)
fprintf("\n===== Statistical Comparison (NSGA-II vs SPEA2) =====\n");
comparison_results = struct();

for m = 1:length(metrics)
    metric_name = metrics{m};
    fprintf("\n--- %s Comparison ---\n", metric_name);
    
    data_nsga2 = results.NSGA_II.(metric_name);
    data_spea2 = results.SPEA2.(metric_name);
    
    % Remove zeros (failed runs)
    data_nsga2 = data_nsga2(data_nsga2 ~= 0);
    data_spea2 = data_spea2(data_spea2 ~= 0);
    
    if isempty(data_nsga2) || isempty(data_spea2)
        fprintf("  Insufficient data for comparison.\n");
        continue;
    end
    
    % Perform Wilcoxon rank-sum test (non-parametric)
    [p, h, stats] = ranksum(data_nsga2, data_spea2);
    
    fprintf("  Wilcoxon Rank-Sum Test p-value: %.4f\n", p);
    comparison_results.(metric_name).p_value = p;
    comparison_results.(metric_name).stats = stats;
    
    if h
        fprintf("  Statistically significant difference found (p < 0.05)\n");
        median_nsga2 = median(data_nsga2);
        median_spea2 = median(data_spea2);
        fprintf("  Median NSGA-II: %.4f\n", median_nsga2);
        fprintf("  Median SPEA2: %.4f\n", median_spea2);
        
        if strcmp(metric_name, "Risk") % Lower is better for Risk
            winner = ternary(median_nsga2 < median_spea2, "NSGA-II", "SPEA2");
        else % Higher is better for others
            winner = ternary(median_nsga2 > median_spea2, "NSGA-II", "SPEA2");
        end
        fprintf("  %s performs significantly better.\n", winner);
        comparison_results.(metric_name).winner = winner;
    else
        fprintf("  No statistically significant difference found (p >= 0.05)\n");
        comparison_results.(metric_name).winner = "None";
    end
end

%% Visualization

% 1. Side-by-side Box Plots
for m = 1:length(metrics)
    metric_name = metrics{m};
    figure("Name", sprintf("Comparison: %s", metric_name), "Position", [100, 100, 600, 600]);
    
    data_nsga2 = results.NSGA_II.(metric_name);
    data_spea2 = results.SPEA2.(metric_name);
    data_nsga2 = data_nsga2(data_nsga2 ~= 0);
    data_spea2 = data_spea2(data_spea2 ~= 0);
    
    if isempty(data_nsga2) && isempty(data_spea2)
        close(gcf);
        continue;
    end
    
    boxplot_data = [data_nsga2; data_spea2];
    boxplot_groups = [repmat({"NSGA-II"}, length(data_nsga2), 1); repmat({"SPEA2"}, length(data_spea2), 1)];
    
    boxplot(boxplot_data, boxplot_groups);
    title(sprintf("Comparison: %s", metric_name));
    ylabel(metric_name);
    grid on;
    
    % Add mean markers
    hold on;
    if ~isempty(data_nsga2)
        plot(1, mean(data_nsga2), "ro", "MarkerSize", 10, "MarkerFaceColor", "r");
    end
    if ~isempty(data_spea2)
        plot(2, mean(data_spea2), "ro", "MarkerSize", 10, "MarkerFaceColor", "r");
    end
    hold off;
    
    % Save figure
    saveas(gcf, sprintf("Comparison_%s_boxplot.png", metric_name));
    saveas(gcf, sprintf("Comparison_%s_boxplot.fig", metric_name));
end

% 2. Pareto Front Plot (Last Generation - Sample Run)
figure("Name", "Pareto Front Comparison", "Position", [100, 100, 1000, 800]);

% Select a representative run (e.g., the first one)
run_to_plot = 1;

pareto_nsga2 = results.NSGA_II.ParetoFronts{run_to_plot};
pareto_spea2 = results.SPEA2.ParetoFronts{run_to_plot};

if ~isempty(pareto_nsga2)
    scatter3(pareto_nsga2(:,1), pareto_nsga2(:,2), pareto_nsga2(:,3), 50, colors{1}, markers{1}, "filled", "DisplayName", "NSGA-II");
    hold on;
end
if ~isempty(pareto_spea2)
    scatter3(pareto_spea2(:,1), pareto_spea2(:,2), pareto_spea2(:,3), 50, colors{2}, markers{2}, "filled", "DisplayName", "SPEA2");
end

hold off;
xlabel("ROI (%)");
ylabel("Risk (-Sharpe)");
zlabel("Win Rate (%)");
title(sprintf("Pareto Front Comparison (Last Gen, Run %d)", run_to_plot));
legend("Location", "best");
grid on;
view(3); % 3D view

% Save figure
saveas(gcf, "Comparison_ParetoFront.png");
saveas(gcf, "Comparison_ParetoFront.fig");

% 3. Convergence Plots (Hypervolume and ROI)
metrics_to_plot = {"Hypervolume", "ROI"};
for m = 1:length(metrics_to_plot)
    metric_name = metrics_to_plot{m};
    figure("Name", sprintf("Convergence: %s", metric_name), "Position", [100, 100, 1000, 600]);
    hold on;
    legend_entries = {};
    
    for algo_idx = 1:length(algorithms)
        algo_name = algorithms{algo_idx}.name;
        
        if isfield(convergence.(algo_name), metric_name) && ...
           ~isempty(convergence.(algo_name).(metric_name).generations)
            
            generations = convergence.(algo_name).(metric_name).generations;
            mean_values = convergence.(algo_name).(metric_name).mean_values;
            std_values = convergence.(algo_name).(metric_name).std_values;
            
            % Plot mean with shaded error region
            fill([generations, fliplr(generations)], ...
                 [mean_values + std_values, fliplr(mean_values - std_values)], ...
                 colors{algo_idx}, "FaceAlpha", 0.2, "EdgeColor", "none");
            plot(generations, mean_values, ["-" markers{algo_idx}], ...
                "Color", colors{algo_idx}, "LineWidth", 2, "MarkerSize", 6, ...
                "MarkerFaceColor", colors{algo_idx});
            
            legend_entries{end+1} = algo_name;
        end
    end
    
    hold off;
    xlabel("Generation");
    ylabel(metric_name);
    title(sprintf("Convergence Comparison: %s", metric_name));
    legend(legend_entries, "Location", "best");
    grid on;
    xlim([0, max_generations_across_runs]); % Ensure consistent x-axis
    
    % Save figure
    saveas(gcf, sprintf("Comparison_%s_convergence.png", metric_name));
    saveas(gcf, sprintf("Comparison_%s_convergence.fig", metric_name));
end

% 4. Convergence Time Comparison
figure("Name", "Convergence Time Comparison", "Position", [100, 100, 800, 600]);

exec_times = [convergence.NSGA_II.ExecutionTime.mean, convergence.SPEA2.ExecutionTime.mean];
exec_stds = [convergence.NSGA_II.ExecutionTime.std, convergence.SPEA2.ExecutionTime.std];

bar(1:2, exec_times);
hold on;
errorbar(1:2, exec_times, exec_stds, ".k", "LineWidth", 1.5);
hold off;

set(gca, "XTick", 1:2, "XTickLabel", {"NSGA-II", "SPEA2"});
ylabel("Average Execution Time (seconds)");
title("Average Execution Time Comparison");
grid on;

% Save figure
saveas(gcf, "Comparison_ExecutionTime.png");
saveas(gcf, "Comparison_ExecutionTime.fig");

%% Generate Comparative Report
report_file = fopen("comparison_analysis_report.txt", "w");
fprintf(report_file, "NSGA-II vs SPEA2 Comparative Analysis Report (Set 5)\n");
fprintf(report_file, "====================================================\n\n");

% Summary Statistics
fprintf(report_file, "Summary Statistics (Final Generation):\n");
fprintf(report_file, "-------------------------------------\n");
for m = 1:length(metrics)
    metric_name = metrics{m};
    fprintf(report_file, "Metric: %s\n", metric_name);
    
    data_nsga2 = results.NSGA_II.(metric_name);
    data_spea2 = results.SPEA2.(metric_name);
    data_nsga2 = data_nsga2(data_nsga2 ~= 0);
    data_spea2 = data_spea2(data_spea2 ~= 0);
    
    if ~isempty(data_nsga2)
        fprintf(report_file, "  NSGA-II: Mean=%.4f, Median=%.4f, Std=%.4f\n", ...
            mean(data_nsga2), median(data_nsga2), std(data_nsga2));
    else
        fprintf(report_file, "  NSGA-II: No valid data\n");
    end
    
    if ~isempty(data_spea2)
        fprintf(report_file, "  SPEA2:   Mean=%.4f, Median=%.4f, Std=%.4f\n", ...
            mean(data_spea2), median(data_spea2), std(data_spea2));
    else
        fprintf(report_file, "  SPEA2:   No valid data\n");
    end
    fprintf(report_file, "\n");
end

% Statistical Comparison Summary
fprintf(report_file, "Statistical Comparison Summary (Wilcoxon Rank-Sum Test):\n");
fprintf(report_file, "-------------------------------------------------------\n");
for m = 1:length(metrics)
    metric_name = metrics{m};
    if isfield(comparison_results, metric_name) && isfield(comparison_results.(metric_name), "p_value")
        fprintf(report_file, "Metric: %s\n", metric_name);
        fprintf(report_file, "  p-value: %.4f %s\n", ...
            comparison_results.(metric_name).p_value, ...
            ternary(comparison_results.(metric_name).p_value < 0.05, "(Significant)", "(Not Significant)"));
        if comparison_results.(metric_name).p_value < 0.05
            fprintf(report_file, "  Winner: %s\n", comparison_results.(metric_name).winner);
        end
    else
        fprintf(report_file, "Metric: %s - Comparison not performed (insufficient data)\n", metric_name);
    end
end
fprintf(report_file, "\n");

% Convergence Summary
fprintf(report_file, "Convergence Summary:\n");
fprintf(report_file, "-------------------\n");
fprintf(report_file, "Average Execution Time:\n");
fprintf(report_file, "  NSGA-II: %.2f +/- %.2f seconds\n", convergence.NSGA_II.ExecutionTime.mean, convergence.NSGA_II.ExecutionTime.std);
fprintf(report_file, "  SPEA2:   %.2f +/- %.2f seconds\n", convergence.SPEA2.ExecutionTime.mean, convergence.SPEA2.ExecutionTime.std);
fprintf(report_file, "(Note: Execution time is estimated based on generations completed)\n\n");

% Conclusion
fprintf(report_file, "Conclusion:\n");
fprintf(report_file, "-----------\n");

% Determine overall winner based on significant wins
wins_nsga2 = 0;
wins_spea2 = 0;
for m = 1:length(metrics)
    metric_name = metrics{m};
    if isfield(comparison_results, metric_name) && isfield(comparison_results.(metric_name), "winner")
        if strcmp(comparison_results.(metric_name).winner, "NSGA-II")
            wins_nsga2 = wins_nsga2 + 1;
        elseif strcmp(comparison_results.(metric_name).winner, "SPEA2")
            wins_spea2 = wins_spea2 + 1;
        end
    end
end

if wins_nsga2 > wins_spea2
    fprintf(report_file, "NSGA-II demonstrated significantly better performance in %d out of %d metrics compared to SPEA2.\n", wins_nsga2, length(metrics));
elseif wins_spea2 > wins_nsga2
    fprintf(report_file, "SPEA2 demonstrated significantly better performance in %d out of %d metrics compared to NSGA-II.\n", wins_spea2, length(metrics));
else
    fprintf(report_file, "NSGA-II and SPEA2 showed comparable performance, with no clear overall winner based on significant differences.\n");
    fprintf(report_file, "NSGA-II won %d metrics, SPEA2 won %d metrics significantly.\n", wins_nsga2, wins_spea2);
end

fclose(report_file);
fprintf("Comparative analysis report saved to comparison_analysis_report.txt\n");

%% Helper function for hypervolume calculation (same as before)
function hv = calculate_hypervolume(points, reference)
    [n_points, n_objectives] = size(points);
    if n_points == 0, hv = 0; return; end
    normalized_points = points;
    normalized_reference = reference;
    if n_objectives >= 1, normalized_points(:, 1) = -normalized_points(:, 1); normalized_reference(1) = -normalized_reference(1); end
    if n_objectives >= 3, normalized_points(:, 3) = -normalized_points(:, 3); normalized_reference(3) = -normalized_reference(3); end
    if n_objectives == 2, hv = calculate_2d_hypervolume(normalized_points, normalized_reference);
    elseif n_objectives == 3, hv = calculate_3d_hypervolume(normalized_points, normalized_reference);
    else, hv = approximate_hypervolume(normalized_points, normalized_reference); end
end
function hv = calculate_2d_hypervolume(points, reference)
    [~, idx] = sort(points(:, 1)); sorted_points = points(idx, :); hv = 0; prev_x = reference(1);
    for i = 1:size(sorted_points, 1)
        x = sorted_points(i, 1); y = sorted_points(i, 2);
        if y < reference(2), hv = hv + (x - prev_x) * (reference(2) - y); prev_x = x; end
    end
end
function hv = calculate_3d_hypervolume(points, reference)
    [~, idx] = sort(points(:, 1)); sorted_points = points(idx, :); hv = 0;
    for i = 1:size(sorted_points, 1)
        x = sorted_points(i, 1); y = sorted_points(i, 2); z = sorted_points(i, 3);
        if i == 1, prev_x = reference(1); else, prev_x = sorted_points(i-1, 1); end
        if y < reference(2) && z < reference(3), volume = (x - prev_x) * (reference(2) - y) * (reference(3) - z); hv = hv + volume; end
    end
end
function hv = approximate_hypervolume(points, reference)
    hv = 0;
    for i = 1:size(points, 1)
        point_hv = 1; dominated = true;
        for j = 1:size(points, 2)
            if points(i, j) > reference(j), dominated = false; break; end
            point_hv = point_hv * (reference(j) - points(i, j));
        end
        if dominated, hv = hv + point_hv; end
    end
end
function result = ternary(condition, if_true, if_false)
    if condition, result = if_true; else, result = if_false; end
end
