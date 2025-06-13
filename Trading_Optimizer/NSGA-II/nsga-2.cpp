#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <tuple>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// ============= GLOBAL PARAMETERS FOR EASY TUNING =============

// Algorithm parameters
int POP_SIZE = 200;                // Population size
int GENERATIONS = 250;             // Number of generations to run
int TOURNAMENT_SIZE = 2;           // Tournament selection size
double MUTATION_RATE = 0.2;        // Probability of mutation
double CROSSOVER_RATE = 0.9;       // Probability of crossover
double SBX_ETA = 20.0;             // Distribution index for SBX crossover
double PM_ETA = 20.0;              // Distribution index for polynomial mutation

// Strategy parameter bounds
int MIN_SHORT_MA = 8;              // Minimum short moving average period
int MAX_SHORT_MA = 30;             // Maximum short moving average period
int MIN_LONG_MA = 30;              // Minimum long moving average period
int MAX_LONG_MA = 100;              // Maximum long moving average period
double MIN_THRESHOLD = 0.1;        // Minimum threshold value
double MAX_THRESHOLD = 10.0;       // Maximum threshold value

// Trading simulation parameters
double INITIAL_CAPITAL = 1000.0;   // Initial trading capital
std::string DATA_FILE = "ethereum.csv";  // Price data file
int PRICE_COLUMN = 7;              // Column index for price data (0-based)
bool SKIP_HEADER = true;           // Whether to skip the first line of CSV

// Logging parameters
bool ENABLE_LOGGING = true;        // Enable/disable logging
int LOG_FREQUENCY = 10;            // Log every N generations
bool LOG_FINAL_POPULATION = true;  // Log final population details

// Experiment parameters
int PARAM_SET_ID = 1;              // ID for the current parameter set

// ============= DATA STRUCTURES =============

struct Strategy {
    int short_ma, long_ma;
    double buy_thresh, sell_thresh, stop_loss, take_profit;
};

struct PriceData {
    double close;
};

struct Fitness {
    double roi;
    double risk;
    double win_rate;
};

struct Individual {
    Strategy strategy;
    Fitness fitness;
    int rank;
    double crowding_distance;
};

// ============= UTILITY FUNCTIONS =============

// Timer class for performance measurement
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string label;

public:
    Timer(const std::string& label_text) : label(label_text) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << label << ": " << duration << " ms" << std::endl;
    }
};

// Create directory if it doesn't exist
bool create_directory(const std::string& dir_path) {
    if (!fs::exists(dir_path)) {
        try {
            return fs::create_directory(dir_path);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory " << dir_path << ": " << e.what() << std::endl;
            return false;
        }
    }
    return true;
}

// Load price data from CSV file
std::vector<PriceData> load_price_data(const std::string& filename) {
    Timer timer("Data loading");
    std::vector<PriceData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    if (SKIP_HEADER && std::getline(file, line)) {
        // Skip header line
    }

    // Detect delimiter from first data line
    char delimiter = ',';  // Default delimiter
    if (std::getline(file, line)) {
        if (line.find('\t') != std::string::npos) {
            delimiter = '\t';
        }
        
        // Process the first line
        try {
            std::stringstream ss(line);
            std::string token;
            int col_idx = 0;
            
            while (std::getline(ss, token, delimiter)) {
                if (col_idx == PRICE_COLUMN) {
                    data.push_back({std::stod(token)});
                    break;
                }
                col_idx++;
            }
        } catch (...) {
            // Skip malformed line
        }
    }
    
    // Process remaining lines
    while (std::getline(file, line)) {
        try {
            std::stringstream ss(line);
            std::string token;
            int col_idx = 0;
            
            while (std::getline(ss, token, delimiter)) {
                if (col_idx == PRICE_COLUMN) {
                    data.push_back({std::stod(token)});
                    break;
                }
                col_idx++;
            }
        } catch (...) {
            // Skip malformed line
        }
    }

    return data;
}

// Calculate moving average efficiently using a sliding window
double moving_average(const std::vector<PriceData>& data, int index, int window) {
    // Handle edge case: not enough data for full window
    if (index < window - 1) {
        double sum = 0.0;
        for (int i = 0; i <= index; ++i) {
            sum += data[i].close;
        }
        return sum / (index + 1);
    }
    
    // Normal case: full window available
    double sum = 0.0;
    for (int i = index - window + 1; i <= index; ++i) {
        sum += data[i].close;
    }
    return sum / window;
}

// Normalize a value between min and max to [0,1]
double normalize(double x, double min_val, double max_val) {
    if (std::abs(max_val - min_val) < 1e-10) return 0.5;  // Avoid division by zero
    return (x - min_val) / (max_val - min_val);
}

// Denormalize a value from [0,1] to range [min,max]
double denormalize(double x_norm, double min_val, double max_val) {
    return x_norm * (max_val - min_val) + min_val;
}

// ============= EVALUATION FUNCTIONS =============

// Evaluate a trading strategy on historical price data
Fitness evaluate_strategy(const Strategy& s, const std::vector<PriceData>& data) {
    bool position_open = false;
    double entry_price = 0.0;
    double capital = INITIAL_CAPITAL;
    std::vector<double> trade_returns;
    int wins = 0, trades = 0;
    double total_profit = 0.0, total_loss = 0.0;
    
    // Pre-calculate moving averages for efficiency
    std::vector<double> short_mas(data.size());
    std::vector<double> long_mas(data.size());
    
    // Calculate short MA
    double short_sum = 0;
    for (int i = 0; i < data.size(); ++i) {
        short_sum += data[i].close;
        if (i >= s.short_ma) short_sum -= data[i - s.short_ma].close;
        short_mas[i] = (i < s.short_ma - 1) ? data[i].close : short_sum / s.short_ma;
    }
    
    // Calculate long MA
    double long_sum = 0;
    for (int i = 0; i < data.size(); ++i) {
        long_sum += data[i].close;
        if (i >= s.long_ma) long_sum -= data[i - s.long_ma].close;
        long_mas[i] = (i < s.long_ma - 1) ? data[i].close : long_sum / s.long_ma;
    }
    
    // Simulate trading
    for (int i = s.long_ma; i < data.size(); ++i) {
        double diff = (short_mas[i] - long_mas[i]) / long_mas[i] * 100;

        if (!position_open && diff > s.buy_thresh) {
            entry_price = data[i].close;
            position_open = true;
        }

        if (position_open) {
            double change = (data[i].close - entry_price) / entry_price * 100;
            if (change > s.take_profit || change < -s.stop_loss || diff < -s.sell_thresh) {
                capital *= (1 + change / 100.0);
                trade_returns.push_back(change);
                if (change > 0) {
                    ++wins;
                    total_profit += change;
                } else {
                    total_loss += std::abs(change);
                }
                ++trades;
                position_open = false;
            }
        }
    }

    // Calculate metrics
    double avg_return = 0.0;
    double stddev = 0.0;
    
    if (!trade_returns.empty()) {
        // Calculate average return
        for (double r : trade_returns) avg_return += r;
        avg_return /= trade_returns.size();
        
        // Calculate standard deviation
        for (double r : trade_returns) stddev += (r - avg_return) * (r - avg_return);
        stddev = sqrt(stddev / trade_returns.size());
    }
    
    // Sharpe ratio (assuming risk-free rate of 0)
    double sharpe_ratio = stddev > 0 ? avg_return / stddev : 0.0;
    
    // Calculate win rate percentage
    double win_rate = trades > 0 ? (wins * 100.0) / trades : 0.0;
    
    // Calculate profit factor with safeguards
    double profit_factor;
    if (total_loss > 0) {
        profit_factor = total_profit / total_loss;
    } else if (total_profit > 0) {
        profit_factor = 2.0; // If no losses but has profits, give a good profit factor
    } else {
        profit_factor = 1.0; // If no trades or no profits/losses, neutral profit factor
    }
    
    // Ensure profit factor is within reasonable bounds
    profit_factor = std::clamp(profit_factor, 0.1, 10.0);
    
    // Calculate weighted win rate with safeguards
    double weighted_win_rate;
    if (trades > 0) {
        // Base win rate component (50% weight)
        double base_component = win_rate * 0.5;
        
        // Profit factor component (50% weight)
        double profit_component = (profit_factor / (1.0 + profit_factor)) * 100.0 * 0.5;
        
        weighted_win_rate = base_component + profit_component;
    } else {
        weighted_win_rate = 0.0;
    }

    return { 
        (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0,  // ROI
        -sharpe_ratio,                                          // Risk (negative because we minimize it)
        weighted_win_rate                                       // Weighted win rate
    };
}

// ============= GENETIC OPERATORS =============

// Generate a random trading strategy
Strategy random_strategy(std::default_random_engine& rng) {
    std::uniform_int_distribution<int> ma_short(MIN_SHORT_MA, MAX_SHORT_MA);
    std::uniform_int_distribution<int> ma_long(MIN_LONG_MA, MAX_LONG_MA);
    std::uniform_real_distribution<double> thresh_dist(MIN_THRESHOLD, MAX_THRESHOLD);
    
    Strategy s;
    do {
        s.short_ma = ma_short(rng);
        s.long_ma = ma_long(rng);
    } while (s.short_ma >= s.long_ma);
    
    s.buy_thresh = thresh_dist(rng);
    s.sell_thresh = thresh_dist(rng);
    s.stop_loss = thresh_dist(rng);
    s.take_profit = thresh_dist(rng);
    
    return s;
}

// SBX crossover for normalized values
double sbx_crossover(double parent1, double parent2, std::default_random_engine& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    
    // Calculate beta value for SBX
    double beta;
    if (u <= 0.5) {
        beta = pow(2.0 * u, 1.0 / (SBX_ETA + 1.0));
    } else {
        beta = pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (SBX_ETA + 1.0));
    }
    
    // Calculate child value
    double child = 0.5 * ((parent1 + parent2) + beta * (parent1 - parent2));
    
    // Ensure child is within [0,1]
    return std::clamp(child, 0.0, 1.0);
}

// Polynomial mutation for normalized values
double polynomial_mutation(double x, std::default_random_engine& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    
    // Calculate delta value for polynomial mutation
    double delta;
    if (u < 0.5) {
        delta = pow(2.0 * u, 1.0 / (PM_ETA + 1.0)) - 1.0;
    } else {
        delta = 1.0 - pow(2.0 * (1.0 - u), 1.0 / (PM_ETA + 1.0));
    }
    
    // Calculate mutated value
    double mutated = x + delta;
    
    // Ensure mutated value is within [0,1]
    return std::clamp(mutated, 0.0, 1.0);
}

// Perform crossover between two parent strategies
Strategy crossover(const Strategy& p1, const Strategy& p2, std::default_random_engine& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Skip crossover with probability (1 - CROSSOVER_RATE)
    if (dist(rng) > CROSSOVER_RATE) {
        return dist(rng) < 0.5 ? p1 : p2;
    }
    
    Strategy child;
    
    // Crossover for short_ma
    {
        double p1_norm = normalize(p1.short_ma, MIN_SHORT_MA, MAX_SHORT_MA);
        double p2_norm = normalize(p2.short_ma, MIN_SHORT_MA, MAX_SHORT_MA);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.short_ma = static_cast<int>(std::round(denormalize(child_norm, MIN_SHORT_MA, MAX_SHORT_MA)));
    }
    
    // Crossover for long_ma (ensuring long_ma > short_ma)
    {
        int min_long = std::max(MIN_LONG_MA, child.short_ma + 1);
        double p1_norm = normalize(p1.long_ma, min_long, MAX_LONG_MA);
        double p2_norm = normalize(p2.long_ma, min_long, MAX_LONG_MA);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.long_ma = static_cast<int>(std::round(denormalize(child_norm, min_long, MAX_LONG_MA)));
    }
    
    // Crossover for thresholds
    {
        double p1_norm = normalize(p1.buy_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        double p2_norm = normalize(p2.buy_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.buy_thresh = denormalize(child_norm, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    {
        double p1_norm = normalize(p1.sell_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        double p2_norm = normalize(p2.sell_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.sell_thresh = denormalize(child_norm, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    {
        double p1_norm = normalize(p1.stop_loss, MIN_THRESHOLD, MAX_THRESHOLD);
        double p2_norm = normalize(p2.stop_loss, MIN_THRESHOLD, MAX_THRESHOLD);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.stop_loss = denormalize(child_norm, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    {
        double p1_norm = normalize(p1.take_profit, MIN_THRESHOLD, MAX_THRESHOLD);
        double p2_norm = normalize(p2.take_profit, MIN_THRESHOLD, MAX_THRESHOLD);
        double child_norm = sbx_crossover(p1_norm, p2_norm, rng);
        child.take_profit = denormalize(child_norm, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    return child;
}

// Mutate a strategy
Strategy mutate(const Strategy& s, std::default_random_engine& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Strategy mutated = s;
    
    // Apply mutation with probability MUTATION_RATE for each parameter
    if (dist(rng) < MUTATION_RATE) {
        double norm_val = normalize(s.short_ma, MIN_SHORT_MA, MAX_SHORT_MA);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.short_ma = static_cast<int>(std::round(denormalize(norm_val, MIN_SHORT_MA, MAX_SHORT_MA)));
    }
    
    if (dist(rng) < MUTATION_RATE) {
        int min_long = std::max(MIN_LONG_MA, mutated.short_ma + 1);
        double norm_val = normalize(s.long_ma, min_long, MAX_LONG_MA);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.long_ma = static_cast<int>(std::round(denormalize(norm_val, min_long, MAX_LONG_MA)));
    }
    
    if (dist(rng) < MUTATION_RATE) {
        double norm_val = normalize(s.buy_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.buy_thresh = denormalize(norm_val, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    if (dist(rng) < MUTATION_RATE) {
        double norm_val = normalize(s.sell_thresh, MIN_THRESHOLD, MAX_THRESHOLD);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.sell_thresh = denormalize(norm_val, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    if (dist(rng) < MUTATION_RATE) {
        double norm_val = normalize(s.stop_loss, MIN_THRESHOLD, MAX_THRESHOLD);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.stop_loss = denormalize(norm_val, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    if (dist(rng) < MUTATION_RATE) {
        double norm_val = normalize(s.take_profit, MIN_THRESHOLD, MAX_THRESHOLD);
        norm_val = polynomial_mutation(norm_val, rng);
        mutated.take_profit = denormalize(norm_val, MIN_THRESHOLD, MAX_THRESHOLD);
    }
    
    return mutated;
}

// ============= NSGA-II FUNCTIONS =============

// Check if solution a dominates solution b
bool dominates(const Fitness& a, const Fitness& b) {
    return (a.roi >= b.roi && a.risk <= b.risk && a.win_rate >= b.win_rate) &&
           (a.roi > b.roi || a.risk < b.risk || a.win_rate > b.win_rate);
}

// Fast non-dominated sort to identify Pareto fronts
std::vector<std::vector<Individual>> fast_non_dominated_sort(const std::vector<Individual>& pop) {
    Timer timer("Non-dominated sorting");
    int n = pop.size();
    std::vector<std::vector<int>> dominated_by(n);
    std::vector<int> domination_count(n, 0);
    std::vector<int> current_front;
    
    // Calculate domination relationships
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dominates(pop[i].fitness, pop[j].fitness)) {
                dominated_by[i].push_back(j);
                domination_count[j]++;
            } else if (dominates(pop[j].fitness, pop[i].fitness)) {
                dominated_by[j].push_back(i);
                domination_count[i]++;
            }
        }
        
        // Add individuals with no domination to first front
        if (domination_count[i] == 0) {
            current_front.push_back(i);
        }
    }
    
    // Generate all fronts
    std::vector<std::vector<Individual>> fronts;
    int front_index = 0;
    
    while (!current_front.empty()) {
        fronts.push_back(std::vector<Individual>());
        std::vector<int> next_front;
        
        for (int i : current_front) {
            Individual ind = pop[i];
            ind.rank = front_index;
            fronts.back().push_back(ind);
            
            // Update domination counts for dominated solutions
            for (int j : dominated_by[i]) {
                domination_count[j]--;
                if (domination_count[j] == 0) {
                    next_front.push_back(j);
                }
            }
        }
        
        front_index++;
        current_front = next_front;
    }
    
    return fronts;
}

// Calculate crowding distance for individuals in a front
void assign_crowding_distance(std::vector<Individual>& front) {
    int n = front.size();
    if (n <= 1) {
        if (n == 1) front[0].crowding_distance = std::numeric_limits<double>::infinity();
        return;
    }
    
    // Initialize crowding distances
    for (auto& ind : front) {
        ind.crowding_distance = 0;
    }
    
    // For each objective
    for (int obj = 0; obj < 3; ++obj) {
        // Sort by objective value
        std::sort(front.begin(), front.end(), [obj](const Individual& a, const Individual& b) {
            switch (obj) {
                case 0: return a.fitness.roi < b.fitness.roi;
                case 1: return a.fitness.risk < b.fitness.risk;
                case 2: return a.fitness.win_rate < b.fitness.win_rate;
                default: return false;
            }
        });
        
        // Set boundary points to infinity
        front.front().crowding_distance = std::numeric_limits<double>::infinity();
        front.back().crowding_distance = std::numeric_limits<double>::infinity();
        
        // Calculate objective range for normalization
        double obj_min, obj_max;
        switch (obj) {
            case 0:
                obj_min = front.front().fitness.roi;
                obj_max = front.back().fitness.roi;
                break;
            case 1:
                obj_min = front.front().fitness.risk;
                obj_max = front.back().fitness.risk;
                break;
            case 2:
                obj_min = front.front().fitness.win_rate;
                obj_max = front.back().fitness.win_rate;
                break;
        }
        
        // Skip if all values are the same
        if (std::abs(obj_max - obj_min) < 1e-10) continue;
        
        // Calculate crowding distance for intermediate points
        for (int i = 1; i < n - 1; ++i) {
            double prev_val, next_val;
            switch (obj) {
                case 0:
                    prev_val = front[i-1].fitness.roi;
                    next_val = front[i+1].fitness.roi;
                    break;
                case 1:
                    prev_val = front[i-1].fitness.risk;
                    next_val = front[i+1].fitness.risk;
                    break;
                case 2:
                    prev_val = front[i-1].fitness.win_rate;
                    next_val = front[i+1].fitness.win_rate;
                    break;
            }
            
            // Add normalized distance to crowding measure
            front[i].crowding_distance += (next_val - prev_val) / (obj_max - obj_min);
        }
    }
}

// Tournament selection based on rank and crowding distance
Individual tournament_selection(const std::vector<Individual>& pop, std::default_random_engine& rng) {
    std::uniform_int_distribution<int> dist(0, pop.size() - 1);
    
    // Select TOURNAMENT_SIZE random individuals
    std::vector<const Individual*> contestants;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        contestants.push_back(&pop[dist(rng)]);
    }
    
    // Find the best individual in the tournament
    const Individual* best = contestants[0];
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        // Compare based on rank first, then crowding distance
        if (contestants[i]->rank < best->rank || 
            (contestants[i]->rank == best->rank && 
             contestants[i]->crowding_distance > best->crowding_distance)) {
            best = contestants[i];
        }
    }
    
    return *best;
}

// ============= LOGGING FUNCTIONS =============

// Log all data to CSV files with unique run ID
void log_all_data_to_csv(const std::vector<Individual>& population, const std::vector<std::vector<Individual>>& fronts, int generation) {
    if (!ENABLE_LOGGING) return;
    
    // Create a unique run ID based on timestamp if not already created
    static std::string run_id = "";
    if (run_id.empty()) {
        std::time_t now = std::time(nullptr);
        char timestamp[20];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
        run_id = timestamp;
    }
    
    // Create directory structure for parameter set and run
    std::string param_dir = "param_set_" + std::to_string(PARAM_SET_ID);
    std::string run_dir = param_dir + "/run_" + run_id;
    
    // Create directories if they don't exist
    if (!fs::exists(param_dir)) {
        create_directory(param_dir);
    }
    if (!fs::exists(run_dir)) {
        create_directory(run_dir);
    }
    
    // Create generation-specific log file in the run directory
    std::string filename = run_dir + "/gen_" + std::to_string(generation) + "_data.csv";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Generation,IndividualID,FrontRank,ShortMA,LongMA,BuyThresh,SellThresh,StopLoss,TakeProfit,ROI,Risk,WinRate,CrowdingDistance" << std::endl;
    
    // Write data for each individual in the population
    for (size_t i = 0; i < population.size(); ++i) {
        const auto& ind = population[i];
        file << std::fixed << std::setprecision(6)
             << generation << ","
             << i << ","
             << ind.rank << ","
             << ind.strategy.short_ma << ","
             << ind.strategy.long_ma << ","
             << ind.strategy.buy_thresh << ","
             << ind.strategy.sell_thresh << ","
             << ind.strategy.stop_loss << ","
             << ind.strategy.take_profit << ","
             << ind.fitness.roi << ","
             << ind.fitness.risk << ","
             << ind.fitness.win_rate << ","
             << ind.crowding_distance << std::endl;
    }
    
    file.close();
    
    // Log Pareto fronts separately for easier visualization
    std::string fronts_filename = run_dir + "/gen_" + std::to_string(generation) + "_fronts.csv";
    std::ofstream fronts_file(fronts_filename);
    
    if (!fronts_file.is_open()) {
        std::cerr << "Error: Could not open fronts log file " << fronts_filename << std::endl;
        return;
    }
    
    // Write header for fronts file
    fronts_file << "Generation,FrontIndex,IndividualIndex,ROI,Risk,WinRate" << std::endl;
    
    // Write data for each front
    for (size_t i = 0; i < fronts.size(); ++i) {
        for (size_t j = 0; j < fronts[i].size(); ++j) {
            const auto& ind = fronts[i][j];
            fronts_file << std::fixed << std::setprecision(6)
                       << generation << ","
                       << i << ","
                       << j << ","
                       << ind.fitness.roi << ","
                       << ind.fitness.risk << ","
                       << ind.fitness.win_rate << std::endl;
        }
    }
    
    fronts_file.close();
    
    // Also save parameter settings for this run
    if (generation == 0) {
        std::string params_filename = run_dir + "/parameters.csv";
        std::ofstream params_file(params_filename);
        
        if (params_file.is_open()) {
            params_file << "Parameter,Value" << std::endl;
            params_file << "PARAM_SET_ID," << PARAM_SET_ID << std::endl;
            params_file << "POP_SIZE," << POP_SIZE << std::endl;
            params_file << "GENERATIONS," << GENERATIONS << std::endl;
            params_file << "TOURNAMENT_SIZE," << TOURNAMENT_SIZE << std::endl;
            params_file << "MUTATION_RATE," << MUTATION_RATE << std::endl;
            params_file << "CROSSOVER_RATE," << CROSSOVER_RATE << std::endl;
            params_file << "SBX_ETA," << SBX_ETA << std::endl;
            params_file << "PM_ETA," << PM_ETA << std::endl;
            params_file << "MIN_SHORT_MA," << MIN_SHORT_MA << std::endl;
            params_file << "MAX_SHORT_MA," << MAX_SHORT_MA << std::endl;
            params_file << "MIN_LONG_MA," << MIN_LONG_MA << std::endl;
            params_file << "MAX_LONG_MA," << MAX_LONG_MA << std::endl;
            params_file << "MIN_THRESHOLD," << MIN_THRESHOLD << std::endl;
            params_file << "MAX_THRESHOLD," << MAX_THRESHOLD << std::endl;
            params_file.close();
        }
    }
    
    std::cout << "Logged generation " << generation << " data to " << filename << " and " << fronts_filename << std::endl;
}

// ============= MAIN FUNCTION =============

int main() {
    // Initialize random number generator with time-based seed
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    
    // Load price data
    std::vector<PriceData> data = load_price_data(DATA_FILE);
    std::cout << "Loaded " << data.size() << " price entries from file." << std::endl;

    // Check if we have enough data
    if (data.size() < MAX_LONG_MA * 2) {
        std::cerr << "Error: Not enough data loaded. Need at least " << (MAX_LONG_MA * 2) 
                  << " entries, but only found " << data.size() << "." << std::endl;
        return 1;
    }

    // Initialize population
    Timer init_timer("Population initialization");
    std::vector<Individual> population;
    for (int i = 0; i < POP_SIZE; ++i) {
        Strategy s = random_strategy(rng);
        Fitness f = evaluate_strategy(s, data);
        population.push_back({s, f, 0, 0});
    }
    init_timer.~Timer();  // Force timer to display elapsed time

    // Log initial population if enabled
    if (ENABLE_LOGGING) {
        auto fronts = fast_non_dominated_sort(population);
        for (auto& front : fronts) {
            assign_crowding_distance(front);
        }
        log_all_data_to_csv(population, fronts, 0);
    }

    // Main evolutionary loop
    for (int gen = 0; gen < GENERATIONS; ++gen) {
        Timer gen_timer("Generation " + std::to_string(gen));
        
        // Create offspring population
        std::vector<Individual> offspring;
        offspring.reserve(POP_SIZE);
        
        while (offspring.size() < POP_SIZE) {
            // Select parents
            Individual p1 = tournament_selection(population, rng);
            Individual p2 = tournament_selection(population, rng);
            
            // Create and evaluate child
            Strategy child = crossover(p1.strategy, p2.strategy, rng);
            child = mutate(child, rng);
            Fitness f = evaluate_strategy(child, data);
            
            offspring.push_back({child, f, 0, 0});
        }

        // Combine parent and offspring populations
        std::vector<Individual> combined_pop = population;
        combined_pop.insert(combined_pop.end(), offspring.begin(), offspring.end());

        // Non-dominated sort
        auto fronts = fast_non_dominated_sort(combined_pop);

        // Select next generation
        population.clear();
        population.reserve(POP_SIZE);
        
        int front_idx = 0;
        while (population.size() + fronts[front_idx].size() <= POP_SIZE) {
            assign_crowding_distance(fronts[front_idx]);
            population.insert(population.end(), fronts[front_idx].begin(), fronts[front_idx].end());
            front_idx++;
            if (front_idx >= fronts.size()) break;
        }

        // Add remaining individuals from the last front based on crowding distance
        if (population.size() < POP_SIZE && front_idx < fronts.size()) {
            assign_crowding_distance(fronts[front_idx]);
            std::sort(fronts[front_idx].begin(), fronts[front_idx].end(),
                     [](const Individual& a, const Individual& b) {
                         return a.crowding_distance > b.crowding_distance;
                     });
            int remaining = POP_SIZE - population.size();
            population.insert(population.end(),
                            fronts[front_idx].begin(),
                            fronts[front_idx].begin() + remaining);
        }

        // Log population data for MATLAB visualization if enabled and at the specified frequency
        if (ENABLE_LOGGING && (gen % LOG_FREQUENCY == 0 || gen == GENERATIONS - 1)) {
            log_all_data_to_csv(population, fronts, gen + 1);
        }

        // Print generation statistics
        std::cout << "=== Generation " << gen << " ===" << std::endl;
        
        // Find best individual for each objective
        Individual best_roi = *std::max_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness.roi < b.fitness.roi; });
        
        Individual best_risk = *std::min_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness.risk < b.fitness.risk; });
        
        Individual best_win_rate = *std::max_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness.win_rate < b.fitness.win_rate; });
        
        std::cout << "Best ROI: " << best_roi.fitness.roi << "% (Risk: " << best_roi.fitness.risk
                  << ", Win Rate: " << best_roi.fitness.win_rate << "%)" << std::endl;
        
        std::cout << "Best Risk: " << best_risk.fitness.risk << " (ROI: " << best_risk.fitness.roi
                  << "%, Win Rate: " << best_risk.fitness.win_rate << "%)" << std::endl;
        
        std::cout << "Best Win Rate: " << best_win_rate.fitness.win_rate << "% (ROI: " << best_win_rate.fitness.roi
                  << "%, Risk: " << best_win_rate.fitness.risk << ")" << std::endl;
    }

    // Print final results
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Top 5 strategies by ROI:" << std::endl;
    
    // Sort by ROI
    std::sort(population.begin(), population.end(),
             [](const Individual& a, const Individual& b) {
                 return a.fitness.roi > b.fitness.roi;
             });
    
    // Print top 5
    for (int i = 0; i < std::min(5, (int)population.size()); ++i) {
        const auto& ind = population[i];
        std::cout << i+1 << ". ROI: " << ind.fitness.roi << "%, Risk: " << ind.fitness.risk
                  << ", Win Rate: " << ind.fitness.win_rate << "%" << std::endl;
        
        std::cout << "   Strategy: short_ma=" << ind.strategy.short_ma
                  << ", long_ma=" << ind.strategy.long_ma
                  << ", buy_thresh=" << ind.strategy.buy_thresh
                  << ", sell_thresh=" << ind.strategy.sell_thresh
                  << ", stop_loss=" << ind.strategy.stop_loss
                  << ", take_profit=" << ind.strategy.take_profit << std::endl;
    }

    return 0;
}
