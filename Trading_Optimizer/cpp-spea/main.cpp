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
int ARCHIVE_SIZE = 50;            // Archive size
int GENERATIONS = 300;             // Number of generations to run
int K_NEAREST = 1;                 // k value for k-th nearest neighbor density estimation
double MUTATION_RATE = 0.1;        // Probability of mutation
double CROSSOVER_RATE = 1.0;       // Probability of crossover
double SBX_ETA = 30.0;             // Distribution index for SBX crossover
double PM_ETA = 15.0;              // Distribution index for polynomial mutation

// Strategy parameter bounds
int MIN_SHORT_MA = 8;              // Minimum short moving average period
int MAX_SHORT_MA = 30;             // Maximum short moving average period
int MIN_LONG_MA = 30;              // Minimum long moving average period
int MAX_LONG_MA = 100;             // Maximum long moving average period
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
int PARAM_SET_ID = 5;              // ID for the current parameter set

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
    double strength;       // Number of individuals this individual dominates
    double raw_fitness;    // Sum of strengths of individuals that dominate this one
    double density;        // Density estimation value
    double spea2_fitness;  // Final SPEA2 fitness value (raw_fitness + density)
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

// ============= SPEA2 SPECIFIC FUNCTIONS =============

// Check if individual i dominates individual j
bool dominates(const Individual& i, const Individual& j) {
    // For ROI: higher is better (maximize)
    // For Risk: lower (more negative) is better (minimize)
    // For Win Rate: higher is better (maximize)
    
    bool at_least_one_better = false;
    
    // Check if i is better than or equal to j in all objectives
    if (i.fitness.roi < j.fitness.roi) return false;
    if (i.fitness.risk > j.fitness.risk) return false;
    if (i.fitness.win_rate < j.fitness.win_rate) return false;
    
    // Check if i is strictly better than j in at least one objective
    if (i.fitness.roi > j.fitness.roi) at_least_one_better = true;
    if (i.fitness.risk < j.fitness.risk) at_least_one_better = true;
    if (i.fitness.win_rate > j.fitness.win_rate) at_least_one_better = true;
    
    return at_least_one_better;
}

// Calculate Euclidean distance between two individuals in objective space
double calculate_distance(const Individual& i, const Individual& j) {
    // Normalize objectives to [0,1] for fair distance calculation
    // Assuming reasonable ranges for objectives based on typical values
    double roi_range = 2000.0;  // Typical ROI range: 0% to 2000%
    double risk_range = 1.0;    // Typical Risk range: -1.0 to 0.0
    double win_rate_range = 100.0;  // Typical Win Rate range: 0% to 100%
    
    double roi_i_norm = i.fitness.roi / roi_range;
    double roi_j_norm = j.fitness.roi / roi_range;
    
    double risk_i_norm = (i.fitness.risk - (-1.0)) / risk_range;
    double risk_j_norm = (j.fitness.risk - (-1.0)) / risk_range;
    
    double win_rate_i_norm = i.fitness.win_rate / win_rate_range;
    double win_rate_j_norm = j.fitness.win_rate / win_rate_range;
    
    double roi_diff = roi_i_norm - roi_j_norm;
    double risk_diff = risk_i_norm - risk_j_norm;
    double win_rate_diff = win_rate_i_norm - win_rate_j_norm;
    
    return sqrt(roi_diff * roi_diff + risk_diff * risk_diff + win_rate_diff * win_rate_diff);
}

// Calculate strength values for all individuals
void calculate_strength(std::vector<Individual>& population, std::vector<Individual>& archive) {
    std::vector<Individual*> all_individuals;
    for (auto& ind : population) {
        all_individuals.push_back(&ind);
    }
    for (auto& ind : archive) {
        all_individuals.push_back(&ind);
    }
    
    // Calculate strength for each individual
    for (auto ind_ptr : all_individuals) {
        ind_ptr->strength = 0;
        for (auto other_ptr : all_individuals) {
            if (dominates(*ind_ptr, *other_ptr)) {
                ind_ptr->strength += 1.0;
            }
        }
    }
    
    // Calculate raw fitness for each individual
    for (auto ind_ptr : all_individuals) {
        ind_ptr->raw_fitness = 0;
        for (auto other_ptr : all_individuals) {
            if (dominates(*other_ptr, *ind_ptr)) {
                ind_ptr->raw_fitness += other_ptr->strength;
            }
        }
    }
}

// Calculate density values for all individuals
void calculate_density(std::vector<Individual>& population, std::vector<Individual>& archive) {
    std::vector<Individual*> all_individuals;
    for (auto& ind : population) {
        all_individuals.push_back(&ind);
    }
    for (auto& ind : archive) {
        all_individuals.push_back(&ind);
    }
    
    int total_size = all_individuals.size();
    
    // Calculate distances for each individual
    for (auto ind_ptr : all_individuals) {
        std::vector<double> distances;
        for (auto other_ptr : all_individuals) {
            if (ind_ptr != other_ptr) {
                distances.push_back(calculate_distance(*ind_ptr, *other_ptr));
            }
        }
        
        // Sort distances
        std::sort(distances.begin(), distances.end());
        
        // Get k-th nearest neighbor distance
        int k = std::min(K_NEAREST, static_cast<int>(distances.size()));
        double k_distance = (k > 0 && k <= distances.size()) ? distances[k-1] : 0.0;
        
        // Calculate density as 1 / (k_distance + 2) to avoid division by zero
        ind_ptr->density = 1.0 / (k_distance + 2.0);
    }
    
    // Calculate final SPEA2 fitness
    for (auto ind_ptr : all_individuals) {
        ind_ptr->spea2_fitness = ind_ptr->raw_fitness + ind_ptr->density;
    }
}

// Environmental selection for SPEA2
void environmental_selection(std::vector<Individual>& population, std::vector<Individual>& archive) {
    // Calculate strength and density
    calculate_strength(population, archive);
    calculate_density(population, archive);
    
    // Create new archive with all non-dominated individuals from population and old archive
    std::vector<Individual> new_archive;
    
    // Add all individuals with fitness < 1 (non-dominated)
    for (const auto& ind : population) {
        if (ind.spea2_fitness < 1.0) {
            new_archive.push_back(ind);
        }
    }
    
    for (const auto& ind : archive) {
        if (ind.spea2_fitness < 1.0) {
            new_archive.push_back(ind);
        }
    }
    
    // If archive is too small, add dominated individuals
    if (new_archive.size() < ARCHIVE_SIZE) {
        // Sort population and old archive by fitness
        std::vector<Individual> remaining;
        for (const auto& ind : population) {
            if (ind.spea2_fitness >= 1.0) {
                remaining.push_back(ind);
            }
        }
        
        for (const auto& ind : archive) {
            if (ind.spea2_fitness >= 1.0) {
                remaining.push_back(ind);
            }
        }
        
        // Sort by fitness (lower is better)
        std::sort(remaining.begin(), remaining.end(), 
                 [](const Individual& a, const Individual& b) {
                     return a.spea2_fitness < b.spea2_fitness;
                 });
        
        // Fill archive with best dominated individuals
        while (new_archive.size() < ARCHIVE_SIZE && !remaining.empty()) {
            new_archive.push_back(remaining.front());
            remaining.erase(remaining.begin());
        }
    }
    // If archive is too large, truncate using distance-based method
    else if (new_archive.size() > ARCHIVE_SIZE) {
        while (new_archive.size() > ARCHIVE_SIZE) {
            // Find the individual with the minimum distance to another individual
            double min_distance = std::numeric_limits<double>::max();
            int min_idx = -1;
            
            for (int i = 0; i < new_archive.size(); ++i) {
                std::vector<double> distances;
                for (int j = 0; j < new_archive.size(); ++j) {
                    if (i != j) {
                        distances.push_back(calculate_distance(new_archive[i], new_archive[j]));
                    }
                }
                
                // Sort distances
                std::sort(distances.begin(), distances.end());
                
                // Get the smallest distance
                double smallest_dist = distances.empty() ? std::numeric_limits<double>::max() : distances[0];
                
                // If this individual has the smallest distance so far, mark it for removal
                if (smallest_dist < min_distance) {
                    min_distance = smallest_dist;
                    min_idx = i;
                }
                else if (std::abs(smallest_dist - min_distance) < 1e-10) {
                    // If distances are equal, compare the second smallest
                    int k = 1;
                    while (k < distances.size() && k < new_archive[min_idx].spea2_fitness) {
                        double dist_i = k < distances.size() ? distances[k] : std::numeric_limits<double>::max();
                        
                        std::vector<double> min_distances;
                        for (int j = 0; j < new_archive.size(); ++j) {
                            if (min_idx != j) {
                                min_distances.push_back(calculate_distance(new_archive[min_idx], new_archive[j]));
                            }
                        }
                        std::sort(min_distances.begin(), min_distances.end());
                        double dist_min = k < min_distances.size() ? min_distances[k] : std::numeric_limits<double>::max();
                        
                        if (dist_i < dist_min) {
                            min_idx = i;
                            break;
                        }
                        else if (dist_i > dist_min) {
                            break;
                        }
                        
                        ++k;
                    }
                }
            }
            
            // Remove the individual with the minimum distance
            if (min_idx >= 0) {
                new_archive.erase(new_archive.begin() + min_idx);
            }
            else {
                // Fallback: remove the last individual
                new_archive.pop_back();
            }
        }
    }
    
    // Update archive
    archive = new_archive;
}

// Binary tournament selection for SPEA2
Individual tournament_selection(const std::vector<Individual>& archive, std::default_random_engine& rng) {
    std::uniform_int_distribution<int> dist(0, archive.size() - 1);
    
    int idx1 = dist(rng);
    int idx2 = dist(rng);
    
    // Lower fitness is better in SPEA2
    if (archive[idx1].spea2_fitness < archive[idx2].spea2_fitness) {
        return archive[idx1];
    } else {
        return archive[idx2];
    }
}

// ============= LOGGING FUNCTIONS =============

// Log population data to CSV
void log_population(const std::vector<Individual>& population, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "ShortMA,LongMA,BuyThresh,SellThresh,StopLoss,TakeProfit,ROI,Risk,WinRate,Strength,RawFitness,Density,SPEA2Fitness" << std::endl;
    
    // Write data
    for (const auto& ind : population) {
        file << ind.strategy.short_ma << ","
             << ind.strategy.long_ma << ","
             << ind.strategy.buy_thresh << ","
             << ind.strategy.sell_thresh << ","
             << ind.strategy.stop_loss << ","
             << ind.strategy.take_profit << ","
             << ind.fitness.roi << ","
             << ind.fitness.risk << ","
             << ind.fitness.win_rate << ","
             << ind.strength << ","
             << ind.raw_fitness << ","
             << ind.density << ","
             << ind.spea2_fitness << std::endl;
    }
    
    file.close();
}

// Log archive data to CSV
void log_archive(const std::vector<Individual>& archive, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "ShortMA,LongMA,BuyThresh,SellThresh,StopLoss,TakeProfit,ROI,Risk,WinRate,Strength,RawFitness,Density,SPEA2Fitness" << std::endl;
    
    // Write data
    for (const auto& ind : archive) {
        file << ind.strategy.short_ma << ","
             << ind.strategy.long_ma << ","
             << ind.strategy.buy_thresh << ","
             << ind.strategy.sell_thresh << ","
             << ind.strategy.stop_loss << ","
             << ind.strategy.take_profit << ","
             << ind.fitness.roi << ","
             << ind.fitness.risk << ","
             << ind.fitness.win_rate << ","
             << ind.strength << ","
             << ind.raw_fitness << ","
             << ind.density << ","
             << ind.spea2_fitness << std::endl;
    }
    
    file.close();
}

// ============= MAIN ALGORITHM =============

int main(int argc, char* argv[]) {
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            std::string value = argv[i + 1];
            
            if (arg == "--pop-size") POP_SIZE = std::stoi(value);
            else if (arg == "--archive-size") ARCHIVE_SIZE = std::stoi(value);
            else if (arg == "--generations") GENERATIONS = std::stoi(value);
            else if (arg == "--k-nearest") K_NEAREST = std::stoi(value);
            else if (arg == "--mutation-rate") MUTATION_RATE = std::stod(value);
            else if (arg == "--crossover-rate") CROSSOVER_RATE = std::stod(value);
            else if (arg == "--sbx-eta") SBX_ETA = std::stod(value);
            else if (arg == "--pm-eta") PM_ETA = std::stod(value);
            else if (arg == "--data-file") DATA_FILE = value;
            else if (arg == "--param-set") PARAM_SET_ID = std::stoi(value);
        }
    }
    
    // Create timestamp for run ID
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    
    // Create output directory
    std::string base_dir = "param_set_" + std::to_string(PARAM_SET_ID);
    std::string run_dir = base_dir + "/run_" + timestamp;
    
    if (!create_directory(base_dir)) {
        std::cerr << "Error: Could not create directory " << base_dir << std::endl;
        return 1;
    }
    
    if (!create_directory(run_dir)) {
        std::cerr << "Error: Could not create directory " << run_dir << std::endl;
        return 1;
    }
    
    // Initialize random number generator
    std::random_device rd;
    std::default_random_engine rng(rd());
    
    // Load price data
    std::vector<PriceData> price_data = load_price_data(DATA_FILE);
    if (price_data.empty()) {
        std::cerr << "Error: No price data loaded" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << price_data.size() << " price data points" << std::endl;
    
    // Initialize population
    std::vector<Individual> population(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        population[i].strategy = random_strategy(rng);
        population[i].fitness = evaluate_strategy(population[i].strategy, price_data);
    }
    
    // Initialize archive
    std::vector<Individual> archive;
    
    // Main loop
    for (int gen = 0; gen < GENERATIONS; ++gen) {
        Timer gen_timer("Generation " + std::to_string(gen));
        
        // Environmental selection
        environmental_selection(population, archive);
        
        // Log data if needed
        if (ENABLE_LOGGING && (gen % LOG_FREQUENCY == 0 || gen == GENERATIONS - 1)) {
            std::string gen_prefix = run_dir + "/gen_" + std::to_string(gen);
            log_population(population, gen_prefix + "_data.csv");
            log_archive(archive, gen_prefix + "_archive.csv");
        }
        
        // Create offspring population
        std::vector<Individual> offspring(POP_SIZE);
        for (int i = 0; i < POP_SIZE; ++i) {
            // Binary tournament selection
            Individual parent1 = tournament_selection(archive, rng);
            Individual parent2 = tournament_selection(archive, rng);
            
            // Crossover
            Strategy child_strategy = crossover(parent1.strategy, parent2.strategy, rng);
            
            // Mutation
            child_strategy = mutate(child_strategy, rng);
            
            // Evaluate
            offspring[i].strategy = child_strategy;
            offspring[i].fitness = evaluate_strategy(child_strategy, price_data);
        }
        
        // Replace population with offspring
        population = offspring;
        
        // Print progress
        if (gen % 10 == 0 || gen == GENERATIONS - 1) {
            // Find best individual in archive
            double best_roi = -std::numeric_limits<double>::max();
            double best_risk = std::numeric_limits<double>::max();
            double best_win_rate = -std::numeric_limits<double>::max();
            
            for (const auto& ind : archive) {
                best_roi = std::max(best_roi, ind.fitness.roi);
                best_risk = std::min(best_risk, ind.fitness.risk);
                best_win_rate = std::max(best_win_rate, ind.fitness.win_rate);
            }
            
            std::cout << "Generation " << gen << ": "
                      << "Archive size = " << archive.size() << ", "
                      << "Best ROI = " << best_roi << "%, "
                      << "Best Risk = " << best_risk << ", "
                      << "Best Win Rate = " << best_win_rate << "%" << std::endl;
        }
    }
    
    // Final logging
    if (LOG_FINAL_POPULATION) {
        log_population(population, run_dir + "/final_population.csv");
        log_archive(archive, run_dir + "/final_archive.csv");
    }
    
    // Print final results
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "Archive size: " << archive.size() << std::endl;
    
    // Find best individuals for each objective
    Individual best_roi_ind = archive[0];
    Individual best_risk_ind = archive[0];
    Individual best_win_rate_ind = archive[0];
    
    for (const auto& ind : archive) {
        if (ind.fitness.roi > best_roi_ind.fitness.roi) {
            best_roi_ind = ind;
        }
        if (ind.fitness.risk < best_risk_ind.fitness.risk) {
            best_risk_ind = ind;
        }
        if (ind.fitness.win_rate > best_win_rate_ind.fitness.win_rate) {
            best_win_rate_ind = ind;
        }
    }
    
    // Print best individuals
    std::cout << "\nBest ROI Strategy:" << std::endl;
    std::cout << "Short MA: " << best_roi_ind.strategy.short_ma << std::endl;
    std::cout << "Long MA: " << best_roi_ind.strategy.long_ma << std::endl;
    std::cout << "Buy Threshold: " << best_roi_ind.strategy.buy_thresh << std::endl;
    std::cout << "Sell Threshold: " << best_roi_ind.strategy.sell_thresh << std::endl;
    std::cout << "Stop Loss: " << best_roi_ind.strategy.stop_loss << std::endl;
    std::cout << "Take Profit: " << best_roi_ind.strategy.take_profit << std::endl;
    std::cout << "ROI: " << best_roi_ind.fitness.roi << "%" << std::endl;
    std::cout << "Risk: " << best_roi_ind.fitness.risk << std::endl;
    std::cout << "Win Rate: " << best_roi_ind.fitness.win_rate << "%" << std::endl;
    
    std::cout << "\nBest Risk Strategy:" << std::endl;
    std::cout << "Short MA: " << best_risk_ind.strategy.short_ma << std::endl;
    std::cout << "Long MA: " << best_risk_ind.strategy.long_ma << std::endl;
    std::cout << "Buy Threshold: " << best_risk_ind.strategy.buy_thresh << std::endl;
    std::cout << "Sell Threshold: " << best_risk_ind.strategy.sell_thresh << std::endl;
    std::cout << "Stop Loss: " << best_risk_ind.strategy.stop_loss << std::endl;
    std::cout << "Take Profit: " << best_risk_ind.strategy.take_profit << std::endl;
    std::cout << "ROI: " << best_risk_ind.fitness.roi << "%" << std::endl;
    std::cout << "Risk: " << best_risk_ind.fitness.risk << std::endl;
    std::cout << "Win Rate: " << best_risk_ind.fitness.win_rate << "%" << std::endl;
    
    std::cout << "\nBest Win Rate Strategy:" << std::endl;
    std::cout << "Short MA: " << best_win_rate_ind.strategy.short_ma << std::endl;
    std::cout << "Long MA: " << best_win_rate_ind.strategy.long_ma << std::endl;
    std::cout << "Buy Threshold: " << best_win_rate_ind.strategy.buy_thresh << std::endl;
    std::cout << "Sell Threshold: " << best_win_rate_ind.strategy.sell_thresh << std::endl;
    std::cout << "Stop Loss: " << best_win_rate_ind.strategy.stop_loss << std::endl;
    std::cout << "Take Profit: " << best_win_rate_ind.strategy.take_profit << std::endl;
    std::cout << "ROI: " << best_win_rate_ind.fitness.roi << "%" << std::endl;
    std::cout << "Risk: " << best_win_rate_ind.fitness.risk << std::endl;
    std::cout << "Win Rate: " << best_win_rate_ind.fitness.win_rate << "%" << std::endl;

    return 0;
}