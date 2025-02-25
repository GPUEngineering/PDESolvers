//
// Created by Chelsea De Marseilla on 07/02/2025.
//
#include <iostream>
#include <chrono>
#include <iomanip>

/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK 512

/**
 * Determines the number of blocks needed for a given number of tasks, n,
 * and number of threads per block
 *
 * @param n problem size
 * @param threads_per_block threads per block (defaults to THREADS_PER_BLOCK)
 * @return number of blocks
 */
constexpr size_t numBlocks(size_t n, size_t threads_per_block = THREADS_PER_BLOCK) {
    return (n / threads_per_block + (n % threads_per_block != 0));
}

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

template<typename T = DEFAULT_FPX>
inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}

/**
 * Option type (call or put)
 */
enum class OptionType {
    Call,
    Put
};

/**
 * Sets the terminal condition of the Black-Scholes equation, based on option type
 *
 * @tparam type option type
 * @tparam T data type (default: double)
 * @param dev_grid device memory grid storing option values
 * @param s_nodes number of spatial nodes (asset prices)
 * @param t_nodes number of time nodes
 * @param dS grid spacing between asset prices
 * @param strike_price strike price of the option
 **/
template<OptionType type, typename T = DEFAULT_FPX>
__global__ void set_terminal_condition(T *dev_grid,
                                       int s_nodes,
                                       int t_nodes,
                                       T dS,
                                       T strike_price) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (t_nodes) * (s_nodes + 1) + idx;
    if (idx < s_nodes + 1) {
        T current_s = idx * dS;
        if constexpr (type == OptionType::Call) {
            dev_grid[index] = max(current_s - strike_price, 0.0);
        } else if constexpr (type == OptionType::Put) {
            dev_grid[index] = max(strike_price - current_s, 0.0);
        }
    }
}

/**
 * Sets the boundary conditions of the Black-Scholes equation
 *
 * @tparam type option type
 * @tparam T data type (default: double)
 * @param dev_grid device memory grid storing option values
 * @param t_nodes number of time nodes
 * @param s_nodes number of spatial nodes (asset prices)
 * @param s_max the maximum price of the underlying asset
 * @param strike_price strike price of the option
 * @param rate the risk-free interest rate
 * @param expiry the time to expiry
 * @param dt grid spacing between time nodes
 *
**/
template<OptionType type, typename T = DEFAULT_FPX>
__global__ void set_boundary_conditions(T *dev_grid,
                                        int t_nodes,
                                        int s_nodes,
                                        T s_max,
                                        T strike_price,
                                        T rate,
                                        T expiry,
                                        T dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < t_nodes) {
        T current_time_step = idx * dt;
        if constexpr (type == OptionType::Call) {
            dev_grid[idx * (s_nodes + 1)] = 0.0;
            dev_grid[idx * (s_nodes + 1) + s_nodes] = s_max - strike_price * std::exp(
                    -rate * (expiry - current_time_step));
        } else if constexpr (type == OptionType::Put) {
            dev_grid[idx * (s_nodes + 1)] = strike_price * std::exp(-rate * (expiry - current_time_step));
            dev_grid[idx * (s_nodes + 1) + s_nodes] = 0.0;
        }
    }
}

/**
 * Solves the explicit method in parallel
 *
 * @tparam T data type (default: double)
 * @param current points to the current time step
 * @param prev points to the previous time step (to be computed)
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param s_nodes the number of spatial nodes (asset prices)
 * @param dS grid spacing between spatial nodes
 * @param dt grid spacing between time nodes
**/
template<typename T = DEFAULT_FPX>
__global__ void solve_explicit_parallel(const T *current,
                                        T *prev,
                                        T sigma,
                                        T rate,
                                        int s_nodes,
                                        T dS,
                                        T dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1 && tid < s_nodes) {
        T current_s = tid * dS;
        // calculating greeks
        T delta = (current[tid + 1] - current[tid - 1]) / (2 * dS);
        T gamma = (current[tid + 1] - 2 * current[tid] + current[tid - 1]) / pow(dS, 2);
        T theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma
                  - rate * current_s * delta + rate * current[tid];
        prev[tid] = current[tid] - (theta * dt);
    }
}

/**
 * Solution u(x, t)
 *
 * @tparam T data type (default: double)
 */
template<typename T = DEFAULT_FPX>
class Solution {
public:
    T *m_d_data = nullptr;
    size_t m_s_nodes;
    size_t m_t_nodes;
    double m_duration = 0;

    Solution(T *data, size_t s_nodes, size_t t_nodes) {
        m_d_data = data;
        m_s_nodes = s_nodes;
        m_t_nodes = t_nodes;
    };

    ~Solution() {
        if (m_d_data) {
            gpuErrChk(cudaFree(m_d_data));
            m_d_data = 0;
        }
    }

    size_t grid_size() const {
        return (m_t_nodes + 1) * (m_s_nodes + 1);
    }

    void download(T *host_data) const {
        if (m_d_data) {
            gpuErrChk(cudaMemcpy(host_data, m_d_data, grid_size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    // Here we can include more methods (e.g., for printing, finding average value, etc)
};

/**
 *
 * @tparam type data type (default: double)
 * @param s_max the maximum price of the underlying asset
 * @param expiry option expiry
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param strike_price strike price
 * @param s_nodes number of nodes along S-axis
 * @param t_nodes number of nodes along T-axis
 * @return solution as instance of Solution
 */
template<OptionType type, typename T = DEFAULT_FPX>
Solution<T> solve_bse_explicit(T s_max,
                               T expiry,
                               T sigma,
                               T rate,
                               T strike_price,
                               int s_nodes,
                               int t_nodes) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t grid_size = (t_nodes + 1) * (s_nodes + 1);
    T dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    T dS = s_max / s_nodes;
    // calculates appropriate dt value (is it okay to put this here?)
    T dt = static_cast<T>(expiry) / t_nodes;
    if (dt > dt_max) throw std::runtime_error("t_nodes too small");

    // gpu memory allocation
    T *dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    size_t num_threads = s_nodes + 1;
    size_t nBlocks = numBlocks(num_threads);
    set_terminal_condition<type><<<nBlocks, THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
    set_boundary_conditions<type><<<nBlocks, THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate,
                                                                  expiry, dt);
    // solving equation in parallel per time step
    for (int tau = t_nodes; tau > 0; tau--) {
        T *current_time_step = dev_grid + tau * (s_nodes + 1);
        T *prev_time_step = current_time_step - (s_nodes + 1);
        solve_explicit_parallel<<<nBlocks, THREADS_PER_BLOCK>>>(current_time_step, prev_time_step, sigma, rate, s_nodes,
                                                                dS, dt);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Solution<T> sol(dev_grid, s_nodes, t_nodes);
    sol.m_duration = (double) duration.count() / 1e6;
    return sol;
}

/* The definition of DEBUG_PDESOLVERS should be moved to CMakeLists.txt */
#define DEBUG_PDESOLVERS
#ifdef DEBUG_PDESOLVERS

int main() {
    constexpr OptionType type = OptionType::Call;
    double s_max = 300.0;
    double expiry = 0.1;
    double sigma = 0.2;
    double rate = 0.05;
    double strike_price = 100;
    int s_nodes = 1000;
    int t_nodes = 100000;

    /* GPU Computation and timing */
    Solution<double> solution = solve_bse_explicit<type>(s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);

    /* Print duration*/
    std::cout << "[GPU] Explicit method finished in " << std::setprecision(3) << solution.m_duration << "s"
              << std::endl;

    /* Download solution */
    double *host_grid = new double[solution.grid_size()];
    solution.download(host_grid);
    std::cout << "x = " << host_grid[1400] << std::endl;
    delete[] host_grid;
}

#endif
