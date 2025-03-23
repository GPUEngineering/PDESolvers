//
// Created by Chelsea De Marseilla on 05/03/2025.
//

#ifndef BSE_SOLVERS_PARALLEL_CUH
#define BSE_SOLVERS_PARALLEL_CUH

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cusparse.h>
#include <cublas.h>
#include <fstream>
#include <filesystem>

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
    } else if constexpr(std::is_same_v<T, cusparseStatus_t>) {
        if (code != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "cublas error. Code: " << cusparseGetErrorString(code)
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

    if (idx <= t_nodes) {
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
 * Computes the coefficients for the tridiagonal matrix
 *
 * @tparam T data type (default: double)
 * @param d_alpha the alpha coefficients
 * @param d_beta the beta coefficients
 * @param d_gamma the gamma coefficients
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param s_nodes the number of spatial nodes
 * @param dS grid spacing between spatial nodes
 * @param dt grid spacing between time nodes
 */
template <typename T = DEFAULT_FPX>
__global__ static void compute_coefficients(T* d_alpha,
                                            T* d_beta,
                                            T* d_gamma,
                                            T sigma,
                                            T rate,
                                            int s_nodes,
                                            T dS,
                                            T dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < s_nodes)
    {
        T current_s = tid * dS;
        T sigma_sq = pow(sigma, 2);
        T dS_sq_inv = 1 / pow(dS, 2);
        d_alpha[tid] = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv - rate * current_s / dS);
        d_beta[tid] = -dt * 0.5 * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate);
        d_gamma[tid] = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate * current_s / dS);
    }
}

/**
 * Prepares the coefficients for the left-hand side tridiagonal matrix
 *
 * @tparam T data type (default: double)
 * @param d_alpha_lhs the alpha coefficients of the lhs matrix
 * @param d_beta_lhs the beta coefficients of the lhs matrix
 * @param d_gamma_lhs the gamma coefficients of the lhs matrix
 * @param d_alpha the alpha coefficients
 * @param d_beta the beta coefficients
 * @param d_gamma the gamma coefficients
 * @param vector_size the size of the vector
 */
template <typename T = DEFAULT_FPX>
__global__ static void construct_lhs(T* d_alpha_lhs, T* d_beta_lhs, T* d_gamma_lhs,
                                   T* d_alpha, T* d_beta, T* d_gamma, size_t vector_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < vector_size) {
        d_alpha_lhs[tid] = -d_alpha[tid + 1];
        d_beta_lhs[tid] = 1 - d_beta[tid + 1];
    }

    if (tid < vector_size - 1) {
        d_gamma_lhs[tid] = -d_gamma[tid + 1];
    }
}

/**
 * Initialises the first value of the right-hand side vector for the tridiagonal matrix
 *
 * @tparam T data type (default: double)
 * @param rhs_vector the right-hand side vector
 * @param current the current time step
 * @param prev the previous time step
 * @param d_alpha the alpha coefficients
 * @param d_beta the beta coefficients
 * @param d_gamma the gamma coefficients
 */
template <typename T = DEFAULT_FPX>
__global__ static void initialise_rhs_first_val(T* rhs_vector,
                                     const T* current,
                                     T* prev,
                                     T* d_alpha,
                                     T* d_beta,
                                     T* d_gamma)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        rhs_vector[0] = d_alpha[1] * (current[0] + prev[0]) + (1 + d_beta[1]) * current[1] + d_gamma[1] * current[2];
    }
}

/**
 * Initialises the right-hand side vector for the tridiagonal matrix
 *
 * @tparam T data type (default: double)
 * @param rhs_vector the right-hand side vector
 * @param current the current time step
 * @param prev the previous time step
 * @param d_alpha the alpha coefficients
 * @param d_beta the beta coefficients
 * @param d_gamma the gamma coefficients
 * @param vector_size the size of the vector
 * @param s_nodes the number of spatial nodes
 */
template <typename T = DEFAULT_FPX>
__global__ static void initialise_rhs_end_val(T* rhs_vector,
                                     const T* current,
                                     T* prev,
                                     T* d_alpha,
                                     T* d_beta,
                                     T* d_gamma,
                                     int vector_size,
                                     int s_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        rhs_vector[vector_size - 1] = d_alpha[vector_size] * current[s_nodes - 2] + (1 + d_beta[vector_size]) *
            current[s_nodes - 1] + d_gamma[vector_size] * (current[s_nodes] + prev[s_nodes]);
    }
}

/**
 * Constructs the remaining inner values of right-hand side vector for the tridiagonal matrix
 *
 * @tparam T data type (default: double)
 * @param rhs_vector the right-hand side vector
 * @param current the current time step
 * @param d_alpha the alpha coefficients
 * @param d_beta the beta coefficients
 * @param d_gamma the gamma coefficients
 * @param vector_size the size of the vector
 */
template <typename T = DEFAULT_FPX>
__global__ static void construct_rhs(T* rhs_vector,
                                     const T* current,
                                     T* d_alpha,
                                     T* d_beta,
                                     T* d_gamma,
                                     int vector_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= 1 && tid < vector_size - 1)
    {
        int idx = tid + 1;
        rhs_vector[tid] = d_alpha[idx] * current[idx - 1] + (1 + d_beta[idx]) * current[idx] + d_gamma[idx] * current[
            idx + 1];
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
 * Singleton for Cuda library handles
 */
class Session {
public:

    static Session &getInstance() {
        static Session instance;
        return instance;
    }

private:
    Session() {
        gpuErrChk(cusparseCreate(&m_sparseHandle));
    }

    ~Session() {
        gpuErrChk(cusparseDestroy(m_sparseHandle));
    }

    cublasHandle_t m_cublasHandle;
    cusparseHandle_t m_sparseHandle;


public:
    Session(Session const &) = delete;

    void operator=(Session const &) = delete;

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusparseHandle_t &cuSparseHandle() { return m_sparseHandle; }
};

/**
 * Solution u(x, t)
 *
 * @tparam T data type (default: double)
 */
template<typename T = DEFAULT_FPX>
class Solution {
private:
    static std::string get_output_file_path(const std::string& filename)
    {
        std::filesystem::path current_path = std::filesystem::current_path();
        return (current_path / filename).string();
    }

    std::ostream &print(std::ostream &out) const
    {
        size_t nr = m_t_nodes + 1, nc = m_s_nodes + 1;
        T* host_data = new T[grid_size()];
        download(host_data);

        // gets output for file path
        std::string file_path = get_output_file_path("out.csv");

        // exports to csv file
        std::ofstream csv_file(file_path);
        out << "Grid [" << nr << " x " << nc << "]:" << std::endl;
        for (size_t i = 0; i < nr; i++) {
            for (size_t j = 0; j < nc; j++) {
                csv_file << host_data[i * nc + j];
                if (j < nc - 1) {
                    csv_file << ",";
                }
            }
        }
        csv_file << std::endl;
        csv_file.close();

        out << "Data exported to " << file_path <<" successfully" << std::endl;

        delete[] host_data;
        return out;
    }

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

    friend std::ostream &operator << (std::ostream &out, const Solution<T> &data) {
        return data.print(out);
    }

    // Here we can include more methods (e.g., for printing, finding average value, etc)
};

/**
 *
 * Solves the Black-Scholes equation using the explicit method in parallel
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
    T dS = s_max / static_cast<T> (s_nodes);
    // calculates appropriate dt value (is it okay to put this here?)
    T dt = expiry / static_cast<T> (t_nodes);
    if (dt > dt_max) throw std::runtime_error("t_nodes too small");

    // gpu memory allocation
    T *dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    size_t nBlocks_s = numBlocks(s_nodes + 1);
    size_t nBlocks_t = numBlocks(t_nodes + 1);

    set_terminal_condition<type><<<nBlocks_s, THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
    set_boundary_conditions<type><<<nBlocks_t, THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate,
                                                                  expiry, dt);
    // solving equation in parallel per time step
    for (int tau = t_nodes; tau > 0; tau--) {
        T *current_time_step = dev_grid + tau * (s_nodes + 1);
        T *prev_time_step = current_time_step - (s_nodes + 1);
        solve_explicit_parallel<<<nBlocks_s, THREADS_PER_BLOCK>>>(current_time_step, prev_time_step, sigma, rate, s_nodes,
                                                                dS, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    Solution<T> sol(dev_grid, s_nodes, t_nodes);
    sol.m_duration = (double) duration.count() / 1e6;
    return sol;
}

/**
 * Solves the Black-Scholes equation using the Crank-Nicolson method in parallel
 *
 * @tparam type option type
 * @tparam T data type (default: double)
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
Solution<T> solve_bse_cn(T s_max,
                               T expiry,
                               T sigma,
                               T rate,
                               T strike_price,
                               int s_nodes,
                               int t_nodes)
{
    auto start = std::chrono::high_resolution_clock::now();

    size_t nBlocks_s = numBlocks(s_nodes + 1);
    size_t nBlocks_t = numBlocks(t_nodes + 1);

    size_t grid_size = (t_nodes + 1) * (s_nodes + 1);
    size_t vector_size = s_nodes - 1;
    T dt = expiry / static_cast<T> (t_nodes);
    T dS = s_max / static_cast<T> (s_nodes);

    // memory allocation
    T* dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    T *d_alpha, *d_beta, *d_gamma;
    gpuErrChk(cudaMalloc(&d_alpha, (s_nodes + 1) * sizeof(T)));
    gpuErrChk(cudaMalloc(&d_beta, (s_nodes + 1) * sizeof(T)));
    gpuErrChk(cudaMalloc(&d_gamma, (s_nodes + 1) * sizeof(T)));

    T *d_alpha_lhs, *d_beta_lhs, * d_gamma_lhs;
    gpuErrChk(cudaMalloc(&d_alpha_lhs, (vector_size) * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_beta_lhs, vector_size * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_gamma_lhs, (vector_size) * sizeof(double)));

    T* d_rhs_vector;
    gpuErrChk(cudaMalloc(&d_rhs_vector, vector_size * sizeof(T)));

    // computes coefficients for tridiagonal matrix
    compute_coefficients<<<nBlocks_s, THREADS_PER_BLOCK>>>(d_alpha, d_beta, d_gamma, sigma, rate, (s_nodes + 1), dS, dt);

    // prepares lhs tridiagonal matrix
    construct_lhs<<<numBlocks(vector_size), THREADS_PER_BLOCK>>>(d_alpha_lhs, d_beta_lhs, d_gamma_lhs, d_alpha, d_beta, d_gamma, vector_size);

    // allocating buffer
    void* pBuffer = nullptr;
    size_t bufferSize;

    gpuErrChk(cusparseDgtsv2_bufferSizeExt(Session::getInstance().cuSparseHandle(), vector_size, 1, d_alpha_lhs, d_beta_lhs, d_gamma_lhs, nullptr, vector_size, &bufferSize));
    gpuErrChk(cudaMalloc(&pBuffer, bufferSize));

    set_terminal_condition<type><<<nBlocks_s, THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);

    set_boundary_conditions<type><<<nBlocks_t, THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate,
                                                                  expiry, dt);

    for (int tau = t_nodes; tau > 0; tau--)
    {
        // points to the current time step
        T* current = dev_grid + tau * (s_nodes + 1);
        T* prev = current - (s_nodes + 1);

        // constructing RHS vector
        initialise_rhs_first_val<<<1, 1>>>(d_rhs_vector, current, prev, d_alpha, d_beta, d_gamma);
        initialise_rhs_end_val<<<1, 1>>>(d_rhs_vector, current, prev, d_alpha, d_beta, d_gamma, vector_size, s_nodes);
        construct_rhs<<<numBlocks(vector_size), THREADS_PER_BLOCK>>>(d_rhs_vector, current, d_alpha, d_beta, d_gamma,
                                                      vector_size);

        // solves for the next time step (Ax = b)
        gpuErrChk(cusparseDgtsv2(Session::getInstance().cuSparseHandle(), vector_size, 1, d_alpha_lhs, d_beta_lhs, d_gamma_lhs, d_rhs_vector, vector_size, pBuffer));

        gpuErrChk(cudaMemcpy(prev + 1, d_rhs_vector, vector_size * sizeof(T), cudaMemcpyDeviceToDevice));

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    gpuErrChk(cudaFree(pBuffer));
    gpuErrChk(cudaFree(d_rhs_vector));
    gpuErrChk(cudaFree(d_gamma_lhs));
    gpuErrChk(cudaFree(d_beta_lhs));
    gpuErrChk(cudaFree(d_alpha_lhs));
    gpuErrChk(cudaFree(d_gamma));
    gpuErrChk(cudaFree(d_beta));
    gpuErrChk(cudaFree(d_alpha));

    Solution<T> sol(dev_grid, s_nodes, t_nodes);
    sol.m_duration = (double) duration.count() / 1e6;

    return sol;
}

#endif //BSE_SOLVERS_PARALLEL_CUH
