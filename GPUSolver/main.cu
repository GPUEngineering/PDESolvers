#include "gpu/bse_solvers_parallel.cuh"

int main() {
    constexpr OptionType type = OptionType::Call;
    double s_max = 300.0;
    double expiry = 1;
    double sigma = 0.2;
    double rate = 0.05;
    double strike_price = 100;
    int s_nodes = 100;
    int t_nodes = 1000;

    /* GPU Computation and timing */
    Solution<double> solution1 = solve_bse_explicit<type>(s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);
    Solution<double> solution2 = solve_bse_cn<type>(s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);


    /* Print duration*/
    std::cout << "[GPU] Explicit method finished in " << solution1.m_duration << "s"
              << std::endl;

    std::cout << "[GPU] Crank Nicolson method finished in " << solution2.m_duration << "s"
          << std::endl;

    /* Download solution */
    double *host_grid1 = new double[solution1.grid_size()];
    solution1.download(host_grid1);
    double *host_grid2 = new double[solution2.grid_size()];
    solution2.download(host_grid2);

    std::cout << solution2;

    std::cout << "x = " << host_grid1[1400] << std::endl;
    std::cout << "x = " << host_grid2[1400] << std::endl;

    delete[] host_grid1;
    delete[] host_grid2;
}

