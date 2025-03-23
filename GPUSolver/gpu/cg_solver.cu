#include <iostream>           // Printing
#include <vector>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdlib.h>           // EXIT_FAILURE
#include <cublas_v2.h>
#include <memory>             // smart pointers
#include <iomanip>            // for std::setw

#define DEFAULT_FPX double
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif


//* ================================================================================================
// *  ERROR CHECKING
// * ================================================================================================ */

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

TEMPLATE_WITH_TYPE_T inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
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



/* ================================================================================================
 *  SESSION
 * ================================================================================================ */
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
        gpuErrChk(cublasCreate(&m_cublasHandle));
        gpuErrChk(cusparseCreate(&m_sparseHandle));
    }

    ~Session() {
        gpuErrChk(cublasDestroy(m_cublasHandle));
        gpuErrChk(cusparseDestroy(m_sparseHandle));
    }

    cublasHandle_t m_cublasHandle;
    cusparseHandle_t m_sparseHandle;


public:
    Session(Session const &) = delete;

    void operator=(Session const &) = delete;

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusparseHandle_t &cuSpraseHandle() { return m_sparseHandle; }
};


/* ================================================================================================
 *  DVector
 * ================================================================================================ */
TEMPLATE_WITH_TYPE_T
class DVector {
private:
    size_t m_nulEl = 0;
    T *m_d_data = nullptr;
    cusparseDnVecDescr_t m_vecX;

public:

    DVector(size_t n) {
        m_nulEl = n;
        gpuErrChk(cudaMalloc((void **) &m_d_data, m_nulEl * sizeof(T)));
        gpuErrChk(cusparseCreateDnVec(&m_vecX, m_nulEl, m_d_data, CUDA_R_32F));
    }

    DVector(std::vector<T> hostData) {
        m_nulEl = hostData.size();
        gpuErrChk(cudaMalloc((void **) &m_d_data, m_nulEl * sizeof(T)));
        gpuErrChk(cudaMemcpy(m_d_data, hostData.data(), m_nulEl * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrChk(cusparseCreateDnVec(&m_vecX, m_nulEl, m_d_data, CUDA_R_32F));
    }

    ~DVector() {
        if (m_d_data) {
            gpuErrChk(cudaFree(m_d_data));
            m_d_data = nullptr;
            gpuErrChk(cusparseDestroyDnVec(m_vecX));
        }
        m_nulEl = 0;
    }

    const cusparseDnVecDescr_t &cusparseVector() const {
        return m_vecX;
    }

    void downloadTo(T *hostData) const {
        gpuErrChk(cudaMemcpy(hostData, m_d_data, m_nulEl * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void downloadTo(std::vector<T> &vec) const {
        vec.resize(m_nulEl);
        gpuErrChk(cudaMemcpy(vec.data(),
                             m_d_data,
                             m_nulEl * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    size_t numEl() const {
        return m_nulEl;
    }

    void deviceCopyFrom(DVector<T> &other) const {
        gpuErrChk(cudaMemcpy(m_d_data, other.m_d_data, m_nulEl * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    std::ostream &print(std::ostream &out) const {
        std::vector<T> temp;
        downloadTo(temp);
        out << "[DVector] " << m_nulEl << " elements " << std::endl;
        for (size_t i = 0; i < m_nulEl; i++) {
            out << std::setw(10) << temp[i] << std::endl;
        }
        return out;
    }

    friend std::ostream &operator<<(std::ostream &out, const DVector<T> &data) {
        return data.print(out);
    }

    T *raw() const {
        return m_d_data;
    }

    T norm() const;

    T dot(const DVector<T> &other) const;

    /**
     * x += beta z
     * @param z
     * @param alpha
     */
    void axpy(DVector<T> &z,
              const T alpha = 1.);

    /**
     * x *= alpha
     * @param alpha
     */
    void scale(const T alpha = 1.);

};

template<>
void DVector<double>::scale(const double alpha) {
    gpuErrChk(cublasDscal(Session::getInstance().cuBlasHandle(), m_nulEl, &alpha, m_d_data, 1));
}

template<>
void DVector<float>::scale(const float alpha) {
    gpuErrChk(cublasSscal(Session::getInstance().cuBlasHandle(), m_nulEl, &alpha, m_d_data, 1));
}

template<>
inline void DVector<double>::axpy(DVector<double> &z,
                                  const double alpha) {
    gpuErrChk(cublasDaxpy(Session::getInstance().cuBlasHandle(), m_nulEl, &alpha, z.m_d_data, 1, m_d_data, 1));
}

template<>
inline void DVector<float>::axpy(DVector<float> &z,
                                 const float alpha) {
    gpuErrChk(cublasSaxpy(Session::getInstance().cuBlasHandle(), m_nulEl, &alpha, z.m_d_data, 1, m_d_data, 1));
}

template<>
inline double DVector<double>::norm() const {
    double the_norm;
    gpuErrChk(cublasDnrm2(Session::getInstance().cuBlasHandle(), m_nulEl, m_d_data, 1, &the_norm));
    return the_norm;
}

template<>
inline float DVector<float>::norm() const {
    float the_norm;
    gpuErrChk(cublasSnrm2(Session::getInstance().cuBlasHandle(), m_nulEl, m_d_data, 1, &the_norm));
    return the_norm;
}

// T dot(const DVector<T>& other) const;
template<>
inline double DVector<double>::dot(const DVector<double> &other) const {
    size_t n = numEl();
    double result;
    gpuErrChk(cublasDdot(Session::getInstance().cuBlasHandle(), n,
                         raw(), 1,
                         other.raw(), 1,
                         &result));
    return result;
}

template<>
inline float DVector<float>::dot(const DVector<float> &other) const {
    size_t n = numEl();
    float result;
    gpuErrChk(cublasSdot(Session::getInstance().cuBlasHandle(), n,
                         raw(), 1,
                         other.raw(), 1,
                         &result));
    return result;
}

/* ================================================================================================
 *  DSparseCSRMatrix (CSR SPARSE MATRIX)
 * ================================================================================================ */
TEMPLATE_WITH_TYPE_T
class DSparseCSRMatrix {
private:
    /* Metadata */
    size_t m_numRows = 0;  ///< Number of rows
    size_t m_numCols = 0;  ///< Number of columns
    size_t m_numNonZeros = 0;  ///< Number of nonzero elements

    /* Data */
    T *m_d_data = nullptr;  ///< Pointer to device data
    int *m_d_csrOffsets = nullptr;
    int *m_d_csrColumns = nullptr;
    cusparseSpMatDescr_t m_csrMat;

    /* Buffer */
    size_t m_bufferSize = 0;
    void *m_buffer = nullptr;

public:
    DSparseCSRMatrix(const std::vector<T> &data,
                     const std::vector<int> &csrOffsets,
                     const std::vector<int> &csrColumns,
                     size_t nRows,
                     size_t nCols,
                     size_t nNonzero) :
            m_numCols(nCols), m_numRows(nRows), m_numNonZeros(nNonzero) {
        /* allocate memory */
        gpuErrChk(cudaMalloc((void **) &m_d_data, m_numNonZeros * sizeof(T)));
        gpuErrChk(cudaMalloc((void **) &m_d_csrOffsets, (m_numRows + 1) * sizeof(int)));
        gpuErrChk(cudaMalloc((void **) &m_d_csrColumns, m_numNonZeros * sizeof(int)));
        /* copy data to device */
        gpuErrChk(cudaMemcpy(m_d_data, data.data(), m_numNonZeros * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrOffsets, csrOffsets.data(), (m_numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrColumns, csrColumns.data(), m_numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
        /* create CSR */
        gpuErrChk(cusparseCreateCsr(&m_csrMat, m_numRows, m_numCols, m_numNonZeros,
                                    m_d_csrOffsets, m_d_csrColumns, m_d_data,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DSparseCSRMatrix() {
        if (m_d_csrOffsets) {
            gpuErrChk(cudaFree(m_d_csrOffsets));
            m_d_csrOffsets = nullptr;
        }
        if (m_d_csrColumns) {
            gpuErrChk(cudaFree(m_d_csrColumns));
            m_d_csrColumns = nullptr;
        }
        if (m_d_data) {
            gpuErrChk(cudaFree(m_d_data));
            m_d_data = nullptr;
        }
        if (m_buffer) {
            gpuErrChk(cudaFree(m_buffer));
            m_buffer = nullptr;
        }
        if (m_numNonZeros) gpuErrChk(cusparseDestroySpMat(m_csrMat));
    }

    /**
     * Performs y = alpha * A * x + beta * y
     * @param y vector
     * @param x vector
     * @param alpha scalar
     * @param beta scalar
     */
    void axpby(DVector<T> &y,
               DVector<T> &x,
               T alpha = 1.,
               T beta = 0) {
        if (!m_buffer) {
            gpuErrChk(cusparseSpMV_bufferSize(
                    Session::getInstance().cuSpraseHandle(),
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, m_csrMat, x.cusparseVector(), &beta, y.cusparseVector(), CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, &m_bufferSize));
            gpuErrChk(cudaMalloc((void **) &m_buffer, m_bufferSize));
        }
        gpuErrChk(cusparseSpMV(Session::getInstance().cuSpraseHandle(),
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, m_csrMat, x.cusparseVector(), &beta, y.cusparseVector(), CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, m_buffer));
    }

    size_t nRows() const {
        return m_numRows;
    }

    size_t nCols() const {
        return m_numCols;
    }

};


TEMPLATE_WITH_TYPE_T
class CGSolver {
private:
    DSparseCSRMatrix<T> &m_lhs;
    std::unique_ptr<DVector<T>> m_residual = nullptr;
    std::unique_ptr<DVector<T>> m_search_direction = nullptr;
    std::unique_ptr<DVector<T>> m_a_search_direction = nullptr;
    size_t m_maxIters;

public:
    CGSolver(DSparseCSRMatrix<T> &lhsMatrix, size_t maxIters = 0) : m_lhs(lhsMatrix), m_maxIters(maxIters) {
        size_t m = m_lhs.nRows();
        if (maxIters == 0) m_maxIters = m;
        m_residual = std::make_unique<DVector<T>>(m);
        m_search_direction = std::make_unique<DVector<T>>(m);
        m_a_search_direction = std::make_unique<DVector<T>>(m);
    }

    void solve(DVector<T> &rhs, DVector<T> &x, T eps) {
        m_residual->deviceCopyFrom(rhs);
        m_lhs.axpby(*m_residual, x, -1, 1);
        m_search_direction->deviceCopyFrom(*m_residual);
        T norm_x = x.norm();
        T old_resid_norm = m_residual->norm();
        for (size_t i = 0; i < m_maxIters; i++) {
            m_lhs.axpby(*m_a_search_direction, *m_search_direction, 1, 0);
            T num = old_resid_norm * old_resid_norm;
            T den = m_search_direction->dot(*m_a_search_direction);
            T step_size = num / den;
            x.axpy(*m_search_direction, step_size);
            m_residual->axpy(*m_a_search_direction, -step_size);
            T new_resid_norm = m_residual->norm();
            T kappa = new_resid_norm / old_resid_norm;
            kappa *= kappa;
            m_search_direction->scale(kappa);
            m_search_direction->axpy(*m_residual);
            old_resid_norm = new_resid_norm;
        }
    }

};



//* ================================================================================================
// *  MAIN function (for testing only)
// * ================================================================================================ */

int main(void) {

    // MATRIX A DATA (CSR)
    const int nr = 4, nc = 4, nnz = 9;
    DSparseCSRMatrix<float> aCSR(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                                 std::vector<int>{0, 3, 4, 7, 9},
                                 std::vector<int>{0, 2, 3, 1, 0, 2, 3, 1, 3},
                                 nr, nc, nnz);

    // VECTORS
    DVector<float> x(std::vector<float>{1.1, 2.1, 3.1, 4.5});
    DVector<float> b(std::vector<float>{19., 8., 51., 52.});

    /* Call Conjugate Gradient Solver (unlikely to converge!) */
    CGSolver<float> solver(aCSR);
    solver.solve(b, x, 0.01);
    std::cout << x;

    return EXIT_SUCCESS;
}