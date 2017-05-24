// -----------------------------------------------------------------------------
#include <stdio.h>
#include <stdbool.h>
#include <curand.h>
#include <cublas_v2.h>
// -----------------------------------------------------------------------------
#include "matrix.h"
#include "nmf.h"
// -----------------------------------------------------------------------------
#define CHECK_CONVERGENCE (100)
#define EPSILON           (1e-32)
#define MIN_STEPSIZE      (1e-64)
#ifdef DEBUG
#define PRINT_DEBUG       (1000)
#endif
// -----------------------------------------------------------------------------

/**
 * Checks the given CUDA error code and terminates the program if an error
 * occured.
 *
 * @param error CUDA error message.
 */
void checkCUDAError(cudaError_t error)
{
  if (error != cudaSuccess)
  {
    fprintf(stderr, "[ERROR] CUDA error: %s\n", cudaGetErrorString(error));
    fflush(stderr);
    exit(1);
  }
}

/**
 *
 *
 * @param  error [description]
 * @return       [description]
 */
static const char * cublasGetErrorString(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

/**
 *
 *
 * @param status [description]
 */
void checkCUBLASStatus(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "[ERROR] CUDA status: %s\n", cublasGetErrorString(status));
    fflush(stderr);
    exit(1);
  }
}

/**
 * Checks the status of the GPU memory and prints some info on stdout.
 */
void checkGPUMem()
{
  size_t free
       , total
       ;

  checkCUDAError(cudaMemGetInfo(&free, &total));
  fprintf(stdout, "[INFO ] Free memory %g M/ %g M\n", free/pow(10., 6), total/pow(10., 6));
}

/**
 * Allocates memory for a matrix with rows*cols elements on the GPU.
 *
 * @param rows Number of rows.
 * @param cols Number of column.
 *
 * @returns    The created matrix that points to memory in GPU memory.
 */
Matrix onGPU(size_t rows, size_t cols)
{
  Matrix res;
  res.rows = rows;
  res.cols = cols;

  #ifdef DEBUG
  fprintf(stderr, "[DEBUG] Creating matrix with %lu elements on GPU (%lu KB)\n", rows * cols, rows * cols * sizeof(double) / 1024);
  fflush(stderr);
  #endif

  checkCUDAError(cudaMalloc((void**) &res.elements, rows * cols * sizeof(double)));
  return res;
}

/**
 *
 *
 * @param  m [description]
 * @return   [description]
 */
Matrix freeGPU(Matrix m)
{
  m.rows = 0;
  m.cols = 0;
  cudaFree(m.elements);
  return m;
}

/**
 * Fills n slots in gpu memory with value v, starting at d_M.
 *
 * @param d_M Pointer to the start of GPU memory to fill.
 * @param n   Number of elements to fill with v.
 * @param v   Value to store.
 */
__global__ void fillWith(double* d_M, size_t n, double v)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
    d_M[idx] = v;
}

/**
 * Turns n elements starting from d_M positive.
 *
 * @param d_M Pointer to the start of GPU memory.
 * @param n   Number of elements to turn positive.
 */
__global__ void mkPositive(double* d_M, size_t n)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
    d_M[idx] = fabs(d_M[idx]);
}

/**
 * Copies a matrix to GPU memory.
 *
 * @param  h_M       Matrix that should be copied to GPU memory.
 * @param  init      Method of initialisation for the memory on the GPU.
 * @param  generator Random number generator for the GPU.
 * @param  blocksize Number of threads that can execute concurrently on the GPU.
 *
 * @returns          A matrix that holds its elements in GPU memory.
 */
Matrix toGPU(Matrix h_M, MatrixInitialisation init, curandGenerator_t generator, unsigned blocksize)
{
  Matrix M;
  M.rows = h_M.rows;
  M.cols = h_M.cols;

  #ifdef DEBUG
  fprintf(stderr, "[DEBUG] Creating matrix with %lu elements on GPU (%lu KB)\n", size(h_M), size(h_M) * sizeof(double) / 1024);
  fflush(stderr);
  #endif

  // allocate memory for the matrix' data
  checkCUDAError(cudaMalloc((void**) &M.elements, size(h_M) * sizeof(double)));

  // initialise the matrix
  switch (init)
  {
    case INIT_COPY:
      #ifdef DEBUG
      fprintf(stderr, "[DEBUG] Copying matrix data to GPU...\n");
      fflush(stderr);
      #endif

      // copy the matrix' data to the GPU
      checkCUDAError(cudaMemcpy(M.elements, h_M.elements, size(h_M) * sizeof(double), cudaMemcpyHostToDevice));
      break;

    case INIT_RANDOM:
      #ifdef DEBUG
      fprintf(stderr, "[DEBUG] Initialising matrix on GPU with random values...\n");
      fflush(stderr);
      #endif

      // generate random numbers on the gpu
      curandGenerateUniformDouble(generator, M.elements, size(h_M));

      // we have to make sure all numbers are positive, we do that but setting
      // them to their absolute value and adding a small epsilon
      mkPositive<<<1 + size(h_M) / blocksize, blocksize>>>(M.elements, size(h_M));
      break;
    case INIT_UNINITIALISED:
    default:
      break;
  }

  return M;
}

/**
 * Copies a matrix from GPU memory into host memory.
 *
 * @param d_M Pointer to the matrix in GPU memory.
 * @param h_M Matrix in host memory.
 */
void fromGPU(double* d_M, Matrix h_M)
{
  checkCUDAError(cudaMemcpy(h_M.elements, d_M, size(h_M) * sizeof(double), cudaMemcpyDeviceToHost));
}

// -----------------------------------------------------------------------------
// Auxiliary stuff
// -----------------------------------------------------------------------------

/**
 * Performs component-wise exponentiation of the values in M. The values in M
 * are used as bases and raised to the power given by exponent.
 *
 * @param d_M      Bases, in GPU memory.
 * @param d_R      Target for the results.
 * @param exponent Exponent.
 * @param size     Number of elements in M.
 */
__global__ void componentwise_exp(double* d_M, double* d_R, double exponent, size_t size)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < size)
    d_R[idx] = pow(d_M[idx], exponent);
}

/**
 * Perform component-wise multiplication of the values in M and N and stores the
 * results in R.
 *
 * @param d_M  First matrix, in GPU memory.
 * @param d_N  Second matrix, in GPU memory.
 * @param d_R  Result matrix, in GPU memory.
 * @param size Number of elements in M, N and R.
 */
__global__ void componentwise_mul(double* d_M, double* d_N, double* d_R, size_t size)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < size)
    d_R[idx] = d_M[idx] * d_N[idx];
}

/**
 *
 *
 * @param d_M   [description]
 * @param theta [description]
 * @param size  [description]
 */
__global__ void mk_smoothing_matrix(double * d_S, double theta, size_t size)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < size * size)
  {
    unsigned row = idx % size
           , col = idx / size
           ;

    d_S[idx] = (1.0 - theta) * (row == col ? 1 : 0) + theta/size;
  }
}

/**
 *
 *
 * @param d_A    [description]
 * @param d_B    [description]
 * @param d_C    [description]
 * @param handle [description]
 */
void cudaSub(Matrix d_A, Matrix d_B, Matrix d_C, cublasHandle_t handle)
{
  double mone = -1
       , one  =  1
       ;

  checkCUBLASStatus(cublasDgeam( handle
                               , CUBLAS_OP_N
                               , CUBLAS_OP_N
                               , d_C.rows
                               , d_C.cols
                               , &one
                               , d_A.elements
                               , d_A.rows
                               , &mone
                               , d_B.elements
                               , d_B.rows
                               , d_C.elements
                               , d_C.rows
                               ));
}

/**
 * Performs the operation C = A * B where C is a matrix with dimensions m x n,
 * A is a matrix of dimensions m x k and B is a matrix with dimensions k x n.
 *
 * @param d_A    [description]
 * @param opA    [description]
 * @param d_B    [description]
 * @param opB    [description]
 * @param d_C    [description]
 * @param handle [description]
 */
void cudaMul(Matrix d_A, cublasOperation_t opA, Matrix d_B, cublasOperation_t opB, Matrix d_C, cublasHandle_t handle)
{
  double zero = 0
       , one  = 1
       ;

  checkCUBLASStatus(cublasDgemm( handle
                               , opA
                               , opB
                               , d_C.rows
                               , d_C.cols
                               , opA == CUBLAS_OP_N ? d_A.cols : d_A.rows
                               , &one
                               , d_A.elements
                               , d_A.rows
                               , d_B.elements
                               , d_B.rows
                               , &zero
                               , d_C.elements
                               , d_C.rows
                               ));
}

// -----------------------------------------------------------------------------
// L2
// -----------------------------------------------------------------------------

/**
 * [updateL2Kernel description]
 * @param d_M           [description]
 * @param d_numerator   [description]
 * @param d_denominator [description]
 * @param n             [description]
 */
__global__ void updateL2Kernel(double* d_M, double* d_numerator, double* d_denominator, size_t n)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
    d_M[idx] *= d_numerator[idx] / d_denominator[idx];
}

/**
 * [runL2 description]
 * @param V    [description]
 * @param W    [description]
 * @param H    [description]
 * @param handle [description]
 * @param conf   [description]
 *
 * @returns      Reconstruction error after decomposition.
 */
double run_l2(Matrix V, Matrix W, Matrix H, cublasHandle_t handle, NMFConfig conf)
{
  #ifdef DEBUG
  // check GPU memory status
  checkGPUMem();
  #endif

  unsigned r = V.rows
         , c = V.cols
         , k = conf.components
         ;

  // allocate memory for the matrices on the GPU
  Matrix WH       = onGPU(r, c)
       , WtV      = onGPU(k, c)
       , WtWH     = onGPU(k, c)
       , VHt      = onGPU(r, k)
       , WHHt     = onGPU(r, k)
       , WH_V     = onGPU(r, c)
       , S        = onGPU(k, k)
       , W_smooth = onGPU(r, k)
       , H_smooth = onGPU(k, c)
       ;

  double error
       , errorNew
       ;

  mk_smoothing_matrix<<<1 + (k*k) / conf.blocksize, conf.blocksize>>>(S.elements, conf.theta, k);

  // find the reconstruction error
  cudaMul(W,        CUBLAS_OP_N, S, CUBLAS_OP_N, W_smooth, handle); // W_smooth = W * S
  cudaMul(S,        CUBLAS_OP_N, H, CUBLAS_OP_N, H_smooth, handle); // H_smooth = S * H
  cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH,       handle); // WH       = W * S * H
  cudaSub(WH,                    V,              WH_V,     handle); // WH_V     = WH - V
  cublasDnrm2(handle, r*c, WH_V.elements, 1, &errorNew);            // e        = ||WH_V||
  error = 1 + errorNew;

  unsigned iter = 0;
  while (errorNew < error && iter < conf.iterations)
  {
    #ifdef DEBUG
    if (!(iter % PRINT_DEBUG))
      fprintf(stderr, "[DEBUG] Iteration %u,  error: %lf\n", iter, errorNew);
      fflush(stderr);
    #endif

    error = errorNew;

    // update H
    cudaMul(W_smooth, CUBLAS_OP_N, H,  CUBLAS_OP_N, WH,   handle); // WH   = W_smooth * H
    cudaMul(W_smooth, CUBLAS_OP_T, V,  CUBLAS_OP_N, WtV,  handle); // WtV  = W_smooth_t * V
    cudaMul(W,        CUBLAS_OP_T, WH, CUBLAS_OP_N, WtWH, handle); // WtWH = Wt * WH
    updateL2Kernel<<<1 + (k*c) / conf.blocksize, conf.blocksize>>>(H.elements, WtV.elements, WtWH.elements, k*c);

    // smoothen H
    cudaMul(S, CUBLAS_OP_N, H, CUBLAS_OP_N, H_smooth, handle); // H_smooth = S * H

    // don't perform updates on W if we're going supervised
    if (!conf.supervised)
    {
      cudaMul(W,  CUBLAS_OP_N, H_smooth, CUBLAS_OP_N, WH,   handle); // WH   = W * H_smooth
      cudaMul(V,  CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, VHt,  handle); // VHt  = V * H_smooth_t
      cudaMul(WH, CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, WHHt, handle); // WHHt = WH * H_smooth_t

      updateL2Kernel<<<1 + (r*k) / conf.blocksize, conf.blocksize>>>(W.elements, VHt.elements, WHHt.elements, r*k);

      // smoothen W
      cudaMul(W, CUBLAS_OP_N, S, CUBLAS_OP_N, W_smooth, handle);      // W_smooth = W * S
    }

    // find the new reconstruction error
    cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH, handle);       // WH   = W * S * H
    cudaSub(WH,                    V,              WH_V,     handle); // WH_V     = WH - V
    cublasDnrm2(handle, r*c, WH_V.elements, 1, &errorNew);            // e    = ||WH_V||

    cudaDeviceSynchronize();
    ++iter;
  }

  #ifdef DEBUG
  fprintf(stderr, "[DEBUG] Error: %lf\n", errorNew);
  fprintf(stderr, "[DEBUG] Converged after %u iterations.\n", iter);
  fflush(stderr);
  #endif

  freeGPU(WH);
  freeGPU(WtV);
  freeGPU(WtWH);
  freeGPU(VHt);
  freeGPU(WHHt);
  freeGPU(WH_V);
  freeGPU(S);
  freeGPU(W_smooth);
  freeGPU(H_smooth);

  return errorNew;
}

// -----------------------------------------------------------------------------
// KL and IS
// -----------------------------------------------------------------------------

/**
 * Performs the component-wise operation VWHB2 = V*(WH^exp).
 *
 * @param d_WH    First matrix, in GPU memory.
 * @param d_V     Second matrix, in GPU memory.
 * @param d_VWHb2 Result matrix, in GPU memory.
 * @param exp     Exponent.
 * @param size    Number of element in the three matrices.
 */
__global__ void componentwise_exp_mul(double* d_WH, double* d_V, double* d_VWHb2, double exp, size_t size)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < size)
    d_VWHb2[idx] = d_V[idx] * pow(d_WH[idx], exp);
}

/**
 * Performs update steps for minimising the KL divergence. The performed
 * component-wise operation is M <- M * (N/D)^eta.
 *
 * @param d_M First matrix, in GPU memory. This matrix contains the
 *            reconstruction and is updated in this step.
 * @param n   Number of elements in M, N and D.
 * @param d_N Matrix that contains the numerators for the operation, in GPU memory.
 * @param d_D Matrix that contains the denominators for the operation, in GPU memory.
 * @param eta Step width for the operation.
 */
__global__ void update_matrix_kl(double* d_M, size_t n, double* d_N, double* d_D, double eta)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
  {
    d_M[idx] = d_M[idx] * pow(d_N[idx] / d_D[idx], eta);

    if (d_M[idx] < EPSILON)
      d_M[idx] += EPSILON;
  }
}

/**
 * Calculates the component-wise KL divergence between matrices A and B and
 * stores the results in matrix C. Strictly speaking, KL divergence is only
 * defined if all values are >0. As a workaround, we set the component-wise
 * divergence to 0 whenever we encounter an entry that is 0. For simplicity, we
 * treat all three matrices as vectors.
 *
 * @param d_A First matrix.
 * @param d_B Second matrix.
 * @param d_C Result matrix that stores the component-wise KL divergence between
 *            matrices A and B after the calculation.
 * @param n   Number of elements in A, B and C.
 */
__global__ void build_divergence(double* d_A, double* d_B, double* d_C, size_t n)
{
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
    d_C[idx] = d_A[idx] != 0 && d_B[idx] != 0
             ? d_A[idx] * log(d_A[idx] / d_B[idx]) - d_A[idx] + d_B[idx]
             : 0
             ;
}

/**
 * Run the nmf decomposition using KL divergence. We perform one iteration of
 * the decomposition in order to be able to compute the reconstruction error.
 * The we either perform as many iterations as specified by the configuration
 * or we terminate the decomposition as soon as the reconstruction error does
 * not improve for 1000 iterations. (We wait for 1000 iterations because the
 * reconstruction error as given by KL divergence might be unstable and start
 * to oscillate.)
 *
 * @param V      Matrix that should be decomposed, in GPU memory.
 * @param W      Components matrix, in GPU memory.
 * @param H      Weights matrix, in GPU memory.
 * @param handle The cuBLAS handle that is needed in order to perform cuBLAS
 *               calculations.
 * @param conf   The configuration for the nmf.
 *
 * @returns      Reconstruction error after decomposition.
 */
double run_kl(Matrix V, Matrix W, Matrix H, cublasHandle_t handle, NMFConfig conf)
{
  #ifdef DEBUG
  checkGPUMem();
  #endif

  unsigned r           = V.rows
         , c           = V.cols
         , k           = conf.components
         , notImproved = 0
         ;

  Matrix WH         = onGPU(r, c)
       , WHb1       = onGPU(r, c)
       , VWHb2      = onGPU(r, c)
       , VWHb2Ht    = onGPU(r, k)
       , WHb1Ht     = onGPU(r, k)
       , WtVWHb2    = onGPU(k, c)
       , WtWHb1     = onGPU(k, c)
       , divergence = onGPU(1, r*c)
       , ones       = onGPU(r*c, 1)
       , d_error    = onGPU(1, 1)
       , S          = onGPU(k, k)
       , W_smooth   = onGPU(r, k)
       , H_smooth   = onGPU(k, c)
       ;

  Matrix h_error = mk_zeros(1, 1);

  double error
       , errorNew
       ;

  mk_smoothing_matrix<<<1 + (k*k) / conf.blocksize, conf.blocksize>>>(S.elements, conf.theta, k);

  // prepare a matrix filled with 1 entries that is used to calculate the reconstruction error
  fillWith<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(ones.elements, r*c, 1.0);

  // ----------
  // update W
  if (!conf.supervised)
  {
    cudaMul(S, CUBLAS_OP_N, H,        CUBLAS_OP_N, H_smooth, handle); // H_smooth = S * H
    cudaMul(W, CUBLAS_OP_N, H_smooth, CUBLAS_OP_N, WH,       handle); // WH = W * H_smooth

    // numerator
    componentwise_exp_mul<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, V.elements, VWHb2.elements, conf.beta-2, r*c);
    cudaMul(VWHb2, CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, VWHb2Ht, handle); // VWHb2Ht = VWHb2 * H_smooth_t

    // denominator
    componentwise_exp<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, WHb1.elements, conf.beta-1, r*c);      // WHb1 = WH^(b-1)
    cudaMul(WHb1, CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, WHb1Ht, handle); // WHb1Ht = WHb1 * H_smooth_t

    update_matrix_kl<<<1 + (r*k) / conf.blocksize, conf.blocksize>>>(W.elements, size(W), VWHb2Ht.elements, WHb1Ht.elements, conf.eta);
  }

  // -----
  // update H
  cudaMul(W,        CUBLAS_OP_N, S, CUBLAS_OP_N, W_smooth, handle); // W_smooth = W * S
  cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH,       handle); // WH = W_smooth * H

  // numerator
  componentwise_exp_mul<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, V.elements, VWHb2.elements, conf.beta-2, size(WH));
  cudaMul(W, CUBLAS_OP_T, VWHb2, CUBLAS_OP_N, WtVWHb2, handle); // WtVWHb2 = Wt * VWHb2

  // denominator
  componentwise_exp<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, WHb1.elements, conf.beta-1, size(WH));      // WHb1 = WH^(b-1)
  cudaMul(W, CUBLAS_OP_T, WHb1, CUBLAS_OP_N, WtWHb1, handle); // WtWHb1 = Wt * VWHb2

  update_matrix_kl<<<1 + (k*c) / conf.blocksize, conf.blocksize>>>(H.elements, size(H), WtVWHb2.elements, WtWHb1.elements, conf.eta);

  // -----
  // find the reconstruction error
  cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH, handle); // WH = W * S * H
  build_divergence<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(V.elements, WH.elements, divergence.elements, r*c);
  cudaMul(divergence, CUBLAS_OP_N, ones, CUBLAS_OP_N, d_error, handle); // error = divergence * ones
  fromGPU(d_error.elements, h_error);
  errorNew = h_error.elements[0];

  unsigned iter = 0;
  while (iter < conf.iterations && notImproved < 1000)
  {
    error = errorNew;
    // -----
    // update W
    if (!conf.supervised)
    {
      cudaMul(S, CUBLAS_OP_N, H,        CUBLAS_OP_N, H_smooth, handle); // H_smooth = S * H
      cudaMul(W, CUBLAS_OP_N, H_smooth, CUBLAS_OP_N, WH,       handle); // WH = W * H_smooth

      // numerator
      componentwise_exp_mul<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, V.elements, VWHb2.elements, conf.beta-2, size(WH));
      cudaMul(VWHb2, CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, VWHb2Ht, handle); // VWHb2Ht = VWHb2 * H_smooth_t

      // denominator
      componentwise_exp<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, WHb1.elements, conf.beta-1, size(WH));      // WHb1 = WH^(b-1)
      cudaMul(WHb1, CUBLAS_OP_N, H_smooth, CUBLAS_OP_T, WHb1Ht, handle); // WHb1Ht = WHb1 * H_smooth_t

      update_matrix_kl<<<1 + (r*k) / conf.blocksize, conf.blocksize>>>(W.elements, size(W), VWHb2Ht.elements, WHb1Ht.elements, conf.eta);
    }

    // -----
    // update H
    cudaMul(W,        CUBLAS_OP_N, S, CUBLAS_OP_N, W_smooth, handle); // W_smooth = W * S
    cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH,       handle); // WH = W_smooth * H

    // numerator
    componentwise_exp_mul<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, V.elements, VWHb2.elements, conf.beta-2, size(WH));
    cudaMul(W, CUBLAS_OP_T, VWHb2, CUBLAS_OP_N, WtVWHb2, handle); // WtVWHb2 = Wt * VWHb2

    // denominator
    componentwise_exp<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(WH.elements, WHb1.elements, conf.beta-1, size(WH));      // WHb1 = WH^(b-1)
    cudaMul(W, CUBLAS_OP_T, WHb1, CUBLAS_OP_N, WtWHb1, handle); // WtWHb1 = Wt * WHb1

    update_matrix_kl<<<1 + (k*c) / conf.blocksize, conf.blocksize>>>(H.elements, size(H), WtVWHb2.elements, WtWHb1.elements, conf.eta);

    // -----
    // find the reconstruction error
    cudaMul(W_smooth, CUBLAS_OP_N, H, CUBLAS_OP_N, WH, handle); // WH = W * S * H
    cudaMul(W, CUBLAS_OP_N, H, CUBLAS_OP_N, WH, handle); // WH = W * S * H
    build_divergence<<<1 + (r*c) / conf.blocksize, conf.blocksize>>>(V.elements, WH.elements, divergence.elements, size(V));
    cudaMul(divergence, CUBLAS_OP_N, ones, CUBLAS_OP_N, d_error, handle); // error = divergence * ones
    fromGPU(d_error.elements, h_error);
    errorNew = h_error.elements[0];

    notImproved = errorNew < error
                ? 0
                : notImproved + 1
                ;

    #ifdef DEBUG
    if (!(iter % PRINT_DEBUG))
      fprintf(stderr, "[DEBUG] Iteration %u, error = %lf\n", iter, error);
    #endif

    ++iter;
  }

  #ifdef DEBUG
  fprintf(stderr, "[DEBUG] Error: %lf\n", errorNew);
  fprintf(stderr, "[DEBUG] Converged after %u iterations.\n", iter);
  fflush(stderr);
  #endif

  free_matrix(h_error);

  freeGPU(WH);
  freeGPU(WHb1);
  freeGPU(VWHb2);
  freeGPU(VWHb2Ht);
  freeGPU(WHb1Ht);
  freeGPU(WtVWHb2);
  freeGPU(WtWHb1);
  freeGPU(divergence);
  freeGPU(ones);
  freeGPU(d_error);
  freeGPU(S);
  freeGPU(W_smooth);
  freeGPU(H_smooth);

  return errorNew;
}

// -----------------------------------------------------------------------------

/**
 * Prepare everything for runnin the nmf. Setup cuBLAS and copy the data into
 * GPU memory. Then invoke the actual nmf.
 *
 * @param config The configuration for the nmf. We have to pass it as a void*
 *               because of pthread.
 */
void * nmf(void * config)
{
  NMFConfig conf = *((NMFConfig*) config);

  fprintf(stderr, "[DEBUG] Running on GPU %u\n", conf.onGPU);
  checkCUDAError(cudaSetDevice(conf.onGPU));

  Matrix h_V = conf.h_V
       , h_W = conf.h_W
       , h_H = conf.h_H
       ;

  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, conf.seed);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  conf.blocksize = prop.maxThreadsPerBlock;

  // copy all the data into GPU memory
  Matrix V = toGPU(h_V, INIT_COPY,   generator, conf.blocksize)
       , W = toGPU(h_W, conf.supervised
                        ? INIT_COPY
                        : INIT_RANDOM, generator, conf.blocksize)
       , H = toGPU(h_H, INIT_COPY, generator, conf.blocksize)
       ;

  double error;

  // create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Find a reasonable starting point by running 100 iterations of L2
  if (conf.warm && (conf.norm == NORM_KL || conf.norm == NORM_IS))
  {
    unsigned iter = conf.iterations;
    conf.iterations = 100;
    run_l2(V, W, H, handle, conf);
    conf.iterations = iter;
  }

  switch (conf.norm)
  {
    // KL and IS divergence are basically the same and only need different
    // parameters for beta
    case NORM_KL:
    case NORM_IS:
      error = run_kl(V, W, H, handle, conf);
      break;

    case NORM_L2:
    default:
      error = run_l2(V, W, H, handle, conf);
  }

  // destroy the handle
  cublasDestroy(handle);

  // copy the results into host memory
  fromGPU(W.elements, h_W);
  fromGPU(H.elements, h_H);

  // scale the result matrices
  unsigned r, c;
  double sum;
  for (c = 0; c < h_W.cols; ++c)
  {
    sum = 0;
    for (r = 0; r < h_W.rows; ++r)
      sum += get_matrix(h_W, r, c);

    for (r = 0; r < h_W.rows; ++r)
      h_W = set_matrix(&h_W, r, c, get_matrix(h_W, r, c) / sum);

    for (r = 0; r < h_H.cols; ++r)
      h_H = set_matrix(&h_H, c, r, get_matrix(h_H, c, r) * sum);
  }

  freeGPU(V);
  freeGPU(W);
  freeGPU(H);

  ((NMFConfig*) config)->reconstructionError = error;

  return NULL;
}
