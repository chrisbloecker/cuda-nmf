#ifndef __NMF_H__
#define __NMF_H__
// -----------------------------------------------------------------------------
#include <cublas_v2.h>
#include <stdbool.h>
// -----------------------------------------------------------------------------
#include "matrix.h"
// -----------------------------------------------------------------------------

/**
 * Enumeration to define which matrix initialisation should be used.
 */
typedef enum {
  /**
   * Copy the data from host memory into GPU memory
   */
  INIT_COPY,

  /**
   * Initialise the matrix in GPU memory with random values
   */
  INIT_RANDOM,

  /**
   * Leave the matrix in GPU memory uninitialised. This is likely to result in
   * non-deterministic behaviour regarding the result of the decomposition.
   */
  INIT_UNINITIALISED
} MatrixInitialisation
;

/**
 * Enumeration that specifies the norm that should be used for the nmf. Note
 * that KL and IS are not proper norms.
 */
typedef enum {
  /**
   * L2 norm (equivalent to Frobenius and Eucledian norm)
   */
  NORM_L2,

  /**
   * Kullback-Leibler divergence
   */
  NORM_KL,

  /**
   * Itakura-Saito divergence
   */
  NORM_IS,

  /**
   * When the norm has not been defined
   */
  NORM_UNDEFINED
} Norm
;

/**
 * @struct NMFConfig
 * @brief The configuration for the nmf.
 *
 * @var NMFConfig::blocksize
 * Number of threads that can execute concurrently on the available GPU.
 *
 * @var NMFConfig::components
 * Number of components the input matrix should be decomposed into.
 *
 * @var NMFConfig::iterations
 * Number of iterations to be performed before the decomposition terminates.
 *
 * @var NMFConfig::seed
 * Random number seed for matrix initialisations.
 *
 * @var NMFConfig::norm
 * Defines which "norm" to use for the nmf.
 *
 * @var NMFConfig::supervised
 * Defines whether the decomposition should be performed supervised or unsupervised.
 * If the decomposition is performed in supervised mode, there will be no updates
 * made to the W matrix during decomposition. W will be read from a given file.
 *
 * @var NMFConfig::w
 * Components that should be used for supervised decomposition.
 *
 * @var NMFConfig::warm
 * Defines whether a warm start should be done. Only applies to KL and IS divergence.
 * For a warm start, 100 iterations of nmf using L2 norm will be performed.
 * The decomposition found by that will be used as the starting point for the
 * decomposition using KL or IS divergence.
 *
 * @var NMFConfig::beta
 * Exponent for the update formulas. Different values of beta correspond to
 * different norm or divergences, respectively.
 *   0: Itakura-Saito divergence
 *   1: Kullback-Leibler divergence
 *   2: L2 / Frobenius / Eucledian norm
 *
 * @var NMFConfig::eta
 * The step size for the "gradient descent"
 *
 * @var NMFConfig::theta
 *
 *
 * @var NMFConfig::h_V
 *
 *
 * @var NMFConfig::h_W
 *
 *
 * @var NMFConfig::h_H
 *
 *
 */
typedef struct {
  unsigned blocksize
         , components
         , iterations
         ;

  // random number seed for matrix initialisation
  unsigned seed;

  // norm that should be minimised
  Norm norm;

  // for supervised learning
  bool supervised;

  // path to the file that contains the components for supervised nmf
  char* w;

  // perform a warm start, only applicable for KL and IS divergence
  bool warm;

  double beta                // to define the norm
       , eta                 // the step size for the "gradient descent"
       , theta               // controls the smoothness
       , reconstructionError // Reconstruction error after decomposition
       ;

  // The matrices in host memory
  Matrix h_V
       , h_W
       , h_H
       ;

  // defines on which GPU the decomposition should be execured
  int onGPU;
} NMFConfig;

// -----------------------------------------------------------------------------

/**
 * Prepare everything for runnin the nmf. Setup cuBLAS and copy the data into
 * GPU memory. Then invoke the actual nmf.
 *
 * @param config The configuration for the nmf. We have to pass it as a void*
 *               because of pthread.
 */
void * nmf(void * config);

#endif
