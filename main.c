#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
// -----------------------------------------------------------------------------
#include "config.h"
#include "matrix.h"
#include "nmf.h"
// -----------------------------------------------------------------------------
#define max(m,n) ((m)>(n)?(m):(n))
#define min(m,n) ((m)<(n)?(m):(n))
// -----------------------------------------------------------------------------

void * do_something(void* arg)
{
  fprintf(stderr, "%i\n", *((int*) arg));
  return NULL;
}

/**
 * [main description]
 * @param  argc [description]
 * @param  argv [description]
 * @return      [description]
 */
int main(int argc, char* argv[])
{
  Configuration conf = parseArgs(argc, argv);
  bool err = checkConfig(conf);

  if (err)
    help(stderr);

  int gpus = 0
    , gpu
    ;
  cudaGetDeviceCount(&gpus);
  conf.par = gpus;

  #ifdef DEBUG
  fprintf(stderr, "[DEBUG] Number of CUDA capable GPUs detected %u\n", gpus);

  // display info on each GPU^
  for (gpu = 0; gpu < gpus; ++gpu)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);

    fprintf(stderr, "[DEBUG] GPU #%i\n", gpu);
    fprintf(stderr, "[DEBUG]   GPU model: %s\n", prop.name);
    fprintf(stderr, "[DEBUG]   Available memory on device: %lf GB\n", prop.totalGlobalMem / 1e9);
    fprintf(stderr, "[DEBUG]   Number of multi processors: %i\n", prop.multiProcessorCount);
    fprintf(stderr, "[DEBUG]   Maximum number of threads per block: %i\n", prop.maxThreadsPerBlock);
    fprintf(stderr, "[DEBUG]   Compute mode is set to: %i\n", prop.computeMode);
  }
  fflush(stderr);
  #endif


  if (!err)
  {
    #ifdef DEBUG
    fprintf(stderr, "[DEBUG] Reading input from file %s. Decomposing into %u components.\n", conf.inFile, conf.nmfConfig.components);
    #endif

    srand(conf.nmfConfig.seed);

    pthread_t * threads = (pthread_t*) calloc(conf.par, sizeof(pthread_t*));
    NMFConfig * threadArgs = (NMFConfig*) calloc(conf.par, sizeof(NMFConfig));
    int threadError;

    double reconstructionError = 1e32;

    unsigned run
           , thread
           , numThreads
           ;

    // the mutation counts
    Matrix h_V = read_matrix(conf.inFile, 1, 4)
         , h_W = conf.nmfConfig.supervised
               ? read_matrix(conf.nmfConfig.w, 1, 4)
               : mk_matrix_fill(h_V.rows, conf.nmfConfig.components, 1.0)
         , h_H = mk_matrix_fill(conf.nmfConfig.components, h_V.cols, 1.0)
         , h_W_best = copy_matrix(h_W)
         , h_H_best = copy_matrix(h_H)
         ;

    fprintf(stdout, "[INFO ] Read input matrix with dimension %u x %u\n", h_V.rows, h_V.cols);
    fprintf(stdout, "[INFO ] Decomposing into %u components\n", conf.nmfConfig.components);

    // prepare the data structures for all threads first. we will reuse the
    // memory so we don't have to allocate and free memory all the time.
    for (thread = 0; thread < conf.par; ++thread)
    {
      threadArgs[thread]     = conf.nmfConfig;
      threadArgs[thread].h_V = copy_matrix(h_V);
      threadArgs[thread].h_W = copy_matrix(h_W);
      threadArgs[thread].h_H = copy_matrix(h_H);
    }

    for (run = 0; run < int((double) conf.runs / (double) conf.par + 0.5); ++run)
    {
      numThreads = min(conf.par,conf.runs-(run*conf.par));

      // start threads after setting their data back to default
      for (thread = 0; thread < numThreads; ++thread)
      {
        threadArgs[thread].onGPU = thread;
        threadArgs[thread].seed  = random();
        memcpy(threadArgs[thread].h_W.elements, h_W.elements, size(h_W));
        memcpy(threadArgs[thread].h_H.elements, h_H.elements, size(h_H));

        threadError = pthread_create(threads+thread, NULL, nmf, threadArgs+thread);
        assert(threadError == 0);
      }

      // wait for all threads to finish
      for (thread = 0; thread < numThreads; ++thread)
      {
        threadError = pthread_join(threads[thread], NULL);
        assert(threadError == 0);

        if (threadArgs[thread].reconstructionError < reconstructionError)
        {
          reconstructionError = threadArgs[thread].reconstructionError;
          memcpy(h_W_best.elements, threadArgs[thread].h_W.elements, size(h_W)*sizeof(double));
          memcpy(h_H_best.elements, threadArgs[thread].h_H.elements, size(h_H)*sizeof(double));

          #ifdef DEBUG
          fprintf(stderr, "[DEBUG] Found better decomposition, reconstruction error: %lf\n", reconstructionError);
          #endif
        }
      }
    }

    // clean up the data from the threads
    for (thread = 0; thread < conf.par; ++thread)
    {
      free_matrix(threadArgs[thread].h_V);
      free_matrix(threadArgs[thread].h_W);
      free_matrix(threadArgs[thread].h_H);
    }

    // write the results to the specified files
    fprintf(stdout, "[INFO ] Writing W to %s\n", conf.outW);
    write_components(conf.outW, conf.inFile, h_W_best);

    fprintf(stdout, "[INFO ] Writing Ht to %s\n", conf.outH);
    Matrix h_Ht = transpose_matrix(h_H_best);
    write_weights(conf.outH, conf.inFile, h_Ht, h_V.colnames);

    // clean up
    free_matrix(h_V);
    free_matrix(h_W);
    free_matrix(h_H);
    free_matrix(h_Ht);
    free_matrix(h_W_best);
    free_matrix(h_H_best);
  }

  return err;
}
