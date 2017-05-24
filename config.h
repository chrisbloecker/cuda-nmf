#ifndef __CONFIG_H__
#define __CONFIG_H__
// -----------------------------------------------------------------------------
#include "nmf.h"
// -----------------------------------------------------------------------------
#define DEFAULT_COMPONENTS (30)
#define DEFAULT_ITERATIONS (100000)
#define DEFAULT_SEED       (42)
#define DEFAULT_NORM       (NORM_L2)
#define DEFAULT_SPARSITY   (SPARSE_NO)
#define DEFAULT_STEP       (1.0)
#define DEFAULT_SMOOTHNESS (0.0)
#define DEFAULT_RUNS       (4)
#define DEFAULT_PAR        (1)
// -----------------------------------------------------------------------------

/**
 * @struct Configuration
 * @brief The configuration for a run.
 *
 * @var Configuration::inFile
 * Input file that contains a matrix that should be decomposed with nmf.
 *
 * @var Configuration::outW
 * Path to the output file for the discovered components.
 *
 * @var Configuration::outH
 * Path to the output file for the discovered weights.
 *
 * @var Configuration::runs
 * Number of decomposition runs to perform.
 *
 * @var Configuration::par
 * Number of decompositions to run in parallel.
 *
 * @var Configuration::nmfConfig
 * Configuration for the nmf algorithm.
 */
typedef struct {
  char * inFile
     , * outW
     , * outH
     ;

  unsigned runs
         , par
         ;

  NMFConfig nmfConfig;
} Configuration;

// -----------------------------------------------------------------------------

/**
 * Prints the help message to stream.
 *
 * @param stream Where to print the help message.
 */
void help(FILE* stream);

/**
 * Parses the command line arguments and returns a configuration.
 *
 * @param argc Number of command line arguments.
 * @param argv The command line arguments.
 *
 * @returns    A configuration obtained from the command line arguments.
 */
Configuration parseArgs(int argc, char* argv[]);

/**
 * Checks whether the given configuration is sane.
 *
 * @param conf The configuration.
 *
 * @returns    True  if the given configuration is sane
 *             False otherwise.
 */
bool checkConfig(Configuration conf);

#endif
