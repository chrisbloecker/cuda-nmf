// -----------------------------------------------------------------------------
#include <string.h>
#include <stdlib.h>
// -----------------------------------------------------------------------------
#include "config.h"
// -----------------------------------------------------------------------------

/**
 * Prints the help message to stream.
 *
 * @param stream Where to print the help message.
 */
void help(FILE* stream)
{
  fprintf(stream, "Usage: nmf <matrix.in> [options]\n");
  fprintf(stream, "  Required arguments:\n");
  fprintf(stream, "    matrix.in       File that contains the input matrix.\n");
  fprintf(stream, "\n");
  fprintf(stream, "  Options:\n");
  fprintf(stream, "    -k <number>         Number of components for the decomposition.                               (default: 30)\n");
  fprintf(stream, "    -i <number>         Number of iterations to perform.                                          (default: 100,000)\n");
  fprintf(stream, "    --norm [l2|kl|is]   The norm to be used for the decomposition. Valid choices are: l2, kl, is. (default: l2)\n");
  fprintf(stream, "                          - l2: Eucledian norm\n");
  fprintf(stream, "                          - kl: Kullback-Leibler divergence\n");
  fprintf(stream, "                          - is: Itakura-Saito divergence\n");
  fprintf(stream, "    --wout <wout>       Output file for W matrix.                                                 (default: w.txt)\n");
  fprintf(stream, "    --hout <hout>       Output file for H matrix.                                                 (default: h.txt)\n");
  fprintf(stream, "    --seed <number>     Seed for the random number generator for initialising the W matrix.       (default: 42)\n");
  fprintf(stream, "    --step <number>     Step size for the decomposition, should be in [0,1]                       (default: 1)\n");
  fprintf(stream, "    --smooth <number>   Smoothness of the decomposition, should be in [0,1]                       (default: 0)\n");
  fprintf(stream, "    --supervised <sigs> Use sigs file as W matrix.\n");
  fprintf(stream, "    --warm              Run with a warm start, only applies to KL and IS divergence\n");
  fprintf(stream, "    --runs <number>     Number of decompositions to run, best result will be used                 (default: 4)\n");
}

/**
 * Checks whether filename is a path to an existing file.
 *
 * @param filename Path to check.
 *
 * @returns        True  if filename is a path to an existing file
 *                 False otherwise.
 */
bool fileExists(char* filename)
{
  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
    return false;
  fclose(fp);
  return true;
}

/**
 * Parses the command line arguments and returns a configuration.
 *
 * @param argc Number of command line arguments.
 * @param argv The command line arguments.
 *
 * @returns    A configuration obtained from the command line arguments.
 */
Configuration parseArgs(int argc, char* argv[])
{
  if (argc < 2)
  {
    help(stderr);
    exit(1);
  }

  Configuration conf;
  conf.inFile = argv[1];
  conf.outW   = (char*) "w.txt";
  conf.outH   = (char*) "h.txt";
  conf.runs   = DEFAULT_RUNS;
  conf.par    = DEFAULT_PAR;

  conf.nmfConfig.components  = DEFAULT_COMPONENTS;
  conf.nmfConfig.iterations  = DEFAULT_ITERATIONS;
  conf.nmfConfig.seed        = DEFAULT_SEED;
  conf.nmfConfig.norm        = DEFAULT_NORM;
  conf.nmfConfig.beta        = 2; // for L2 norm
  conf.nmfConfig.eta         = DEFAULT_STEP;
  conf.nmfConfig.theta       = DEFAULT_SMOOTHNESS;
  conf.nmfConfig.supervised  = false;
  conf.nmfConfig.w           = NULL;
  conf.nmfConfig.warm        = false;

  int i = 2;
  while (i < argc)
  {
    if (strcmp(argv[i], "-k") == 0)
      sscanf(argv[++i], "%u", &conf.nmfConfig.components);

    else if (strcmp(argv[i], "-i") == 0)
      sscanf(argv[++i], "%u", &conf.nmfConfig.iterations);

    else if (strcmp(argv[i], "--norm") == 0)
    {
      ++i;
      if (strcmp(argv[i], "l2") == 0)
        conf.nmfConfig.norm = NORM_L2;
      else if (strcmp(argv[i], "kl") == 0)
        conf.nmfConfig.norm = NORM_KL;
      else if (strcmp(argv[i], "is") == 0)
        conf.nmfConfig.norm = NORM_IS;
      else
        conf.nmfConfig.norm = NORM_UNDEFINED;

      switch (conf.nmfConfig.norm)
      {
        case NORM_L2:
          conf.nmfConfig.beta = 2;
          break;
        case NORM_KL:
          conf.nmfConfig.beta = 1;
          break;
        case NORM_IS:
          conf.nmfConfig.beta = 0;
          break;
        case NORM_UNDEFINED:
        default:
          ;
      }
    }

    else if (strcmp(argv[i], "--wout") == 0)
      conf.outW = argv[++i];

    else if (strcmp(argv[i], "--hout") == 0)
      conf.outH = argv[++i];

    else if (strcmp(argv[i], "--seed") == 0)
      sscanf(argv[++i], "%i", &conf.nmfConfig.seed);

    else if (strcmp (argv[i], "--step") == 0)
      sscanf(argv[++i], "%lf", &conf.nmfConfig.eta);

    else if (strcmp(argv[i], "--smooth") == 0)
      sscanf(argv[++i], "%lf", &conf.nmfConfig.theta);

    else if (strcmp (argv[i], "--runs") == 0)
      sscanf(argv[++i], "%u", &conf.runs);

    else if (strcmp(argv[i], "--supervised") == 0)
    {
      conf.nmfConfig.supervised = true;
      conf.nmfConfig.w = argv[++i];
    }

    else
    {
      fprintf(stdout, "[ERROR] Option not recognised: %s. Exiting.\n", argv[i]);
      help(stderr);
      exit(-1);
    }

    ++i;
  }

  // make sure the number of parallel processes is smaller or equal to the
  // number of runs
  conf.par = conf.runs < conf.par
           ? conf.runs
           : conf.par
           ;

  return conf;
}

/**
 * Checks whether the given configuration is sane.
 *
 * @param conf The configuration.
 *
 * @returns    True  if the given configuration is sane
 *             False otherwise.
 */
bool checkConfig(Configuration conf)
{
  bool err = false;

  if (!fileExists(conf.inFile))
  {
    err = true;
    fprintf(stdout, "[ERROR] Input file %s doesn't exist!\n", conf.inFile);
  }

  return err;
}
