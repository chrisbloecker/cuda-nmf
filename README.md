# cuda-nmf
Non-negative matrix factorisation with CUDA

# Current status
At the moment, we can perform non-negative matrix factorisation minimising the L2 norm, Kullback-Leibler divergence and Itakura-Saito divergence. The parameters for the factorisation may be given as command line arguments following the below given usage.

# Usage
```
Usage: nmf <matrix.in> [options]
  Required arguments:
    matrix.in       File that contains the input matrix.

  Options:
    -k <number>         Number of components for the decomposition.                               (default: 30)
    -i <number>         Number of iterations to perform.                                          (default: 100,000)
    --norm [l2|kl|is]   The norm to be used for the decomposition. Valid choices are: l2, kl, is. (default: l2)
                          - l2: Eucledian norm
                          - kl: Kullback-Leibler divergence
                          - is: Itakura-Saito divergence
    --wout <wout>       Output file for W matrix.                                                 (default: w.txt)
    --hout <hout>       Output file for H matrix.                                                 (default: h.txt)
    --seed <number>     Seed for the random number generator for initialising the W matrix.       (default: 42)
    --step <number>     Step size for the decomposition, should be in [0,1]                       (default: 1)
    --supervised <sigs> Use sigs file as W matrix.
    --warm              Run with a warm start, only applies to KL and IS divergence
    --runs <number>     Number of decompositions to run, best result will be used                 (default: 4)
```

# Setup
To use this program, you'll need to have __g++__ and __nvcc__ from the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed. Then, simply run `make` to compile the program or `make debug` to compile with debug output. __g++__ is used to compile the ordinary .c files, __nvcc__ is used to compile the CUDA .cu files and for linking. Note that `make` looks for the CUDA headers in `/usr/local/cuda/include/`.

# Input file format
The input file format for this program is that of a catalogue file. A catalogue file is a __tab-delimited__ text file __with header__. The header contains sample names which are used as row names in the weights output file. Each of the lines following the header starts with four columns, i.e. _Before_, _Ref_, _Alt_, and _Var_, identifying the respective trinucleotide context. The remainder of the line contains observation counts for the given trinucleotide context for each sample.

# Bibliography
* __Algorithms for Non-negative Matrix Factorization__, _Daniel D. Lee and Seung, H. Sebastian_, Advances in Neural Information Processing Systems 13, p. 556--562, 2001, MIT Press
* __Stability Analysis of Multiplicative Update Algorithms and Application to Nonnegative Matrix Factorization__, _Roland Badeau and Emmanuel Vincent_, IEEE Transactions on Neural Networks, Vok. 21, No. 12, December 2010
* __Nonsmooth Nonnegative Matrix Factorization (nsNMF)__, _Pascual-Montano, Alberto and Carazo, J. M. and Kochi, Kieko and Lehmann, Dietrich and Pascual-Marqui, Roberto D._, IEEE Trans. Pattern Anal. Mach. Intell., Vol. 28, No. 3, p. 403--415, March 2006
