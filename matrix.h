#ifndef __MATRIX_H__
#define __MATRIX_H__
// -----------------------------------------------------------------------------
#include <stdio.h>
// -----------------------------------------------------------------------------

/**
 * We only consider "real"-valued, two-dimensional matrices.
 *
 * @field rows     Number of rows of the matrix.
 * @field cols     Number of columns of the matrix.
 * @field elements The elements of the matrix.
 */
typedef struct {
  unsigned rows
         , cols
         ;
  double* elements;
  char** colnames;
} Matrix;

// -----------------------------------------------------------------------------

/**
 * Creates a matrix with the given number of rows and columns and fills all
 * entries with 0.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @returns    The created matrix.
 */
Matrix mk_zeros(unsigned rows, unsigned cols);

/**
 * Creates a matrix with the given number of rows and columns and fills all
 * entries with the given value v.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param v    Value to be used to fill all elements of the matrix with.
 *
 * @returns    The created matrix.
 */
Matrix mk_matrix_fill(unsigned rows, unsigned cols, double v);

/**
 * Creates a matrix and fills the entries with random values. Note that random
 * numbers should have been seeded _before_.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @returns    The created matrix.
 */
Matrix mk_random(unsigned rows, unsigned cols);

/**
 * Copies a matrix.
 *
 * @param M Matrix that should be copied.
 *
 * @returns A copy of matrix M.
 */
Matrix copy_matrix(Matrix M);

/**
 * Frees the memory that is occupied by the given matrix.
 *
 * @param M Matrix that should be freed.
 */
void free_matrix(Matrix M);

/**
 * Reads in a matrix from filename.
 * For cuBLAS, we have to store it in column-major order.
 *
 * @param filename File that contains a matrix and should be read.
 * @param skiprows Number of rows to skip.
 * @param skipcols Number of columns to skip.
 *
 * @return The read matrix.
 */
Matrix read_matrix(char* filename, int skiprows, int skipcols);

/**
 * Writes the given matrix to the given stream.
 * Again, for cuBLAS, we have to work with column-major order.
 *
 * @param  stream Where to write the matrix.
 * @param  M      The matrix that should be written.
 */
void write_matrix(FILE* stream, Matrix M);

/**
 * Writes the components found by nmf to the given output file.
 *
 * @param outFile Path to the output file.
 * @param inFile  Path to the input file.
 * @param M       Matrix that should be written to outFile.
 */
void write_components(char* outFile, char* inFile, Matrix M);

/**
 * Writes the weights found by nmf to the given output file.
 *
 * @param outFile  Path to the ouput file.
 * @param inFile   Path to the input file.
 * @param M        Matrix that should be written to outFile.
 * @param rownames Names of the rows.
 */
void write_weights(char* outFile, char* inFile, Matrix M, char** rownames);

/**
 * Gets the element at position (row, col) from the given matrix.
 *
 * @param M   Matrix.
 * @param row Row.
 * @param col Column.
 *
 * @returns   The element at (row, col) in matrix M.
 */
double get_matrix(Matrix M, unsigned row, unsigned col);

/**
 * Sets the element at position (row, col) in matrix M to v.
 *
 * @param M   Matrix that should be modified.
 * @param row Row.
 * @param col Column.
 * @param v   Value to store at position (row, col) in M.
 *
 * @returns   The modified matrix.
 */
Matrix set_matrix(Matrix * M, unsigned row, unsigned col, double v);

/**
 * Calculates the size of the given matrix, i.e. the number of elements.
 *
 * @param  M Matrix.
 *
 * @returns  The size of M, i.e. the number of elements in M.
 */
size_t size(Matrix M);

/**
 * Transposes the given matrix.
 *
 * @param   M Matrix that should be transposed.
 *
 * @returns   The transposed matrix.
 */
Matrix transpose_matrix(Matrix M);

/**
 * Performs the matrix multiplication M*N, given the dimensions of M and N are
 * compatible.
 *
 * @param  M First matrix.
 * @param  N Second matrix.
 *
 * @returns  The matrix product of M*N.
 */
Matrix multiply_matrix(Matrix M, Matrix N);

#endif
