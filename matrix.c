// -----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
// -----------------------------------------------------------------------------
#include "matrix.h"
// -----------------------------------------------------------------------------
#define EPSILON (1e-16)
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
Matrix mk_zeros(unsigned rows, unsigned cols)
{
  Matrix res;
  res.rows = rows;
  res.cols = cols;
  res.elements = (double*) calloc(rows*cols, sizeof(double));
  return res;
}

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
Matrix mk_matrix_fill(unsigned rows, unsigned cols, double v)
{
  Matrix res;
  res.rows = rows;
  res.cols = cols;
  res.elements = (double*) malloc(rows * cols * sizeof(double));

  unsigned i;
  for (i = 0; i < rows*cols; ++i)
    res.elements[i] = v;

  return res;
}

/**
 * Creates a matrix and fills the entries with random values. Note that random
 * numbers should have been seeded _before_.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @returns    The created matrix.
 */
Matrix mk_random(unsigned rows, unsigned cols)
{
  Matrix res;
  res.rows = rows;
  res.cols = cols;
  res.elements = (double*) malloc(rows * cols * sizeof(double));

  unsigned i;
  for (i = 0; i < rows*cols; ++i)
    res.elements[i] = ((double) rand() / (double) RAND_MAX);

  return res;
}

/**
 * Copies a matrix.
 *
 * @param M Matrix that should be copied.
 *
 * @returns A copy of matrix M.
 */
Matrix copy_matrix(Matrix M)
{
  Matrix res;
  res.rows = M.rows;
  res.cols = M.cols;
  res.elements = (double*) malloc(M.rows * M.cols * sizeof(double));
  memcpy(res.elements, M.elements, M.rows * M.cols * sizeof(double));
  return res;
}

/**
 * Frees the memory that is occupied by the given matrix.
 *
 * @param M Matrix that should be freed.
 */
void free_matrix(Matrix M)
{
  M.rows = 0;
  M.cols = 0;

  free(M.elements);
  M.elements = NULL;
}

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
Matrix read_matrix(char* filename, int skiprows, int skipcols)
{
  unsigned rows = 0
         , cols = 0
         , pos
         , len
         , i
         , j
         ;

  int chr
    , _
    ;
  FILE* fp = fopen(filename, "r");

  // first, we'll have to read the whole file once to figure out how many rows and columns we have
  while ((chr = fgetc(fp)) != '\n')
    if (chr == '\t')
      ++cols;
  ++cols;
  ++rows;
  while ((chr = fgetc(fp)) != EOF)
    if (chr == int('\n'))
      ++rows;

  // we have to ignore the first row and the first 4 columns
  rows -= skiprows; cols -= skipcols;
  Matrix res = mk_zeros(rows, cols);
  res.colnames = (char**) calloc(cols, sizeof(char *));

#ifdef DEBUG
  fprintf(stderr, "[DEBUG] Found %u rows and %u columns\n", rows, cols);
#endif

  // reset the file pointer and read the column names
  fseek(fp, 0, 0);

  // skip the first 4 columns
  for (i = 0; i < skipcols; ++i)
    while (fgetc(fp) != '\t');

  // read the column names
  i = 0;
  while (i < cols)
  {
    // remember the offset of where the column name starts
    len = 0;
    pos = ftell(fp);

    // count the length of the column
    chr = fgetc(fp);
    while (chr != '\t' && chr != '\r' && chr != '\n')
    {
      len += 1;
      chr = fgetc(fp);
    }

    res.colnames[i] = (char*) calloc(len+1, sizeof(char));

    fseek(fp, pos, 0);
    for (j = 0; j < len; ++j)
      res.colnames[i][j] = fgetc(fp);
    res.colnames[i][j] = '\0';

    fgetc(fp);
    ++i;
  }

  for (i = 0; i < rows; ++i)
  {
    // skip the first 4 columns
    for (j = 0; j < skipcols; ++j)
      while ((chr = fgetc(fp)) != '\t');

    for (j = 0; j < cols-1; ++j)
      _ = fscanf(fp, "%lf ", &res.elements[j*rows+i]);
    _ = fscanf(fp, "%lf\n", &res.elements[j*rows+i]);
  }

  fclose(fp);
  return res;
}

/**
 * Writes the given matrix to the given stream.
 * Again, for cuBLAS, we have to work with column-major order.
 *
 * @param  stream Where to write the matrix.
 * @param  M      The matrix that should be written.
 */
void write_matrix(FILE* stream, Matrix M)
{
  unsigned i, j;
  for (i = 0; i < M.rows; ++i)
  {
    for (j = 0; j < M.cols-1; ++j)
      fprintf(stream, "%lf\t", M.elements[j*M.rows+i]);
    fprintf(stream, "%lf\n", M.elements[j*M.rows+i]);
  }
}

/**
 * Writes the components found by nmf to the given output file.
 *
 * @param outFile Path to the output file.
 * @param inFile  Path to the input file.
 * @param M       Matrix that should be written to outFile.
 */
void write_components(char* outFile, char* inFile, Matrix M)
{
  FILE* out = fopen(outFile, "w");
  FILE* in  = fopen(inFile,  "r");

  int chr;
  unsigned i, j;

  // write the header. first copy the first 4 columns
  for (j = 0; j < 4; ++j)
  {
    while ((chr = fgetc(in)) != '\t')
      fputc(chr, out);
    fputc(chr, out);
  }
  // skip until the end of the line
  while (fgetc(in) != '\n');

  // now add an entry for each component
  for (j = 1; j < M.cols; ++j)
    fprintf(out, "%u\t", j);
  fprintf(out, "%u\n", j);

  // now write the matrix, but prepend the first 4 columns from the input file
  for (i = 0; i < M.rows; ++i)
  {
    // copy the first 4 columns
    for (j = 0; j < 4; ++j)
    {
      while ((chr = fgetc(in)) != '\t')
        fputc(chr, out);
      fputc(chr, out);
    }
    // skip until the end of the line
    while (fgetc(in) != '\n');

    for (j = 0; j < M.cols-1; ++j)
      fprintf(out, "%lf\t", M.elements[j*M.rows+i]);
    fprintf(out, "%lf\n", M.elements[j*M.rows+i]);
  }

  fclose(out);
  fclose(in);
}

/**
 * Writes the weights found by nmf to the given output file.
 *
 * @param outFile  Path to the ouput file.
 * @param inFile   Path to the input file.
 * @param M        Matrix that should be written to outFile.
 * @param rownames Names of the rows.
 */
void write_weights(char* outFile, char* inFile, Matrix M, char** rownames)
{
  FILE* out = fopen(outFile, "w");

  int chr;
  unsigned i, j;

  // set numbers for the header
  fprintf(out, "\t");
  for (j = 1; j < M.cols; ++j)
    fprintf(out, "%u\t", j);
  fprintf(out, "%u\n", j);

  // now write the matrix
  for (i = 0; i < M.rows; ++i)
  {
    fprintf(out, "%s\t", rownames[i]);
    for (j = 0; j < M.cols-1; ++j)
      fprintf(out, "%lf\t", M.elements[j*M.rows+i]);
    fprintf(out, "%lf\n", M.elements[j*M.rows+i]);
  }

  fclose(out);
}

/**
 * Gets the element at position (row, col) from the given matrix.
 *
 * @param M   Matrix.
 * @param row Row.
 * @param col Column.
 *
 * @returns   The element at (row, col) in matrix M.
 */
double get_matrix(Matrix M, unsigned row, unsigned col)
{
  // make sure we don't access undefined memory.
  assert(row < M.rows);
  assert(col < M.cols);

  return M.elements[col * M.rows + row];
}

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
Matrix set_matrix(Matrix * M, unsigned row, unsigned col, double v)
{
  // make sure we don't access undefined memory.
  assert(row < M->rows);
  assert(col < M->cols);

  M->elements[col * M->rows + row] = v;

  return *M;
}

/**
 * Calculates the size of the given matrix, i.e. the number of elements.
 *
 * @param  M Matrix.
 *
 * @returns  The size of M, i.e. the number of elements in M.
 */
size_t size(Matrix M)
{
  return M.rows * M.cols;
}

/**
 * Transposes the given matrix.
 *
 * @param   M Matrix that should be transposed.
 *
 * @returns   The transposed matrix.
 */
Matrix transpose_matrix(Matrix M)
{
  Matrix res;
  res.rows = M.cols;
  res.cols = M.rows;
  res.elements = (double*) malloc(res.rows * res.cols * sizeof(double));

  unsigned i, j;
  for (i = 0; i < res.cols; ++i)
    for (j = 0; j < res.rows; ++j)
      res.elements[i*res.rows+j] = M.elements[j*M.rows+i];

  return res;
}

/**
 * Performs the matrix multiplication M*N, given the dimensions of M and N are
 * compatible.
 *
 * @param  M First matrix.
 * @param  N Second matrix.
 *
 * @returns  The matrix product of M*N.
 */
Matrix multiply_matrix(Matrix M, Matrix N)
{
  // check for compatibility
  assert(M.cols == N.rows);

  Matrix res = mk_zeros(M.rows, N.cols);

  double v;

  unsigned i, j, k;
  for (i = 0; i < res.rows; ++i)
    for (j = 0; j < res.cols; ++j)
    {
      v = 0;
      for (k = 0; k < M.cols; ++ k)
        v += get_matrix(M, i, k) * get_matrix(N, k, j);

      set_matrix(&res, i, j, v);
    }
  return res;
}
