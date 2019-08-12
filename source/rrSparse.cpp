/*
 * rrCSparse.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: andy
 */

#pragma hdrstop
#include "rrSparse.h"
#include "rrLogger.h"
#include "rrExecutableModel.h"
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <limits>
#include "rrStringUtils.h"

#if defined(_WIN32) || defined(__WIN32__)
#define isnan _isnan
#else
#define isnan std::isnan
#endif

namespace rr {

using namespace std;

// sorting predicate for vector of column values,
// sort on the index, take value along for the ride.
struct sort_pred
{
    bool operator()(const std::pair<unsigned,double> &left, const std::pair<unsigned,double> &right)
    {
        return left.first < right.first;
    }
};

csr_matrix* csr_matrix_new(unsigned m, unsigned n,
	const std::vector<unsigned>& rowidx, const std::vector<unsigned>& colidx,
	const std::vector<double>& values)
{
	char err[64];
	unsigned nnz = rowidx.size();

	if (colidx.size() != nnz || values.size() != nnz)
	{
		throw runtime_error("rowidx, colidx and values must be the same length");
	}

	for (unsigned i = 0; i < nnz; i++)
	{
		if (rowidx[i] >= m)
		{
			snprintf(err, sizeof(err) / sizeof(char),
				"rowidx[%i] == %i >= row count %i", i, rowidx[i], m);
			throw runtime_error(err);
		}
		if (colidx[i] >= n)
		{
			snprintf(err, sizeof(err) / sizeof(char),
				"colidx[%i] == %i >= column count %i", i, colidx[i], n);
			throw runtime_error(err);
		}
	}

	csr_matrix* mat = (csr_matrix*)calloc(1, sizeof(csr_matrix));

	// values that will get stuffed into struct
	vector<double> mvalues;
	vector<unsigned> mcolidx;
	vector<unsigned> mrowptr;

	mat->m = m;
	mat->n = n;
	mat->nnz = 0;

	// first index
	mrowptr.push_back(0);

	for (unsigned i = 0; i < m; i++)
	{
		// find all the values in this row
		vector<pair<unsigned, double> > cols;
		for (unsigned j = 0; j < nnz; j++)
		{
			if (i == rowidx[j])
			{
				cols.push_back(pair<unsigned, double>(colidx[j], values[j]));
			}
		}

		// sort by the column index
		sort(cols.begin(), cols.end(), sort_pred());

		for (vector<pair<unsigned, double> >::const_iterator j = cols.begin(); j != cols.end(); j++)
		{
			mcolidx.push_back(j->first);
			mvalues.push_back(j->second);
		}
		mrowptr.push_back((mat->nnz += cols.size()));
	}

	assert(mat->nnz == nnz);
	assert(mcolidx.size() == nnz);
	assert(mvalues.size() == nnz);
	assert(mrowptr.size() == mat->m + 1);

	mat->rowptr = (unsigned*)calloc(mat->m + 1, sizeof(unsigned));
	mat->colidx = (unsigned*)calloc(nnz, sizeof(unsigned));
	mat->values = (double*)calloc(nnz, sizeof(double));

	memcpy(mat->rowptr, &mrowptr[0], (mat->m + 1) * sizeof(unsigned));
	if (mcolidx.size() > 0) {
		memcpy(mat->colidx, &mcolidx[0], nnz * sizeof(unsigned));
	}
	if (mvalues.size() > 0) {
		memcpy(mat->values, &mvalues[0], nnz * sizeof(double));
	}

    return mat;
}

bool csr_matrix_set_nz(csr_matrix* mat, unsigned row, unsigned col, double val)
{
    if (mat && row <= mat->m && col <= mat->n)
    {
        for (unsigned k = mat->rowptr[row]; k < mat->rowptr[row + 1]; k++)
        {
            if (mat->colidx[k] == col)
            {
                mat->values[k] = val;
                return true;
            }
        }
    }
    return false;
}

double csr_matrix_get_nz(const csr_matrix* mat, unsigned row, unsigned col)
{
    if (mat && row <= mat->m && col <= mat->n)
    {
        for (unsigned k = mat->rowptr[row]; k < mat->rowptr[row + 1]; k++)
        {
            if (mat->colidx[k] == col)
            {
                return mat->values[k];
            }
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void csr_matrix_delete(csr_matrix* mat)
{
    if (mat)
    {
        free(mat->colidx);
        free(mat->rowptr);
        free(mat->values);
        free(mat);
    }
}

void csr_matrix_dgemv(double alpha, const csr_matrix* A, const double* x,
        double beta, double* y)
{
    const unsigned m = A->m;
    unsigned *rowptr = A->rowptr;
    unsigned *colidx = A->colidx;
    double *values = A->values;
    if (beta == 0.0)
    {
        for (unsigned i = 0; i < m; i++)
        {
            double yi = 0.0;
            for (unsigned k = rowptr[i]; k < rowptr[i + 1]; k++)
            {
                yi = yi + alpha * values[k] * x[colidx[k]];
            }
            y[i] = yi;
        }
    }
    else
    {
        for (unsigned i = 0; i < m; i++)
        {
            double yi = beta * y[i];
            for (unsigned k = rowptr[i]; k < rowptr[i + 1]; k++)
            {
                yi = yi + alpha * values[k] * x[colidx[k]];
            }
            y[i] = yi;
        }
    }
}


double csr_matrix_ddot(int row, const csr_matrix *A, const double *x)
{
    assert(row < A->m && "invalid row");
    unsigned *rowptr = A->rowptr;
    unsigned *colidx = A->colidx;
    double *values = A->values;

    double sum = 0.0;
    for (unsigned k = rowptr[row]; k < rowptr[row + 1]; k++)
    {
        sum = sum + values[k] * x[colidx[k]];
    }
    return sum;
}

void csr_matrix_fill_dense(const csr_matrix *A, double *dense)
{
    unsigned *rowptr = A->rowptr;
    unsigned *colidx = A->colidx;
    double *values = A->values;

    memset(dense, 0, A->m * A->n * sizeof(double));

    // m rows x n cols
    // offset = row*NUMCOLS + column

    // loop over rows
    for (unsigned i = 0; i < A->m; i++)
    {
        // get the columns
        for (unsigned k = rowptr[i]; k < rowptr[i + 1]; k++)
        {
            uint col = colidx[k];
            dense[i * A->n + col] = values[k];
        }
    }
}

void csr_matrix_dump_binary(const csr_matrix *x, std::ostream& out) 
{
	rr::saveBinary(out, x->m);
	rr::saveBinary(out, x->n);
	rr::saveBinary(out, x->nnz);
	out.write((char*)(x->values), x->nnz * sizeof(double));
	out.write((char*)(x->colidx), x->nnz * sizeof(unsigned));
	out.write((char*)(x->rowptr), (x->m + 1) * sizeof(unsigned));
}

csr_matrix* csr_matrix_new_from_binary(std::istream& in) 
{
	csr_matrix* x = (csr_matrix*)malloc(sizeof(csr_matrix));
	rr::loadBinary(in, x->m);
	rr::loadBinary(in, x->n);
	rr::loadBinary(in, x->nnz);

	x->values = (double*)malloc(x->nnz * sizeof(double));
	in.read((char*)(x->values), x->nnz * sizeof(double));

	x->colidx = (unsigned*)malloc(x->nnz*sizeof(unsigned));
	in.read((char*)(x->colidx), x->nnz * sizeof(unsigned));

	x->rowptr = (unsigned*)malloc((x->m + 1) * sizeof(unsigned));
	in.read((char*)(x->rowptr), (x->m + 1) * sizeof(unsigned));
	return x;
}

std::ostream& operator <<(std::ostream& os, const csr_matrix* mat)
{
    os.precision(2);                   // Set 2 digits past the decimal
    os.flags(ios::right | ios::fixed); // Fixed point, right justified

    os << "csr_matrix\n";

    if (mat == 0)
    {
        os << "NULL" << endl;
        return os;
    }

    os << "rows: " << mat->m << ", columns: " << mat->n;
    os << ", non-zero entries: " << mat->nnz << endl;
    if (mat->nnz > 0)
    {
        os << '[';
        for (unsigned m = 0; m < mat->m; ++m)
        {
            if (m != 0)
                os << ' ';
            os << '[';
            for (unsigned n = 0; n < mat->n; ++n)
            {
                double val = csr_matrix_get_nz(mat, m, n);
                os.width(7);
                os << (isnan(val) ? 0 : val);
                if (n < mat->n - 1)
                {
                    os << ", ";
                }
                else
                {
                    os << ']';
                }
            }
            if (m < mat->m - 1)
            {
                os << endl;
            }
            else
            {
                os << ']' << endl;
            }
        }
    }
    else
    {
        os << "[[]]" << endl;
    }

    return os;
}

}


