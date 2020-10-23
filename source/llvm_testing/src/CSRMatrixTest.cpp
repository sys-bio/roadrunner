/*
 * CSRMatrixTest.cpp
 *
 *  Created on: Jul 11, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "CSRMatrixTest.h"

#include "tests.h"
#include "llvm/ModelGeneratorContext.h"
#include "llvm/LLVMModelDataSymbols.h"
#include "llvm/ModelDataIRBuilder.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrSparse.h"


#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif
#include "llvm/LLVMIncludes.h"
#include "llvm/AssignmentRuleEvaluator.h"
#include "llvm/EvalInitialConditionsCodeGen.h"
#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif


#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/SBMLReader.h>

#include "llvm/LLVMModelData.h"

#include <utility>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "cpplapack.h"

using namespace std;
using namespace rr;
using namespace llvm;
using namespace libsbml;


bool runSparseTest(const int m, const int n, const int nnz)
{
    double *A = (double*)calloc(m*n, sizeof(double));

    double *x = (double*)calloc(n, sizeof(double));

    double *y = (double*)calloc(m, sizeof(double));
    double *ys = (double*)calloc(m, sizeof(double));

    int true_nnz = 0;


    // row major:    offset = row*NUMCOLS + column
    // column major: offset = row + column*NUMROWS


    // set x and first row of A to ones.
    //for (int i = 0; i < n; i++) {
    //    A[0 + i*m] = 1;
    //    x[i] = 1;
    //}

    for (int k = 0; k < nnz; k++) {
        int i = rand() % m;
        int j = rand() % n;

        if (A[i + j*m] == 0) {
            A[i + j*m] = 1;
            true_nnz++;
        }
    }

    for (int k = 0; k < n; k++) {
        x[k] = rand() % 10;
    }

    //cout << "A: ";
    //for(int i = 0; i < m*n; i++) {
    //    cout << A[i] << ",";
    //}
    //cout << "\n";

    cout << "x: ";
    for(int i = 0; i < n; i++) {
        cout << x[i] << ",";
    }
    cout << "\n";


    cpp_dgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);

    cout << "dense y:  ";
    for(int i = 0; i < m; i++) {
        cout << y[i] << ",";
    }
    cout << "\n";


    vector<uint> rowidx;
    vector<uint> colidx;
    vector<double> values;

    for(uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            double value = A[i + j*m];

            if (value != 0)
            {
                rowidx.push_back(i);
                colidx.push_back(j);
                values.push_back(value);
            }
        }
    }

    assert(rowidx.size() == true_nnz);
    assert(colidx.size() == true_nnz);
    assert(values.size() == true_nnz);

    csr_matrix *mat = csr_matrix_new(m, n, rowidx, colidx, values);

    csr_matrix_dgemv(1, mat, x, 0, ys);

    cout << "sparse y: ";
    for(int i = 0; i < m; i++) {
        cout << ys[i] << ",";
    }
    cout << "\n";

    // change some values
    for(int k = 0; k < true_nnz; k++)
    {
        int i = rowidx[k];
        int j = colidx[k];
        double value = rand() % 10;

        csr_matrix_set_nz(mat, i, j, value);
        A[i + j*m] = value;

        assert(csr_matrix_get_nz(mat, i, j) == value);
    }

    memset(y, 0, m*sizeof(double));
    cpp_dgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);

    cout << "dense y:  ";
    for (int i = 0; i < m; i++)
    {
        cout << y[i] << ",";
    }
    cout << "\n";


    memset(ys, 0, m*sizeof(double));
    csr_matrix_dgemv(1, mat, x, 0, ys);

    cout << "sparse y: ";
    for (int i = 0; i < m; i++)
    {
        cout << ys[i] << ",";
    }
    cout << "\n";

    csr_matrix_delete(mat);


    free(A);
    free(x);
    free(y);
    free(ys);

    return true;
}
