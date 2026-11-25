//
// Created by Ciaran on 10/03/2021.
//

#include "SVD.h"
#include "rrException.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include "rr-libstruct/lsMatrix.h"


namespace rr {

    SVD::SVD(ls::DoubleMatrix &matrix) : inputMatrix_(matrix) {
        nRows_ = (int) inputMatrix_.numRows();
        nCols_ = (int) inputMatrix_.numCols();
        lda_ = (int) inputMatrix_.numRows();
        ldu_ = (int) inputMatrix_.numRows();
        ldvt_ = (int) inputMatrix_.numCols();

        // resize out data types
        singularValues_.resize(1, std::min(nRows_, nCols_));
        leftSingularVectors_.resize(nRows_, nRows_);
        rightSingularVectors_.resize(nCols_, nCols_);

        /**
         * The lapack rountine at the core of this class
         * seems to expect the transpose of our input matrix
         * so we do that here in the constructor.
         * transpose should be done *after* collecting the initial dimensions
         * Moreover, the output matrices left and right require transposing
         * back before we give them to user
         */
        inputMatrixTranspose_ = std::unique_ptr<ls::DoubleMatrix> (inputMatrix_.getTranspose());

        compute();
    };

    /**
     * Example program:
     *  https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.html
     */
    void SVD::compute() {
        ls_int info, lwork;
        double wkopt;
        double *work;
        /* Query and allocate the optimal workspace */
        lwork = -1;
        // dgesdd_ or dgesvd_?
        // note: "dgesdd_" is an alternative algorithm. Implement only if needed.
        dgesvd_((char*)"All", (char*)"All", &nRows_, &nCols_, (*inputMatrixTranspose_).getArray(),
                &lda_, singularValues_.getArray(), leftSingularVectors_.getArray(),
                &ldu_, rightSingularVectors_.getArray(), &ldvt_, &wkopt, &lwork, &info);
        lwork = (int) wkopt;

        work = (double *) malloc(lwork * sizeof(double));
        /* Compute SVD */
        dgesvd_((char*)"All", (char*)"All", &nRows_, &nCols_, (*inputMatrixTranspose_).getArray(),
                &lda_, singularValues_.getArray(), leftSingularVectors_.getArray(),
                &ldu_, rightSingularVectors_.getArray(), &ldvt_, work, &lwork, &info);

        /* Check for convergence */
        if (info > 0) {
            throw DidNotConvergeException("The algorithm computing SVD failed to converge.\n");
        }
        // we do it this way to prevent memory leak from getTranspose
        ls::DoubleMatrix* tmp = leftSingularVectors_.getTranspose();
        leftSingularVectors_ = *tmp;
        delete tmp;

        tmp = rightSingularVectors_.getTranspose();
        rightSingularVectors_ = *tmp;
        delete tmp;
        free((void *) work);
    }

    const ls::DoubleMatrix &SVD::getSingularValues() const {
        return singularValues_;
    }

    const ls::DoubleMatrix &SVD::getLeftSingularVectors() const {
        return leftSingularVectors_;
    }

    const ls::DoubleMatrix &SVD::getRightSingularVectors() const {
        return rightSingularVectors_;
    }

    int SVD::rank(double tol) const {
        const auto& s = getSingularValues();
        int rank = 0;
        for (int i=0; i<s.numCols(); i++){
            const double& value = s(0, i);
            if (value - tol > 0 ){
                // we consider this close enough to be called 0
                rank += 1;
            }
        }
        return rank;
    }

    bool SVD::isSingular(){
        return rank() < std::min(nRows_,nCols_);
    }
}