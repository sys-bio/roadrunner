//
// Created by Ciaran on 10/03/2021.
//

#ifndef ROADRUNNER_SVD_H
#define ROADRUNNER_SVD_H

#include "rr-libstruct/lsMatrix.h"
#if defined(__has_include)
#if !__has_include(<rr-libstruct/lsLapack.h>)
#error "libroadrunner-deps is too old: it must provide rr-libstruct/lsLapack.h (BLAS and LAPACK from OpenBLAS). Update libroadrunner-deps."
#endif
#endif
#include "rr-libstruct/lsLapack.h"
#include <memory>

namespace rr {

    /**
     * @brief The routine computes the singular value decomposition
     * (SVD) of a real m-by-n matrix.
     * @details
     * Often when computing the steady state of a system the
     * jacobian matrix is singular. SVD is a way of computing the
     * rank of a matrix, which in turn can be used to determine
     * whether a rank is singular:
     *      - if rank < min(N, M) of an NxM matrix then it is singular.
     *
     */
    class SVD {

    public:

        /**
         * @brief constructor for SVD
         * @param matrix a ls::Matrix<double>
         * @details SVG is conducted on instantiation
         *      @see SVD::getSingularValues
         *      @see SVD::getLeftSingularVectors
         *      @see SVD::getRightSingularVectors
         */
        explicit SVD(ls::DoubleMatrix& matrix);

        /**
         * @brief returns the singular values computed for
         * input matrix
         */
        const ls::DoubleMatrix &getSingularValues() const;

        /**
         * @brief returns left singular vectors where columns
         * are the left vectors
         */
        const ls::DoubleMatrix &getLeftSingularVectors() const;

        /**
         * @brief returns right singular vectors where rows
         * are the right vectors
         */
        const ls::DoubleMatrix &getRightSingularVectors() const;

        /**
         * @brief estimates the rank of the matrix
         * by counting non 0 singular values.
         * @details the rank of a matrix is equal to the number
         * of non-zero singular values!
         * @param tol tolerance by which we consider close enough
         * to zero to be called zero
         */
        int rank(double tol = 1e-15) const;

        /**
         * @brief return true when the input matrix
         * is singular.
         * @details this is determined by testing
         * whether than rank of the matrix is less than
         * the smaller of the number of rows or number of
         * columns.
         */
        bool isSingular();

    private:

        void compute();

        ls::DoubleMatrix inputMatrix_;
        std::unique_ptr<ls::DoubleMatrix> inputMatrixTranspose_ = nullptr;
        ls_int nRows_;
        ls_int nCols_;
        ls_int lda_;
        ls_int ldu_;
        ls_int ldvt_;

        ls::DoubleMatrix singularValues_;
        ls::DoubleMatrix leftSingularVectors_;
        ls::DoubleMatrix rightSingularVectors_;

    };
}


#endif //ROADRUNNER_SVD_H
