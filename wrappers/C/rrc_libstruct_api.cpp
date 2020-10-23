#pragma hdrstop

#include "rrc_libstruct_api.h"
#include "rrc_utilities.h"
#include "rrc_macros.h"

#include "rrRoadRunner.h"
#include "rrException.h"
#include "rr-libstruct/lsLibla.h"
#include "rrc_cpp_support.h"


//---------------------------------------------------------------------------
namespace rrc
{
using namespace rr;
using namespace ls;    //Libstruct namespace
using namespace std;

RRDoubleMatrixPtr rrcCallConv getL0Matrix(RRHandle handle)
{
    start_try
        rr::RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix     tempMat     = rri->getL0Matrix();
        return createMatrix(&tempMat);

    catch_ptr_macro;
}

RRVectorPtr getConservedSums(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle); //Will throw if it can't cast
        vector<double> tempMat = rri->getConservedMoietyValues();
        return rrc::createVector(tempMat);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getEigenvaluesMatrix(const RRDoubleMatrixPtr mat)
{
    start_try
        if (mat == NULL)
        {
            stringstream msg;
            msg<<"RoadRunner exception: "<< "Matrix argument to getEigenvaluesMatrix is NULL" <<endl;
            setError(msg.str());
            return NULL;
        }

        // Convert RRDoubleMatrixPtr mat to a DoubleMatrix
        DoubleMatrix *dmat = createMatrix (mat);

        vector<Complex> vals = ls::getEigenValues((*dmat));
        DoubleMatrix result(static_cast<unsigned int>(vals.size()), 2);

        for (int i = 0; i < vals.size(); i++)
        {
            result[i][0] = real(vals[i]);
            result[i][1] = imag(vals[i]);
        }
        // Convert the DoubleMatrix result to a RRDoubleMatrixPtr type
        return createMatrix(&result);
    catch_ptr_macro
}

RRComplexVectorPtr rrcCallConv getEigenvaluesVector(const RRDoubleMatrixPtr mat)
{
    start_try
        if (mat == NULL)
        {
            stringstream msg;
            msg<<"RoadRunner exception: "<< "Argument to getEigenvaluesVector is NULL" <<endl;
            setError(msg.str());
            return NULL;
        }

        // Convert RRDoubleMatrixPtr mat to a DoubleMatrix
        DoubleMatrix *dmat = createMatrix (mat);
        vector<Complex> vals = ls::getEigenValues((*dmat));

        // Convert the DoubleMatrix result to a RRDoubleMatrixPtr type
        return createVector(vals);
    catch_ptr_macro
}

//LIB_EXTERN int LibLA_getEigenValues(double** inMatrix, int numRows, int numCols,
//                                    double* *outReal, double * *outImag, int *outLength)
//{
//    try
//    {
//        DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//        vector< Complex > oVector = getInstance()->getEigenValues( oMatrix );
//
//        ls::CopyComplexVector(oVector, *outReal, *outImag, *outLength);
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}
//
//LIB_EXTERN int LibLA_ZgetEigenValues(double** inMatrixReal, double** inMatrixImag, int numRows, int numCols,
//                                     double* *outReal, double * *outImag, int *outLength)
//{
//    try
//    {
//        ComplexMatrix oMatrix(numRows, numCols);
//
//        for (int i = 0; i < numRows; i++)
//        {
//            for (int j = 0; j < numCols; j++)
//            {
//                oMatrix(i,j).set( inMatrixReal[i][j],inMatrixImag[i][j]);
//            }
//        }
//
//        vector< Complex > oVector = getInstance()->ZgetEigenValues( oMatrix );
//
//        ls::CopyComplexVector(oVector, *outReal, *outImag, *outLength);
//
//        oVector.clear();
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}
//
//LIB_EXTERN int LibLA_getQRWithPivot(double** inMatrix, int numRows, int numCols,
//                                    double** *outQ, int *outQRows, int * outQCols,
//                                    double** *outR, int *outRRows, int * outRCols,
//                                    double** *outP, int *outPRows, int * outPCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    vector< DoubleMatrix * > oVector = LibLA::getInstance()->getQRWithPivot( oMatrix );
//
//    ls::CopyMatrix(*oVector[0], *outQ, *outQRows, *outQCols); delete oVector[0];
//    ls::CopyMatrix(*oVector[1], *outR, *outRRows, *outRCols); delete oVector[1];
//    ls::CopyMatrix(*oVector[2], *outP, *outPRows, *outPCols); delete oVector[2];
//
//
//    return 0;
//}
//
//LIB_EXTERN int LibLA_getQR(double** inMatrix, int numRows, int numCols,
//                           double** *outQ, int *outQRows, int * outQCols,
//                           double** *outR, int *outRRows, int * outRCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    vector< DoubleMatrix * > oVector = LibLA::getInstance()->getQR( oMatrix );
//
//    ls::CopyMatrix(*oVector[0], *outQ, *outQRows, *outQCols); delete oVector[0];
//    ls::CopyMatrix(*oVector[1], *outR, *outRRows, *outRCols); delete oVector[1];
//
//    return 0;
//}
//
//LIB_EXTERN int LibLA_getSingularValsBySVD(double** inMatrix, int numRows, int numCols,
//                                          double* *outSingularVals, int *outLength)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    vector< double > oVector = LibLA::getInstance()->getSingularValsBySVD( oMatrix );
//
//    ls::CopyDoubleVector(oVector, *outSingularVals, *outLength);
//
//    return 0;
//}
//
//LIB_EXTERN int LibLA_getLUwithFullPivoting(double** inMatrix, int numRows, int numCols,
//                                           double** *outL, int *outLRows, int * outLCols,
//                                           double** *outU, int *outURows, int * outUCols,
//                                           int** *outP, int *outPRows, int * outPCols,
//                                           int** *outQ, int *outQRows, int * outQCols,
//                                           int *info)
//{
//
//    try
//    {
//        DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//        LU_Result* oResult = LibLA::getInstance()->getLUwithFullPivoting( oMatrix );
//
//        ls::CopyMatrix(*(oResult->L), *outL, *outLRows, *outLCols);
//        ls::CopyMatrix(*(oResult->U), *outU, *outURows, *outUCols);
//        ls::CopyMatrix(*(oResult->P), *outP, *outPRows, *outPCols);
//        ls::CopyMatrix(*(oResult->Q), *outQ, *outQRows, *outQCols);
//
//        *info = oResult->nInfo;
//
//        delete oResult;
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}
//
//LIB_EXTERN int LibLA_getLU(double** inMatrix, int numRows, int numCols,
//                           double** *outL, int *outLRows, int * outLCols,
//                           double** *outU, int *outURows, int * outUCols,
//                           int** *outP, int *outPRows, int * outPCols,
//                           int *info)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    LU_Result* oResult = LibLA::getInstance()->getLU( oMatrix );
//
//    ls::CopyMatrix(*(oResult->L), *outL, *outLRows, *outLCols);
//    ls::CopyMatrix(*(oResult->U), *outU, *outURows, *outUCols);
//    ls::CopyMatrix(*(oResult->P), *outP, *outPRows, *outPCols);
//
//    *info = oResult->nInfo;
//
//    delete oResult;
//    return 0;
//}
//
//LIB_EXTERN int LibLA_inverse(double** inMatrix, int numRows, int numCols,
//                             double** *outMatrix, int *outRows, int *outCols)
//{
//
//    try
//    {
//        DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//        DoubleMatrix *oResult = LibLA::getInstance()->inverse( oMatrix );
//
//        ls::CopyMatrix(*oResult, *outMatrix, *outRows, *outCols); delete oResult;
//
//        return 0;
//    }
//    catch (...)
//    {
//        return -1;
//    }
//}
//
//
//LIB_EXTERN int LibLA_leftNullspace(double** inMatrix, int numRows, int numCols,
//                                   double** *outMatrix, int *outRows, int *outCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    DoubleMatrix *oResult = LibLA::getInstance()->getLeftNullSpace( oMatrix );
//
//    ls::CopyMatrix(*oResult, *outMatrix, *outRows, *outCols); delete oResult;
//
//    return 0;
//}
//LIB_EXTERN int LibLA_rightNullspace(double** inMatrix, int numRows, int numCols,
//                                    double** *outMatrix, int *outRows, int *outCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    DoubleMatrix *oResult = LibLA::getInstance()->getRightNullSpace( oMatrix );
//
//    ls::CopyMatrix(*oResult, *outMatrix, *outRows, *outCols); delete oResult;
//
//    return 0;
//}
//
//LIB_EXTERN int LibLA_scaledLeftNullspace(double** inMatrix, int numRows, int numCols,
//                                         double** *outMatrix, int *outRows, int *outCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    DoubleMatrix *oResult = LibLA::getInstance()->getScaledLeftNullSpace( oMatrix );
//
//    ls::CopyMatrix(*oResult, *outMatrix, *outRows, *outCols); delete oResult;
//
//    return 0;
//}
//LIB_EXTERN int LibLA_scaledRightNullspace(double** inMatrix, int numRows, int numCols,
//                                          double** *outMatrix, int *outRows, int *outCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    DoubleMatrix *oResult = LibLA::getInstance()->getScaledRightNullSpace( oMatrix );
//
//    ls::CopyMatrix(*oResult, *outMatrix, *outRows, *outCols); delete oResult;
//
//    return 0;
//}
//
//LIB_EXTERN int LibLA_getRank(double** inMatrix, int numRows, int numCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    int nRank = LibLA::getInstance()->getRank( oMatrix );
//    return nRank;
//}
//
//
//LIB_EXTERN double LibLA_getRCond(double** inMatrix, int numRows, int numCols)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    double dRcond = LibLA::getInstance()->getRCond( oMatrix );
//    return dRcond;
//}
//
//
//LIB_EXTERN int LibLA_Zinverse(double** inMatrixReal, double **inMatrixImag, int numRows, int numCols,
//                              double** *outMatrixReal, double ** *outMatrixImag, int *outRows, int *outCols)
//{
//    try
//    {
//        ComplexMatrix oMatrix(numRows, numCols);
//
//        for (int i = 0; i < numRows; i++)
//        {
//            for (int j = 0; j < numCols; j++)
//            {
//                oMatrix(i,j).set(inMatrixReal[i][j],inMatrixImag[i][j]);
//            }
//        }
//
//        ComplexMatrix *oResult = LibLA::getInstance()->Zinverse( oMatrix );
//        ls::CopyMatrix(*oResult, *outMatrixReal, *outMatrixImag, *outRows, *outCols); delete oResult;
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}
//
//LIB_EXTERN void LibLA_freeVector(void* vector)
//{
//    if (vector) free(vector);
//}
//
//LIB_EXTERN void LibLA_freeMatrix(void** matrix, int numRows)
//{
//    for (int i = 0; i < numRows; i++)
//    {
//        if (matrix[i]) free(matrix[i]);
//    }
//    free(matrix);
//}
//
//
//
//LIB_EXTERN int LibLA_gaussJordan(double** inMatrix, int numRows, int numCols,
//                                 double** *outMatrix, int *outRows, int *outCols, int* *oPivots, int *nLength)
//{
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    vector<int> oPivotVec  = ls::GaussJordan(oMatrix, LibLA::getInstance()->getTolerance());
//    ls::CopyMatrix(oMatrix, *outMatrix, *outRows, *outCols);
//
//    ls::CopyIntVector(oPivotVec, *oPivots, *nLength);
//    return 0;
//}
//
//std::vector<int> LibLA::gaussJordan(DoubleMatrix &oMatrix)
//{
//    return ls::GaussJordan(oMatrix, gLapackTolerance);
//}
//
//
////LIB_EXTERN int LibLA_gaussJordan2(double** inMatrix, int numRows, int numCols,
////                                  double** *outMatrix, int *outRows, int *outCols)
////{
////    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
////    ls::gaussJordan(oMatrix, LibLA::getInstance()->getTolerance());
////    ls::CopyMatrix(oMatrix, outMatrix, outRows, outCols);
////    return 0;
////}
//
//
//void LibLA::fullyPivotedGaussJordan(DoubleMatrix &oMatrix, vector<int> &rowPivots, vector<int> &colPivots)
//{
//    return ls::FullyPivotedGaussJordan(oMatrix, gLapackTolerance, rowPivots, colPivots);
//}
//
//
//LIB_EXTERN int LibLA_fullyPivotedGaussJordan(double** inMatrix, int numRows, int numCols,
//                                             double** *outMatrix, int *outRows, int *outCols,
//                                             int* *oRowPivots, int *nRowLength,
//                                             int* *oColPivots, int *nColLength)
//{
//
//    DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//    vector< int > vecRowPivots;  vector< int> vecColPivots;
//    ls::FullyPivotedGaussJordan(oMatrix, LibLA::getInstance()->getTolerance(), vecRowPivots, vecColPivots);
//
//    ls::CopyMatrix(oMatrix, *outMatrix, *outRows, *outCols);
//
//    ls::CopyIntVector(vecRowPivots, *oRowPivots, *nRowLength);
//    ls::CopyIntVector(vecColPivots, *oColPivots, *nColLength);
//
//    return 0;
//
//}

//LIB_EXTERN int LibLA_getEigenVectors(double** inMatrix, int numRows, int numCols,
//                                     double** *outMatrixReal, double** *outMatrixImag, int *outRows, int *outCols)
//{
//
//    try
//    {
//        DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//        ComplexMatrix *oResult = LibLA::getInstance()->getEigenVectors( oMatrix );
//
//        ls::CopyMatrix(*oResult, *outMatrixReal, *outMatrixImag, *outRows, *outCols); delete oResult;
//
//        return 0;
//    }
//    catch (...)
//    {
//        return -1;
//    }
//}

RRComplexMatrixPtr rrcCallConv getEigenVectors(const RRDoubleMatrixPtr inMatrix)
{
    start_try
        DoubleMatrix* mat =  createMatrix(inMatrix);
        if(mat)
        {
            ComplexMatrix* cMat  = ls::getEigenVectors((*mat));
            return createMatrix(cMat);
        }
        else
        {
            return NULL;
        }
    catch_ptr_macro
}

//LIB_EXTERN int LibLA_ZgetEigenVectors(double** inMatrixReal,   double **  inMatrixImag, int numRows, int numCols,
//                                      double** *outMatrixReal, double** *outMatrixImag, int *outRows, int *outCols)
//{
//    try
//    {
//        ComplexMatrix oMatrix(numRows, numCols);
//
//        for (int i = 0; i < numRows; i++)
//        {
//            for (int j = 0; j < numCols; j++)
//            {
//                oMatrix(i,j).set(inMatrixReal[i][j],inMatrixImag[i][j]);
//            }
//        }
//
//        ComplexMatrix *oResult = LibLA::getInstance()->ZgetEigenVectors( oMatrix );
//        ls::CopyMatrix(*oResult, *outMatrixReal, *outMatrixImag, *outRows, *outCols); delete oResult;
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}

RRComplexMatrixPtr rrcCallConv getZEigenVectors(RRComplexMatrixPtr inMatrix)
{
    start_try
        ComplexMatrix mat(inMatrix->RSize, inMatrix->CSize);
        for (int i = 0; i < inMatrix->RSize; i++)
        {
            for (int j = 0; j < inMatrix->CSize; j++)
            {
                RRComplex value;
                if(!getComplexMatrixElement(inMatrix, i, j, &value))
                {
                    throw(Exception("Failed getting element in Complex matrix"));
                }

                mat(i,j) = Complex(value.re, value.imag);
            }
        }

        ComplexMatrix *oResult = ls::ZgetEigenVectors(mat);
        RRComplexMatrixPtr out = createMatrix(oResult);
        delete oResult;
        return out;
    catch_ptr_macro
}

//LIB_EXTERN int LibLA_getSVD(double** inMatrix, int numRows, int numCols,
//                            double** *outU, int *outRowsU, int *outColsU,
//                            double* *outSingVals, int *outLength,
//                            double** *outV, int *outRowsV, int *outColsV)
//{
//    try
//    {
//        DoubleMatrix oMatrix(inMatrix, numRows, numCols);
//
//        DoubleMatrix*   matrixU  = NULL;
//        vector<double>* singVals = NULL;
//        DoubleMatrix*   matrixV  = NULL;
//
//        LibLA::getInstance()->getSVD(oMatrix, matrixU, singVals, matrixV);
//
//        ls::CopyMatrix(*matrixU, *outU, *outRowsU, *outColsU); delete matrixU;
//        ls::CopyDoubleVector(*singVals, *outSingVals, *outLength); delete singVals;
//        ls::CopyMatrix(*matrixV, *outV, *outRowsV, *outColsV); delete matrixV;
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//
//}
//
//LIB_EXTERN int LibLA_ZgetSVD(double** inMatrixReal, double** inMatrixImag, int numRows, int numCols,
//                             double** *outUReal, double** *outUImag, int *outRowsU, int *outColsU,
//                             double* *outSingVals, int *outLength,
//                             double** *outVReal, double** *outVImag, int *outRowsV, int *outColsV)
//{
//    try
//    {
//        ComplexMatrix oMatrix(numRows, numCols);
//
//        for (int i = 0; i < numRows; i++)
//        {
//            for (int j = 0; j < numCols; j++)
//            {
//                oMatrix(i,j).set(inMatrixReal[i][j],inMatrixImag[i][j]);
//            }
//        }
//
//        ComplexMatrix *matrixU = NULL; vector<double>* singVals= NULL; ComplexMatrix *matrixV= NULL;
//
//        LibLA::getInstance()->ZgetSVD(oMatrix, matrixU, singVals, matrixV);
//
//        ls::CopyMatrix(*matrixU, *outUReal, *outUImag, *outRowsU, *outColsU); delete matrixU;
//        ls::CopyDoubleVector(*singVals, *outSingVals, *outLength); delete singVals;
//        ls::CopyMatrix(*matrixV, *outVReal, *outVImag, *outRowsV, *outColsV); delete matrixV;
//
//        return 0;
//    }
//    catch(...)
//    {
//        return -1;
//    }
//}

} //namespace rrc

