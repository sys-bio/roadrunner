/**
 * LibStruct, original author: Frank Bergmann.
 * Fixes and improvments: Totte Karsson
 */
#pragma hdrstop
#include <string.h>
#include <stdlib.h>
#include <complex>
#include "telMatrix.h"

//---------------------------------------------------------------------------
namespace tlp
{

//DoubleMatrix instantiate;

ostream& operator<<(ostream& stream, const IntMatrix& mat)
{
    for(unsigned row = 0; row < mat.RSize(); row++)
    {
        for(unsigned col = 0; col < mat.CSize(); col++)
        {
            int val = mat(row,col);
            stream<<val<<"\t";
        }
        stream<<"\n";
    }
    return stream;
}

ostream& operator<<(ostream& stream, const DoubleMatrix& mat)
{
    for(unsigned row = 0; row < mat.RSize(); row++)
    {
        for(unsigned col = 0; col < mat.CSize(); col++)
        {
            double val = mat(row,col);
            stream<<val<<"\t";
        }
        stream<<"\n";
    }
    return stream;
}

ostream& operator<<(ostream& stream, const ComplexMatrix& mat)
{
    for(unsigned row = 0; row < mat.RSize(); row++)
    {
        for(unsigned col = 0; col < mat.CSize(); col++)
        {
            Complex val = mat(row,col);
               stream<<val<<"\t";
        }
        stream<<"\n";
    }
    return stream;
}


template<typename T> void telMatrix<T>::initializeFrom2DMatrix(T** &oRawData, int rows, int cols)
{
    resize(rows, cols);
    for (unsigned int i = 0; i < mRSize; i++)
    {
        for (unsigned int j = 0; j < mCSize; j++)
        {
            this->operator ()(i,j) = oRawData[i][j];
        }
    }

}
template<typename T> void telMatrix<T>::initializeFromConst2DMatrix(const T** oRawData, int rows, int cols)
{
    resize(rows, cols);
    for (unsigned int i = 0; i < mRSize; i++)
    {
        for (unsigned int j = 0; j < mCSize; j++)
        {
            (*this)(i,j) = oRawData[i][j];
        }
    }
}

template<typename T> T** telMatrix<T>::get2DMatrix(int &nRows, int &nCols)
{
    T** oBuffer = (T**) malloc(sizeof(T*)*mRSize);
    for (unsigned int i = 0; i < mRSize; i++)
    {
        oBuffer[i] = (T*) malloc(sizeof(T)*mCSize);
    }

    for (unsigned int i = 0; i < mRSize; i++)
    {
        for (unsigned int j = 0; j < mCSize; j++)
        {
            oBuffer[i][j] = this->operator ()(i,j);
        }
    }

    nRows = mRSize;
    nCols = mCSize;

    return oBuffer;
}


// ******************************************************************** }
// Multiply matrix 'm1' by 'm2' - returns a DoubleMatrix
//                                                                      }
// Usage:  A = mult (A1, A2); multiply A1 by A2 giving A                  }
//                                                                      }
// ******************************************************************** }
ls::DoubleMatrix mult(ls::DoubleMatrix& m1, ls::DoubleMatrix& m2)
{
    ls::DoubleMatrix result(0,0);

    //  Check dimensions
    unsigned int m1_nRows = m1.numRows();
    unsigned int m2_nRows = m2.numRows();

    unsigned int m1_nColumns = m1.numCols();
    unsigned int m2_nColumns = m2.numCols();

    if (m1.size() == 0)
    {
        return m1;
    }
    if (m2.size() == 0)
    {
        return m2;
    }

    if (m1_nColumns == m2_nRows)
    {
        result.resize(m1_nRows, m2_nColumns);
        for (unsigned int row = 0; row < result.numRows(); row++)
        {
            for (unsigned int col = 0; col < m2_nColumns; col++)
            {
                double sum = 0.0;
                for (unsigned int k = 0; k < m1_nColumns; k++)
                {
                    sum = sum + (m1[row][k] * m2[k][col]);
                }
                result[row][col] = sum;
            }
        }
        return result;
    }

    if (m1_nRows == m2_nColumns)
    {
        return mult(m2, m1);
    }

    throw ("Incompatible matrix operands to multiply");
}

//Double matrix is a special case of a complex matrix for which the imag part is all zero..
//so it makes sense that return value is a Double matrix..
DoubleMatrix mult(ComplexMatrix& m1, DoubleMatrix& m2)
{
    //  Check dimensions
    unsigned int m1_nRows = m1.numRows();
    unsigned int m2_nRows = m2.numRows();
    unsigned int m1_nColumns = m1.numCols();
    unsigned int m2_nColumns = m2.numCols();

    if (m1.size() == 0)
    {
        return real(m1);
    }

    if (m2.size() == 0)
    {
        return m2;
    }

    DoubleMatrix result(m1_nRows, m2_nColumns);
    if (m1_nColumns == m2_nRows)
    {
        for (unsigned int row = 0; row < result.numRows(); row++)
        {
            for (unsigned int col = 0; col < m2_nColumns; col++)
            {
                double sum = 0.0;
                for (unsigned int k = 0; k < m1_nColumns; k++)
                {
                    sum = sum + (real(m1[row][k]) * m2[k][col]);
                }
                result[row][col] = sum;
            }
        }
        return result;
    }

    if (m1_nRows == m2_nColumns)
    {
        return mult(m2, m1);
    }

    throw ("Incompatible matrix operands to multiply");
}

DoubleMatrix mult(DoubleMatrix& m2, ComplexMatrix& m1)
{
    //  Check dimensions
    unsigned int m1_nRows = m1.numRows();
    unsigned int m2_nRows = m2.numRows();
    unsigned int m1_nColumns = m1.numCols();
    unsigned int m2_nColumns = m2.numCols();

    if (m1.size() == 0)
    {
        return real(m1);
    }

    if (m2.size() == 0)
    {
        return m2;
    }

    DoubleMatrix result(m1_nRows, m2_nColumns);
    if (m1_nColumns == m2_nRows)
    {
        for (unsigned int row = 0; row < result.numRows(); row++)
        {
            for (unsigned int col = 0; col < m2_nColumns; col++)
            {
                double sum = 0.0;
                for (unsigned int k = 0; k < m1_nColumns; k++)
                {
                    sum = sum + (real(m1[row][k]) * m2[k][col]);
                }
                result[row][col] = sum;
            }
        }
        return result;
    }

    if (m1_nRows == m2_nColumns)
    {
        return mult(m2, m1);
    }

    throw ("Incompatible matrix operands to multiply");
}


DoubleMatrix real(const ComplexMatrix& m2)
{
    DoubleMatrix result(m2);
    return result;
}

DoubleMatrix imag(const ComplexMatrix& m2)
{
    DoubleMatrix result(m2, false); //This will copy imag part of complex matrix to the double one
    return result;
}

ComplexMatrix subtract(ComplexMatrix& x, ComplexMatrix& y)
{
    if(sameDimensions(x,y))
    {
        ComplexMatrix result(x.RSize(), x.CSize());

        for (int i = 0; i < x.RSize(); i++)
        {
            for (int j = 0; j < x.CSize(); j++)
            {
                result(i, j) = x(i, j) - y(i, j);
            }
        }
        return result;
    }
    else
    {
       throw ("Matrices must be the same dimension to perform subtraction");
    }
}


//From Fransk CSharp code... a bug was fixed in this code..
bool sameDimensions(ComplexMatrix& x, ComplexMatrix& y)
{
    return ((x.RSize() == y.RSize()) && (x.CSize() == y.CSize())) ? true : false;
}

ls::ComplexMatrix mult(ls::ComplexMatrix& m1, ls::ComplexMatrix& m2)
{
    if(m1.CSize() != m2.RSize())
    {
        throw("Matrix product not defined, incompatible sizes..\n");
      }

      ComplexMatrix temp(m1.RSize(), m2.CSize());
      Complex tempVal(0,0);
      for(int row1 = 0; row1 < m1.RSize(); row1++)
    {
        for(int rescol = 0; rescol < m2.CSize(); rescol++)
        {
              for(int col1 = 0; col1 < m1.CSize(); col1++)
            {
                Complex val1 = m1(row1,col1);
                Complex val2 = m2(col1,rescol);
                Complex test = val1*val2;
                tempVal = test + tempVal;
              }
              temp(row1,rescol) = tempVal;
              tempVal = Complex(0,0);
        }
      }
      return temp;
}


//The following instantiate each matrix type, AFAIK
//template class Matrix<double>;
//template class Matrix<int>;
//template class Matrix< Complex >;

}
