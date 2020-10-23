#ifndef telMatrixH
#define telMatrixH
/**
 * Original author: Frank Bergmann.
 * Fixes and improvements: Totte Karlsson
 */


#if defined __unix || defined(__APPLE__)
#   include <cstring> //for memset on unix platforms
#endif
#include <string>
#include <sstream>
#include <complex>
#include "telProperty.h"
#include "telMathExporter.h"

namespace tlp
{

/*! \class ls::Matrix
    \brief ls::Matrix is the matrix class used by ls::LibLA and LIB_STRUTURAL::LibStructural
    This class implements a template to hold real, ls::Complex and integer matrices. It also implements basic
    operations on matrices.
*/

using std::string;
using std::stringstream;
typedef std::complex<double> Complex;

template <class T>
class telMatrix
{
    protected:
        unsigned int                mRSize;
        unsigned int                mCSize;
        T*                          mArray;

    public:

                                    //! Creates a new matrix with the given numbers of rows and columns
                                    telMatrix(unsigned int rows = 0, unsigned int cols = 0);

                                    //! Copy constructors
                                    telMatrix(const telMatrix< double >& src);
                                    telMatrix(const telMatrix< Complex >& src, bool real = true);

                                    //! Constructor taking a matrix mapped to a vector and reconstructing the 2D form
                                    telMatrix(T* &oRawData, int nRows, int nCols, bool transpose = true);

                                    //! constructs a matrix from 2D data
                                    telMatrix(T** &oRawData, int nRows, int nCols);

                                    //! constructs a matrix from 2D const data
                                    telMatrix(const T** oRawData, int nRows, int nCols);

                                    //!  destructor
                                    ~telMatrix();

        T*                          GetPointer();
        bool                        isAllocated() const;
        unsigned int                CSize() const;                      //! returns the number of columns
        unsigned int                RSize() const;
        unsigned int                Length() const;

        T*                          getArray();                         //! returns a pointer to the underlying 1D array
        T*                          getCopy(bool transpose = false);    //! returns a copy of the data, optionally transposing it
        void                        initializeFrom2DtelMatrix( T** &oRawData, int nRows, int nCols); //! initializes the matrix from 2D data
        void                        initializeFromConst2DtelMatrix(const T** oRawData, int nRows, int nCols); //! initializes the matrix from 2D const data


                                    //! returns a 2D data array
        T**                         get2DtelMatrix(int &nRows, int &nCols);

                                    //! swaps the given rows
        void                        swapRows(unsigned int row1, unsigned int row2);

                                    //! swaps the given columns
        void                        swapCols(unsigned int col1, unsigned int col2);

                                    //! resizes the matrix to the given number of rows and columns
        void                        resize(unsigned int rows, unsigned int cols);
        bool                        Allocate(unsigned int rows, unsigned int cols);

                                    //! creates a new matrix holding the transpose
        telMatrix<T>*               getTranspose();

                                    //! assignment operator
        telMatrix<T>&               operator = (const telMatrix <T>& rhs);

                                    //! scalar assignment operator
        telMatrix<T>&               operator = (const T & value);

                                    //! returns the size of the matrix ??
        unsigned int                size() const;

                                    //! returns the number of rows
        unsigned int                numRows() const;

                                    //! returns the number of columns
        unsigned int                numCols() const;

                                    //! returns the selected row
        T*                          operator[](unsigned int row);

                                    //! returns the selected row
        const T*                    operator[](unsigned int row) const;

                                    //! returns the selected matrix element
        T&                          operator()(const unsigned int & row, const unsigned int & col);

                                    //! returns the selected matrix element (const)
        const T&                    operator()(const unsigned int & row, const unsigned int & col) const;
        const T&                    Element(const unsigned int & row, const unsigned int & col) const;

};

//! hide templates in signatures
typedef telMatrix< double >         Matrix;
typedef telMatrix< int >            IntMatrix;
typedef telMatrix< Complex >        ComplexMatrix;

//Exported Utility telMatrix functions
LIB_EXTERN Matrix                   real(const ComplexMatrix& m2);               //Return real part of complex matrix
LIB_EXTERN Matrix                   imag(const ComplexMatrix& m2);               //Return imag part of complex matrix

LIB_EXTERN Matrix                   mult(Matrix& m1, Matrix& m2);
LIB_EXTERN Matrix                   mult(ComplexMatrix& m1, Matrix& m2);
LIB_EXTERN Matrix                   mult(Matrix& m1, ComplexMatrix& m2);

LIB_EXTERN ComplexMatrix            subtract(ComplexMatrix& x, ComplexMatrix& y);
LIB_EXTERN ComplexMatrix            mult(ComplexMatrix& m1, ComplexMatrix& m2);
LIB_EXTERN bool                     sameDimensions(ComplexMatrix& x, ComplexMatrix& y);


LIB_EXTERN std::ostream&            operator<<(std::ostream& stream, const IntMatrix& mat);
LIB_EXTERN std::ostream&            operator<<(std::ostream& stream, const Matrix& mat);
LIB_EXTERN std::ostream&            operator<<(std::ostream& stream, const ComplexMatrix& mat);

template<class T>
telMatrix<T>::telMatrix(unsigned int rows, unsigned int cols) :
mRSize(rows),
mCSize(cols),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new T[mRSize * mCSize]; //Todo: memoryleak
        memset(mArray, 0, (sizeof(T)*mRSize*mCSize));
    }
}

//We can have various init scenarios
template<class T>
inline telMatrix<T>::telMatrix(const telMatrix<double> & src)
:
mRSize(src.RSize()),
mCSize(src.CSize()),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new T[mRSize * mCSize]; //Todo: memoryleak
        //memcpy(mArray, src.mArray, mRSize * mCSize * sizeof(double));
    }
}

template<>
inline telMatrix<double>::telMatrix(const telMatrix<double> & src)
:
mRSize(src.mRSize),
mCSize(src.mCSize),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new double[mRSize * mCSize]; //Todo: memoryleak
        memcpy(mArray, src.mArray, mRSize * mCSize * sizeof(double));
    }
}

template<>
inline telMatrix<double>::telMatrix(const telMatrix<Complex> & src, bool copyReal) :
mRSize(src.RSize()),
mCSize(src.CSize()),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new double[mRSize * mCSize];
    }

    telMatrix<double>& refMat = *this;
    for(unsigned int r = 0; r < mRSize; r++)
    {
        for(unsigned int c = 0; c < mCSize; c++)
        {
            if(copyReal)
            {
                refMat(r,c) = std::real(src(r,c));
            }
            else
            {
                refMat(r,c) = std::imag(src(r,c));
            }
        }
    }
}

template<class T>
inline telMatrix<T>::telMatrix(const telMatrix<Complex> & src, bool copyReal) :
mRSize(src.RSize()),
mCSize(src.CSize()),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new T[mRSize * mCSize];
    }

    telMatrix<T>& refMat = *this;
    for(unsigned int r = 0; r < mRSize; r++)
    {
        for(unsigned int c = 0; c < mCSize; c++)
        {
            //refMat(r,c) = src(r,c);
        }
    }
}

template<>
inline telMatrix<Complex>::telMatrix(const telMatrix<Complex> & src, bool copyReal) :
mRSize(src.RSize()),
mCSize(src.CSize()),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new Complex[mRSize * mCSize];
    }

    telMatrix<Complex>& refMat = *this;
    for(unsigned int r = 0; r < mRSize; r++)
    {
        for(unsigned int c = 0; c < mCSize; c++)
        {
            refMat(r,c) = src(r,c);
        }
    }
}

template<>
inline telMatrix< Complex >::telMatrix(const telMatrix<double> & src):
mRSize(src.RSize()),
mCSize(src.CSize()),
mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new Complex[mRSize * mCSize];
        telMatrix< Complex >& refMat = *this;
        for(unsigned int r = 0; r < mRSize; r++)
        {
            for(unsigned int c = 0; c < mCSize; c++)
            {
                refMat(r,c) = src(r,c);
            }
        }
    }
}

template<class T>
inline telMatrix<T>::telMatrix( T* &oRawData, int nRows, int nCols, bool transpose) :
    mRSize(nRows),
    mCSize(nCols),
    mArray(NULL)
{
    if (mRSize && mCSize)
    {
        mArray = new T[mRSize * mCSize];
        if (!transpose)
        {
            memcpy(mArray, oRawData, sizeof(T)*nRows*nCols);
        }
        else
        {
            for (unsigned int i = 0; i < mRSize; i++)
            {
                for (unsigned int j = 0; j < mCSize; j++)
                {
                    (*this)(i,j) = oRawData[i+mRSize*j];
                }
          }
        }
    }
}

template<class T>
inline telMatrix<T>::~telMatrix()
{
  if (mArray)
  {
      delete [] mArray;
//      mArray = NULL;
  }
}

template<class T>
inline unsigned int telMatrix<T>::Length() const
{
    return mRSize*mCSize;
}

template<class T>
inline bool telMatrix<T>::isAllocated() const
{
    return (mArray != NULL) ? true : false;
}

template<class T>
T* telMatrix<T>::GetPointer()
{
    return mArray;
}

template<class T>
unsigned int telMatrix<T>::CSize() const
{
  return mCSize;
}

template<class T>
unsigned int telMatrix<T>::RSize() const
{
  return mRSize;
}

template<class T>
telMatrix<T>::telMatrix( T** &oRawData, int nRows, int nCols) : mArray(NULL), mRSize(0), mCSize(0)
{
  initializeFrom2DtelMatrix(oRawData, nRows, nCols);
}

template<class T>
telMatrix<T>::telMatrix(const T** oRawData, int nRows, int nCols) : mArray(NULL), mRSize(0), mCSize(0)
{
  initializeFromConst2DtelMatrix(oRawData, nRows, nCols);
}

template<class T>
T* telMatrix<T>::getArray() { return mArray; };

template<class T>
T* telMatrix<T>::getCopy(bool transpose)
{
    T* result = new T[mRSize * mCSize];
    if (mRSize * mCSize == 0)
    {
        return result;
    }

    if (!transpose)
    {
        memcpy(result, mArray, sizeof(T)*mRSize*mCSize);
    }
    else
    {
        for (unsigned int i = 0; i < mRSize; i++)
        {
            for (unsigned int j = 0; j < mCSize; j++)
            {
                result[i+mRSize*j] = (*this)(i,j) ;
            }
        }
    }
    return result;
}

template<class T>
void telMatrix<T>::swapRows(unsigned int row1, unsigned int row2)
{
  for (unsigned int i = 0; i < mCSize; i++)
  {
      T tmp = (*this)(row1,i);
      (*this)(row1,i)=(*this)(row2,i);
      (*this)(row2,i)=tmp;
  }
}

template<class T>
void telMatrix<T>::swapCols(unsigned int col1, unsigned int col2)
{
    for (unsigned int i = 0; i < mRSize; i++)
    {
        T tmp = (*this)(i,col1);
        (*this)(i,col1)=(*this)(i,col2);
        (*this)(i,col2)=tmp;
    }
}

template<class T>
void telMatrix<T>::resize(unsigned int rows, unsigned int cols)
{
    if (rows * cols != mRSize * mCSize)
    {
        if (mArray)
        {
            delete [] mArray;
            mArray = NULL;
        }
        if (rows && cols)
        {
            mArray = new T[rows * cols];
        }
    }

    mRSize = rows;
    mCSize = cols;
}

template<class T>
bool telMatrix<T>::Allocate(unsigned int rows, unsigned int cols)
{
    resize(rows,cols);
    return mArray ? true : false;
}

template<class T>
telMatrix<T>* telMatrix<T>::getTranspose()
{
  telMatrix <T> *oResult = new telMatrix <T>(mCSize, mRSize);  //Todo: memoryleak
  for (unsigned int i = 0; i < mCSize; i++)
  {
      for (unsigned int j = 0; j <mRSize; j++)
      {
          (*oResult)(i,j) = (*this)(j,i);
      }
  }
  return oResult;
}

template<class T>
telMatrix<T>& telMatrix<T>::operator = (const telMatrix <T>& rhs)
{
    if (this == &rhs)      // Same object?
    {
      return *this;
    }

    if (mRSize != rhs.mRSize || mCSize != rhs.mCSize)
    {
      resize(rhs.mRSize, rhs.mCSize);
    }

  memcpy(mArray, rhs.mArray, mRSize * mCSize * sizeof(T));
  return *this;
}

template<class T>
telMatrix <T> & telMatrix<T>::operator = (const T & value)
{
    unsigned int i, imax = mRSize * mCSize;
    T * tmp = mArray;

    for (i = 0; i < imax; i++, tmp++) *tmp = value;

    return *this;
}

template<class T>
unsigned int telMatrix<T>::size() const
{
    return mRSize * mCSize;
}

template<class T>
unsigned int telMatrix<T>::numRows() const
{
    return mRSize;
}

template<class T>
unsigned int telMatrix<T>::numCols() const
{
    return mCSize;
}

template<class T>
T* telMatrix<T>::operator[](unsigned int row)
{
    return mArray + row * mCSize;
}

template<class T>
const T* telMatrix<T>::operator[](unsigned int row) const
{
    return mArray + row * mCSize;
}

template<class T>
T& telMatrix<T>::operator()(const unsigned int & row, const unsigned int & col)
{
    return *(mArray + row * mCSize + col);
}

template<class T>
const T& telMatrix<T>::operator()(const unsigned int & row, const unsigned int & col) const
{
    return *(mArray + row * mCSize + col);
}

template<class T>
const T& telMatrix<T>::Element(const unsigned int & row, const unsigned int & col) const
{
    return *(mArray + row * mCSize + col);
}

template<class T>
const telMatrix<T> operator*(telMatrix<T>& lhs, const double& rhs)
{
    telMatrix<T> result(lhs.RSize(), lhs.CSize());
    for(unsigned int i = 0; i < lhs.RSize(); i++)
    {
        for(unsigned int j = 0; j < lhs.CSize(); j++)
        {
            result(i,j) = lhs(i,j) * rhs;
        }
    }
    return result;
}

template<class T>
telMatrix<T>    multDiag(telMatrix< T >& mat, T& val)
{
    telMatrix<T> result(mat.RSize(), mat.CSize());
    for (int i = 0; i < mat.RSize(); i++)
    {
        result(i, i) = mat(i, i) *  val;
    }
    return result;
}

//When a Matrix is a property =====================
template<>
inline string Property< ls::Matrix<double> >::getValueAsString() const
{
    stringstream ss;
    ss << mValue;

    return ss.str();
}
//
//template<>
//inline void Property< Matrix >::setValueFromString(const string& val)
//{
//    stringstream msg;
//
//}
//
//template<>
//inline string getPropertyType< Matrix >(const Matrix& value)
//{
//    return "Matrix";
//}

}

#endif

