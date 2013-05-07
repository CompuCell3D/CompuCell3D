#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <string.h>
#include <stdlib.h>
#include <complex>

#include "lsMatrix.h"
#include "lsComplex.h"
#include "lsUtils.h"

//---------------------------------------------------------------------------
namespace ls
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
            //Complex val = mat(row,col);
           //stream<<val<<"\t";
        }
        stream<<"\n";
    }
    return stream;
}


template<typename T> void Matrix<T>::initializeFrom2DMatrix(T** &oRawData, int rows, int cols)
{
    resize(rows, cols);
    for (unsigned int i = 0; i < _Rows; i++)
    {
        for (unsigned int j = 0; j < _Cols; j++)
        {
            this->operator ()(i,j) = oRawData[i][j];
        }
    }

}
template<typename T> void Matrix<T>::initializeFromConst2DMatrix(const T** oRawData, int rows, int cols)
{
    resize(rows, cols);
    for (unsigned int i = 0; i < _Rows; i++)
    {
        for (unsigned int j = 0; j < _Cols; j++)
        {
            (*this)(i,j) = oRawData[i][j];
        }
    }
}

template<typename T> T** Matrix<T>::get2DMatrix(int &nRows, int &nCols)
{
    T** oBuffer = (T**) malloc(sizeof(T*)*_Rows);
    for (unsigned int i = 0; i < _Rows; i++)
    {
        oBuffer[i] = (T*) malloc(sizeof(T)*_Cols);
    }

    for (unsigned int i = 0; i < _Rows; i++)
    {
        for (unsigned int j = 0; j < _Cols; j++)
        {
            oBuffer[i][j] = this->operator ()(i,j);
        }
    }

    nRows = _Rows;
    nCols = _Cols;

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

//The following instantiate each matrix type, AFAIK
template class Matrix<double>;
template class Matrix<int>;
template class Matrix< ls::Complex >;
//template class Matrix< std::complex<double> >;

}
