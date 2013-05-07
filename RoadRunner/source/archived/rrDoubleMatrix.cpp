#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrDoubleMatrix.h"
//---------------------------------------------------------------------------

namespace rr
{
//
//DoubleMatrix::DoubleMatrix(const unsigned& rows, const unsigned& cols, const string& name)
//:
//mRowCount (rows),
//mColCount (cols),
//mMatrix(NULL),
//mIsOwner(true),
//mNamePtr(NULL)
//{
//    if (rows != 0 && cols != 0)
//    {
//        mMatrix = new double[rows * cols];
//        for(unsigned i = 0; i < rows * cols; i++)
//        {
//            mMatrix[i] = i + 1;
//        }
//    }
//}
//
//DoubleMatrix::DoubleMatrix(const DoubleMatrix& m)        // Copy constructor
//{
//    mMatrix = NULL;
//    *this = m;
//}
//
//DoubleMatrix::DoubleMatrix(double* ptrToArray, const unsigned& rowCount, const unsigned& colCount)
//{
//    mRowCount = rowCount;
//    mColCount = colCount;
//
//    //Shallow or deep copy?
//    mMatrix = ptrToArray; //Thats is pretty shallow...
//    mIsOwner = false;       //Somebody else allocatesd this one
//}
//
//DoubleMatrix::~DoubleMatrix()
//{
//    if(mIsOwner)
//    {
//        delete [] mMatrix;
//    }
//}
//
//bool DoubleMatrix::Allocate(unsigned rows, unsigned cols)
//{
//    if(mMatrix)
//    {
//        delete [] mMatrix;
//    }
//
//    mMatrix = new double[rows * cols];
//    mRowCount = rows;
//    mColCount = cols;
//    return mMatrix ? true : false;
//}
//
////=========== OPERATORS
//double& DoubleMatrix::operator() (const unsigned& row, const unsigned& col)
//{
//    if (row >= mRowCount || col >= mColCount)//If doing a lot of these, don't check, its to costly..
//    {
//        strstream msg;
//        string matName = mNamePtr != NULL ? *mNamePtr : string("");
//        msg << "Subscript out of bounds in matrix ("<<matName<<") RSize, CSize = "<<mRowCount<<", "<<mColCount<<" Row, Col = "<<row<<", "<<col;
//        throw Exception(msg.str());
//    }
//    return mMatrix[mColCount*row + col];
//}
//
//double DoubleMatrix::operator() (const unsigned& row, const unsigned& col) const
//{
//    if (row >= mRowCount || col >= mColCount)
//    {
//          strstream msg;
//        string matName = mNamePtr != NULL ? *mNamePtr : string("");
//        msg << "Subscript out of bounds in matrix ("<<matName<<") RSize, CSize = "<<mRowCount<<", "<<mColCount<<" Row, Col = "<<row<<", "<<col;
//        throw Exception(msg.str());
//    }
//    return mMatrix[mColCount*row + col];
//}
//
//
//DoubleMatrix& DoubleMatrix::operator = (const DoubleMatrix& rhs)
//{
//    if (this == &rhs)      // Same object?
//        return *this;
//
//    Allocate(rhs.RSize(), rhs.CSize());
//
//    for(unsigned row = 0; row < RSize(); row++)
//    {
//        for(unsigned col = 0; col < CSize(); col++)
//        {
//            this->operator()(row, col) = rhs(row, col);
//        }
//    }
//    return *this;
//}
//
//////        internal static double[][] GetDoubleMatrixFromPtr(IntPtr pointer, int nRows, int nCols)
//////        {
//////            IntPtr[] rawRows = new IntPtr[nRows];
//////            double[][] oResult = new double[nRows][];
//////            Marshal.Copy(pointer, rawRows, 0, nRows);
//////            for (int i = 0; i < nRows; i++)
//////            {
//////                oResult[i] = new double[nCols];
//////                Marshal.Copy(rawRows[i], oResult[i], 0, nCols);
//////            } // for (int)
//////            StructAnalysis.FreeMatrix(pointer, nRows);
//////            return oResult;
//////        } // GetDoubleMatrixFromPtr(pointer, nRows, nCols)
//
//DoubleMatrix RR_DECLSPEC GetDoubleMatrixFromPtr(double** *pointer, const unsigned& nRows, const unsigned& nCols)
//{
//    DoubleMatrix mat(nRows, nCols);
//    for(unsigned col = 0; col < mat.CSize(); col++)
//    {
//        for(unsigned row = 0; row < mat.RSize(); row++)
//        {
//            double val = **pointer[col*row + col];
//            mat(row,col) = val ;
//        }
//    }
//
//    return mat;
//}
//
//ostream& operator<<(ostream& stream, const DoubleMatrix& mat)
//{
//    for(unsigned row = 0; row < mat.RSize(); row++)
//    {
//        for(unsigned col = 0; col < mat.CSize(); col++)
//        {
//            double val = mat(row,col);
//            stream<<val<<"\t";
//        }
//        stream<<std::endl;
//    }
//    return stream;
//}

} //namespace rr
