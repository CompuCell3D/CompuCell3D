#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop

#ifdef WIN32
#pragma warning (disable: 4996)
#define WIN32_LEAN_AND_MEAN        // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <stdlib.h>
#include "lsUtils.h"
//---------------------------------------------------------------------------

using namespace std;
using namespace ls;

namespace ls
{

DoubleMatrix* getSubMatrix(int /*Mb*/, int /*Nb*/, int ms, int ns, int mi, int nj, DoubleMatrix& A)
{
    DoubleMatrix* subMatrix = new DoubleMatrix(ms, ns);
    for (int i=0; i<ms; i++)
    {
        for (int j=0; j<ns; j++)
        {
            (*subMatrix)(i,j) = A(i+mi,j+nj);
        }
    }
    return subMatrix;
}


DoubleMatrix* matMult(unsigned int mA, unsigned int nA, DoubleMatrix &A, DoubleMatrix &B,unsigned  int nB)
{
    DoubleMatrix* oResult = new DoubleMatrix(mA, nB);
    double sum = 0;
    for (unsigned int i = 0; i < mA; i++)
    {
        for (unsigned int j = 0; j < nB; j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < nA; k++)
            {
                sum += A(i,k)*B(k,j);
            }
            (*oResult)(i,j) = sum;
        }
    }
    return oResult;

}

DoubleMatrix* matMult(DoubleMatrix &A, DoubleMatrix &B)
{
    return matMult(A.numRows(), A.numCols(), A, B, B.numCols());
}

DoubleMatrix* matMult(IntMatrix &A, DoubleMatrix &B)
{
    DoubleMatrix* oResult = new DoubleMatrix(A.numRows(), B.numCols());
    double sum = 0;
    for (unsigned int i = 0; i < A.numRows(); i++)
    {
        for (unsigned int j = 0; j < B.numCols(); j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < A.numCols(); k++)
            {
                sum += (double)A(i,k)*B(k,j);
            }
            (*oResult)(i,j) = sum;
        }
    }
    return oResult;
}

IntMatrix* matMult(IntMatrix &A, IntMatrix &B)
{
    IntMatrix* oResult = new IntMatrix(A.numRows(), B.numCols());
    int sum = 0;
    for (unsigned int i = 0; i < A.numRows(); i++)
    {
        for (unsigned int j = 0; j < B.numCols(); j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < A.numCols(); k++)
            {
                sum += (int)A(i,k)*B(k,j);
            }
            (*oResult)(i,j) = sum;
        }
    }
    return oResult;
}


// ----------------------------------------------------------------------------
// double[][] matMult(int, int, int**, int**, int)
// 
// Integer Matrix Multiplication
// ----------------------------------------------------------------------------
int** matMult(int mA, int nA, int** A, int** B, int nB) {
    int sum;
    int** prod;
    prod = new int*[mA];
    for (int i=0; i<mA; i++) {
        prod[i] = new int[nB];
        for (int j=0; j<nB; j++) {
            sum = 0;
            for (int k=0; k<nA; k++) {
                sum = sum + A[i][k]*B[k][j];
            }
            prod[i][j] = sum;
        }
    }
    return prod;
}

// ----------------------------------------------------------------------------
// double[][] matMult(int, int, double**, double**, int)
//
// Matrix Multiplication
// ----------------------------------------------------------------------------
double** matMult(int mA, int nA, double** A, double** B, int nB) {
    double sum;
    double** prod;
    prod = new double*[mA];
    for (int i=0; i<mA; i++) {
        prod[i] = new double[nB];
        for (int j=0; j<nB; j++) {
            sum = 0.0;
            for (int k=0; k<nA; k++) {
                sum = sum + A[i][k]*B[k][j];
            }
            prod[i][j] = sum;
        }
    }
    return prod;
}

void checkTolerance(int nrows, double* A, double dTolerance) {
    for (int i=0; i<nrows; i++) 
    {
        A[i] = RoundToTolerance(A[i], dTolerance);
    }
}

double RoundToTolerance(double dValue, double dTolerance)
{
    if (fabs(dValue) < dTolerance)
    {
        dValue = 0.0;
    }
    else if (fabs(ceil(dValue) - dValue) < dTolerance)
    {
        dValue = ceil(dValue);
    }
    else if (fabs(dValue - floor(dValue)) < dTolerance)
    {
        dValue = floor(dValue);
    }
    return dValue;
}

void RoundMatrixToTolerance(DoubleMatrix& oMatrix, double dTolerance)
{
    for (unsigned int i = 0; i < oMatrix.numRows(); i++)
    {
        for (unsigned int j = 0; j < oMatrix.numCols(); j++)
        {
            oMatrix(i,j) = RoundToTolerance(oMatrix(i,j), dTolerance);
        }
    }
}

void checkTolerance(int nrows, int ncols, double** A, double dTolerance) {
    for (int i=0; i<nrows; i++) 
    {
        for (int j=0; j<ncols; j++) 
        {
            A[i][j] =RoundToTolerance(A[i][j], dTolerance);
        }
    }
}

int findRank(DoubleMatrix &oMatrix, double dTolerance)
{
    int i;
    int rank = oMatrix.numRows();        
    double sumj;

    i = oMatrix.numRows()-1;
    while (i != 0) 
    {
        sumj = 0.0;
        for (unsigned int j=0; j<oMatrix.numCols(); j++) 
        {
            sumj = sumj + std::abs(oMatrix(i,j));
        }
        if (sumj < dTolerance) 
        {
            rank--;
            i--;
        }
        else break;
    }
    return rank;
}

// fully pivoted gauss jordan, returning a pair of row pivots and column pivots
void FullyPivotedGaussJordan(DoubleMatrix &oMatrix, double dTolerance, std::vector< int > &rowPivots, std::vector< int > &colPivots)
{

    DoubleMatrix *oTranspose = oMatrix.getTranspose();

    colPivots = GaussJordan(*oTranspose, dTolerance);
    std::vector<int> oColCopy(colPivots.begin(), colPivots.end());    // take a copy 
    // take permutations and swap columns in original matrix

    bool bChanged = true; unsigned int nLast = 0;
    while (bChanged)
    {
        bChanged = false;
        for (unsigned int i = nLast; i < oColCopy.size(); i++)
        {
            nLast++;
            int nVal = oColCopy[i];            // take current value
            if (nVal != (int)i)                // if not equal swap colums
            {
                oMatrix.swapCols(i, nVal);
                oColCopy[i] = oColCopy[nVal];
                oColCopy[nVal] = nVal;
                bChanged = true;
                break;
            }
        }
        
    }

    delete oTranspose;
    rowPivots = GaussJordan(oMatrix, dTolerance);    
}
bool isPositive(DoubleMatrix &A, double dTolerance)
{
    for (unsigned int i = 0; i < A.numRows(); i++)
    {
        for (unsigned int j = 0; j < A.numCols(); j++)
        {
            if (A(i,j) < dTolerance*-1)
            {
                //print(A);
                return false;
            }
        }
    }
    return true;
}

std::vector<int> GaussJordan(DoubleMatrix &oMatrix, double dTolerance)
{
    std::vector<int> oPivots;

    // here the pseudo code: 
    /*
    il = 0 (number of leading 1s created)
    for j = 1 ... m do:
        if exists i > il such that aij != 0 then:
            il = il + 1
            interchange row i and row il (this brings the new row just below the row you have just done)
            divide row i by ailj creating a leading 1
            Reduce all other entries in the column to 0
    */

    int nRows = oMatrix.numRows();
    int nCols = oMatrix.numCols();

    for (int i = 0; i < nRows; i++)
    {
        oPivots.push_back(i);
    }

    if (nRows == 0 || nCols == 0) return oPivots;

    int nCurrentRow = 0; int nTempPivotRow = 0; double dPivot;
    for    (int nCurrentCol = 0; nCurrentCol < nCols; nCurrentCol ++)
    {
        // search for alternative pivot
        nTempPivotRow = nCurrentRow;
        for (int nRow = nCurrentRow; nRow < nRows; nRow++)
        {
            if (fabs(oMatrix(nRow, nCurrentCol)) > fabs(oMatrix(nTempPivotRow, nCurrentCol)))
            {                
                nTempPivotRow = nRow;
            }
        }

        // found better pivot so lets swap rows
        if (nCurrentRow != nTempPivotRow)
        {            
            int nTemp = oPivots[nCurrentRow];
            oPivots[nCurrentRow] = oPivots[nTempPivotRow];
            oPivots[nTempPivotRow] = nTemp;

            oMatrix.swapRows(nCurrentRow, nTempPivotRow);
        }

        // get the pivot
        dPivot = oMatrix(nCurrentRow, nCurrentCol);
        //Log(lDebug5) << "pivot: " << dPivot << " row: " << nCurrentRow << " col: " << nCurrentCol<< endl;

        if (dPivot == 0.0)
        {
            //nCurrentRow ++;    if (nCurrentRow >= nRows) break;
            continue;
        }
        // divide current row by pivot (yielding a leading 1)
        for (int nCol = 0; nCol < nCols; nCol ++)
        {            
            oMatrix(nCurrentRow, nCol) = oMatrix(nCurrentRow, nCol)/dPivot;
        }

        // reduce all other columns in the current row to zero
        for (int nRow = 0; nRow < nRows; nRow ++)
        {
            if (nRow == nCurrentRow) continue;
            double dTemp = oMatrix(nRow, nCurrentCol);
            if (fabs(dTemp) > dTolerance)
            {
                for (int nCol = 0; nCol < nCols; nCol++)
                {            
                    oMatrix(nRow, nCol) = oMatrix(nRow, nCol) -  oMatrix(nCurrentRow,nCol)*dTemp;
                }
            }
        }

        // advance to next row
        nCurrentRow ++;    if (nCurrentRow >= nRows) break;
    }
    // round matrix for good measure
    RoundMatrixToTolerance(oMatrix, dTolerance);
    return oPivots;
}
void gaussJordan(DoubleMatrix &oMatrix,double dTolerance)
{
    //GaussJordan(oMatrix, dTolerance);
    double dPivot, dTemp;
    int nRows = oMatrix.numRows();
    int nCols = oMatrix.numCols();

    //Log(lDebug5) << "INSIDE GAUSSIAN ELIMINATION METHOD \n";
    //Log(lDebug5) << " \nInput matrix : \n";
    //Log(lDebug5)<<print(oMatrix);

    int x;     int nPivotRow = 0;     int nPivotCol = 0;
    while ((nPivotRow < nRows) && (nPivotCol < nCols))
    {
        // Find largest pivot.  Search for a number below
        // the current pivot with a greater absolute value.
        x = nPivotRow;

        for (int C=nPivotRow; C<nRows; C++) 
        {
            if (fabs(oMatrix(C,nPivotCol)) > fabs(oMatrix(x,nPivotCol))) x = C;
        }

        if(x != nPivotRow) 
        {
            // If here, there is a better pivot choice somewhere
            // below the current pivot.
            // Interchange the pivot row with the row
            // containing the largest pivot, x
            for (int b=0; b<nCols; b++) 
            {
                dTemp = oMatrix(nPivotRow,b);
                oMatrix(nPivotRow,b) = oMatrix(x,b);
                oMatrix(x,b) = dTemp;
            }
        }

        dPivot = oMatrix(nPivotRow,nPivotCol);
        //Log(lDebug5) << "pivot: " << dPivot << endl;
        if(fabs(dPivot) > dTolerance) 
        {
            // Introduce a '1' at the pivot point
            for (int b=0; b < nCols; b++)
                oMatrix(nPivotRow,b) = oMatrix(nPivotRow,b)/dPivot;

            for (int b=0; b < nPivotRow; b++) 
            {
                // Eliminate (make zero) all elements above
                // and below the pivot.
                
                // Skip over the pivot row when we come to it.
                if ((b != nPivotRow) ) //|| (fabs(oMatrix(b,nPivotCol)) > dTolerance)) 
                {
                    dPivot = oMatrix(b,nPivotCol);
                    for (int d = nPivotRow; d < nCols; d++)
                        oMatrix(b,d) = oMatrix(b,d) - oMatrix(nPivotRow,d)*dPivot;
                }
            }
            nPivotRow++; // Next row
        }

        nPivotCol++;  // Next column
        //Log(lDebug5) << "Printing matrices PivotCol = " << nPivotCol << " \n";
//        print(m, n, A, EM);
        //Log(lDebug5) << "-----------------------------------------------------------------\n";
    }
    RoundMatrixToTolerance(oMatrix, dTolerance);
}

// ----------------------------------------------------------------------------
// void print (int, int, int*)
//
// Prints to Log(lDebug5) an arrray in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(int mr, int nc, int* A)
{
    stringstream stream;
    stream << "[";
    for (int i=0; i<mr; i++) {
        stream << "[";
        for (int j=0; j<nc; j++) {
            stream << A[i+j*mr] << (j+1<nc? ",    " : "    ");
        }
        stream << (i + 1 < mr ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}


// ----------------------------------------------------------------------------
// void print (int, int, int**)
//
// Prints to stream a matrix in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(IntMatrix& A)
{
    stringstream stream;
    stream << "[";
    for (unsigned int i=0; i < A.numRows(); i++)
    {
        stream << "[";
        for (unsigned int j=0; j < A.numCols(); j++)
        {
            stream << A(i,j) << (j+1 < A.numCols()? ",    " : "");
        }
        stream << (i + 1 < A.numRows() ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

string print(DoubleMatrix& A)
{
    stringstream stream;
    stream << "[";
    for (unsigned int i=0; i < A.numRows(); i++)
    {
        stream << "[";
        for (unsigned int j=0; j < A.numCols(); j++)
        {
            stream << A(i,j) << (j+1<A.numCols()? ",    " : "");
        }
        stream << (i + 1 < A.numRows() ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

string print(ComplexMatrix& A)
{
    stringstream stream;
    stream << "[";
    for (unsigned int i=0; i < A.numRows(); i++)
    {
        stream << "[";
        for (unsigned int j=0; j < A.numCols(); j++)
        {
            stream << "(" << A(i,j).Real << " + " << A(i,j).Imag << "i)" << (j+1<A.numCols()? ",    " : "");
        }
        stream << (i + 1 < A.numRows() ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

string print(int mr, int nc, int** A)
{
    stringstream stream;
    stream << "[";
    for (int i=0; i<mr; i++) {
        stream << "[";
        for (int j=0; j<nc; j++) {
            stream << A[i][j] << (j+1<nc? ",    " : "    ");
        }
        stream << (i + 1 < mr ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, double*)
//
// Prints to stream an arrray in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(int mr, int nc, double* A)
{
    stringstream stream;
    for (int i=0; i<mr; i++)
    {
        for (int j=0; j<nc; j++)
        {
            stream <<left<<setw(10)<< setprecision(3)<<A[i+j*mr];
        }
        stream << "\n";
    }
    stream << endl << endl;
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, double**)
//
// Prints to stream a matrix in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(int mr, int nc, double** A)
{
    stringstream stream;
    stream << "[";
    for (int i=0; i<mr; i++) {
        stream << "[";
        for (int j=0; j<nc; j++) {
            stream << A[i][j] << (j+1<nc? ",    " : "    ");
        }
        stream << (i + 1 < mr ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, Complex*)
//
// Prints to stream an arrray in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(int mr, int nc, ls::Complex* A)
{
    stringstream stream;
    stream << "[";
    for (int i=0; i<mr; i++) {
        stream << "[";
        for (int j=0; j<nc; j++) {
            stream << "(" << A[i+j*mr].Real << ", " << A[i+j*mr].Imag << ")  ";
        }
        stream << (i + 1 < mr ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, Complex**)
//
// Prints to stream a matrix in Nrows x Ncols
// ----------------------------------------------------------------------------
string print(int mr, int nc, ls::Complex** A)
{
    stringstream stream;
    stream << "[";
    for (int i=0; i<mr; i++) {
        stream << "[";
        for (int j=0; j<nc; j++) {
            stream << "(" << A[i][j].Real << ", " << A[i][j].Imag << ")  ";
        }
        stream << (i + 1 < mr ? "],\n" : "]\n");
    }
    stream << "]" << endl << endl;
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, int**, int** )
//
// Prints two matrices next to each other Nrows x Ncols | Nrows x Nrows
// ----------------------------------------------------------------------------
string print(int mr, int nc, int* A, int* B)
{
    stringstream stream;
    for (int i=0; i<mr; i++) {
        for (int j=0; j<nc; j++) {
            stream << A[j+i*mr] << ",   ";
        }
        stream << "  |  ";
        for (int j=0; j<nc; j++) {
            stream << B[j+i*mr] << ",   ";
        }
        stream << "\n ";
    }
    return stream.str();
}

// ----------------------------------------------------------------------------
// void print (int, int, int**, int** )
//
// Prints two matrices next to each other Nrows x Ncols | Nrows x Nrows
// ----------------------------------------------------------------------------
string print(int mr, int nc, int** A, int** B)
{
    stringstream stream;
    stream.precision(8);
    for (int i=0; i<mr; i++) {
        for (int j=0; j<nc; j++) {
            stream << A[i][j] << ",   ";
        }
        stream << "  |  ";
        for (int j=0; j<mr; j++) {
            stream << B[i][j] << ",   ";
        }
        stream << "\n ";
    }
    return stream.str();
}

// ----------------------------------------------------------------------------
// string print (int, int, double**, double** )
//
// Prints two matrices next to each other Nrows x Ncols | Nrows x Nrows
// ----------------------------------------------------------------------------
string print(int mr, int nc, double** A, double** B)
{
    stringstream stream;
    stream.precision(8);
    for (int i=0; i<mr; i++) {
        for (int j=0; j<nc; j++) {
            stream << A[i][j] << ",   ";
        }
        stream << "  |  ";
        for (int j=0; j<mr; j++) {
            stream << B[i][j] << ",   ";
        }
        stream << "\n ";
    }
    return stream.str();
}

void CopyMatrix(IntMatrix& oMatrix, int** &outMatrix, int &outNumRows, int &outNumCols)
{
    outNumRows = oMatrix.numRows();
    outNumCols = oMatrix.numCols();

    outMatrix = (int **) malloc(sizeof(int*) *outNumRows); memset(outMatrix, 0, sizeof(int*)*outNumRows);
    for (int i = 0; i < outNumRows; i++)
    {
        outMatrix[i] = (int*) malloc(sizeof(int)*outNumCols); memset(outMatrix[i], 0, sizeof(int)*outNumCols);
    }

    for (int i = 0; i < outNumRows; i++)
    {
        for (int j = 0; j < outNumCols; j++)
        {
            outMatrix[i][j] = oMatrix(i,j);
        }
    }
}
void CopyMatrix(DoubleMatrix& oMatrix, double** &outMatrix, int &outNumRows, int &outNumCols)
{
    if(!oMatrix.getArray())
    {
        outMatrix = NULL;
        outNumRows = 0;
        outNumCols = 0;
        return;
    }
    int numRows = oMatrix.numRows();
    int numCols = oMatrix.numCols();

    outMatrix = (double **) malloc(sizeof(double*) *numRows);
    if (outMatrix == NULL)
    {
        throw ApplicationException("Out of Memory during Matrix copy");
    }
    memset(outMatrix, 0, sizeof(double*)*numRows);
    for (int i = 0; i < numRows; i++)
    {
        outMatrix[i] = (double*) malloc(sizeof(double)*numCols);
        if (outMatrix[i] == NULL)
        {
            throw ApplicationException("Out of Memory during Matrix copy");
        }
        memset(outMatrix[i], 0, sizeof(double)*numCols);
    }

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            outMatrix[i][j] = oMatrix(i,j);
        }
    }

    outNumRows = numRows;
    outNumCols = numCols;

}

void CopyMatrix(ComplexMatrix& oMatrix, double** &outMatrixReal,double** &outMatrixImag, int &outNumRows, int &outNumCols)
{
    outNumRows = oMatrix.numRows();
    outNumCols = oMatrix.numCols();

    outMatrixReal = (double **) malloc(sizeof(double*) *outNumRows); if (outMatrixReal == NULL) throw new ApplicationException("Out of Memory during Matrix copy"); memset(outMatrixReal, 0, sizeof(double*)*outNumRows);
    outMatrixImag = (double **) malloc(sizeof(double*) *outNumRows); if (outMatrixImag == NULL) throw new ApplicationException("Out of Memory during Matrix copy"); memset(outMatrixImag, 0, sizeof(double*)*outNumRows);
    for (int i = 0; i < outNumRows; i++)
    {
        outMatrixReal[i] = (double*) malloc(sizeof(double)*outNumCols); if (outMatrixReal[i] == NULL) throw new ApplicationException("Out of Memory during Matrix copy"); memset(outMatrixReal[i], 0, sizeof(double)*outNumCols);
        outMatrixImag[i] = (double*) malloc(sizeof(double)*outNumCols); if (outMatrixImag[i] == NULL) throw new ApplicationException("Out of Memory during Matrix copy");memset(outMatrixImag[i], 0, sizeof(double)*outNumCols);
    }

    for (int i = 0; i < outNumRows; i++)
    {
        for (int j = 0; j < outNumCols; j++)
        {
            outMatrixReal[i][j] = oMatrix(i,j).Real;
            outMatrixImag[i][j] = oMatrix(i,j).Imag;
        }
    }
}

void CopyIntVector(const std::vector< int > &vector, int* &outVector, int &outLength)
{
    outLength = vector.size();
    outVector = (int*)malloc(sizeof(int)*outLength); if (outVector == NULL) throw new ApplicationException("Out of Memory during Vector copy"); memset(outVector, 0, sizeof(int)*outLength);
    for (int i = 0; i < outLength; i++)
    {
        outVector[i] = vector[i];
    }
}

void CopyComplexVector(const std::vector< Complex> &vector, double* &outVectorReal, double* &outVectorImag, int &outLength)
{
    outLength = vector.size();
    outVectorReal = (double*)malloc(sizeof(double)*outLength); memset(outVectorReal, 0, sizeof(double)*outLength);
    outVectorImag = (double*)malloc(sizeof(double)*outLength); memset(outVectorImag, 0, sizeof(double)*outLength);
    for (int i = 0; i < outLength; i++)
    {
        outVectorReal[i] = vector[i].Real;
        outVectorImag[i] = vector[i].Imag;
    }
}

void CopyDoubleVector(const std::vector< double > &vector, double* &outVector, int &outLength)
{
    outLength = vector.size();
    outVector = (double*)malloc(sizeof(double)*outLength); memset(outVector, 0, sizeof(double)*outLength);
    for (int i = 0; i < outLength; i++)
    {
        outVector[i] = vector[i];
    }
}

void CopyStringVector(const std::vector< std::string > &vector, char** &outVector, int &outLength)
{
    outLength = vector.size();
    outVector = (char**) malloc(sizeof(char*) *outLength);
    memset(outVector, 0, sizeof(char*)*outLength);
    for (int i = 0; i < outLength; i++)
    {
        outVector[i] = strdup(vector[i].c_str());
    }
}

}//namespace ls
