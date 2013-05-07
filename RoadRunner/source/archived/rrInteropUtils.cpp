#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrInteropUtils.h"
//---------------------------------------------------------------------------


namespace rr
{
//        internal static void FreePtrMatrix(IntPtr pointer, int nRows)
//        {
//            IntPtr[] rows = new IntPtr[nRows];
//            Marshal.Copy(pointer, rows, 0, nRows);
//            for (int i = 0; i < nRows; i++)
//            {
//                Marshal.FreeHGlobal(rows[i]);
//            } // for (int)
//            Marshal.FreeHGlobal(pointer);
//        } // FreePtrMatrix(pointer, nRows)
//
//        /// <summary>
//        /// Free ptr vector
//        /// </summary>
//        internal static void FreePtrVector(IntPtr ptrValues)
//        {
//            Marshal.FreeHGlobal(ptrValues);
//        } // FreePtrVector(ptrValues)
//
//        /// <summary>
//        /// Get complex array from ptr
//        /// </summary>
//        internal static Complex[] GetComplexArrayFromPtr(IntPtr pointerReal, IntPtr pointerImag, int nLength)
//        {
//            Complex[] oResult = new Complex[nLength];
//            double[] realTemp = GetDoubleArrayFromPtr(pointerReal, nLength);
//            double[] imagTemp = GetDoubleArrayFromPtr(pointerImag, nLength);
//
//            for (int i = 0; i < nLength; i++)
//            {
//                oResult[i] = new Complex(realTemp[i], imagTemp[i]);
//            } // for (int)
//            return oResult;
//        } // GetComplexArrayFromPtr(pointerReal, pointerImag, nLength)
//
//        /// <summary>
//        /// Get complex array from ptr
//        /// </summary>
//        internal static Complex[] GetComplexArrayFromPtrLA(IntPtr pointerReal, IntPtr pointerImag, int nLength)
//        {
//            Complex[] oResult = new Complex[nLength];
//            double[] realTemp = GetDoubleArrayFromPtrLA(pointerReal, nLength);
//            double[] imagTemp = GetDoubleArrayFromPtrLA(pointerImag, nLength);
//
//            for (int i = 0; i < nLength; i++)
//            {
//                oResult[i] = new Complex(realTemp[i], imagTemp[i]);
//            } // for (int)
//            return oResult;
//        } // GetComplexArrayFromPtr(pointerReal, pointerImag, nLength)
//
//
//        /// <summary>
//        /// Get complex matrix from ptr
//        /// </summary>
//        internal static Complex[][] GetComplexMatrixFromPtr(IntPtr pointerReal, IntPtr pointerImag, int nRows, int nCols)
//        {
//            Complex[][] oResult = new Complex[nRows][];
//            double[][] realTemp = GetDoubleMatrixFromPtr(pointerReal, nRows, nCols);
//            double[][] imagTemp = GetDoubleMatrixFromPtr(pointerImag, nRows, nCols);
//
//            for (int i = 0; i < nRows; i++)
//            {
//                oResult[i] = new Complex[nCols];
//                for (int j = 0; j < nCols; j++)
//                {
//                    oResult[i][j] = new Complex(realTemp[i][j], imagTemp[i][j]);
//                } // for (int)
//            } // for (int)
//            return oResult;
//        } // GetComplexMatrixFromPtr(pointerReal, pointerImag, nRows)
//
//        /// <summary>
//        /// Get complex matrix from ptr
//        /// </summary>
//        internal static Complex[][] GetComplexMatrixFromPtrLA(IntPtr pointerReal, IntPtr pointerImag, int nRows, int nCols)
//        {
//            Complex[][] oResult = new Complex[nRows][];
//            double[][] realTemp = GetDoubleMatrixFromPtrLA(pointerReal, nRows, nCols);
//            double[][] imagTemp = GetDoubleMatrixFromPtrLA(pointerImag, nRows, nCols);
//
//            for (int i = 0; i < nRows; i++)
//            {
//                oResult[i] = new Complex[nCols];
//                for (int j = 0; j < nCols; j++)
//                {
//                    oResult[i][j] = new Complex(realTemp[i][j], imagTemp[i][j]);
//                } // for (int)
//            } // for (int)
//            return oResult;
//        } // GetComplexMatrixFromPtrLA(pointerReal, pointerImag, nRows)
//
//        /// <summary>
//        /// Get double array from ptr
//        /// </summary>
//        internal static double[] GetDoubleArrayFromPtr(IntPtr pointer, int nLength)
//        {
//            double[] oResult = new double[nLength];
//            Marshal.Copy(pointer, oResult, 0, nLength);
//            StructAnalysis.FreeVector(pointer);
//            return oResult;
//        } // GetDoubleArrayFromPtr(pointer, nLength)
//
//        /// <summary>
//        /// Get double array from ptr
//        /// </summary>
//        internal static double[] GetDoubleArrayFromPtrLA(IntPtr pointer, int nLength)
//        {
//            double[] oResult = new double[nLength];
//            Marshal.Copy(pointer, oResult, 0, nLength);
//            LA.FreeVector(pointer);
//            return oResult;
//        } // GetDoubleArrayFromPtr(pointer, nLength)
//
/// <summary>
/// Get double matrix from ptr
/// </summary>
ls::DoubleMatrix GetDoubleMatrixFromPtr(IntPtr pointer, int nRows, int nCols)
{
//    double* oResult = new double[nRows*nCols];
    ls::DoubleMatrix oResult(nRows, nCols);

//    double* Matrix = (double*) pointer;
    double** Matrix = (double**) pointer;
    for(int row = 0; row < nRows; row++)
    {
        for(int col = 0; col < nCols; col++)
        {
            oResult(row, col) = Matrix[row][col];
        }
    }

//    StructAnalysis.FreeMatrix(pointer, nRows);
    return oResult;
}

//        /// <summary>
//        /// Get double matrix from ptr
//        /// </summary>
//        internal static double[][] GetDoubleMatrixFromPtrLA(IntPtr pointer, int nRows, int nCols)
//        {
//            IntPtr[] rawRows = new IntPtr[nRows];
//            double[][] oResult = new double[nRows][];
//            Marshal.Copy(pointer, rawRows, 0, nRows);
//            for (int i = 0; i < nRows; i++)
//            {
//                oResult[i] = new double[nCols];
//                Marshal.Copy(rawRows[i], oResult[i], 0, nCols);
//            } // for (int)
//            LA.FreeMatrix(pointer, nRows);
//            return oResult;
//        } // GetDoubleMatrixFromPtrLA(pointer, nRows, nCols)
//
//        /// <summary>
//        /// Get int array from ptr
//        /// </summary>
//        internal static int[] GetIntArrayFromPtr(IntPtr pointer, int nLength)
//        {
//            int[] oResult = new int[nLength];
//            Marshal.Copy(pointer, oResult, 0, nLength);
//            StructAnalysis.FreeVector(pointer);
//            return oResult;
//        } // GetIntArrayFromPtr(pointer, nLength)
//
//        /// <summary>
//        /// Get int array from ptr
//        /// </summary>
//        internal static int[] GetIntArrayFromPtrLA(IntPtr pointer, int nLength)
//        {
//            int[] oResult = new int[nLength];
//            Marshal.Copy(pointer, oResult, 0, nLength);
//            LA.FreeVector(pointer);
//            return oResult;
//        } // GetIntArrayFromPtr(pointer, nLength)
//
//
//        /// <summary>
//        /// Get int matrix from ptr
//        /// </summary>
//        internal static int[][] GetIntMatrixFromPtr(IntPtr pointer, int nRows, int nCols)
//        {
//            IntPtr[] rawRows = new IntPtr[nRows];
//            int[][] oResult = new int[nRows][];
//            Marshal.Copy(pointer, rawRows, 0, nRows);
//            for (int i = 0; i < nRows; i++)
//            {
//                oResult[i] = new int[nCols];
//                Marshal.Copy(rawRows[i], oResult[i], 0, nCols);
//            } // for (int)
//            StructAnalysis.FreeMatrix(pointer, nRows);
//            return oResult;
//        } // GetIntMatrixFromPtr(pointer, nRows, nCols)
//
//        /// <summary>
//        /// Get int matrix from ptr
//        /// </summary>
//        internal static int[][] GetIntMatrixFromPtrLA(IntPtr pointer, int nRows, int nCols)
//        {
//            IntPtr[] rawRows = new IntPtr[nRows];
//            int[][] oResult = new int[nRows][];
//            Marshal.Copy(pointer, rawRows, 0, nRows);
//            for (int i = 0; i < nRows; i++)
//            {
//                oResult[i] = new int[nCols];
//                Marshal.Copy(rawRows[i], oResult[i], 0, nCols);
//            } // for (int)
//            LA.FreeMatrix(pointer, nRows);
//            return oResult;
//        } // GetIntMatrixFromPtr(pointer, nRows, nCols)
//
//
/// <summary>
/// Get string array from ptr
/// </summary>
vector<string> GetStringArrayFromPtr(IntPtr &pointer, int nStrings)
{
    vector<string> oResult;
    oResult.resize(nStrings);
    char** stringRows = (char**) pointer;

    for(int i = 0; i < nStrings; i++)
    {
        char* oneString = stringRows[i];
        string aString(oneString);
        oResult[i] = aString;
    }

//    LibStructural_freeMatrix(pointer, nStrings);
    //StructAnalysis::FreeMatrix(pointer, nStrings);
    return oResult;
}

//        /// <summary>
//        /// Get string array from ptr
//        /// </summary>
//        internal static string[] GetStringArrayFromPtrLA(IntPtr pointer, int nLength)
//        {
//            IntPtr[] rawRows = new IntPtr[nLength];
//            string[] oResult = new string[nLength];
//            Marshal.Copy(pointer, rawRows, 0, nLength);
//
//            for (int i = 0; i < nLength; i++)
//            {
//                oResult[i] = Marshal.PtrToStringAnsi(rawRows[i]);
//            } // for (int)
//
//            LA.FreeMatrix(pointer, nLength);
//            return oResult;
//        } // GetStringArrayFromPtrLA(pointer, nLength)
//
/// <summary>
/// Get string from ptr
/// </summary>
string GetStringFromPtr(IntPtr &pointer, int nLength)
{
    char* oneString = (char*) pointer;
//    string sResult = Marshal.PtrToStringAnsi(pointer, nLength);
//    StructAnalysis.FreeVector(pointer);
    string sResult(oneString);
    return sResult;
}
//
//        /// <summary>
//        /// Get string from ptr
//        /// </summary>
//        internal static string GetStringFromPtrLA(IntPtr pointer, int nLength)
//        {
//            string sResult = Marshal.PtrToStringAnsi(pointer, nLength);
//            LA.FreeVector(pointer);
//            return sResult;
//        } // GetStringFromPtrLA(pointer, nLength)
//
//        /// <summary>
//        /// Map double array to pointer
//        /// </summary>
//        internal static void MapDoubleArrayToPointer(double[] oValues, out IntPtr ptrValues, out int nLength)
//        {
//            nLength = oValues.Length;
//            ptrValues = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(double)) * nLength);
//            Marshal.Copy(oValues, 0, ptrValues, nLength);
//        } // MapDoubleArrayToPointer(oValues, ptrValues, nLength)
//
//        /// <summary>
//        /// Map double array to pointer
//        /// </summary>
//        internal static void MapIntArrayToPointer(int[] oValues, out IntPtr ptrValues, out int nLength)
//        {
//            nLength = oValues.Length;
//            ptrValues = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(int)) * nLength);
//            Marshal.Copy(oValues, 0, ptrValues, nLength);
//        } // MapDoubleArrayToPointer(oValues, ptrValues, nLength)
//
//
//
//        /// <summary>
//        /// Map matrix to pointer
//        /// </summary>
//        internal static void MapMatrixToPointer(double[][] oMatrix, out IntPtr pointer, out int nRows, out int nCols)
//        {
//            nRows = oMatrix.Length;
//            nCols = oMatrix[0].Length;
//
//            // allocate memory
//            pointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(IntPtr)) * nRows);
//
//            IntPtr[] rows = new IntPtr[nRows];
//            for (int i = 0; i < nRows; i++)
//            {
//                rows[i] = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Double)) * nCols);
//                Marshal.Copy(oMatrix[i], 0, rows[i], nCols);
//            } // for (int)
//            Marshal.Copy(rows, 0, pointer, nRows);
//        } // MapMatrixToPointer(oMatrix, pointer, nRows)
//
//        /// <summary>
//        /// Map string array to pointer
//        /// </summary>
//        internal static void MapStringArrayToPointer(string[] sNames, out IntPtr pointer, out int nLength)
//        {
//            nLength = sNames.Length;
//
//            // allocate memory
//            pointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(IntPtr)) * nLength);
//
//            IntPtr[] rows = new IntPtr[nLength];
//            for (int i = 0; i < nLength; i++)
//            {
//                rows[i] = Marshal.StringToHGlobalAnsi(sNames[i]);
//            } // for (int)
//            Marshal.Copy(rows, 0, pointer, nLength);
//        } // MapStringArrayToPointer(sNames, pointer, nLength)
//
//
//    } // class InteropUtil
}//namespace rr
