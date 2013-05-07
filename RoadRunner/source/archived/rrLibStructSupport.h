#ifndef rrLibStructSupportH
#define rrLibStructSupportH
#include <vector>
#include <string>
#include "rrStringList.h"
#include "rrObject.h"
#include "libstruct/lsLibstructural.h"


using std::vector;
using std::string;

//---------------------------------------------------------------------------
typedef long* IntPtr;
namespace rr
{
using namespace ls;

//The following class is a translation of C#'s StructuralAnalysis.cs in LibStructurlaCSharp
class RR_DECLSPEC StructAnalysis : public rrObject
{
    protected:
        static LibStructural           *mInstance;

//        void WriteMatlabForm(System.IO.TextWriter oWriter, double[][] oMatrix)

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrixGJ")]
//        int LibStructural_getGammaMatrixGJ(System.IntPtr inMatrix, int numRows, int numCols,
//            out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_findPositiveGammaMatrix")]
//        int LibStructural_findPositiveGammaMatrix(
//            System.IntPtr inMatrix, int numRows, int numCols,
//            IntPtr inRowLabels,
//            out System.IntPtr outMatrix, out int outRows, out int outCols,
//            out System.IntPtr outRowLabels, out int outRowCount);


       public:
                                                        StructAnalysis();
        virtual                                        ~StructAnalysis();
        LibStructural*                                  GetInstance(){return mInstance;}
        void                                            reset();

//        internal const string LIBRARY_FILE = "LibStructural";
//        string AnalyzeWithFullyPivotedLU()
//        string AnalyzeWithFullyPivotedLUAndRunTests()
//        string AnalyzeWithLU()
//        string AnalyzeWithLUAndRunTests()
//        string AnalyzeWithQR()
//        double[][] GetColumnReorderedNrMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetColumnReorderedNrMatrix()
//        void GetColumnReorderedNrMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        string[] getConservedLaws()
//        double[] GetConservedSums()
//        string[] GetDependentReactionIds()
        StringList GetDependentSpeciesIds();
//        double[][] GetFullyReorderedStoichiometryMatrix()
//        double[][] GetFullyReorderedStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetFullyReorderedStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
        ls::DoubleMatrix*    GetGammaMatrix();
//        double[][] GetGammaMatrixGJ(double[][] oMatrix)
//        bool FindPositiveGammaMatrix(double[][] oMatrix,  out double[][] gammaMatrix, ref string[] rowNames, out string[] colNames)
//        bool FindPositiveGammaMatrix(double[][] oMatrix, out double[][] gammaMatrix)
//        double[][] GetGammaMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetGammaMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        string[] GetIndependentReactionIds()
        StringList GetIndependentSpeciesIds();
//        void GetInitialConditions(out string[] variableNames, out double[] initialValues)
//        double[][] GetK0Matrix()
//        double[][] GetK0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetK0MatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetKMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetKMatrix()
//        void GetKMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
        ls::DoubleMatrix*   GetL0Matrix();
        ls::DoubleMatrix*   GetL0Matrix(vector<string>& sRowLabels, vector<string>& sColumnLabels);
//        double[][] GetL0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
        void GetL0MatrixLabels(vector<string>& sRowLabels, vector<string>& sColumnLabels);
        ls::DoubleMatrix*   GetLinkMatrix();
//        double[][] GetLinkMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetLinkMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetLMatrix()
//        double[][] GetLMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetLMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        string GetModelName()
//        double[][] GetN0Matrix()
//        double[][] GetN0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetN0MatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetNDCMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetNDCMatrix()
//        void GetNDCMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetNICMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetNICMatrix()
//        void GetNICMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNmatrixSparsity")]
//        extern double GetNmatrixSparsity();

//        double[][] GetNrMatrix()
//        double[][] GetNrMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetNrMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthConservedEntity")]
//        //internal static extern int LibStructural_getNthConservedEntity(int n, out IntPtr result, out int nLength);
//        //public static string GetNthConservedEntity(int n)
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumConservedSums")]
//        int GetNumConservedSums();

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumDepSpecies")]
//        int GetNumDependentSpecies();
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumIndSpecies")]
        int GetNumIndependentSpecies();
//
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumReactions")]
//        int GetNumReactions();
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumDepReactions")]
//        int GetNumDependentReactions();

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumIndReactions")]
//        int GetNumIndependentReactions();
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumSpecies")]
        int GetNumSpecies();
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getRank")]
//        int GetRank();
//        string[] GetReactionIds()
//        string[] GetReorderedReactionIds()
        StringList              GetReorderedSpeciesIds();
        ls::DoubleMatrix*   GetReorderedStoichiometryMatrix();
//        double[][] GetReorderedStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        void GetReorderedStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
        StringList GetSpeciesIds();
//        double[][] GetStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
//        double[][] GetStoichiometryMatrix()
//        void GetStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
//        string GetTestDetails(  )
//        void LoadReactionNames(string[] reactionNames)
          string LoadSBML(const string& sbml);
//        string LoadSBMLFromFile(string sFileName)
//        string LoadSBMLWithTests(string sbml)
//        void LoadSpeciesNames(string[] speciesNames, double[] speciesValues)
//        void LoadSpeciesNames(string[] speciesNames)
//        void LoadStoichiometryMatrix(double[][] oMatrix)
//        void LoadStoichiometryMatrix(double[][] oMatrix, string[] speciesNames, string[] reactionNames)
//        void LoadStoichiometryMatrix(double[][] oMatrix, string[] speciesNames, double[] initialValues, string[] reactionNames)
//        void PrintDoubleMatrix(double[][] oMatrix)
//        void PrintIntMatrix(int[][] oMatrix)
//        void PrintComplexMatrix(Complex[][] oMatrix)
//        void PrintDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix)

//        void PrintIntMatrix(System.IO.TextWriter oWriter, int[][] oMatrix)
//        void PrintComplexMatrix(System.IO.TextWriter oWriter, Complex[][] oMatrix)
//        void PrintLabledDoubleMatrix(double[][] oMatrix, IEnumerable<string> sRowLables, IEnumerable<string> sColumnLables)
//        void PrintLabledDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix, IEnumerable<string> sRowLablesIn, IEnumerable<string> sColumnLablesIn)
//        void PrintSortedLabledDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix, IEnumerable<string> sRowLablesIn, IEnumerable<string> sColumnLablesIn)
//        void PrintStringArray(System.IO.TextWriter oWriter, string[] oArray)
//        void PrintStringArray(string[] oArray)

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_setTolerance")]
//        void SetTolerance(double dTolerance);
//        bool[] ValidateStructuralMatrices(  )

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_freeMatrix")]
        void FreeMatrix(IntPtr matrix, int numRows);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_freeVector")]
//        void FreeVector(IntPtr vector);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithFullyPivotedLU")]
//        int LibStructural_analyzeWithFullyPivotedLU(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithFullyPivotedLUwithTests")]
//        int LibStructural_analyzeWithFullyPivotedLUwithTests(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithLU")]
//        int LibStructural_analyzeWithLU(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithLUandRunTests")]
//        int LibStructural_analyzeWithLUandRunTests(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithQR")]
//        int LibStructural_analyzeWithQR(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getColumnReorderedNrMatrix")]
//        int LibStructural_getColumnReorderedNrMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getColumnReorderedNrMatrixLabels")]
//        int LibStructural_getColumnReorderedNrMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getConservedLaws")]
//        int LibStructural_getConservedLaws(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getConservedSums")]
//        int LibStructural_getConservedSums(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getDependentReactionIds")]
//        int LibStructural_getDependentReactionIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getDependentSpeciesIds")]
//        LibStructural_getDependentSpeciesIds(out IntPtr outArray, out int outLength);

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthReorderedSpeciesId")]
//        //internal static extern int LibStructural_getNthReorderedSpeciesId(int n, out IntPtr result, out int nLength);
//        //public static string GetNthReorderedSpeciesId(int n)

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthIndependentSpeciesId")]
//        //internal static extern int LibStructural_getNthIndependentSpeciesId(int n, out IntPtr result, out int nLength);
//        //public static string GetNthIndependentSpeciesId(int n)

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthDependentSpeciesId")]
//        //internal static extern int LibStructural_getNthDependentSpeciesId(int n, out IntPtr result, out int nLength);
//        //public static string GetNthDependentSpeciesId(int n)

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthReactionId")]
//        //internal static extern int LibStructural_getNthReactionId(int n, out IntPtr result, out int nLength);
//        //public static string GetNthReactionId(int n)

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrix")]
//        int LibStructural_getGammaMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrixLabels")]
//        int LibStructural_getGammaMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getIndependentReactionIds")]
//        int LibStructural_getIndependentReactionIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getIndependentSpeciesIds")]
//        int LibStructural_getIndependentSpeciesIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getInitialConditions")]
//        int LibStructural_getInitialConditions(out IntPtr outVariableNames, out IntPtr outValues, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getK0Matrix")]
//        int LibStructural_getK0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getK0MatrixLabels")]
//        int LibStructural_getK0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getKMatrix")]
//        int LibStructural_getKMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getKMatrixLabels")]
//        int LibStructural_getKMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getL0Matrix")]
//        int LibStructural_getL0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getL0MatrixLabels")]
//        int LibStructural_getL0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getLinkMatrix")]
//        int LibStructural_getLinkMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getLinkMatrixLabels")]
//        int LibStructural_getLMatrixLabels(out System.IntPtr outRowLabels,
//            out int outRowCount,
//            out IntPtr outColLabels,
//            out int outColCount);
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getModelName")]
//        int LibStructural_getModelName(out IntPtr result, out int nLength);
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getN0Matrix")]
//        int LibStructural_getN0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getN0MatrixLabels")]
//        int LibStructural_getN0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNDCMatrix")]
//        int LibStructural_getNDCMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNDCMatrixLabels")]
//        int LibStructural_getNDCMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNICMatrix")]
//        int LibStructural_getNICMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
//
//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNICMatrixLabels")]
//        int LibStructural_getNICMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNrMatrix")]
//        int LibStructural_getNrMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNrMatrixLabels")]
//        int LibStructural_getNrMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReactionIds")]
//        int LibStructural_getReactionIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedReactionIds")]
//        int LibStructural_getReorderedReactionIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedSpeciesIds")]
//        int LibStructural_getReorderedSpeciesIds(out IntPtr outArray, out int outLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedStoichiometryMatrix")]
//        int LibStructural_getReorderedStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedStoichiometryMatrixLabels")]
//        int LibStructural_getReorderedStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getFullyReorderedStoichiometryMatrix", CallingConvention=CallingConvention.Cdecl)]
//        int LibStructural_getFullyReorderedStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getFullyReorderedStoichiometryMatrixLabels")]
//        int LibStructural_getFullyReorderedStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getSpeciesIds")]
//        int LibStructural_getSpeciesIds(out IntPtr outArray, out int outLength);

//        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthConservedSum")]
//        //public static extern double GetNthConservedSum(int n);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getStoichiometryMatrix")]
//        int LibStructural_getStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getStoichiometryMatrixLabels")]
//        int LibStructural_getStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getTestDetails")]
//        int LibStructural_getTestDetails(out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadReactionNames", CharSet=CharSet.Ansi)]
//        void LibStructural_loadReactionNames(IntPtr reactionNames, int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBML", CharSet=CharSet.Ansi)]
//        int LibStructural_loadSBML(string sSBML, out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBMLFromFile", CharSet = CharSet.Ansi)]
//        int LibStructural_loadSBMLFromFile(string sFileName, out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBMLwithTests", CharSet = CharSet.Ansi)]
//        int LibStructural_loadSBMLwithTests(string sSbml, out IntPtr result, out int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSpecies",CharSet=CharSet.Ansi)]
//        void LibStructural_loadSpecies(IntPtr speciesNames, IntPtr speciesValues, int nLength);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadStoichiometryMatrix")]
//        int LibStructural_loadStoichiometryMatrix (IntPtr oMatrix, int nRows, int nCols);

//        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_validateStructuralMatrices")]
//        int LibStructural_validateStructuralMatrices(out System.IntPtr outResults, out int outLength);
};

}
#endif


//////////////////////////////////////////////
//////////////////////////////////////////////
////C# originals...  from StructuralAnalysis.cs
////namespace libstructural
////{
////    /// <summary>
////    /// <para>StructAnalysis represents the main class for all structural analysis on
////    /// either <a href="http://sbml.org/">SBML</a> models or directly on a
////    /// provided stoichiometry matrix.</para>
////    /// <para>The model can be either analyzed employing QR factorization with
////    /// householder reflections, or with LU(P) factorization. Though clearly
////    /// the QR factorization is the superior method.</para>
////    /// <para>For further information please see also:
////    /// <a href="http://bioinformatics.oxfordjournals.org/cgi/content/abstract/bti800v1">
////    ///        Vallabhajosyula RR, Chickarmane V, Sauro HM.<em>Conservation analysis of large biochemical
////    ///        networks</em>. <b>Bioinformatics</b> 2005 Nov 29</a></para>
////    /// <para>For examples on how to use the library see <see cref="LoadSBML"/> and <see cref="LoadStoichiometryMatrix(double[][])"/></para>
////    /// </summary>
////    public static class StructAnalysis
////    {
////        /// <summary>
////        /// The library file to where the members will be found. This will be
////        /// libstructural.dll on windows, and libLibStructural.so on linux and
////        /// OSX under mono.
////        /// </summary>
////        internal const string LIBRARY_FILE = "LibStructural";
////
////
////        /// <summary>
////        /// Analyze the prior loaded model with fully pivoted LU factorization.
////        /// <remarks>
////        /// This will only work if the stoichiometry matrix represents a N-by-N
////        /// matrix.</remarks>
////        /// </summary>
////        /// <returns>Any warnings that occurred while analyzing the Stoichiometry
////        /// matrix, such as 'No floating species', 'No Reactions' or an empty
////        /// stoichiometry matrix.</returns>
////        public static string AnalyzeWithFullyPivotedLU()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_analyzeWithFullyPivotedLU(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // AnalyzeWithFullyPivotedLU()
////
////        /// <summary>
////        /// Analyze the prior loaded model with fully pivoted LU factorization.
////        /// After analyzing all the computed matrices will be tested for their
////        /// validity. (See also <see cref="GetTestDetails"/>.)
////        /// <remarks>
////        /// This will only work if the stoichiometry matrix represents a N-by-N
////        /// matrix.</remarks>
////        /// </summary>
////        /// <returns>Any warnings that occurred while analyzing the Stoichiometry
////        /// matrix, such as 'No floating species', 'No Reactions' or an empty
////        /// stoichiometry matrix.</returns>
////        public static string AnalyzeWithFullyPivotedLUAndRunTests()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_analyzeWithFullyPivotedLUwithTests(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // AnalyzeWithFullyPivotedLUAndRunTests()
////
////        /// <summary>
////        /// Analyze the prior loaded model with partial pivoted LU factorization.
////        /// </summary>
////        /// <returns>Any warnings that occurred while analyzing the Stoichiometry
////        /// matrix, such as 'No floating species', 'No Reactions' or an empty
////        /// stoichiometry matrix.</returns>
////        public static string AnalyzeWithLU()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_analyzeWithLU(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // AnalyzeWithLU()
////
////        /// <summary>
////        /// Analyze the prior loaded model with partial pivoted LU factorization.
////        /// After analyzing all the computed matrices will be tested for their
////        /// validity. (See also <see cref="GetTestDetails"/>.)
////        /// </summary>
////        /// <returns>Any warnings that occurred while analyzing the Stoichiometry
////        /// matrix, such as 'No floating species', 'No Reactions' or an empty
////        /// stoichiometry matrix.</returns>
////        public static string AnalyzeWithLUAndRunTests()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_analyzeWithLUandRunTests(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // AnalyzeWithLUAndRunTests()
////
////        /// <summary>
////        /// Analyze the prior loaded model with QR factorization (employing pivoting
////        /// through householder reflections).
////        /// </summary>
////        /// <returns>Any warnings that occurred while analyzing the Stoichiometry
////        /// matrix, such as 'No floating species', 'No Reactions' or an empty
////        /// stoichiometry matrix.</returns>
////        public static string AnalyzeWithQR()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_analyzeWithQR(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // AnalyzeWithQR()
////
////        /// <summary>
////        /// Get column reordered Nr matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetColumnReorderedNrMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getColumnReorderedNrMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////            return GetColumnReorderedNrMatrix();
////
////        } // GetColumnReorderedNrMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get column reordered Nr matrix
////        /// </summary>
////        public static double[][] GetColumnReorderedNrMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getColumnReorderedNrMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Nr Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetColumnReorderedNrMatrix()
////
////        /// <summary>
////        /// Get column reordered Nr matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetColumnReorderedNrMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getColumnReorderedNrMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetColumnReorderedNrMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get conserved entities
////        /// </summary>
////        public static string[] getConservedLaws()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getConservedLaws(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // getConservedLaws()
////
////        /// <summary>
////        /// Get conserved sums
////        /// </summary>
////        public static double[] GetConservedSums()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getConservedSums(out pointer, out nLength);
////            return InteropUtil.GetDoubleArrayFromPtr(pointer, nLength);
////        } // GetConservedSums()
////
////        /// <summary>
////        /// Get dependent reaction ids
////        /// </summary>
////        public static string[] GetDependentReactionIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getDependentReactionIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetDependentReactionIds()
////
////        /// <summary>
////        /// Get dependent species ids
////        /// </summary>
////        public static string[] GetDependentSpeciesIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getDependentSpeciesIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetDependentSpeciesIds()
////
////
////        /// <summary>
////        /// Get fully reordered stoichiometry matrix
////        /// </summary>
////        public static double[][] GetFullyReorderedStoichiometryMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getFullyReorderedStoichiometryMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The FullyReordered Stoichiometry matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetFullyReorderedStoichiometryMatrix()
////
////        /// <summary>
////        /// Get fully reordered stoichiometry matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetFullyReorderedStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getFullyReorderedStoichiometryMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////            return GetFullyReorderedStoichiometryMatrix();
////        } // GetFullyReorderedStoichiometryMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get fully reordered stoichiometry matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetFullyReorderedStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getFullyReorderedStoichiometryMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetFullyReorderedStoichiometryMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get Gamma matrix
////        /// </summary>
////        public static double[][] GetGammaMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getGammaMatrix(out pointer, out nRows, out nCols) < 0 )
////                throw new Exception("The Conservation Law Array has not yet been calculated, please call one of the analyze methods first.");
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetGammaMatrix()
////
////
////        public static double[][] GetGammaMatrixGJ(double[][] oMatrix)
////        {
////            IntPtr pointer; int nRows; int nCols;
////            InteropUtil.MapMatrixToPointer(oMatrix, out pointer, out nRows, out nCols);
////
////            IntPtr pointerMatrix; int nMatrixRows; int nMatrixCols;
////            LibStructural_getGammaMatrixGJ(pointer, nRows, nCols, out pointerMatrix, out nMatrixRows, out nMatrixCols);
////
////            InteropUtil.FreePtrMatrix(pointer, nRows);
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointerMatrix, nMatrixRows, nMatrixCols);
////
////        } // GetGammaMatrixGJ(oMatrix)
////
////        public static bool FindPositiveGammaMatrix(double[][] oMatrix,  out double[][] gammaMatrix, ref string[] rowNames, out string[] colNames)
////        {
////            IntPtr pointer; int nRows; int nCols;
////            InteropUtil.MapMatrixToPointer(oMatrix, out pointer, out nRows, out nCols);
////
////            IntPtr ptrRowNames; int nRowLength;
////            InteropUtil.MapStringArrayToPointer(rowNames, out ptrRowNames, out nRowLength);
////
////            IntPtr pointerMatrix; int nMatrixRows; int nMatrixCols;
////            IntPtr outRowLabels; int outRowCount;
////            if (LibStructural_findPositiveGammaMatrix(pointer, nRows, nCols,
////                ptrRowNames,
////                out pointerMatrix, out nMatrixRows, out nMatrixCols,
////                out outRowLabels, out outRowCount) < 0)
////            {
////                InteropUtil.FreePtrMatrix(ptrRowNames, nRowLength);
////                InteropUtil.FreePtrMatrix(pointer, nRows);
////                gammaMatrix = null;
////                colNames = null;
////                return false;
////                //throw new Exception("Could not find a permutation of the given stoichiometry matrix that would yield a positive gamma matrix.");
////            }
////
////            InteropUtil.FreePtrMatrix(ptrRowNames, nRowLength);
////            InteropUtil.FreePtrMatrix(pointer, nRows);
////            colNames = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            gammaMatrix = InteropUtil.GetDoubleMatrixFromPtr(pointerMatrix, nMatrixRows, nMatrixCols);
////            rowNames = new string[gammaMatrix.Length];
////            for (int i = 0; i < rowNames.Length; i++)
////            {
////                rowNames[i] = i.ToString();
////            }
////            return true;
////        }
////
////
////        public static bool FindPositiveGammaMatrix(double[][] oMatrix, out double[][] gammaMatrix)
////        {
////            string[] rowNames = new string[oMatrix.Length];
////            for (int i = 0; i < rowNames.Length; i++)
////            {
////                rowNames[i] = i.ToString();
////            }
////            string[] colNames;
////            return FindPositiveGammaMatrix(oMatrix, out gammaMatrix, ref rowNames, out colNames);
////
////
////        } // FindPositiveGammaMatrix(oMatrix)
////
////
////        /// <summary>
////        /// Get Gamma matrix
////        /// </summary>
////        public static double[][] GetGammaMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetGammaMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetGammaMatrix();
////        } // GetGammaMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get Gamma matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetGammaMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getGammaMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetGammaMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get independent reaction ids
////        /// </summary>
////        public static string[] GetIndependentReactionIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getIndependentReactionIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetIndependentReactionIds()
////
////        /// <summary>
////        /// Get independent species ids
////        /// </summary>
////        public static string[] GetIndependentSpeciesIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getIndependentSpeciesIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetIndependentSpeciesIds()
////
////        /// <summary>
////        /// Get initial conditions
////        /// </summary>
////        public static void GetInitialConditions(out string[] variableNames, out double[] initialValues)
////        {
////            IntPtr outVariableNames; IntPtr outValues; int outLength;
////            LibStructural_getInitialConditions(out outVariableNames, out outValues, out outLength);
////
////            variableNames = InteropUtil.GetStringArrayFromPtr(outVariableNames, outLength);
////            initialValues = InteropUtil.GetDoubleArrayFromPtr(outValues, outLength);
////        } // GetInitialConditions(variableNames, initialValues)
////
////
////        /// <summary>
////        /// Get K0 matrix
////        /// </summary>
////        public static double[][] GetK0Matrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getK0Matrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The K0 Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetK0Matrix()
////
////        /// <summary>
////        /// Get K0 matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetK0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetK0MatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetK0Matrix();
////
////        } // GetK0Matrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get K0 matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetK0MatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getK0MatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetK0MatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get K matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetKMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetKMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetKMatrix();
////        } // GetKMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get K matrix
////        /// </summary>
////        public static double[][] GetKMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getKMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Nullspace has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetKMatrix()
////
////        /// <summary>
////        /// Get K matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetKMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getKMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetKMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L0 matrix
////        /// </summary>
////        public static double[][] GetL0Matrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getL0Matrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The L0 Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetL0Matrix()
////
////        /// <summary>
////        /// Get L0 matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetL0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetL0MatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetL0Matrix();
////        } // GetL0Matrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L0 matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetL0MatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getL0MatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetL0MatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L matrix (link matrix)
////        /// </summary>
////        public static double[][] GetLinkMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getLinkMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Link Matrix has not yet been calculated, please call one of the analyze methods first.");
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetLinkMatrix()
////
////        /// <summary>
////        /// Get L matrix (link matrix)
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetLinkMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetLMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetLinkMatrix();
////        } // GetLinkMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L matrix (link matrix) labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetLinkMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetLMatrixLabels(out sRowLabels, out sColumnLabels);
////        } // GetLinkMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L matrix
////        /// </summary>
////        public static double[][] GetLMatrix()
////        {
////            return GetLinkMatrix();
////        } // GetLMatrix()
////
////        /// <summary>
////        /// Get L matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetLMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            return GetLinkMatrix(out sRowLabels, out sColumnLabels);
////        } // GetLMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get L matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetLMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getLMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetLMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get model name
////        /// </summary>
////        public static string GetModelName()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getModelName(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // GetModelName()
////
////        /// <summary>
////        /// Get N0 matrix
////        /// </summary>
////        public static double[][] GetN0Matrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getN0Matrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The N0 Matrix has not yet been calculated, please call one of the analyze methods first.");
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetN0Matrix()
////
////
////        /// <summary>
////        /// Get N0 matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetN0Matrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetN0MatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetN0Matrix();
////        } // GetN0Matrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get N0 matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetN0MatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getN0MatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetN0MatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get dependent reaction  matrix
////        /// </summary>
////        public static double[][] GetNDCMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetNDCMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetNDCMatrix();
////        } // GetNDCMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get dependent reaction  matrix
////        /// </summary>
////        public static double[][] GetNDCMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getNDCMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Nr Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetNDCMatrix()
////
////        /// <summary>
////        /// Get dependent reaction  matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetNDCMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getNDCMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetNDCMatrixLabels(sRowLabels, sColumnLabels)
////
////
////        /// <summary>
////        /// Get independent reaction  matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetNICMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetNICMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetNICMatrix();
////        } // GetNICMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get independent reaction  matrix
////        /// </summary>
////        public static double[][] GetNICMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getNICMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Nr Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetNICMatrix()
////
////        /// <summary>
////        /// Get independent reaction  matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetNICMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getNICMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetNICMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Returns the sparsity of the stoichiometry matrix. (i.e: the percentage
////        /// of non-zero elements in the stoichiometry matrix).
////        /// </summary>
////        /// <returns>the sparsity of the stoichiometry matrix</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNmatrixSparsity")]
////        public static extern double GetNmatrixSparsity();
////
////        /// <summary>
////        /// Get Nr matrix
////        /// </summary>
////        public static double[][] GetNrMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getNrMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Nr Matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetNrMatrix()
////
////        /// <summary>
////        /// Get Nr matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetNrMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetNrMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetNrMatrix();
////        } // GetNrMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get number matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetNrMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getNrMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetNrMatrixLabels(sRowLabels, sColumnLabels)
////
////
////        ///// Return Type: char*
////        /////n: int
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthConservedEntity")]
////        //internal static extern int LibStructural_getNthConservedEntity(int n, out IntPtr result, out int nLength);
////        //public static string GetNthConservedEntity(int n)
////        //{
////        //    IntPtr pointer; int nLength;
////        //    LibStructural_getNthConservedEntity(n, out pointer, out nLength);
////        //    return InteropUtil.GetStringFromPtr(pointer, nLength);
////        //}
////
////
////        /// <summary>
////        /// Returns the number of conserved sums.
////        /// </summary>
////        /// <returns>the number of conserved sums</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumConservedSums")]
////        public static extern int GetNumConservedSums();
////
////
////        /// <summary>
////        /// Returns the number of dependent species.
////        /// </summary>
////        /// <returns>the number of dependent species</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumDepSpecies")]
////        public static extern int GetNumDependentSpecies();
////
////
////        /// <summary>
////        /// Returns the number of independent species.
////        /// </summary>
////        /// <returns>the number of independent species</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumIndSpecies")]
////        public static extern int GetNumIndependentSpecies();
////
////
////        /// <summary>
////        /// Returns the number of reactions.
////        /// </summary>
////        /// <returns>the number of reactions</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumReactions")]
////        public static extern int GetNumReactions();
////
////        /// <summary>
////        /// Returns the number of dependent reactions.
////        /// </summary>
////        /// <returns>the number of dependent reactions</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumDepReactions")]
////        public static extern int GetNumDependentReactions();
////
////
////        /// <summary>
////        /// Returns the number of independent reactions.
////        /// </summary>
////        /// <returns>the number of independent reactions</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumIndReactions")]
////        public static extern int GetNumIndependentReactions();
////
////        /// <summary>
////        /// Returns the number of species.
////        /// </summary>
////        /// <returns>the number of species</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNumSpecies")]
////        public static extern int GetNumSpecies();
////
////
////        /// <summary>
////        /// Returns the rank of the stoichiometry matrix
////        /// </summary>
////        /// <returns>the rank of the stoichiometry matrix</returns>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getRank")]
////        public static extern int GetRank();
////
////        /// <summary>
////        /// Get reaction ids
////        /// </summary>
////        public static string[] GetReactionIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getReactionIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetReactionIds()
////
////        /// <summary>
////        /// Get reordered reaction ids
////        /// </summary>
////        public static string[] GetReorderedReactionIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getReorderedReactionIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetReorderedReactionIds()
////
////        /// <summary>
////        /// Get reordered species ids
////        /// </summary>
////        public static string[] GetReorderedSpeciesIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getReorderedSpeciesIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetReorderedSpeciesIds()
////
////        /// <summary>
////        /// Get reordered stoichiometry matrix
////        /// </summary>
////        public static double[][] GetReorderedStoichiometryMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getReorderedStoichiometryMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Reordered Stoichiometry matrix has not yet been calculated, please call one of the analyze methods first.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetReorderedStoichiometryMatrix()
////
////        /// <summary>
////        /// Get reordered stoichiometry matrix
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static double[][] GetReorderedStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetReorderedStoichiometryMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetReorderedStoichiometryMatrix();
////        } // GetReorderedStoichiometryMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get reordered stoichiometry matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetReorderedStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getReorderedStoichiometryMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetReorderedStoichiometryMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get species ids
////        /// </summary>
////        public static string[] GetSpeciesIds()
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getSpeciesIds(out pointer, out nLength);
////            return InteropUtil.GetStringArrayFromPtr(pointer, nLength);
////        } // GetSpeciesIds()
////
////
////        /// <summary>
////        /// Get stoichiometry matrix
////        /// </summary>
////        public static double[][] GetStoichiometryMatrix(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            GetStoichiometryMatrixLabels(out sRowLabels, out sColumnLabels);
////            return GetStoichiometryMatrix();
////        } // GetStoichiometryMatrix(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get stoichiometry matrix
////        /// </summary>
////        public static double[][] GetStoichiometryMatrix()
////        {
////            IntPtr pointer; int nRows; int nCols;
////            if (LibStructural_getStoichiometryMatrix(out pointer, out nRows, out nCols) < 0)
////                throw new Exception("The Stoichiometry Matrix is not yet available, please call one of the load methods first and then one of the analyze methods.");
////
////            return InteropUtil.GetDoubleMatrixFromPtr(pointer, nRows, nCols);
////        } // GetStoichiometryMatrix()
////
////        /// <summary>
////        /// Get stoichiometry matrix labels
////        /// </summary>
////        /// <param name="sColumnLabels">after calling the method, this string array
////        /// will hold the column lables for the matrix.</param>
////        /// <param name="sRowLabels">after calling the method, this string array
////        /// will hold the row lables for the matrix.</param>
////        public static void GetStoichiometryMatrixLabels(out string[] sRowLabels, out string[] sColumnLabels)
////        {
////            IntPtr outRowLabels; IntPtr outColLabels; int outRowCount; int outColCount;
////            LibStructural_getStoichiometryMatrixLabels(out outRowLabels, out outRowCount, out outColLabels, out outColCount);
////            sRowLabels = InteropUtil.GetStringArrayFromPtr(outRowLabels, outRowCount);
////            sColumnLabels = InteropUtil.GetStringArrayFromPtr(outColLabels, outColCount);
////        } // GetStoichiometryMatrixLabels(sRowLabels, sColumnLabels)
////
////        /// <summary>
////        /// Get test details
////        /// </summary>
////        public static string GetTestDetails(  )
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_getTestDetails(out pointer, out nLength);
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // GetTestDetails()
////
////        /// <summary>
////        /// Load reaction names
////        /// </summary>
////        public static void LoadReactionNames(string[] reactionNames)
////        {
////            IntPtr ptrNames; int nLength;
////            InteropUtil.MapStringArrayToPointer(reactionNames, out ptrNames, out nLength);
////            LibStructural_loadReactionNames(ptrNames, nLength);
////            InteropUtil.FreePtrMatrix(ptrNames, nLength);
////        } // LoadReactionNames(reactionNames)
////
////        /// <summary>
////        /// Load SBML from a string into the library.
////        /// <example>This shows how to load and analyze an SBML file
////        /// <code>
////        ///    using libstructural;
////        ///
////        ///    class Program
////        ///    {
////        ///        static void Main(string[] args)
////        ///        {
////        ///            // read the SBML from a file
////        ///            StreamReader oReader = new StreamReader("c:/Users/fbergman/Documents/SBML Models/BorisEJB.xml");
////        ///            string sSBML = oReader.ReadToEnd(); oReader.Close();
////        ///
////        ///            // load it into the Structural Library
////        ///            StructAnalysis.LoadSBML(sSBML);
////        ///
////        ///            // Print Test Details
////        ///            Console.WriteLine(StructAnalysis.GetTestDetails());
////        ///
////        ///            // get reordered stoichiometry matrix for further analysis
////        ///            double[][] reorderedStoichiometry = StructAnalysis.GetReorderedStoichiometryMatrix();
////        ///
////        ///            // ...
////        ///        }
////        ///
////        ///    }
////        /// </code>
////        /// </example>
////        /// </summary>
////        /// <param name="sbml">the SBML string</param>
////        public static string LoadSBML(string sbml)
////        {
////            IntPtr pointer; int nLength;
////            if (LibStructural_loadSBML(sbml, out pointer, out nLength) < 0)
////                throw new Exception("The SBML could not be loaded, please verify that it is a valid SBML file.");
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // LoadSBML(sbml)
////
////
////        /// <summary>
////        /// Load SBML from a string into the library.
////        /// <example>This shows how to load and analyze an SBML file
////        /// <code>
////        ///    using libstructural;
////        ///
////        ///    class Program
////        ///    {
////        ///        static void Main(string[] args)
////        ///        {
////        ///
////        ///            // load SBML model into the Structural Library
////        ///            StructAnalysis.LoadSBMLFromFile("c:/Users/fbergman/Documents/SBML Models/BorisEJB.xml");
////        ///
////        ///            // Print Test Details
////        ///            Console.WriteLine(StructAnalysis.GetTestDetails());
////        ///
////        ///            // get reordered stoichiometry matrix for further analysis
////        ///            double[][] reorderedStoichiometry = StructAnalysis.GetReorderedStoichiometryMatrix();
////        ///
////        ///            // ...
////        ///        }
////        ///
////        ///    }
////        /// </code>
////        /// </example>
////        /// </summary>
////        /// <param name="sFileName">the full path to a SBML file</param>
////        public static string LoadSBMLFromFile(string sFileName)
////        {
////            IntPtr pointer; int nLength;
////            if (LibStructural_loadSBMLFromFile(sFileName, out pointer, out nLength) < 0)
////                throw new Exception("The SBML could not be loaded, please verify that it is a valid SBML file.");
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // LoadSBMLFromFile(sbml)
////
////
////        /// <summary>
////        /// Load the SBML and perform validity tests. For an example <see cref="LoadSBML"/>.
////        /// </summary>
////        /// <param name="sbml">The SBML to be loaded</param>
////        public static string LoadSBMLWithTests(string sbml)
////        {
////            IntPtr pointer; int nLength;
////            if (LibStructural_loadSBMLwithTests(sbml, out pointer, out nLength) < 0)
////                throw new Exception("The SBML could not be loaded, please verify that it is a valid SBML file.");
////            return InteropUtil.GetStringFromPtr(pointer, nLength);
////        } // LoadSBMLWithTests(sbml)
////
////        /// <summary>
////        /// Load species names and initial conditions. This should be done after <see cref="LoadStoichiometryMatrix(double[][])"/>.
////        /// </summary>
////        public static void LoadSpeciesNames(string[] speciesNames, double[] speciesValues)
////        {
////            IntPtr ptrSpeciesNames; int nLength;
////            InteropUtil.MapStringArrayToPointer(speciesNames, out ptrSpeciesNames, out nLength);
////            IntPtr ptrSpeciesValues;
////            InteropUtil.MapDoubleArrayToPointer(speciesValues, out ptrSpeciesValues, out nLength);
////            LibStructural_loadSpecies ( ptrSpeciesNames, ptrSpeciesValues, nLength);
////            InteropUtil.FreePtrMatrix(ptrSpeciesNames, nLength);
////            InteropUtil.FreePtrVector(ptrSpeciesValues);
////        } // LoadSpeciesNames(speciesNames, speciesValues)
////
////        /// <summary>
////        /// Load species names. This should be done after <see cref="LoadStoichiometryMatrix(double[][])"/>.
////        /// </summary>
////        public static void LoadSpeciesNames(string[] speciesNames)
////        {
////            LoadSpeciesNames(speciesNames, new double[speciesNames.Length]);
////        } // LoadSpeciesNames(speciesNames)
////
////        /// <summary>
////        /// Load stoichiometry matrix into the library.
////        /// <example>This example shows how to use the library with a stoichiometry matrix
////        /// <code>
////        ///    using libstructural;
////        ///
////        ///    class Program
////        ///    {
////        ///        static void Main(string[] args)
////        ///        {
////        ///
////        ///            // load a Stoichiometry Matrix
////        ///            StructAnalysis.LoadStoichiometryMatrix(
////        ///                new double[][]
////        ///                {
////        ///                     new double[] {1.0,  -1.0, -1.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0},
////        ///                     new double[] {0.0,   1.0,  0.0, -1.0, 0.0, -1.0,  0.0,  0.0,  0.0},
////        ///                     new double[] {0.0,   0.0,  1.0,  0.0, 0.0,  1.0, -1.0,  0.0,  0.0},
////        ///                     new double[] {0.0,   0.0,  0.0,  2.0, 0.0,  0.0,  1.0, -1.0,  0.0},
////        ///                     new double[] {0.0,   0.0,  0.0, -1.0, 1.0,  0.0,  0.0,  0.0,  0.0},
////        ///                     new double[] {0.0,   0.0,  0.0,  0.0, 0.0,  1.0,  0.0,  0.0, -1.0}
////        ///                }
////        ///                );
////        ///            // load species and reaction names
////        ///            StructAnalysis.LoadSpeciesNames(new string[] { "A", "B", "C", "D", "E", "F" });
////        ///            StructAnalysis.LoadReactionNames(new string[] { "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9" });
////        ///
////        ///            // analyze with QR factorization and print test results
////        ///            Console.WriteLine(StructAnalysis.AnalyzeWithQR());
////        ///            Console.WriteLine(StructAnalysis.GetTestDetails());
////        ///            // ...
////        ///        }
////        ///
////        ///    }
////        /// </code>
////        /// </example>
////        /// </summary>
////        public static void LoadStoichiometryMatrix(double[][] oMatrix)
////        {
////            IntPtr pointer; int nRows; int nCols;
////            InteropUtil.MapMatrixToPointer(oMatrix, out pointer, out nRows, out nCols);
////            LibStructural_loadStoichiometryMatrix (pointer, nRows, nCols);
////            InteropUtil.FreePtrMatrix(pointer, nRows);
////        } // LoadStoichiometryMatrix(oMatrix)
////
////        /// <summary>
////        /// Load stoichiometry matrix with the given species names and reaction names. For an example see: <see cref="LoadStoichiometryMatrix(double[][])"/>.
////        /// </summary>
////        public static void LoadStoichiometryMatrix(double[][] oMatrix, string[] speciesNames, string[] reactionNames)
////        {
////            LoadStoichiometryMatrix(oMatrix, speciesNames, new double[speciesNames.Length], reactionNames);
////        } // LoadStoichiometryMatrix(oMatrix, speciesNames, reactionNames)
////
////        /// <summary>
////        /// Load stoichiometry matrix with the given species names, initial conditions and reaction names. For an example see: <see cref="LoadStoichiometryMatrix(double[][])"/>.
////        /// </summary>
////        public static void LoadStoichiometryMatrix(double[][] oMatrix, string[] speciesNames, double[] initialValues, string[] reactionNames)
////        {
////            LoadStoichiometryMatrix(oMatrix);
////            LoadSpeciesNames(speciesNames, initialValues);
////            LoadReactionNames(reactionNames);
////            AnalyzeWithQR();
////        } // LoadStoichiometryMatrix(oMatrix, speciesNames, initialValues)
////
////
////        /// <summary>
////        /// Print double matrix
////        /// </summary>
////        public static void PrintDoubleMatrix(double[][] oMatrix)
////        {
////            PrintDoubleMatrix(Console.Out, oMatrix);
////        } // PrintDoubleMatrix(oMatrix)
////
////        /// <summary>
////        /// Print int matrix
////        /// </summary>
////        public static void PrintIntMatrix(int[][] oMatrix)
////        {
////            PrintIntMatrix(Console.Out, oMatrix);
////        } // PrintDoubleMatrix(oMatrix)
////
////
////        /// <summary>
////        /// Print complex matrix
////        /// </summary>
////        public static void PrintComplexMatrix(Complex[][] oMatrix)
////        {
////            PrintComplexMatrix(Console.Out, oMatrix);
////        } // PrintComplexMatrix(oMatrix)
////        /// <summary>
////        /// Print double matrix
////        /// </summary>
////        public static void PrintDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix)
////        {
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oMatrix.Length)
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oWriter.Write(oMatrix[i][j]);
////                    if (j + 1 < oMatrix[i].Length) Console.Write(", ");
////                } // for (int)
////                oWriter.WriteLine();
////            } // for (int)
////
////            WriteMatlabForm(oWriter,oMatrix);
////
////        } // PrintDoubleMatrix(oWriter, oMatrix)
////
////        private static void WriteMatlabForm(System.IO.TextWriter oWriter, double[][] oMatrix)
////        {
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(Matlab: [] )");
////                return;
////            } // if (oMatrix.Length)
////            oWriter.Write("(Matlab: [");
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oWriter.Write(oMatrix[i][j]);
////                    if (j + 1 < oMatrix[i].Length) Console.Write(" ");
////                } // for (int)
////                if (i + 1 < oMatrix.Length)
////                oWriter.Write(" ;");
////            } // for (int)
////            oWriter.WriteLine("])");
////        }
////        /// <summary>
////        /// Print int matrix
////        /// </summary>
////        public static void PrintIntMatrix(System.IO.TextWriter oWriter, int[][] oMatrix)
////        {
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oMatrix.Length)
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oWriter.Write(oMatrix[i][j]);
////                    if (j + 1 < oMatrix[i].Length) Console.Write(", ");
////                } // for (int)
////                oWriter.WriteLine();
////            } // for (int)
////        } // PrintIntMatrix(oWriter, oMatrix)
////
////
////        /// <summary>
////        /// Print complex matrix
////        /// </summary>
////        public static void PrintComplexMatrix(System.IO.TextWriter oWriter, Complex[][] oMatrix)
////        {
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oMatrix.Length)
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oWriter.Write(oMatrix[i][j]);
////                    if (j + 1 < oMatrix[i].Length) Console.Write(", ");
////                } // for (int)
////                oWriter.WriteLine();
////            } // for (int)
////        }
////
////        /// <summary>
////        /// Print labeled double matrix
////        /// </summary>
////        public static void PrintLabledDoubleMatrix(double[][] oMatrix, IEnumerable<string> sRowLables, IEnumerable<string> sColumnLables)
////        {
////            PrintLabledDoubleMatrix(Console.Out, oMatrix, sRowLables, sColumnLables);
////        } // PrintLabledDoubleMatrix(oMatrix, sRowLables, sColumnLables)
////
////        /// <summary>
////        /// Print labeled double matrix
////        /// </summary>
////        public static void PrintLabledDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix, IEnumerable<string> sRowLablesIn, IEnumerable<string> sColumnLablesIn)
////        {
////
////            List<string> sRowLables = new List<string>(sRowLablesIn);
////            List<string> sColumnLables = new List<string>(sColumnLablesIn);
////
////
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oMatrix.Length)
////            oWriter.Write("\t");
////
////            foreach (string s in sColumnLables)
////                oWriter.Write(s + "\t");
////            oWriter.WriteLine();
////
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                if (sRowLables.Count > i)
////                oWriter.Write(sRowLables[i] + "\t");
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oWriter.Write(oMatrix[i][j].ToString("G3"));
////                    if (j + 1 < oMatrix[i].Length)
////                        oWriter.Write(",\t");
////                } // for (int)
////                oWriter.WriteLine();
////            } // for (int)
////
////        } // PrintLabledDoubleMatrix(oWriter, oMatrix, sRowLablesIn)
////        /// <summary>
////        /// Print sorted labeled double matrix
////        /// </summary>
////        public static void PrintSortedLabledDoubleMatrix(System.IO.TextWriter oWriter, double[][] oMatrix, IEnumerable<string> sRowLablesIn, IEnumerable<string> sColumnLablesIn)
////        {
////
////            List<string> sRowLables = new List<string>(sRowLablesIn);
////            List<string> sRowLablesSorted = new List<string>(sRowLables); sRowLablesSorted.Sort();
////            List<string> sColumnLables = new List<string>(sColumnLablesIn);
////            List<string> sColumnLablesSorted = new List<string>(sColumnLables); //sColumnLablesSorted.Sort();
////
////
////            if (oMatrix.Length == 0 || oMatrix[0].Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oMatrix.Length)
////            oWriter.Write("\t");
////
////            foreach (string s in sColumnLablesSorted)
////                oWriter.Write(s + "\t");
////            oWriter.WriteLine();
////
////
////            foreach (string sRow in sRowLablesSorted)
////            {
////                int i = sRowLables.IndexOf(sRow);
////                oWriter.Write(sRow + "\t");
////                for (int j = 0; j < sColumnLablesSorted.Count; j++)
////                {
////                    //int nIndex = sColumnLables.IndexOf(sColumnLablesSorted[j]);
////                    int nIndex = j;
////                    oWriter.Write(oMatrix[i][nIndex].ToString("G3"));
////                    if (j + 1 < sColumnLablesSorted.Count)
////                        oWriter.Write(",\t");
////                } // for (int)
////                oWriter.WriteLine();
////            } // foreach (sRow)
////
////        } // PrintSortedLabledDoubleMatrix(oWriter, oMatrix, sRowLablesIn)
////
////        /// <summary>
////        /// Print string array
////        /// </summary>
////        public static void PrintStringArray(System.IO.TextWriter oWriter, string[] oArray)
////        {
////            if (oArray.Length == 0)
////            {
////                oWriter.WriteLine("(empty)");
////                return;
////            } // if (oArray.Length)
////            for (int i = 0; i < oArray.Length; i++)
////            {
////                oWriter.Write(oArray[i]);
////                if (i + 1 < oArray.Length) oWriter.Write(", ");
////            } // for (int)
////        } // PrintStringArray(oWriter, oArray)
////
////        /// <summary>
////        /// Print string array
////        /// </summary>
////        public static void PrintStringArray(string[] oArray)
////        {
////            PrintStringArray(Console.Out, oArray);
////        } // PrintStringArray(oArray)
////
////
////
////        /// <summary>
////        /// Set tolerance
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_setTolerance")]
////        public static extern void SetTolerance(double dTolerance);
////        /// <summary>
////        /// Validate structural matrices
////        /// </summary>
////        public static bool[] ValidateStructuralMatrices(  )
////        {
////            IntPtr pointer; int nLength;
////            LibStructural_validateStructuralMatrices(out pointer, out nLength);
////            int[] buffer = new int[nLength];
////            Marshal.Copy(pointer, buffer, 0, nLength);
////            bool[] oResult = new bool[nLength];
////            for (int i = 0; i < nLength; i++)
////            {
////                if (buffer[i] == 1) oResult[i] = true;
////                else oResult[i] = false;
////            } // for (int)
////            FreeVector(pointer);
////            return oResult;
////        } // ValidateStructuralMatrices()
////
////
////        /// <summary>
////        /// Free matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_freeMatrix")]
////        internal static extern void FreeMatrix(IntPtr matrix, int numRows);
////
////
////        /// <summary>
////        /// Free vector
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_freeVector")]
////        internal static extern void FreeVector(IntPtr vector);
////
////
////        /// Return Type: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithFullyPivotedLU")]
////        internal static extern int LibStructural_analyzeWithFullyPivotedLU(out IntPtr result, out int nLength);
////
////
////        /// Return Type: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithFullyPivotedLUwithTests")]
////        internal static extern int LibStructural_analyzeWithFullyPivotedLUwithTests(out IntPtr result, out int nLength);
////
////        /// Return Type: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithLU")]
////        internal static extern int LibStructural_analyzeWithLU(out IntPtr result, out int nLength);
////
////
////        /// Return Type: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithLUandRunTests")]
////        internal static extern int LibStructural_analyzeWithLUandRunTests(out IntPtr result, out int nLength);
////
////
////        /// Return Type: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_analyzeWithQR")]
////        internal static extern int LibStructural_analyzeWithQR(out IntPtr result, out int nLength);
////
////
////
////        /// <summary>
////        /// get column reordered number matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getColumnReorderedNrMatrix")]
////        internal static extern int LibStructural_getColumnReorderedNrMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get column reordered number matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getColumnReorderedNrMatrixLabels")]
////        internal static extern int LibStructural_getColumnReorderedNrMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        //Returns algebraic expressions for conserved cycles
////        //LIB_EXTERN  int LibStructural_getConservedLaws(char** &outArray, int &outLength);
////        /// <summary>
////        /// get conserved entities
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getConservedLaws")]
////        internal static extern int LibStructural_getConservedLaws(out IntPtr outArray, out int outLength);
////
////
////        //Returns values for conserved cycles using Initial conditions
////        //LIB_EXTERN int LibStructural_getConservedSums(double* &outArray, int &outLength);
////        /// <summary>
////        /// get conserved sums
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getConservedSums")]
////        internal static extern int LibStructural_getConservedSums(out IntPtr outArray, out int outLength);
////
////        /// <summary>
////        /// get dependent reaction ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getDependentReactionIds")]
////        internal static extern int LibStructural_getDependentReactionIds(out IntPtr outArray, out int outLength);
////
////        /// <summary>
////        /// get dependent species ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getDependentSpeciesIds")]
////        internal static extern int LibStructural_getDependentSpeciesIds(out IntPtr outArray, out int outLength);
////
////
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthReorderedSpeciesId")]
////        //internal static extern int LibStructural_getNthReorderedSpeciesId(int n, out IntPtr result, out int nLength);
////        //public static string GetNthReorderedSpeciesId(int n)
////        //{
////        //    IntPtr pointer; int nLength;
////        //    LibStructural_getNthReorderedSpeciesId(n, out pointer, out nLength);
////        //    return InteropUtil.GetStringFromPtr(pointer, nLength);
////        //}
////
////
////        ///// Return Type: char*
////        /////n: int
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthIndependentSpeciesId")]
////        //internal static extern int LibStructural_getNthIndependentSpeciesId(int n, out IntPtr result, out int nLength);
////        //public static string GetNthIndependentSpeciesId(int n)
////        //{
////        //    IntPtr pointer; int nLength;
////        //    LibStructural_getNthIndependentSpeciesId(n, out pointer, out nLength);
////        //    return InteropUtil.GetStringFromPtr(pointer, nLength);
////        //}
////
////        ///// Return Type: char*
////        /////n: int
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthDependentSpeciesId")]
////        //internal static extern int LibStructural_getNthDependentSpeciesId(int n, out IntPtr result, out int nLength);
////        //public static string GetNthDependentSpeciesId(int n)
////        //{
////        //    IntPtr pointer; int nLength;
////        //    LibStructural_getNthDependentSpeciesId(n, out pointer, out nLength);
////        //    return InteropUtil.GetStringFromPtr(pointer, nLength);
////        //}
////
////        ///// Return Type: char*
////        /////n: int
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthReactionId")]
////        //internal static extern int LibStructural_getNthReactionId(int n, out IntPtr result, out int nLength);
////        //public static string GetNthReactionId(int n)
////        //{
////        //    IntPtr pointer; int nLength;
////        //    LibStructural_getNthReactionId(n, out pointer, out nLength);
////        //    return InteropUtil.GetStringFromPtr(pointer, nLength);
////        //}
////
////        /// <summary>
////        /// get Gamma matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrix")]
////        internal static extern int LibStructural_getGammaMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get Gamma matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrixLabels")]
////        internal static extern int LibStructural_getGammaMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        /// <summary>
////        /// get independent reaction ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getIndependentReactionIds")]
////        internal static extern int LibStructural_getIndependentReactionIds(out IntPtr outArray, out int outLength);
////
////        /// <summary>
////        /// get independent species ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getIndependentSpeciesIds")]
////        internal static extern int LibStructural_getIndependentSpeciesIds(out IntPtr outArray, out int outLength);
////
////
////        /// Return Type: void
////        ///dTolerance: double
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getInitialConditions")]
////        internal static extern int LibStructural_getInitialConditions(out IntPtr outVariableNames, out IntPtr outValues, out int outLength);
////
////
////        /// <summary>
////        /// get K0 matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getK0Matrix")]
////        internal static extern int LibStructural_getK0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get K0 matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getK0MatrixLabels")]
////        internal static extern int LibStructural_getK0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////
////        /// <summary>
////        /// get K matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getKMatrix")]
////        internal static extern int LibStructural_getKMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get K matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getKMatrixLabels")]
////        internal static extern int LibStructural_getKMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////
////        /// Return Type: int
////        ///outMatrix: double**
////        ///outRows: int*
////        ///outCols: int*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getL0Matrix")]
////        internal static extern int LibStructural_getL0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get L0 matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getL0MatrixLabels")]
////        internal static extern int LibStructural_getL0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////
////        /// <summary>
////        /// get L matrix (link matrix)
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getLinkMatrix")]
////        internal static extern int LibStructural_getLinkMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get L matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getLinkMatrixLabels")]
////        internal extern static int LibStructural_getLMatrixLabels(out System.IntPtr outRowLabels,
////            out int outRowCount,
////            out IntPtr outColLabels,
////            out int outColCount);
////
////        /// <summary>
////        /// get model name
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getModelName")]
////        internal static extern int LibStructural_getModelName(out IntPtr result, out int nLength);
////
////        /// Return Type: int
////        ///outMatrix: double**
////        ///outRows: int*
////        ///outCols: int*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getN0Matrix")]
////        internal static extern int LibStructural_getN0Matrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get N0 matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getN0MatrixLabels")]
////        internal static extern int LibStructural_getN0MatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNDCMatrix")]
////        internal static extern int LibStructural_getNDCMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get NDC matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNDCMatrixLabels")]
////        internal static extern int LibStructural_getNDCMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNICMatrix")]
////        internal static extern int LibStructural_getNICMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get N0 matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNICMatrixLabels")]
////        internal static extern int LibStructural_getNICMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////
////
////        /// <summary>
////        /// get number matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNrMatrix")]
////        internal static extern int LibStructural_getNrMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get number matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNrMatrixLabels")]
////        internal static extern int LibStructural_getNrMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        //Returns the list of Reactions
////        //LIB_EXTERN  int LibStructural_getReactionIds(char** &outArray, int &outLength);
////        /// <summary>
////        /// get reaction ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReactionIds")]
////        internal static extern int LibStructural_getReactionIds(out IntPtr outArray, out int outLength);
////
////        /// <summary>
////        /// get reordered reaction ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedReactionIds")]
////        internal static extern int LibStructural_getReorderedReactionIds(out IntPtr outArray, out int outLength);
////
////
////        //Returns the reordered list of species
////        //LIB_EXTERN  int LibStructural_getReorderedSpeciesIds(char** &outArray, int &outLength);
////        /// <summary>
////        /// get reordered species ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedSpeciesIds")]
////        internal static extern int LibStructural_getReorderedSpeciesIds(out IntPtr outArray, out int outLength);
////
////
////        /// <summary>
////        /// get reordered stoichiometry matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedStoichiometryMatrix")]
////        internal static extern int LibStructural_getReorderedStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get reordered stoichiometry matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getReorderedStoichiometryMatrixLabels")]
////        internal static extern int LibStructural_getReorderedStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        /// <summary>
////        /// get fully reordered stoichiometry matrix
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getFullyReorderedStoichiometryMatrix", CallingConvention=CallingConvention.Cdecl)]
////        internal static extern int LibStructural_getFullyReorderedStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get fully reordered stoichiometry matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getFullyReorderedStoichiometryMatrixLabels")]
////        internal static extern int LibStructural_getFullyReorderedStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////        /// <summary>
////        /// get species ids
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getSpeciesIds")]
////        internal static extern int LibStructural_getSpeciesIds(out IntPtr outArray, out int outLength);
////
////
////        ///// Return Type: double
////        /////n: int
////        //[DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getNthConservedSum")]
////        //public static extern double GetNthConservedSum(int n);
////
////
////        /// Return Type: int
////        ///outMatrix: double**
////        ///outRows: int*
////        ///outCols: int*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getStoichiometryMatrix")]
////        internal static extern int LibStructural_getStoichiometryMatrix(out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////        /// <summary>
////        /// get stoichiometry matrix labels
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getStoichiometryMatrixLabels")]
////        internal static extern int LibStructural_getStoichiometryMatrixLabels(out System.IntPtr outRowLabels, out int outRowCount, out IntPtr outColLabels, out int outColCount);
////
////
////        /// <summary>
////        /// get test details
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getTestDetails")]
////        internal static extern int LibStructural_getTestDetails(out IntPtr result, out int nLength);
////
////
////        /// <summary>
////        /// load reaction names
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadReactionNames", CharSet=CharSet.Ansi)]
////        internal static extern void LibStructural_loadReactionNames(IntPtr reactionNames, int nLength);
////
////        /// Return Type: char*
////        ///sSBML: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBML", CharSet=CharSet.Ansi)]
////        internal static extern int LibStructural_loadSBML(string sSBML, out IntPtr result, out int nLength);
////
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBMLFromFile", CharSet = CharSet.Ansi)]
////        internal static extern int LibStructural_loadSBMLFromFile(string sFileName, out IntPtr result, out int nLength);
////
////
////        /// Return Type: char*
////        ///sSBML: char*
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSBMLwithTests", CharSet = CharSet.Ansi)]
////        internal static extern int LibStructural_loadSBMLwithTests(string sSbml, out IntPtr result, out int nLength);
////
////
////        /// <summary>
////        /// load species names and initial values
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadSpecies",CharSet=CharSet.Ansi)]
////        internal static extern void LibStructural_loadSpecies(IntPtr speciesNames, IntPtr speciesValues, int nLength);
////
////
////        /// <summary>
////        /// load a new stoichiometry matrix and reset current loaded model
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_loadStoichiometryMatrix")]
////        internal static extern int LibStructural_loadStoichiometryMatrix (IntPtr oMatrix, int nRows, int nCols);
////
////
////
////        /// <summary>
////        /// validate structural matrices
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_validateStructuralMatrices")]
////        internal static extern int LibStructural_validateStructuralMatrices(out System.IntPtr outResults, out int outLength);
////
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_getGammaMatrixGJ")]
////        private static extern int LibStructural_getGammaMatrixGJ(System.IntPtr inMatrix, int numRows, int numCols,
////            out System.IntPtr outMatrix, out int outRows, out int outCols);
////
////
////        /// <summary>
////        /// double** inMatrix, int numRows, int numCols,
////        /// const char** inRowLabels,  const char** inColLabels,
////        /// double** *outMatrix, int *outRows, int *outCols,
////        /// char** *outRowLabels, int *outRowCount, char** *outColLabels, int *outColCount
////        /// </summary>
////        [DllImportAttribute(LIBRARY_FILE, EntryPoint = "LibStructural_findPositiveGammaMatrix")]
////        private static extern int LibStructural_findPositiveGammaMatrix(
////            System.IntPtr inMatrix, int numRows, int numCols,
////            IntPtr inRowLabels,
////            out System.IntPtr outMatrix, out int outRows, out int outCols,
////            out System.IntPtr outRowLabels, out int outRowCount);
////
////
////    } // class Conservation
////} // namespace libstructural

