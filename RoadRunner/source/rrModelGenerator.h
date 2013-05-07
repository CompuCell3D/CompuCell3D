#ifndef rrModelGeneratorH
#define rrModelGeneratorH
#include <string>
#include <vector>
#include <list>
#include "rrObject.h"
#include "rrStringList.h"
#include "rrSymbolList.h"
#include "rrCodeBuilder.h"
#include "rrNOMSupport.h"
#include "rrScanner.h"
#include "rr-libstruct/lsMatrix.h"
#include "rr-libstruct/lsLibStructural.h"

using std::string;
using std::vector;
using std::list;
using namespace ls;
namespace rr
{
class RoadRunner;

class RR_DECLSPEC ModelGenerator : public rrObject
{
    public:
        bool								mComputeAndAssignConsevationLaws;
        const string                        mDoubleFormat;
        const string                        mFixAmountCompartments;
        vector<int>                         mLocalParameterDimensions;
        string                              mModelName;
        int                                 mNumBoundarySpecies;
        int                                 mNumCompartments;
        int                                 mNumDependentSpecies;
        int                                 mNumEvents;
        int                                 mNumFloatingSpecies;
        int                                 mNumGlobalParameters;
        int                                 mNumIndependentSpecies;
        int                                 mNumReactions;

        int                                 mTotalLocalParmeters;
        StringList                          mFunctionNames;
        StringList                          mFunctionParameters;
        StringList                          mDependentSpeciesList;
        StringList                          mIndependentSpeciesList;
        int                                 mNumModifiableSpeciesReferences;
		LibStructural&                      mLibStruct;                          //Refernce to libstruct library
        NOMSupport&                         mNOM;                                //Object that provide some wrappers and new "NOM" functions.
        IntStringHashTable                  mMapRateRule;
        SymbolList                          mBoundarySpeciesList;
        SymbolList                          mCompartmentList;
        SymbolList                          mConservationList;
        SymbolList                          mFloatingSpeciesAmountsList;
        SymbolList                          mFloatingSpeciesConcentrationList;
        SymbolList                          mGlobalParameterList;
        vector<SymbolList>                  mLocalParameterList;
        SymbolList                          mReactionList;
        StringList                          mWarnings;

        //Pure Virtual functions... =====================================
        virtual string                      convertUserFunctionExpression(const string& equation) = 0;
        virtual void                        substituteEquation(const string& reactionName, Scanner& s, CodeBuilder& sb) = 0;
        virtual void                        substituteWords(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& sb) = 0;
        virtual void                        substituteToken(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& sb) = 0;
        virtual string                      findSymbol(const string& varName) = 0;
        virtual int                         readFloatingSpecies() = 0;
        virtual int                         readBoundarySpecies() = 0;
        virtual void                        writeOutSymbolTables(CodeBuilder& sb) = 0;
        virtual void                        writeComputeAllRatesOfChange(CodeBuilder& sb, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0) = 0;
        virtual void                        writeComputeConservedTotals(CodeBuilder& sb, const int& numFloatingSpecies, const int& numDependentSpecies) = 0;
        virtual void                        writeUpdateDependentSpecies(CodeBuilder& sb, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0) = 0;
        virtual void                        writeUserDefinedFunctions(CodeBuilder& sb) = 0;
        virtual void                        writeResetEvents(CodeBuilder& sb, const int& numEvents) = 0;
        virtual void                        writeSetConcentration(CodeBuilder& sb) = 0;
        virtual void                        writeGetConcentration(CodeBuilder& sb) = 0;
        virtual void                        writeConvertToAmounts(CodeBuilder& sb) = 0;
        virtual void                        writeConvertToConcentrations(CodeBuilder& sb) = 0;
        virtual void                        writeProperties(CodeBuilder& sb) = 0;
        virtual void                        writeAccessors(CodeBuilder& sb) = 0;
        virtual void                        writeOutVariables(CodeBuilder& sb) = 0;
        virtual void                        writeClassHeader(CodeBuilder& sb) = 0;
        virtual void                        writeTestConstraints(CodeBuilder& sb) = 0;
        virtual void                        writeEvalInitialAssignments(CodeBuilder& sb, const int& numReactions) = 0;
        virtual int                         writeComputeRules(CodeBuilder& sb, const int& numReactions) = 0;
        virtual void                        writeComputeReactionRates(CodeBuilder& sb, const int& numReactions) = 0;
        virtual void                        writeEvalEvents(CodeBuilder& sb, const int& numEvents, const int& numFloatingSpecies) = 0;
        virtual void                        writeEvalModel(CodeBuilder& sb, const int& numReactions, const int& numIndependentSpecies, const int& numFloatingSpecies, const int& numOfRules) = 0;
        virtual void                        writeEventAssignments(CodeBuilder& sb, const int& numReactions, const int& numEvents) = 0;
        virtual void                        writeSetParameterValues(CodeBuilder& sb, const int& numReactions) = 0;
        virtual void                        writeSetCompartmentVolumes(CodeBuilder& sb) = 0;
        virtual void                        writeSetBoundaryConditions(CodeBuilder& sb) = 0;
        virtual void                        writeSetInitialConditions(CodeBuilder& sb, const int& numFloatingSpecies) = 0;
        virtual string                      convertCompartmentToC(const string& compartmentName) = 0;
        virtual string                      convertSpeciesToBc(const string& speciesName) = 0;
        virtual string                      convertSpeciesToY(const string& speciesName) = 0;
        virtual string                      convertSymbolToC(const string& compartmentName) = 0;
        virtual string                      convertSymbolToGP(const string& parameterName) = 0;

		//Non virtuals..
        string                              substituteTerms(const int& numReactions, const string& reactionName, const string& equation);
        ASTNode*                            cleanEquation(ASTNode* ast);
        string                              cleanEquation(const string& equation);
        string                              substituteTerms(const string& reactionName, const string& inputEquation, bool bFixAmounts);
        ls::DoubleMatrix*               	initializeL0(int& nrRows, int& nrCols);
        bool                                expressionContainsSymbol(ASTNode* ast, const string& symbol);
        bool                                expressionContainsSymbol(const string& expression, const string& symbol);
        Symbol*                             getSpecies(const string& id);
        int                                 readGlobalParameters();
        void                                readLocalParameters(const int& numReactions,  vector<int>& localParameterDimensions, int& totalLocalParmeters);
        int                                 readCompartments();
        int                                 readModifiableSpeciesReferences();
        SymbolList                          mModifiableSpeciesReferenceList;

    public:
                                            ModelGenerator(LibStructural& ls, NOMSupport& nom);
        virtual                             ~ModelGenerator();
        void                                reset();
        int                                 getNumberOfReactions();
        int                                 numAdditionalRates();        //this variable is the size of moMapRateRule

        StringList                          getCompartmentList();
        StringList                          getConservationList();

        StringList                          getGlobalParameterList();
        StringList                          getLocalParameterList(int reactionId);

        StringList                          getReactionIds();
        SymbolList&                         getReactionListReference();

        StringList                          getFloatingSpeciesConcentrationList();    //Just returns the Ids...!
        SymbolList&                         getFloatingSpeciesConcentrationListReference();

        StringList                          getBoundarySpeciesList();
        SymbolList&                         getBoundarySpeciesListReference();
        SymbolList&                         getGlobalParameterListReference();
        SymbolList&                         getConservationListReference();
        string                              writeDouble(const double& value, const string& format = "%G");

        // Generates the Model Code from theSBML string
        virtual string                      generateModelCode(const string& sbmlStr, const bool& _computeAndAssignConsevationLaws) = 0;    //Any decendant need to implement at least this one
        virtual bool                     	saveSourceCodeToFolder(const string& folder, const string& codeBaseName);
//        void                                SetXMLModelFileName(const string& name);
};
}

#endif



////using System;
////using System.Collections;
////using System.Collections.Generic;
////using System.Collections.Specialized;
////using System.Globalization;
////using System.IO;
////using System.Text;
////using LibRoadRunner.Scanner;
////using LibRoadRunner.Util;
////using libsbmlcs;
////using libstructural;
////using SBMLSupport;
////using SBW;
////
/////************************************************************************
////  * Filename    : modelGenerator.cs
////  * Description : CSharp Simulator
////  * Author(s)   : Herbert Sauro (Based on code from Ravishankar R. Vallabhajosyula)
////  * Organization: Keck Graduate Institute (KGI)
////  * Created     :
////  * Revision    : $Id:
////  * Source      : $Source:
////  *
////  * Copyright 2005 Keck Graduate Institute
////  *
////  * This library is free software; you can redistribute it and/or modify it
////  * under the terms of the GNU Lesser General Public License as published
////  * by the Free Software Foundation; either version 2.1 of the License, or
////  * any later version.
////  *
////  * This library is distributed in the hope that it will be useful, but
////  * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
////  * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
////  * documentation provided hereunder is on an "as is" basis, and the
////  * California Institute of Technology and Japan Science and Technology
////  * Corporation have no obligations to provide maintenance, support,
////  * updates, enhancements or modifications.  In no event shall the
////  * California Institute of Technology or the Japan Science and Technology
////  * Corporation be liable to any party for direct, indirect, special,
////  * incidental or consequential damages, including lost profits, arising
////  * out of the use of this software and its documentation, even if the
////  * California Institute of Technology and/or Japan Science and Technology
////  * Corporation have been advised of the possibility of such damage.  See
////  * the GNU Lesser General Public License for more details.
////  *
////  * You should have received a copy of the GNU Lesser General Public License
////  * along with this library; if not, write to the Free Software Foundation,
////  * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
////  *
////  * The original code contained here was initially developed by:
////  *
////  *     Herbert M Sauro
////  *     The Systems Biology Markup Language Development Group
////  *     Keck Graduate Institute of Applied Life Sciences
////  *     535 Watson Drive
////  *     Claremont, CA, 91711, USA
////  *
////  *     http://www.cds.caltech.edu/erato
////  *     http://www.kgi.edu
////  *     mailto:sysbio-team@caltech.edu
////  *
////  * Contributor(s):
////  *     Ravishankar Rao Vallabhajosyula
////  *     Frank Bergmann
////  ***************************************************************************/
////
////namespace LibRoadRunner
////{
////    /// <summary>
////    /// Summary description for modelGenerator.
////    /// </summary>
////    public class ModelGenerator
////    {
////        private const string STR_DoubleFormat = "G"; //"G17";
////        private const string STR_FixAmountCompartments = "*";
////
////        private static ModelGenerator _instance;
////
////        private static NumberFormatInfo oInfo = new CultureInfo("en-US").NumberFormat;
////
////
////        private int[] _LocalParameterDimensions;
////        private string _ModelName;
////        private int _NumBoundarySpecies;
////        private int _NumCompartments;
////        private int _NumDependentSpecies;
////        private int _NumEvents;
////        private int _NumFloatingSpecies;
////        private int _NumGlobalParameters;
////        private int _NumIndependentSpecies;
////        private int _NumReactions;
////        private int _TotalLocalParmeters;
////        private ArrayList _functionNames;
////        private StringCollection _functionParameters;
////        private Hashtable _oMapRateRule = new Hashtable();
////        public SymbolList boundarySpeciesList;
////        public SymbolList compartmentList;
////        public SymbolList conservationList;
////        private string[] dependentSpeciesList;
////        public SymbolList floatingSpeciesAmountsList;
////        public SymbolList floatingSpeciesConcentrationList;
////        public SymbolList globalParameterList;
////        private string[] independentSpeciesList;
////        public SymbolList[] localParameterList;
////        public SymbolList reactionList;
////        private int _NumModifiableSpeciesReferences;
////
////        private ModelGenerator()
////        {
////            oInfo = new CultureInfo("en-US").NumberFormat;
////            oInfo.NumberDecimalSeparator = ".";
////        }
////
////        public static ModelGenerator Instance
////        {
////            get
////            {
////                if (_instance == null)
////                    _instance = new ModelGenerator();
////                return _instance;
////            }
////        }
////
////        public SymbolList ModifiableSpeciesReferenceList { get; set; }
////
////        public List<string> Warnings { get; set; }
////
////        public int NumAdditionalRates
////        {
////            get { return _oMapRateRule.Count; }
////        }
////
////
////        public int getNumberOfReactions()
////        {
////            return reactionList.Count;
////        }
////
////        private string convertSpeciesToY(string speciesName)
////        {
////            int index;
////            if (floatingSpeciesConcentrationList.find(speciesName, out index))
////            {
////                return "_y[" + index + "]";
////            }
////            throw new SBWApplicationException("Internal Error: Unable to locate species: " + speciesName);
////        }
////
////        private string convertSpeciesToBc(string speciesName)
////        {
////            int index;
////            if (boundarySpeciesList.find(speciesName, out index))
////                return "_bc[" + index + "]";
////            throw new SBWApplicationException("Internal Error: Unable to locate species: " + speciesName);
////        }
////
////        private string convertCompartmentToC(string compartmentName)
////        {
////            int index;
////            if (compartmentList.find(compartmentName, out index))
////                return "_c[" + index + "]";
////            throw new SBWApplicationException("Internal Error: Unable to locate compartment: " + compartmentName);
////        }
////
////        private string convertSymbolToGP(string parameterName)
////        {
////            int index;
////            if (globalParameterList.find(parameterName, out index))
////            {
////                return "_gp[" + index + "]";
////            }
////            throw new SBWApplicationException("Internal Error: Unable to locate parameter: " + parameterName);
////        }
////
////        private string convertSymbolToC(string compartmentName)
////        {
////            int index;
////            if (compartmentList.find(compartmentName, out index))
////            {
////                return "_c[" + index + "]";
////            }
////            throw new SBWApplicationException("Internal Error: Unable to locate compartment: " + compartmentName);
////        }
////
////        public ArrayList getCompartmentList()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < compartmentList.Count; i++)
////                tmp.Add(compartmentList[i].name);
////            return tmp;
////        }
////
////        public ArrayList getFloatingSpeciesConcentrationList()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////                tmp.Add(floatingSpeciesConcentrationList[i].name);
////            return tmp;
////        }
////
////        public ArrayList getConservationList()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < conservationList.Count; i++)
////                tmp.Add(conservationList[i].name);
////            return tmp;
////        }
////
////
////        public ArrayList getBoundarySpeciesList()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < boundarySpeciesList.Count; i++)
////                tmp.Add(boundarySpeciesList[i].name);
////            return tmp;
////        }
////
////        public ArrayList getGlobalParameterList()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < globalParameterList.Count; i++)
////                tmp.Add(globalParameterList[i].name);
////
////            for (int i = 0; i < conservationList.Count; i++)
////                tmp.Add(conservationList[i].name);
////
////            return tmp;
////        }
////
////        public ArrayList getLocalParameterList(int reactionId)
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < localParameterList[reactionId].Count; i++)
////                tmp.Add(localParameterList[reactionId][i].name);
////            return tmp;
////        }
////
////        public ArrayList getReactionNames()
////        {
////            var tmp = new ArrayList();
////            for (int i = 0; i < reactionList.Count; i++)
////                tmp.Add(reactionList[i].name);
////            return tmp;
////        }
////
////        private string convertUserFunctionExpression(string equation)
////        {
////            var s = new Scanner.Scanner();
////            Stream ss = new MemoryStream(Encoding.Default.GetBytes(equation));
////            s.stream = ss;
////            s.startScanner();
////            s.nextToken();
////            var sb = new CodeBuilder();
////
////            try
////            {
////                while (s.token != CodeTypes.tEndOfStreamToken)
////                {
////                    switch (s.token)
////                    {
////                        case CodeTypes.tWordToken:
////
////                            switch (s.tokenString)
////                            {
////                                case "pow":
////                                    sb.Append("Math.Pow");
////                                    break;
////                                case "sqrt":
////                                    sb.Append("Math.Sqrt");
////                                    break;
////                                case "log":
////                                    sb.Append("supportFunctions._log");
////                                    break;
////                                case "log10":
////                                    sb.Append("Math.Log10");
////                                    break;
////                                case "floor":
////                                    sb.Append("Math.Floor");
////                                    break;
////                                case "ceil":
////                                    sb.Append("Math.Ceiling");
////                                    break;
////                                case "factorial":
////                                    sb.Append("supportFunctions._factorial");
////                                    break;
////                                case "exp":
////                                    sb.Append("Math.Exp");
////                                    break;
////                                case "sin":
////                                    sb.Append("Math.Sin");
////                                    break;
////                                case "cos":
////                                    sb.Append("Math.Cos");
////                                    break;
////                                case "tan":
////                                    sb.Append("Math.Tan");
////                                    break;
////                                case "abs":
////                                    sb.Append("Math.Abs");
////                                    break;
////                                case "asin":
////                                    sb.Append("Math.Asin");
////                                    break;
////                                case "acos":
////                                    sb.Append("Math.Acos");
////                                    break;
////                                case "atan":
////                                    sb.Append("Math.Atan");
////                                    break;
////                                case "sec":
////                                    sb.Append("MathKGI.Sec");
////                                    break;
////                                case "csc":
////                                    sb.Append("MathKGI.Csc");
////                                    break;
////                                case "cot":
////                                    sb.Append("MathKGI.Cot");
////                                    break;
////                                case "arcsec":
////                                    sb.Append("MathKGI.Asec");
////                                    break;
////                                case "arccsc":
////                                    sb.Append("MathKGI.Acsc");
////                                    break;
////                                case "arccot":
////                                    sb.Append("MathKGI.Acot");
////                                    break;
////                                case "sinh":
////                                    sb.Append("Math.Sinh");
////                                    break;
////                                case "cosh":
////                                    sb.Append("Math.Cosh");
////                                    break;
////                                case "tanh":
////                                    sb.Append("Math.Tanh");
////                                    break;
////                                case "arcsinh":
////                                    sb.Append("MathKGI.Asinh");
////                                    break;
////                                case "arccosh":
////                                    sb.Append("MathKGI.Acosh");
////                                    break;
////                                case "arctanh":
////                                    sb.Append("MathKGI.Atanh");
////                                    break;
////                                case "sech":
////                                    sb.Append("MathKGI.Sech");
////                                    break;
////                                case "csch":
////                                    sb.Append("MathKGI.Csch");
////                                    break;
////                                case "coth":
////                                    sb.Append("MathKGI.Coth");
////                                    break;
////                                case "arcsech":
////                                    sb.Append("MathKGI.Asech");
////                                    break;
////                                case "arccsch":
////                                    sb.Append("MathKGI.Acsch");
////                                    break;
////                                case "arccoth":
////                                    sb.Append("MathKGI.Acoth");
////                                    break;
////                                case "pi":
////                                    sb.Append("Math.PI");
////                                    break;
////                                case "exponentiale":
////                                    sb.Append("Math.E");
////                                    break;
////                                case "avogadro":
////                                    sb.Append("6.02214179e23");
////                                    break;
////                                case "true":
////                                    //sb.Append("true");
////                                    sb.Append("1.0");
////                                    break;
////                                case "false":
////                                    //sb.Append("false");
////                                    sb.Append("0.0");
////                                    break;
////                                case "gt":
////                                    sb.Append("supportFunctions._gt");
////                                    break;
////                                case "lt":
////                                    sb.Append("supportFunctions._lt");
////                                    break;
////                                case "eq":
////                                    sb.Append("supportFunctions._eq");
////                                    break;
////                                case "neq":
////                                    sb.Append("supportFunctions._neq");
////                                    break;
////                                case "geq":
////                                    sb.Append("supportFunctions._geq");
////                                    break;
////                                case "leq":
////                                    sb.Append("supportFunctions._leq");
////                                    break;
////                                case "and":
////                                    sb.Append("supportFunction._and");
////                                    break;
////                                case "or":
////                                    sb.Append("supportFunction._or");
////                                    break;
////                                case "not":
////                                    sb.Append("supportFunction._not");
////                                    break;
////                                case "xor":
////                                    sb.Append("supportFunction._xor");
////                                    break;
////                                case "root":
////                                    sb.Append("supportFunctions._root");
////                                    break;
////                                case "piecewise":
////                                    sb.Append("supportFunctions._piecewise");
////                                    break;
////                                default:
////                                    //if (!_functionParameters.Contains(s.tokenString))
////                                    //    throw new ArgumentException("Token '" + s.tokenString + "' not recognized.");
////                                    //else
////                                    sb.Append(s.tokenString);
////                                    break;
////                            }
////                            break;
////
////                        case CodeTypes.tDoubleToken:
////                            sb.Append(WriteDouble(s.tokenDouble));
////                            break;
////                        case CodeTypes.tIntToken:
////                            sb.Append(s.tokenInteger.ToString());
////                            break;
////                        case CodeTypes.tPlusToken:
////                            sb.Append("+");
////                            break;
////                        case CodeTypes.tMinusToken:
////                            sb.Append("-");
////                            break;
////                        case CodeTypes.tDivToken:
////                            sb.Append("/");
////                            break;
////                        case CodeTypes.tMultToken:
////                            sb.Append(STR_FixAmountCompartments);
////                            break;
////                        case CodeTypes.tPowerToken:
////                            sb.Append("^");
////                            break;
////                        case CodeTypes.tLParenToken:
////                            sb.Append("(");
////                            break;
////                        case CodeTypes.tRParenToken:
////                            sb.Append(")");
////                            break;
////                        case CodeTypes.tCommaToken:
////                            sb.Append(",");
////                            break;
////                        case CodeTypes.tEqualsToken:
////                            sb.Append(" = ");
////                            break;
////                        case CodeTypes.tTimeWord1:
////                            sb.Append("time");
////                            break;
////                        case CodeTypes.tTimeWord2:
////                            sb.Append("time");
////                            break;
////                        case CodeTypes.tTimeWord3:
////                            sb.Append("time");
////                            break;
////                        case CodeTypes.tAndToken:
////                            sb.Append("supportFunctions._and");
////                            break;
////                        case CodeTypes.tOrToken:
////                            sb.Append("supportFunctions._or");
////                            break;
////                        case CodeTypes.tNotToken:
////                            sb.Append("supportFunctions._not");
////                            break;
////                        case CodeTypes.tLessThanToken:
////                            sb.Append("supportFunctions._lt");
////                            break;
////                        case CodeTypes.tLessThanOrEqualToken:
////                            sb.Append("supportFunctions._leq");
////                            break;
////                        case CodeTypes.tMoreThanOrEqualToken:
////                            sb.Append("supportFunctions._geq");
////                            break;
////                        case CodeTypes.tMoreThanToken:
////                            sb.Append("supportFunctions._gt");
////                            break;
////                        case CodeTypes.tXorToken:
////                            sb.Append("supportFunctions._xor");
////                            break;
////                        default:
////                            var ae =
////                                new SBWApplicationException(
////                                    "Unknown token in convertUserFunctionExpression: " + s.tokenToString(s.token),
////                                    "Exception raised in Module:roadRunner, Method:convertUserFunctionExpression");
////                            throw ae;
////                    }
////                    s.nextToken();
////                }
////            }
////            catch (SBWApplicationException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException(e.Message);
////            }
////            return sb.ToString();
////        }
////
////
////        private string substituteTerms(int numReactions, string reactionName, string equation)
////        {
////            return substituteTerms(reactionName, equation, false);
////        }
////
////        private void SubstituteEquation(string reactionName, LibRoadRunner.Scanner.Scanner s, StringBuilder sb)
////        {
////
////            switch (s.tokenString)
////            {
////                case "pow":
////                    sb.Append("Math.Pow");
////                    break;
////                case "sqrt":
////                    sb.Append("Math.Sqrt");
////                    break;
////                case "log":
////                    sb.Append("supportFunctions._log");
////                    break;
////                case "floor":
////                    sb.Append("Math.Floor");
////                    break;
////                case "ceil":
////                    sb.Append("Math.Ceiling");
////                    break;
////                case "factorial":
////                    sb.Append("supportFunctions._factorial");
////                    break;
////                case "log10":
////                    sb.Append("Math.Log10");
////                    break;
////                case "exp":
////                    sb.Append("Math.Exp");
////                    break;
////                case "abs":
////                    sb.Append("Math.Abs");
////                    break;
////                case "sin":
////                    sb.Append("Math.Sin");
////                    break;
////                case "cos":
////                    sb.Append("Math.Cos");
////                    break;
////                case "tan":
////                    sb.Append("Math.Tan");
////                    break;
////                case "asin":
////                    sb.Append("Math.Asin");
////                    break;
////                case "acos":
////                    sb.Append("Math.Acos");
////                    break;
////                case "atan":
////                    sb.Append("Math.Atan");
////                    break;
////                case "sec":
////                    sb.Append("MathKGI.Sec");
////                    break;
////                case "csc":
////                    sb.Append("MathKGI.Csc");
////                    break;
////                case "cot":
////                    sb.Append("MathKGI.Cot");
////                    break;
////                case "arcsec":
////                    sb.Append("MathKGI.Asec");
////                    break;
////                case "arccsc":
////                    sb.Append("MathKGI.Acsc");
////                    break;
////                case "arccot":
////                    sb.Append("MathKGI.Acot");
////                    break;
////                case "sinh":
////                    sb.Append("Math.Sinh");
////                    break;
////                case "cosh":
////                    sb.Append("Math.Cosh");
////                    break;
////                case "tanh":
////                    sb.Append("Math.Tanh");
////                    break;
////                case "arcsinh":
////                    sb.Append("MathKGI.Asinh");
////                    break;
////                case "arccosh":
////                    sb.Append("MathKGI.Acosh");
////                    break;
////                case "arctanh":
////                    sb.Append("MathKGI.Atanh");
////                    break;
////                case "sech":
////                    sb.Append("MathKGI.Sech");
////                    break;
////                case "csch":
////                    sb.Append("MathKGI.Csch");
////                    break;
////                case "coth":
////                    sb.Append("MathKGI.Coth");
////                    break;
////                case "arcsech":
////                    sb.Append("MathKGI.Asech");
////                    break;
////                case "arccsch":
////                    sb.Append("MathKGI.Acsch");
////                    break;
////                case "arccoth":
////                    sb.Append("MathKGI.Acoth");
////                    break;
////                case "pi":
////                    sb.Append("Math.PI");
////                    break;
////                case "avogadro":
////                    sb.Append("6.02214179e23");
////                    break;
////                case "exponentiale":
////                    sb.Append("Math.E");
////                    break;
////                case "true":
////                    //sb.Append("true");
////                    sb.Append("1.0");
////                    break;
////                case "false":
////                    //sb.Append("false");
////                    sb.Append("0.0");
////                    break;
////                case "NaN":
////                    sb.Append("double.NaN");
////                    break;
////                case "INF":
////                    sb.Append("double.PositiveInfinity");
////                    break;
////                case "geq":
////                    sb.Append("supportFunctions._geq");
////                    break;
////                case "leq":
////                    sb.Append("supportFunctions._leq");
////                    break;
////                case "gt":
////                    sb.Append("supportFunctions._gt");
////                    break;
////                case "lt":
////                    sb.Append("supportFunctions._lt");
////                    break;
////                case "eq":
////                    sb.Append("supportFunctions._eq");
////                    break;
////                case "neq":
////                    sb.Append("supportFunctions._neq");
////                    break;
////                case "and":
////                    sb.Append("supportFunction._and");
////                    break;
////                case "or":
////                    sb.Append("supportFunction._or");
////                    break;
////                case "not":
////                    sb.Append("supportFunction._not");
////                    break;
////                case "xor":
////                    sb.Append("supportFunction._xor");
////                    break;
////                case "root":
////                    sb.Append("supportFunctions._root");
////                    break;
////                case "piecewise":
////                    sb.Append("supportFunctions._piecewise");
////                    break;
////                case "delay":
////                    sb.Append("supportFunctions._delay");
////                    Warnings.Add("RoadRunner does not yet support delay differential equations in SBML, they will be ignored (i.e. treated as delay = 0).");
////                    break;
////                default:
////                    bool bReplaced = false;
////                    int index;
////                    if (reactionList.find(reactionName, out index))
////                    {
////                        int nParamIndex = 0;
////                        if (localParameterList[index].find(s.tokenString, out nParamIndex))
////                        {
////                            sb.Append("_lp[" + index + "][" + nParamIndex + "]");
////                            bReplaced = true;
////                            break;
////                        }
////                    }
////
////                    if (boundarySpeciesList.find(s.tokenString, out index))
////                    {
////                        sb.Append("_bc[" + index + "]");
////                        bReplaced = true;
////                        break;
////                    }
////                    if (!bReplaced &&
////                        (_functionParameters != null && !_functionParameters.Contains(s.tokenString)))
////                    {
////                        throw new ArgumentException("Token '" + s.tokenString + "' not recognized.");
////                    }
////                    break;
////            }
////        }
////        private void SubstituteWords(string reactionName, bool bFixAmounts, LibRoadRunner.Scanner.Scanner s, StringBuilder sb)
////        {
////            // Global parameters have priority
////            int index;
////            if (globalParameterList.find(s.tokenString, out index))
////            {
////                sb.AppendFormat("_gp[{0}]", index);
////            }
////            else if (boundarySpeciesList.find(s.tokenString, out index))
////            {
////                sb.AppendFormat("_bc[{0}]", index);
////
////                var symbol = boundarySpeciesList[index];
////                if (symbol.hasOnlySubstance)
////                {
////                    // we only store concentration for the boundary so we better
////                    // fix that.
////                    int nCompIndex = 0;
////                    if (compartmentList.find(symbol.compartmentName, out nCompIndex))
////                    {
////                        sb.AppendFormat("{0}_c[{1}]", STR_FixAmountCompartments, nCompIndex);
////                    }
////                }
////
////
////                //if (!bFixAmounts) return;
////                //
////                //string compartmentId = "";
////                //if (NOM.MultiplyCompartment(s.tokenString, out compartmentId))
////                //{
////                //    int nCompIndex = 0;
////                //    if (compartmentId != null && compartmentList.find(compartmentId, out nCompIndex))
////                //    {
////                //        sb.AppendFormat("{0}_c[{1}]", STR_FixAmountCompartments, nCompIndex);
////                //    }
////                //}
////            }
////            else if (floatingSpeciesConcentrationList.find(s.tokenString, out index))
////            {
////                var floating1 = floatingSpeciesConcentrationList[index];
////                if (floating1.hasOnlySubstance)
////                {
////                    sb.AppendFormat("amounts[{0}]", index);
////                }
////                else
////                {
////                    sb.AppendFormat("_y[{0}]", index);
////                }
////
////                //if (!bFixAmounts) return;
////                //
////                //
////                //string compartmentId = "";
////                //if (NOM.MultiplyCompartment(s.tokenString, out compartmentId))
////                //{
////                //    int nCompIndex = 0;
////                //    if (compartmentId != null && compartmentList.find(compartmentId, out nCompIndex))
////                //    {
////                //        sb.AppendFormat("{0}_c[{1}]", STR_FixAmountCompartments, nCompIndex);
////                //    }
////                //}
////            }
////            else if (compartmentList.find(s.tokenString, out index))
////            {
////                sb.AppendFormat("_c[{0}]", index);
////            }
////            else if (_functionNames.Contains(s.tokenString))
////            {
////                sb.AppendFormat("{0} ", s.tokenString);
////            }
////            else if (ModifiableSpeciesReferenceList.find(s.tokenString, out index))
////            {
////                sb.AppendFormat("_sr[{0}]", index);
////            }
////            else if (reactionList.find(s.tokenString, out index))
////            {
////                sb.AppendFormat("_rates[{0}]", index);
////            }
////            else
////            {
////                SubstituteEquation(reactionName, s, sb);
////            }
////        }
////        private void SubstituteToken(string reactionName, bool bFixAmounts, Scanner.Scanner s, StringBuilder sb)
////        {
////            switch (s.token)
////            {
////                case CodeTypes.tWordToken:
////                case CodeTypes.tExternalToken:
////                case CodeTypes.tExtToken:
////
////                    SubstituteWords(reactionName, bFixAmounts, s, sb);
////                    break;
////
////                case CodeTypes.tDoubleToken:
////                    sb.Append("(double)" + WriteDouble(s.tokenDouble));
////                    break;
////                case CodeTypes.tIntToken:
////                    sb.Append("(double)" + WriteDouble((double)s.tokenInteger));
////                    break;
////                case CodeTypes.tPlusToken:
////                    sb.AppendFormat("+{0}\t", NL());
////                    break;
////                case CodeTypes.tMinusToken:
////                    sb.AppendFormat("-{0}\t", NL());
////                    break;
////                case CodeTypes.tDivToken:
////                    sb.AppendFormat("/{0}\t", NL());
////                    break;
////                case CodeTypes.tMultToken:
////                    sb.AppendFormat("*{0}\t", NL());
////                    break;
////                case CodeTypes.tPowerToken:
////                    sb.AppendFormat("^{0}\t", NL());
////                    break;
////                case CodeTypes.tLParenToken:
////                    sb.Append("(");
////                    break;
////                case CodeTypes.tRParenToken:
////                    sb.AppendFormat("){0}\t", NL());
////                    break;
////                case CodeTypes.tCommaToken:
////                    sb.Append(",");
////                    break;
////                case CodeTypes.tEqualsToken:
////                    sb.AppendFormat(" = {0}\t", NL());
////                    break;
////                case CodeTypes.tTimeWord1:
////                    sb.Append("time");
////                    break;
////                case CodeTypes.tTimeWord2:
////                    sb.Append("time");
////                    break;
////                case CodeTypes.tTimeWord3:
////                    sb.Append("time");
////                    break;
////                case CodeTypes.tAndToken:
////                    sb.AppendFormat("{0}supportFunctions._and", NL());
////                    break;
////                case CodeTypes.tOrToken:
////                    sb.AppendFormat("{0}supportFunctions._or", NL());
////                    break;
////                case CodeTypes.tNotToken:
////                    sb.AppendFormat("{0}supportFunctions._not", NL());
////                    break;
////                case CodeTypes.tLessThanToken:
////                    sb.AppendFormat("{0}supportFunctions._lt", NL());
////                    break;
////                case CodeTypes.tLessThanOrEqualToken:
////                    sb.AppendFormat("{0}supportFunctions._leq", NL());
////                    break;
////                case CodeTypes.tMoreThanOrEqualToken:
////                    sb.AppendFormat("{0}supportFunctions._geq", NL());
////                    break;
////                case CodeTypes.tMoreThanToken:
////                    sb.AppendFormat("{0}supportFunctions._gt", NL());
////                    break;
////                case CodeTypes.tXorToken:
////                    sb.AppendFormat("{0}supportFunctions._xor", NL());
////                    break;
////                default:
////                    var ae =
////                        new SBWApplicationException(
////                            string.Format("Unknown token in substituteTerms: {0}", s.tokenToString(s.token)),
////                            "Exception raised in Module:roadRunner, Method:substituteTerms");
////                    throw ae;
////            }
////        }
////        private static ASTNode CleanEquation(ASTNode ast)
////        {
////            if (ast.getType() == libsbml.AST_PLUS && ast.getNumChildren() == 0)
////            {
////                var result = new ASTNode(libsbml.AST_INTEGER);
////                result.setValue(0);
////                return result;
////            }
////            else if (ast.getType() == libsbml.AST_TIMES && ast.getNumChildren() == 0)
////            {
////                var result = new ASTNode(libsbml.AST_INTEGER);
////                result.setValue(1);
////                return result;
////            }
////            else if (ast.getType() == libsbml.AST_PLUS && ast.getNumChildren() == 1)
////            {
////                return ast.getChild(0);
////            }
////            else if (ast.getType() == libsbml.AST_TIMES && ast.getNumChildren() == 1)
////            {
////                return ast.getChild(0);
////            }
////
////            for (long i = ast.getNumChildren() - 1; i >= 0; i--)
////                ast.replaceChild(i, CleanEquation(ast.getChild(i)));
////
////            return ast;
////
////        }
////        private static string CleanEquation(string equation)
////        {
////            if (string.IsNullOrEmpty(equation)) return "0";
////
////            if (equation == " + ") return "0";
////            if (equation == " * ") return "1";
////
////            var ast = libsbml.parseFormula(equation);
////            if (ast == null)
////            {
////                // we are in trouble!
////                if (equation.EndsWith("* "))
////                    equation = equation.Substring(0, equation.Length - 2);
////
////                equation = equation.Replace("*  +", "+");
////                equation = equation.Replace("*  -", "-");
////
////                ast = libsbml.parseFormula(equation);
////                if (ast == null)
////                return equation;
////            }
////
////            ast = CleanEquation(ast);
////
////            return libsbml.formulaToString(ast);
////
////        }
////        private string substituteTerms(string reactionName, string inputEquation, bool bFixAmounts)
////        {
////            string equation = CleanEquation(inputEquation);
////            if (string.IsNullOrEmpty(equation)) return "0";
////
////            var s = new Scanner.Scanner();
////            Stream ss = new MemoryStream(Encoding.Default.GetBytes(equation));
////            s.stream = ss;
////            s.startScanner();
////            s.nextToken();
////            var sb = new StringBuilder();
////
////            try
////            {
////                while (s.token != CodeTypes.tEndOfStreamToken)
////                {
////                    SubstituteToken(reactionName, bFixAmounts, s, sb);
////                    s.nextToken();
////                }
////            }
////            catch (SBWApplicationException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException(e.Message);
////            }
////            return sb.ToString();
////        }
////
////        private string NL()
////        {
////            return Environment.NewLine;
////        }
////
////
////        private double[][] InitializeL0()
////        {
////            double[][] L0;
////            try
////            {
////                if (_NumDependentSpecies > 0)
////                    L0 = StructAnalysis.GetL0Matrix();
////                else L0 = new double[0][];
////            }
////            catch (Exception)
////            {
////                L0 = new double[0][];
////            }
////            return L0;
////        }
////
////        private void WriteOutSymbolTables(StringBuilder sb)
////        {
////            sb.Append("\tvoid loadSymbolTables() {" + NL());
////
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////                sb.AppendFormat("\t\tvariableTable[{0}] = \"{1}\";{2}", i, floatingSpeciesConcentrationList[i].name, NL());
////
////            for (int i = 0; i < boundarySpeciesList.Count; i++)
////                sb.AppendFormat("\t\tboundaryTable[{0}] = \"{1}\";{2}", i, boundarySpeciesList[i].name, NL());
////
////            for (int i = 0; i < globalParameterList.Count; i++)
////            {
////                sb.AppendFormat("\t\tglobalParameterTable[{0}] = \"{1}\";{2}", i, globalParameterList[i].name, NL());
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////
////        private int ReadFloatingSpecies()
////        {
////            // Load a reordered list into the variable list.
////            string[] reOrderedList;
////            if ((RoadRunner._bComputeAndAssignConservationLaws))
////                reOrderedList = StructAnalysis.GetReorderedSpeciesIds();
////            else
////                reOrderedList = StructAnalysis.GetSpeciesIds();
////
////            ArrayList oFloatingSpecies = NOM.getListOfFloatingSpecies();
////
////
////            ArrayList oTempList;
////            for (int i = 0; i < reOrderedList.Length; i++)
////            {
////                for (int j = 0; j < oFloatingSpecies.Count; j++)
////                {
////                    oTempList = (ArrayList)oFloatingSpecies[j];
////                    if (reOrderedList[i] != (string)oTempList[0]) continue;
////
////                    string compartmentName = NOM.getNthFloatingSpeciesCompartmentName(j);
////                    var bIsConcentration = (bool)oTempList[2];
////                    var dValue = (double)oTempList[1];
////                    if (double.IsNaN(dValue))
////                        dValue = 0;
////                    Symbol symbol = null;
////                    if (bIsConcentration)
////                    {
////                        symbol = new Symbol(reOrderedList[i], dValue, compartmentName);
////                    }
////                    else
////                    {
////                        int nCompartmentIndex;
////                        compartmentList.find(compartmentName, out nCompartmentIndex);
////                        double dVolume = compartmentList[nCompartmentIndex].value;
////                        if (double.IsNaN(dVolume)) dVolume = 1;
////                        symbol = new Symbol(reOrderedList[i],
////                            dValue / dVolume,
////                            compartmentName,
////                            string.Format("{0}/ _c[{1}]", dValue, nCompartmentIndex));
////                    }
////                    symbol.hasOnlySubstance = NOM.SbmlModel.getSpecies(reOrderedList[i]).getHasOnlySubstanceUnits();
////                    symbol.constant = NOM.SbmlModel.getSpecies(reOrderedList[i]).getConstant();
////                    floatingSpeciesConcentrationList.Add(symbol);
////                    break;
////                }
////                //throw new SBWApplicationException("Reordered Species " + reOrderedList[i] + " not found.");
////            }
////            return oFloatingSpecies.Count;
////        }
////
////        private int ReadBoundarySpecies()
////        {
////            int numBoundarySpecies;
////            ArrayList oBoundarySpecies = NOM.getListOfBoundarySpecies();
////            numBoundarySpecies = oBoundarySpecies.Count; // sp1.Count;
////            for (int i = 0; i < numBoundarySpecies; i++)
////            {
////                var oTempList = (ArrayList)oBoundarySpecies[i];
////                var sName = (string)oTempList[0];
////                string compartmentName = NOM.getNthBoundarySpeciesCompartmentName(i);
////                var bIsConcentration = (bool)oTempList[2];
////                var dValue = (double)oTempList[1];
////                if (double.IsNaN(dValue)) dValue = 0;
////                Symbol symbol = null;
////                if (bIsConcentration)
////                    symbol = new Symbol(sName, dValue, compartmentName);
////                else
////                {
////                    int nCompartmentIndex;
////                    compartmentList.find(compartmentName, out nCompartmentIndex);
////                    double dVolume = compartmentList[nCompartmentIndex].value;
////                    if (double.IsNaN(dVolume)) dVolume = 1;
////                    symbol = new Symbol(sName, dValue / dVolume, compartmentName,
////                                                       string.Format("{0}/ _c[{1}]", dValue, nCompartmentIndex));
////                }
////                symbol.hasOnlySubstance = NOM.SbmlModel.getSpecies(sName).getHasOnlySubstanceUnits();
////                symbol.constant = NOM.SbmlModel.getSpecies(sName).getConstant();
////                boundarySpeciesList.Add(symbol);
////            }
////            return numBoundarySpecies;
////        }
////
////        private int ReadGlobalParameters()
////        {
////            string name;
////            double value;
////            int numGlobalParameters;
////            ArrayList oParameters = NOM.getListOfParameters();
////            numGlobalParameters = oParameters.Count;
////            for (int i = 0; i < numGlobalParameters; i++)
////            {
////                name = (string)((ArrayList)oParameters[i])[0];
////                value = (double)((ArrayList)oParameters[i])[1];
////                globalParameterList.Add(new Symbol(name, value));
////            }
////            return numGlobalParameters;
////        }
////
////        private void ReadLocalParameters(int numReactions, out int[] localParameterDimensions,
////                                         out int totalLocalParmeters)
////        {
////            string name;
////            double value;
////            int numLocalParameters;
////            totalLocalParmeters = 0;
////            string reactionName;
////            localParameterDimensions = new int[numReactions];
////            for (int i = 0; i < numReactions; i++)
////            {
////                numLocalParameters = NOM.getNumParameters(i);
////                reactionName = NOM.getNthReactionId(i);
////                reactionList.Add(new Symbol(reactionName, 0.0));
////                localParameterList[i] = new SymbolList();
////                for (int j = 0; j < numLocalParameters; j++)
////                {
////                    localParameterDimensions[i] = numLocalParameters;
////                    name = NOM.getNthParameterId(i, j);
////                    value = NOM.getNthParameterValue(i, j);
////                    localParameterList[i].Add(new Symbol(reactionName, name, value));
////                }
////            }
////        }
////
////        private void WriteComputeAllRatesOfChange(StringBuilder sb, int numIndependentSpecies, int numDependentSpecies,
////                                                  double[][] L0)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\t// Uses the equation: dSd/dt = L0 dSi/dt" + NL());
////            sb.Append("\tpublic void computeAllRatesOfChange ()" + NL());
////            sb.Append("\t{" + NL());
////            sb.Append("\t\tdouble[] dTemp = new double[amounts.Length + rateRules.Length];" + NL());
////            for (int i = 0; i < NumAdditionalRates; i++)
////            {
////                sb.AppendFormat("\t\tdTemp[{0}] = {1};{2}", i, _oMapRateRule[i], NL());
////            }
////            //sb.Append("\t\trateRules.CopyTo(dTemp, 0);" + NL());
////            sb.Append("\t\tamounts.CopyTo(dTemp, rateRules.Length);" + NL());
////            sb.Append("\t\tevalModel (time, dTemp);" + NL());
////            bool isThereAnEntry = false;
////            for (int i = 0; i < numDependentSpecies; i++)
////            {
////                sb.AppendFormat("\t\t_dydt[{0}] = ", (numIndependentSpecies + i));
////                isThereAnEntry = false;
////                for (int j = 0; j < numIndependentSpecies; j++)
////                {
////                    string dyName = string.Format("_dydt[{0}]", j);
////
////                    if (L0[i][j] > 0)
////                    {
////                        isThereAnEntry = true;
////                        if (L0[i][j] == 1)
////                        {
////                            sb.AppendFormat(" + {0}{1}", dyName, NL());
////                        }
////                        else
////                        {
////                            sb.AppendFormat(" + (double){0}{1}{2}{3}", WriteDouble(L0[i][j]), STR_FixAmountCompartments, dyName, NL());
////                        }
////                    }
////                    else if (L0[i][j] < 0)
////                    {
////                        isThereAnEntry = true;
////                        if (L0[i][j] == -1)
////                        {
////                            sb.AppendFormat(" - {0}{1}", dyName, NL());
////                        }
////                        else
////                        {
////                            sb.AppendFormat(" - (double){0}{1}{2}{3}", WriteDouble(Math.Abs(L0[i][j])), STR_FixAmountCompartments, dyName, NL());
////                        }
////                    }
////                }
////                if (!isThereAnEntry)
////                    sb.Append("0");
////                sb.AppendFormat(";{0}", NL());
////            }
////
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteComputeConservedTotals(StringBuilder sb, int numFloatingSpecies, int numDependentSpecies)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\t// Uses the equation: C = Sd - L0*Si" + NL());
////            sb.Append("\tpublic void computeConservedTotals ()" + NL());
////            sb.Append("\t{" + NL());
////            if (numDependentSpecies > 0)
////            {
////                string factor;
////                double[][] gamma = StructAnalysis.GetGammaMatrix();
////                for (int i = 0; i < numDependentSpecies; i++)
////                {
////                    sb.AppendFormat("\t\t_ct[{0}] = ", i);
////                    for (int j = 0; j < numFloatingSpecies; j++)
////                    {
////                        double current = gamma[i][j];
////
////
////                        if (current != 0.0)
////                        {
////                            if (double.IsNaN(current))
////                            {
////                                // TODO: fix this
////                                factor = "";
////                            }
////                            else if (Math.Abs(current) == 1.0)
////                                factor = "";
////                            else
////                                factor = WriteDouble(Math.Abs(current)) +
////                                         STR_FixAmountCompartments;
////
////                            if (current > 0)
////                                sb.Append(" + " + factor + convertSpeciesToY(floatingSpeciesConcentrationList[j].name) +
////                                          STR_FixAmountCompartments +
////                                          convertCompartmentToC(floatingSpeciesConcentrationList[j].compartmentName) +
////                                          NL());
////                            else
////                                sb.Append(" - " + factor + convertSpeciesToY(floatingSpeciesConcentrationList[j].name) +
////                                          STR_FixAmountCompartments +
////                                          convertCompartmentToC(floatingSpeciesConcentrationList[j].compartmentName) +
////                                          NL());
////                        }
////                    }
////                    sb.Append(";" + NL());
////
////                    conservationList.Add(new Symbol("CSUM" + i, double.NaN));
////                }
////            }
////            sb.Append("    }" + NL() + NL());
////        }
////
////        private void WriteUpdateDependentSpecies(StringBuilder sb, int numIndependentSpecies, int numDependentSpecies,
////                                                 double[][] L0)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\t// Compute values of dependent species " + NL());
////            sb.Append("\t// Uses the equation: Sd = C + L0*Si" + NL());
////            sb.Append("\tpublic void updateDependentSpeciesValues (double[] y)" + NL());
////            sb.Append("\t{" + NL());
////
////            if (numDependentSpecies > 0)
////            {
////                // Use the equation: Sd = C + L0*Si to compute dependent concentrations
////                if (numDependentSpecies > 0)
////                {
////                    for (int i = 0; i < numDependentSpecies; i++)
////                    {
////                        sb.AppendFormat("\t\t_y[{0}] = {1}\t", (i + numIndependentSpecies), NL());
////                        sb.AppendFormat("(_ct[{0}]", i);
////                        string cLeftName =
////                            convertCompartmentToC(
////                                floatingSpeciesConcentrationList[i + numIndependentSpecies].compartmentName);
////
////                        for (int j = 0; j < numIndependentSpecies; j++)
////                        {
////                            string yName = string.Format("y[{0}]", j);
////                            string cName = convertCompartmentToC(floatingSpeciesConcentrationList[j].compartmentName);
////
////                            if (L0[i][j] > 0)
////                            {
////                                if (L0[i][j] == 1)
////                                {
////                                    sb.AppendFormat(" + {0}\t{1}{2}{3}{0}\t",
////                                        NL(),
////                                        yName,
////                                        STR_FixAmountCompartments,
////                                        cName);
////                                }
////                                else
////                                {
////                                    sb.AppendFormat("{0}\t" + " + (double){1}{2}{3}{2}{4}",
////                                        NL(),
////                                        WriteDouble(L0[i][j]),
////                                        STR_FixAmountCompartments,
////                                        yName,
////                                        cName);
////                                }
////                            }
////                            else if (L0[i][j] < 0)
////                            {
////                                if (L0[i][j] == -1)
////                                {
////                                    sb.AppendFormat("{0}\t" + " - {1}{2}{3}",
////                                        NL(),
////                                        yName,
////                                        STR_FixAmountCompartments,
////                                        cName);
////                                }
////                                else
////                                {
////                                    sb.AppendFormat("{0}\t" + " - (double){1}{2}{3}{2}{4}",
////                                        NL(),
////                                        WriteDouble(Math.Abs(L0[i][j])),
////                                        STR_FixAmountCompartments,
////                                        yName,
////                                        cName);
////                                }
////                            }
////                        }
////                        sb.AppendFormat(")/{0};{1}", cLeftName, NL());
////                    }
////                }
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteUserDefinedFunctions(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            for (int i = 0; i < NOM.getNumFunctionDefinitions(); i++)
////            {
////                try
////                {
////                    ArrayList oList = NOM.getNthFunctionDefinition(i);
////                    var sName = (string)oList[0];
////                    sName.Trim();
////                    _functionNames.Add(sName);
////                    var oArguments = (ArrayList)oList[1];
////                    var sBody = (string)oList[2];
////
////
////                    sb.AppendFormat("\t// User defined function:  {0}{1}", sName, NL());
////
////                    sb.AppendFormat("\tpublic double {0} (", sName);
////                    for (int j = 0; j < oArguments.Count; j++)
////                    {
////                        sb.Append("double " + (string)oArguments[j]);
////                        _functionParameters.Add((string)oArguments[j]);
////                        if (j < oArguments.Count - 1)
////                            sb.Append(", ");
////                    }
////                    sb.Append(")" + NL() + "\t{" + NL() + "\t\t return " +
////                              convertUserFunctionExpression(sBody)
////                              + ";" + NL() + "\t}" + NL() + NL());
////                }
////                catch (SBWException ex)
////                {
////                    throw new SBWApplicationException("Error while trying to get Function Definition #" + i,
////                                                      ex.Message + "\r\n\r\n" + ex.DetailedMessage);
////                }
////                catch (Exception ex)
////                {
////                    throw new SBWApplicationException("Error while trying to get Function Definition #" + i, ex.Message);
////                }
////            }
////        }
////
////        private void WriteResetEvents(StringBuilder sb, int numEvents)
////        {
////            sb.AppendFormat("{0}\tpublic void resetEvents() {{{0}", NL());
////            for (int i = 0; i < numEvents; i++)
////            {
////                sb.AppendFormat("\t\t_eventStatusArray[{0}] = false;{1}", i, NL());
////                sb.AppendFormat("\t\t_previousEventStatusArray[{0}] = false;{1}", i, NL());
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteSetConcentration(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.AppendFormat("\tpublic void setConcentration(int index, double value) {{{0}", NL());
////            sb.AppendFormat("\t\tdouble volume = 0.0;{0}", NL());
////            sb.AppendFormat("\t\t_y[index] = value;{0}", NL());
////            sb.AppendFormat("\t\tswitch (index) {{{0}", NL());
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////            {
////                sb.AppendFormat("\t\t\tcase {0}: volume = {1};{2}",
////                    i,
////                    convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName),
////                    NL());
////                sb.AppendFormat("\t\t\t\tbreak;{0}", NL());
////            }
////            sb.AppendFormat("\t\t}}{0}", NL());
////            sb.AppendFormat("\t\t_amounts[index] = _y[index]*volume;{0}", NL());
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteGetConcentration(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.AppendFormat("\tpublic double getConcentration(int index) {{{0}", NL());
////            sb.AppendFormat("\t\treturn _y[index];{0}", NL());
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteConvertToAmounts(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.AppendFormat("\tpublic void convertToAmounts() {{{0}", NL());
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////            {
////                sb.AppendFormat("\t\t_amounts[{0}] = _y[{0}]*{1};{2}",
////                    i,
////                    convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName),
////                    NL());
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteConvertToConcentrations(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic void convertToConcentrations() {" + NL());
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////            {
////                sb.Append("\t\t_y[" + i + "] = _amounts[" + i + "]/" +
////                          convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName) + ";" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteProperties(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] y {" + NL());
////            sb.Append("\t\tget { return _y; } " + NL());
////            sb.Append("\t\tset { _y = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] init_y {" + NL());
////            sb.Append("\t\tget { return _init_y; } " + NL());
////            sb.Append("\t\tset { _init_y = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] amounts {" + NL());
////            sb.Append("\t\tget { return _amounts; } " + NL());
////            sb.Append("\t\tset { _amounts = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] bc {" + NL());
////            sb.Append("\t\tget { return _bc; } " + NL());
////            sb.Append("\t\tset { _bc = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] gp {" + NL());
////            sb.Append("\t\tget { return _gp; } " + NL());
////            sb.Append("\t\tset { _gp = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] sr {" + NL());
////            sb.Append("\t\tget { return _sr; } " + NL());
////            sb.Append("\t\tset { _sr = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[][] lp {" + NL());
////            sb.Append("\t\tget { return _lp; } " + NL());
////            sb.Append("\t\tset { _lp = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] c {" + NL());
////            sb.Append("\t\tget { return _c; } " + NL());
////            sb.Append("\t\tset { _c = value; } " + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] dydt {" + NL());
////            sb.Append("\t\tget { return _dydt; }" + NL());
////            sb.Append("\t\tset { _dydt = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] rateRules {" + NL());
////            sb.Append("\t\tget { return _rateRules; }" + NL());
////            sb.Append("\t\tset { _rateRules = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] rates {" + NL());
////            sb.Append("\t\tget { return _rates; }" + NL());
////            sb.Append("\t\tset { _rates = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] ct {" + NL());
////            sb.Append("\t\tget { return _ct; }" + NL());
////            sb.Append("\t\tset { _ct = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] eventTests {" + NL());
////            sb.Append("\t\tget { return _eventTests; }" + NL());
////            sb.Append("\t\tset { _eventTests = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic TEventDelayDelegate[] eventDelay {" + NL());
////            sb.Append("\t\tget { return _eventDelay; }" + NL());
////            sb.Append("\t\tset { _eventDelay = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic bool[] eventType {" + NL());
////            sb.Append("\t\tget { return _eventType; }" + NL());
////            sb.Append("\t\tset { _eventType = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic bool[] eventPersistentType {" + NL());
////            sb.Append("\t\tget { return _eventPersistentType; }" + NL());
////            sb.Append("\t\tset { _eventPersistentType = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic bool[] eventStatusArray {" + NL());
////            sb.Append("\t\tget { return _eventStatusArray; }" + NL());
////            sb.Append("\t\tset { _eventStatusArray = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic bool[] previousEventStatusArray {" + NL());
////            sb.Append("\t\tget { return _previousEventStatusArray; }" + NL());
////            sb.Append("\t\tset { _previousEventStatusArray = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double[] eventPriorities {" + NL());
////            sb.Append("\t\tget { return _eventPriorities; }" + NL());
////            sb.Append("\t\tset { _eventPriorities = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic TEventAssignmentDelegate[] eventAssignments {" + NL());
////            sb.Append("\t\tget { return _eventAssignments; }" + NL());
////            sb.Append("\t\tset { _eventAssignments = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic TComputeEventAssignmentDelegate[] computeEventAssignments {" + NL());
////            sb.Append("\t\tget { return _computeEventAssignments; }" + NL());
////            sb.Append("\t\tset { _computeEventAssignments = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic TPerformEventAssignmentDelegate[] performEventAssignments {" + NL());
////            sb.Append("\t\tget { return _performEventAssignments; }" + NL());
////            sb.Append("\t\tset { _performEventAssignments = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic double time {" + NL());
////            sb.Append("\t\tget { return _time; }" + NL());
////            sb.Append("\t\tset { _time = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteAccessors(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumIndependentVariables {" + NL());
////            sb.Append("\t\tget { return numIndependentVariables; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumDependentVariables {" + NL());
////            sb.Append("\t\tget { return numDependentVariables; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumTotalVariables {" + NL());
////            sb.Append("\t\tget { return numTotalVariables; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumBoundarySpecies {" + NL());
////            sb.Append("\t\tget { return numBoundaryVariables; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumGlobalParameters {" + NL());
////            sb.Append("\t\tget { return numGlobalParameters; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumLocalParameters(int reactionId)" + NL());
////            sb.Append("\t{" + NL());
////            sb.Append("\t\treturn localParameterDimensions[reactionId];" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumCompartments {" + NL());
////            sb.Append("\t\tget { return numCompartments; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumReactions {" + NL());
////            sb.Append("\t\tget { return numReactions; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumEvents {" + NL());
////            sb.Append("\t\tget { return numEvents; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic int getNumRules {" + NL());
////            sb.Append("\t\tget { return numRules; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic List<string> Warnings {" + NL());
////            sb.Append("\t\tget { return _Warnings; }" + NL());
////            sb.Append("\t\tset { _Warnings = value; }" + NL());
////            sb.Append("\t}" + NL() + NL());
////
////        }
////
////        private void WriteOutVariables(StringBuilder sb)
////        {
////            sb.Append("\tprivate List<string> _Warnings = new List<string>();" + NL());
////            sb.Append("\tprivate double[] _gp = new double[" + (_NumGlobalParameters + _TotalLocalParmeters) +
////                      "];           // Vector containing all the global parameters in the System  " + NL());
////            sb.Append("\tprivate double[] _sr = new double[" + (_NumModifiableSpeciesReferences) +
////                      "];           // Vector containing all the modifiable species references  " + NL());
////            sb.Append("\tprivate double[][] _lp = new double[" + _NumReactions +
////                      "][];       // Vector containing all the local parameters in the System  " + NL());
////            sb.Append("\tprivate double[] _y = new double[" + floatingSpeciesConcentrationList.Count +
////                      "];            // Vector containing the concentrations of all floating species " + NL());
////            sb.Append(String.Format("\tprivate double[] _init_y = new double[{0}];            // Vector containing the initial concentrations of all floating species {1}", floatingSpeciesConcentrationList.Count, NL()));
////            sb.Append("\tprivate double[] _amounts = new double[" + floatingSpeciesConcentrationList.Count +
////                      "];      // Vector containing the amounts of all floating species " + NL());
////            sb.Append("\tprivate double[] _bc = new double[" + _NumBoundarySpecies +
////                      "];           // Vector containing all the boundary species concentration values   " + NL());
////            sb.Append("\tprivate double[] _c = new double[" + _NumCompartments +
////                      "];            // Vector containing all the compartment values   " + NL());
////            sb.Append("\tprivate double[] _dydt = new double[" + floatingSpeciesConcentrationList.Count +
////                      "];         // Vector containing rates of changes of all species   " + NL());
////            sb.Append("\tprivate double[] _rates = new double[" + _NumReactions +
////                      "];        // Vector containing the rate laws of all reactions    " + NL());
////            sb.Append("\tprivate double[] _ct = new double[" + _NumDependentSpecies +
////                      "];           // Vector containing values of all conserved sums      " + NL());
////
////            sb.Append("\tprivate double[] _eventTests = new double[" + _NumEvents +
////                      "];   // Vector containing results of any event tests        " + NL());
////            sb.Append("\tprivate TEventDelayDelegate[] _eventDelay = new TEventDelayDelegate[" + _NumEvents +
////                      "]; // array of trigger function pointers" + NL());
////            sb.Append("\tprivate bool[] _eventType = new bool[" + _NumEvents +
////                      "]; // array holding the status whether events are useValuesFromTriggerTime or not" + NL());
////            sb.Append("\tprivate bool[] _eventPersistentType = new bool[" + _NumEvents +
////                      "]; // array holding the status whether events are persitstent or not" + NL());
////            sb.Append("\tprivate double _time;" + NL());
////            sb.Append("\tprivate int numIndependentVariables;" + NL());
////            sb.Append("\tprivate int numDependentVariables;" + NL());
////            sb.Append("\tprivate int numTotalVariables;" + NL());
////            sb.Append("\tprivate int numBoundaryVariables;" + NL());
////            sb.Append("\tprivate int numGlobalParameters;" + NL());
////            sb.Append("\tprivate int numCompartments;" + NL());
////            sb.Append("\tprivate int numReactions;" + NL());
////            sb.Append("\tprivate int numRules;" + NL());
////            sb.Append("\tprivate int numEvents;" + NL());
////            sb.Append("\tstring[] variableTable = new string[" + floatingSpeciesConcentrationList.Count + "];" + NL());
////            sb.Append("\tstring[] boundaryTable = new string[" + boundarySpeciesList.Count + "];" + NL());
////            sb.Append("\tstring[] globalParameterTable = new string[" + globalParameterList.Count + "];" + NL());
////            sb.Append("\tint[] localParameterDimensions = new int[" + _NumReactions + "];" + NL());
////            sb.Append("\tprivate TEventAssignmentDelegate[] _eventAssignments;" + NL());
////            sb.Append("\tprivate double[] _eventPriorities;" + NL());
////            sb.Append("\tprivate TComputeEventAssignmentDelegate[] _computeEventAssignments;" + NL());
////            sb.Append("\tprivate TPerformEventAssignmentDelegate[] _performEventAssignments;" + NL());
////            sb.Append("\tprivate bool[] _eventStatusArray = new bool[" + _NumEvents + "];" + NL());
////            sb.Append("\tprivate bool[] _previousEventStatusArray = new bool[" + _NumEvents + "];" + NL());
////            sb.Append(NL());
////            sb.Append("\tpublic TModel ()  " + NL());
////            sb.Append("\t{" + NL());
////
////            sb.Append("\t\tnumIndependentVariables = " + _NumIndependentSpecies + ";" + NL());
////            sb.Append("\t\tnumDependentVariables = " + _NumDependentSpecies + ";" + NL());
////            sb.Append("\t\tnumTotalVariables = " + _NumFloatingSpecies + ";" + NL());
////            sb.Append("\t\tnumBoundaryVariables = " + _NumBoundarySpecies + ";" + NL());
////            sb.Append("\t\tnumGlobalParameters = " + globalParameterList.Count + ";" + NL());
////            sb.Append("\t\tnumCompartments = " + compartmentList.Count + ";" + NL());
////            sb.Append("\t\tnumReactions = " + reactionList.Count + ";" + NL());
////            sb.Append("\t\tnumEvents = " + _NumEvents + ";" + NL());
////            sb.Append("\t\tInitializeDelays();" + NL());
////
////            // Declare any eventAssignment delegates
////            if (_NumEvents > 0)
////            {
////                sb.Append("\t\t_eventAssignments = new TEventAssignmentDelegate[numEvents];" + NL());
////                sb.Append("\t\t_eventPriorities = new double[numEvents];" + NL());
////                sb.Append("\t\t_computeEventAssignments= new TComputeEventAssignmentDelegate[numEvents];" + NL());
////                sb.Append("\t\t_performEventAssignments= new TPerformEventAssignmentDelegate[numEvents];" + NL());
////
////                for (int i = 0; i < _NumEvents; i++)
////                {
////                    sb.Append("\t\t_eventAssignments[" + i + "] = new TEventAssignmentDelegate (eventAssignment_" + i +
////                              ");" + NL());
////                    sb.Append("\t\t_computeEventAssignments[" + i +
////                              "] = new TComputeEventAssignmentDelegate (computeEventAssignment_" + i + ");" + NL());
////                    sb.Append("\t\t_performEventAssignments[" + i +
////                              "] = new TPerformEventAssignmentDelegate (performEventAssignment_" + i + ");" + NL());
////                }
////
////                sb.Append("\t\tresetEvents();" + NL());
////                sb.Append(NL());
////            }
////
////            if (_NumModifiableSpeciesReferences > 0)
////            {
////                for (int i = 0; i < ModifiableSpeciesReferenceList.Count; i++)
////                {
////                    sb.Append("\t\t_sr[" + i + "]  = " + WriteDouble(ModifiableSpeciesReferenceList[i].value) + ";" + NL());
////                }
////                sb.Append(NL());
////            }
////
////            // Declare space for local parameters
////            for (int i = 0; i < _NumReactions; i++)
////            {
////                sb.Append("\t\tlocalParameterDimensions[" + i + "] = " + _LocalParameterDimensions[i] + ";" + NL());
////                sb.Append("\t\t_lp[" + i + "] = new double[" + _LocalParameterDimensions[i] + "];" + NL());
////            }
////
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteClassHeader(StringBuilder sb)
////        {
////            sb.Append("using System;" + NL());
////            sb.Append("using System.IO;" + NL());
////            sb.Append("using System.Collections;" + NL());
////            sb.Append("using System.Collections.Generic;" + NL());
////            sb.Append("using LibRoadRunner;" + NL());
////
////            sb.Append(" " + NL() + NL());
////            sb.Append(NL());
////            sb.AppendFormat("class TModel : IModel{0}", NL());
////            sb.Append("{" + NL());
////            sb.AppendFormat("\t// Symbol Mappings{0}{0}", NL());
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////                sb.AppendFormat("\t// y[{0}] = {1}{2}", i, floatingSpeciesConcentrationList[i].name, NL());
////            sb.Append(NL());
////        }
////
////        private string FindSymbol(string varName)
////        {
////            int index = 0;
////            if (floatingSpeciesConcentrationList.find(varName, out index))
////            {
////                return string.Format("\t\t_y[{0}]", index);
////            }
////            else if (globalParameterList.find(varName, out index))
////            {
////                return string.Format("\t\t_gp[{0}]", index);
////            }
////            else if (boundarySpeciesList.find(varName, out index))
////            {
////                return string.Format("\t\t_bc[{0}]", index);
////            }
////            else if (compartmentList.find(varName, out index))
////            {
////                return string.Format("\t\t_c[{0}]", index);
////            }
////            else if (ModifiableSpeciesReferenceList.find(varName, out index))
////                return string.Format("\t\t_sr[{0}]", index);
////
////            else
////                throw new SBWApplicationException(string.Format("Unable to locate lefthand side symbol in assignment[{0}]", varName));
////        }
////
////        private void WriteTestConstraints(StringBuilder sb)
////        {
////            sb.Append("\tpublic void testConstraints()" + NL());
////            sb.Append("\t{" + NL());
////
////            for (int i = 0; i < NOM.getNumConstraints(); i++)
////            {
////                string sMessage;
////                string sCheck = NOM.getNthConstraint(i, out sMessage);
////
////                sb.Append("\t\tif (" + substituteTerms(NOM.getNumReactions(), "", sCheck) + " == 0.0 )" + NL());
////                sb.Append("\t\t\tthrow new Exception(\"" + sMessage + "\");" + NL());
////            }
////
////
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private static bool ExpressionContainsSymbol(ASTNode ast, string symbol)
////        {
////            if (ast == null || string.IsNullOrEmpty(symbol)) return false;
////
////            if (ast.getType() == libsbml.AST_NAME && ast.getName().Trim() == symbol.Trim())
////                return true;
////
////            for (int i = 0; i < ast.getNumChildren(); i++)
////            {
////                if (ExpressionContainsSymbol(ast.getChild(i), symbol))
////                    return true;
////            }
////
////            return false;
////
////        }
////        private static bool ExpressionContainsSymbol(string expression, string symbol)
////        {
////            if (string.IsNullOrEmpty(expression) || string.IsNullOrEmpty(symbol)) return false;
////            var ast = libsbml.parseFormula(expression);
////            return ExpressionContainsSymbol(ast, symbol);
////        }
////        private void WriteEvalInitialAssignments(StringBuilder sb, int numReactions)
////        {
////            sb.Append("\tpublic void evalInitialAssignments()" + NL());
////            sb.Append("\t{" + NL());
////
////            int numInitialAssignments = NOM.getNumInitialAssignments();
////
////            if (numInitialAssignments > 0)
////            {
////                var oList = new List<Pair<string, string>>();
////                for (int i = 0; i < numInitialAssignments; i++)
////                    oList.Add(NOM.getNthInitialAssignmentPair(i));
////
////                // sort them ...
////                bool bChange = true;
////                int nIndex = -1;
////                while (bChange)
////                {
////                    bChange = false;
////
////                    for (int i = 0; i < oList.Count; i++)
////                    {
////                        Pair<string, string> current = oList[i];
////                        for (int j = i + 1; j < oList.Count; j++)
////                        {
////                            if (ExpressionContainsSymbol(current.Second, oList[j].First))
////                            {
////                                bChange = true;
////                                nIndex = j;
////                                break;
////                            }
////                        }
////                        if (bChange) break;
////                    }
////
////                    if (bChange)
////                    {
////                        Pair<string, string> pairToMove = oList[nIndex];
////                        oList.RemoveAt(nIndex);
////                        oList.Insert(0, pairToMove);
////                    }
////                }
////
////                foreach (var pair in oList)
////                {
////                    string leftSideRule = FindSymbol(pair.First);
////                    string rightSideRule = pair.Second;
////                    if (leftSideRule != null)
////                    {
////                        sb.Append(leftSideRule + " = ");
////                        sb.Append(substituteTerms(numReactions, "", rightSideRule) + ";" + NL());
////                    }
////                }
////
////
////            }
////            for (int i = 0; i < NOM.SbmlModel.getNumEvents(); i++)
////            {
////                var current = NOM.SbmlModel.getEvent(i);
////                string initialTriggerValue = current.getTrigger().getInitialValue().ToString().ToLowerInvariant();
////                sb.Append("\t\t_eventStatusArray[" + i + "] = " + initialTriggerValue + ";" + NL());
////                sb.Append("\t\t_previousEventStatusArray[" + i + "] = " + initialTriggerValue + ";" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////        }
////
////
////        private int WriteComputeRules(StringBuilder sb, int numReactions)
////        {
////            int numOfRules = NOM.getNumRules();
////            _oMapRateRule = new Hashtable();
////            var mapVariables = new Hashtable();
////            int numRateRules = 0;
////
////
////            sb.Append("\tpublic void computeRules(double[] y) {" + NL());
////            // ------------------------------------------------------------------------------
////            for (int i = 0; i < numOfRules; i++)
////            {
////                try
////                {
////                    string leftSideRule = "";
////                    string rightSideRule = "";
////                    string ruleType = NOM.getNthRuleType(i);
////                    // We only support assignment and ode rules at the moment
////                    string eqnRule = NOM.getNthRule(i);
////                    int index = eqnRule.IndexOf("=");
////                    string varName = eqnRule.Substring(0, index).Trim();
////                    string rightSide = eqnRule.Substring(index + 1).Trim();
////                    bool isRateRule = false;
////
////                    switch (ruleType)
////                    {
////                        case "Algebraic_Rule":
////                            Warnings.Add("RoadRunner does not yet support algebraic rules in SBML, they will be ignored.");
////                            leftSideRule = null;
////                            break;
////
////
////                        case "Assignment_Rule":
////                            leftSideRule = FindSymbol(varName);
////                            break;
////
////                        case "Rate_Rule":
////                            if (floatingSpeciesConcentrationList.find(varName, out index))
////                            {
////                                leftSideRule = string.Format("\t\t_dydt[{0}]", index);
////                                floatingSpeciesConcentrationList[index].rateRule = true;
////                            }
////                            else
////                            {
////                                leftSideRule = "\t\t_rateRules[" + numRateRules + "]";
////                                _oMapRateRule[numRateRules] = FindSymbol(varName);
////                                mapVariables[numRateRules] = varName;
////                                numRateRules++;
////                            }
////                            isRateRule = true;
////
////                            break;
////                    }
////
////                    // Run the equation through MathML to carry out any conversions (eg ^ to Pow)
////                    string rightSideMathml = NOM.convertStringToMathML(rightSide);
////                    rightSideRule = NOM.convertMathMLToString(rightSideMathml);
////                    if (leftSideRule != null)
////                    {
////                        sb.Append(leftSideRule + " = ");
////
////                        int speciesIndex;
////                        var isSpecies = floatingSpeciesConcentrationList.find(varName, out speciesIndex);
////
////                        var symbol = speciesIndex != -1 ? floatingSpeciesConcentrationList[speciesIndex] : null;
////
////                        //
////
////                        string sCompartment;
////
////                        if (
////                            isRateRule &&
////                            NOM.MultiplyCompartment(varName, out sCompartment) &&
////                            !rightSide.Contains(sCompartment)
////                            )
////                        {
////                            sb.Append(String.Format("({0}) * {1};{2}", substituteTerms(numReactions, "", rightSideRule), FindSymbol(sCompartment), NL()));
////                        }
////                        else
////                        {
////                            if (isSpecies && !isRateRule && symbol != null && symbol.hasOnlySubstance && symbol.compartmentName != null)
////                                sb.Append(String.Format("({0}) / {1};{2}", substituteTerms(numReactions, "", rightSideRule), FindSymbol(symbol.compartmentName), NL()));
////                            else
////                                sb.Append(String.Format("{0};{1}", substituteTerms(numReactions, "", rightSideRule), NL()));
////                        }
////
////
////
////                        // RateRules and species ! again
////                        //
////                        // sb.Append(String.Format("{0};{1}", substituteTerms(numReactions, "", rightSideRule), NL()));
////
////                        if (NOM.IsCompartment(varName))
////                        {
////                            sb.Append("\t\tconvertToConcentrations();");
////                        }
////                    }
////                }
////                catch (SBWException)
////                {
////                    throw;
////                }
////                catch (Exception ex)
////                {
////                    throw new SBWApplicationException("Error while trying to get Rule #" + i, ex.Message);
////                }
////            }
////            sb.Append("\t}" + NL() + NL());
////
////            sb.Append("\tprivate double[] _rateRules = new double[" + numRateRules +
////                      "];           // Vector containing values of additional rate rules      " + NL());
////            sb.Append("\tpublic void InitializeRates()" + NL() + "\t{" + NL());
////            for (int i = 0; i < numRateRules; i++)
////            {
////
////                sb.Append("\t\t_rateRules[" + i + "] = " + _oMapRateRule[i] + ";" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////
////
////            sb.Append("\tpublic void AssignRates()" + NL() + "\t{" + NL());
////            for (int i = 0; i < _oMapRateRule.Count; i++)
////            {
////                sb.Append((string)_oMapRateRule[i] + " = _rateRules[" + i + "];" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////
////
////            sb.Append("\tpublic void InitializeRateRuleSymbols()" + NL() + "\t{" + NL());
////            for (int i = 0; i < _oMapRateRule.Count; i++)
////            {
////                var varName = (string)mapVariables[i];
////                double value = NOM.getValue(varName);
////                if (!double.IsNaN(value))
////                    sb.Append((string)_oMapRateRule[i] + " = " + value + ";" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////
////
////            sb.Append("\tpublic void AssignRates(double[] oRates)" + NL() + "\t{" + NL());
////            for (int i = 0; i < _oMapRateRule.Count; i++)
////            {
////                sb.Append((string)_oMapRateRule[i] + " = oRates[" + i + "];" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////
////            sb.Append("\tpublic double[] GetCurrentValues()" + NL() + "\t{" + NL());
////            sb.Append("\t\tdouble[] dResult = new double[" + NumAdditionalRates + "];" + NL());
////            for (int i = 0; i < _oMapRateRule.Count; i++)
////            {
////                sb.Append("\t\tdResult[" + i + "] = " + (string)_oMapRateRule[i] + ";" + NL());
////            }
////            sb.Append("\t\treturn dResult;" + NL());
////
////            sb.Append("\t}" + NL() + NL());
////            return numOfRules;
////        }
////
////        private void WriteComputeReactionRates(StringBuilder sb, int numReactions)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\t// Compute the reaction rates" + NL());
////            sb.Append("\tpublic void computeReactionRates (double time, double[] y)" + NL());
////            sb.Append("\t{" + NL());
////
////
////            for (int i = 0; i < numReactions; i++)
////            {
////                string kineticLaw = NOM.getKineticLaw(i);
////
////                // The following code is for the case when the kineticLaw contains a ^ in place
////                // of pow for exponent handling. It would not be needed in the case when there is
////                // no ^ in the kineticLaw.
////                string subKineticLaw;
////                if (kineticLaw.IndexOf("^", System.StringComparison.Ordinal) > 0)
////                {
////                    string kineticLaw_mathml = NOM.convertStringToMathML(kineticLaw);
////                    subKineticLaw = NOM.convertMathMLToString(kineticLaw_mathml);
////                }
////                else
////                {
////                    subKineticLaw = kineticLaw;
////                }
////
////                string modKineticLaw = substituteTerms(reactionList[i].name, subKineticLaw, true) + ";";
////
////                // modify to use current y ...
////                modKineticLaw = modKineticLaw.Replace("_y[", "y[");
////
////
////                sb.AppendFormat("\t\t_rates[{0}] = {1}{2}", i, modKineticLaw, NL());
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////        private void WriteEvalEvents(StringBuilder sb, int numEvents, int numFloatingSpecies)
////        {
////            sb.Append("\t// Event handling function" + NL());
////            sb.Append("\tpublic void evalEvents (double timeIn, double[] oAmounts)" + NL());
////            sb.Append("\t{" + NL());
////
////            if (numEvents > 0)
////            {
////                for (int i = 0; i < NumAdditionalRates; i++)
////                {
////                    sb.Append((string)_oMapRateRule[i] + " = oAmounts[" + i + "];" + NL());
////                }
////                for (int i = 0; i < numFloatingSpecies; i++)
////                {
////                    sb.Append("\t\t_y[" + i + "] = oAmounts[" + (i + NumAdditionalRates) + "]/" +
////                              convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName) + ";" + NL());
////                }
////            }
////
////
////            sb.Append("\t\t_time = timeIn;  // Don't remove" + NL());
////            sb.Append("\t\tupdateDependentSpeciesValues(_y);" + NL());
////            sb.Append("\t\tcomputeRules (_y);" + NL());
////
////            for (int i = 0; i < numEvents; i++)
////            {
////                ArrayList ev = NOM.getNthEvent(i);
////                string eventString = (string)ev[0];
////                eventString = substituteTerms(0, "", eventString);
////                sb.Append("\t\tpreviousEventStatusArray[" + i + "] = eventStatusArray[" + i + "];" + NL());
////                sb.Append("\t\tif (" + eventString + " == 1.0) {" + NL());
////                sb.Append("\t\t     eventStatusArray[" + i + "] = true;" + NL());
////                sb.Append("\t\t     eventTests[" + i + "] = 1;" + NL());
////                sb.Append("\t\t} else {" + NL());
////                sb.Append("\t\t     eventStatusArray[" + i + "] = false;" + NL());
////                sb.Append("\t\t     eventTests[" + i + "] = -1;" + NL());
////                sb.Append("\t\t}" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////
////        }
////
////
////        private void WriteEvalModel(StringBuilder sb, int numReactions, int numIndependentSpecies,
////                                    int numFloatingSpecies, int numOfRules)
////        {
////            sb.Append("\t// Model Function" + NL());
////            sb.Append("\tpublic void evalModel (double timein, double[] oAmounts)" + NL());
////            sb.Append("\t{" + NL());
////
////            //sb.Append("\t\tconvertToConcentrations (); " + NL());
////
////
////            for (int i = 0; i < NumAdditionalRates; i++)
////            {
////                sb.Append((string)_oMapRateRule[i] + " = oAmounts[" + i + "];" + NL());
////            }
////            for (int i = 0; i < numFloatingSpecies; i++)
////            {
////                //if (floatingSpeciesConcentrationList[i].rateRule)
////                //    sb.Append("\t\t_y[" + i.ToString() + "] = oAmounts[" + (i + NumAdditionalRates).ToString() + "];");
////                ////sb.Append("\t\t_y[" + i.ToString() + "] = oAmounts[" + (i+NumAdditionalRates).ToString() + "]/" + convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName) + ";" + NL());
////                //else
////                sb.Append("\t\t_y[" + i + "] = oAmounts[" + (i + NumAdditionalRates) + "]/" +
////                          convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName) + ";" + NL());
////            }
////            sb.Append(NL());
////
////
////            sb.Append("\t\tconvertToAmounts();" + NL());
////
////            sb.Append("\t\t_time = timein;  // Don't remove" + NL());
////
////            sb.Append("\t\tupdateDependentSpeciesValues (_y);" + NL());
////            if (numOfRules > 0)
////                sb.Append("\t\tcomputeRules (_y);" + NL());
////
////            sb.Append("\t\tcomputeReactionRates (time, _y);" + NL());
////            //sb.Append("\t\tcomputeReactionRates (time, _y);" + NL());
////
////
////            // Write out the ODE equations
////            string stoich;
////
////            for (int i = 0; i < numIndependentSpecies; i++)
////            {
////                var eqnBuilder = new StringBuilder(" ");
////                string floatingSpeciesName = independentSpeciesList[i];
////                for (int j = 0; j < numReactions; j++)
////                {
////                    libsbmlcs.Reaction oReaction = NOM.SbmlModel.getReaction(j);
////                    int numProducts = (int)oReaction.getNumProducts();
////                    double productStoichiometry;
////                    for (int k1 = 0; k1 < numProducts; k1++)
////                    {
////                        var product = oReaction.getProduct(k1);
////                        string productName = product.getSpecies();
////                        if (floatingSpeciesName == productName)
////                        {
////                            productStoichiometry = product.getStoichiometry();
////
////                            if (product.isSetId() && product.getLevel() > 2)
////                            {
////                                stoich = "(" +
////                                     substituteTerms(numReactions, "",
////                                        product.getId()) +
////                                     ") * ";
////                            }
////                            else if (product.isSetStoichiometry())
////                            {
////                                if (productStoichiometry != 1)
////                                {
////                                    var denom = product.getDenominator();
////                                    if (denom != 1)
////                                        stoich = String.Format("((double){0}/(double){1})*", WriteDouble(productStoichiometry), denom);
////                                    else
////                                        stoich = WriteDouble(productStoichiometry) + '*';
////                                }
////                                else
////                                {
////                                    stoich = "";
////                                }
////                            }
////                            else
////                            {
////                                if (product.isSetStoichiometryMath() && product.getStoichiometryMath().isSetMath())
////                                {
////                                    stoich = "(" +
////                                             substituteTerms(numReactions, "",
////                                                libsbml.formulaToString(product.getStoichiometryMath().getMath())) +
////                                             ") * ";
////                                }
////                                else
////                                {
////                                    stoich = "";
////                                }
////                            }
////                            eqnBuilder.Append(String.Format(" + {0}_rates[{1}]", stoich, j));
////                        }
////                    }
////
////                    int numReactants = (int)oReaction.getNumReactants();
////                    double reactantStoichiometry;
////                    for (int k1 = 0; k1 < numReactants; k1++)
////                    {
////                        var reactant = oReaction.getReactant(k1);
////                        string reactantName = reactant.getSpecies();
////                        if (floatingSpeciesName == reactantName)
////                        {
////                            reactantStoichiometry = reactant.getStoichiometry();
////
////                            if (reactant.isSetId() && reactant.getLevel() > 2)
////                            {
////                                stoich = String.Format("({0}) * ",
////                                    substituteTerms(numReactions, "",
////                                                                                         reactant.getId()));
////                            }
////                            else if (reactant.isSetStoichiometry())
////                            {
////                                if (reactantStoichiometry != 1)
////                                {
////                                    var denom = reactant.getDenominator();
////                                    if (denom != 1)
////                                        stoich = String.Format("((double){0}/(double){1})*", WriteDouble(reactantStoichiometry), denom);
////                                    else
////                                        stoich = WriteDouble(reactantStoichiometry) +
////                                             "*";
////                                }
////                                else
////                                {
////                                    stoich = "";
////                                }
////                            }
////
////                            else
////                            {
////                                if (reactant.isSetStoichiometryMath() && reactant.getStoichiometryMath().isSetMath())
////                                {
////                                    stoich = "(" +
////                                             substituteTerms(numReactions, "",
////                                                libsbml.formulaToString(reactant.getStoichiometryMath().getMath())) +
////                                             ") * ";
////                                }
////                                else
////                                {
////                                    stoich = "";
////                                }
////
////                            }
////
////                            eqnBuilder.Append(String.Format(" - {0}_rates[{1}]", stoich, j));
////                        }
////                    }
////                }
////
////                var final = eqnBuilder.ToString().Trim();
////
////                if (string.IsNullOrEmpty(final))
////                {
////                    final = "    0.0";
////                }
////
////                if (NOM.SbmlDocument.getLevel() > 2)
////                {
////                    // remember to take the conversion factor into account
////                    string factor = "";
////                    var species = NOM.SbmlModel.getSpecies(floatingSpeciesName);
////                    if (species != null)
////                    {
////                        if (species.isSetConversionFactor())
////                            factor = species.getConversionFactor();
////                        else if (NOM.SbmlModel.isSetConversionFactor())
////                            factor = NOM.SbmlModel.getConversionFactor();
////                    }
////
////                    if (!string.IsNullOrEmpty(factor))
////                    {
////                        final = FindSymbol(factor) + " * (" + final + ")";
////                    }
////                }
////
////
////
////
////
////                // If the floating species has a raterule then prevent the dydt
////                // in the model function from overriding it. I think this is expected behavior.
////                if (!floatingSpeciesConcentrationList[i].rateRule)
////                    sb.Append("\t\t_dydt[" + i + "] = " + final + ";" + NL());
////            }
////
////            sb.Append("\t\tconvertToAmounts ();" + NL());
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private Symbol GetSpecies(string id)
////        {
////            int index;
////            if (floatingSpeciesConcentrationList.find(id, out index))
////                return floatingSpeciesConcentrationList[index];
////            if (boundarySpeciesList.find(id, out index))
////                return boundarySpeciesList[index];
////            return null;
////        }
////        private void WriteEventAssignments(StringBuilder sb, int numReactions, int numEvents)
////        {
////            var delays = new ArrayList();
////            var eventType = new List<bool>();
////            var eventPersistentType = new List<bool>();
////            if (numEvents > 0)
////            {
////                sb.Append("\t// Event assignments" + NL());
////                for (int i = 0; i < numEvents; i++)
////                {
////                    var ev = NOM.getNthEvent(i);
////                    eventType.Add(NOM.getNthUseValuesFromTriggerTime(i));
////                    eventPersistentType.Add(NOM.SbmlModel.getEvent(i).getTrigger().getPersistent());
////                    delays.Add(substituteTerms(numReactions, "", (string)ev[1]));
////                    sb.AppendFormat("\tpublic void eventAssignment_{0} () {{{1}", i, NL());
////                    sb.AppendFormat("\t\tperformEventAssignment_{0}( computeEventAssignment_{0}() );{1}", i, NL());
////                    sb.Append("\t}" + NL());
////                    sb.AppendFormat("\tpublic double[] computeEventAssignment_{0} () {{{1}", i, NL());
////                    var oTemp = new ArrayList();
////                    var oValue = new ArrayList();
////                    int nCount = 0;
////                    int numAssignments = ev.Count - 2;
////                    sb.Append(String.Format("\t\tdouble[] values = new double[ {0}];{1}", numAssignments, NL()));
////                    for (int j = 2; j < ev.Count; j++)
////                    {
////                        var asgn = (ArrayList)ev[j];
////                        //string assignmentVar = substituteTerms(numReactions, "", (string)asgn[0]);
////                        string assignmentVar = FindSymbol((string)asgn[0]);
////                        string str;
////                        var species = GetSpecies(assignmentVar);
////
////
////                        if (species != null && species.hasOnlySubstance)
////                        {
////                            str = string.Format("{0} = ({1}) / {2}", assignmentVar, substituteTerms(numReactions, "", (string)asgn[1]), FindSymbol(species.compartmentName));
////                        }
////                        else
////                        {
////                            str = string.Format("{0} = {1}", assignmentVar, substituteTerms(numReactions, "", (string)asgn[1]));
////                        }
////
////                        string sTempVar = string.Format("values[{0}]", nCount);
////
////                        oTemp.Add(assignmentVar);
////                        oValue.Add(sTempVar);
////
////                        str = sTempVar + str.Substring(str.IndexOf(" = ", System.StringComparison.Ordinal));
////
////                        nCount++;
////
////                        sb.AppendFormat("\t\t{0};{1}", str, NL());
////                    }
////                    sb.Append("\t\treturn values;" + NL());
////                    sb.Append("\t}" + NL());
////                    sb.AppendFormat("\tpublic void performEventAssignment_{0} (double[] values) {{{1}", i, NL());
////
////                    for (int j = 0; j < oTemp.Count; j++)
////                    {
////                        sb.AppendFormat("\t\t{0} = values[{1}];{2}", oTemp[j], j, NL());
////                        if (((string)oTemp[j]).Trim().StartsWith("_c["))
////                        {
////                            sb.Append("\t\tconvertToConcentrations();" + NL());
////                        }
////                    }
////
////                    sb.Append("\t}" + NL());
////                }
////                sb.Append("\t" + NL());
////            }
////
////            sb.AppendFormat("{0}{0}\tprivate void InitializeDelays() {{ {0}", NL());
////            for (int i = 0; i < delays.Count; i++)
////            {
////                sb.AppendFormat("\t\t_eventDelay[{0}] = new TEventDelayDelegate(delegate {{ return {1}; }} );{2}", i, delays[i], NL());
////                sb.AppendFormat("\t\t_eventType[{0}] = {1};{2}", i, (eventType[i] ? "true" : "false"), NL());
////                sb.AppendFormat("\t\t_eventPersistentType[{0}] = {1};{2}", i, (eventPersistentType[i] ? "true" : "false"), NL());
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////
////            sb.AppendFormat("{0}{0}\tpublic void computeEventPriorites() {{ {0}", NL());
////            for (int i = 0; i < numEvents; i++)
////            {
////                var current = NOM.SbmlModel.getEvent(i);
////
////                if (current.isSetPriority() && current.getPriority().isSetMath())
////                {
////                    var priority = libsbml.formulaToString(current.getPriority().getMath());
////                    sb.AppendFormat("\t\t_eventPriorities[{0}] = {1};{2}", i, substituteTerms(numReactions, "", priority), NL());
////                }
////                else
////                {
////                    sb.AppendFormat("\t\t_eventPriorities[{0}] = 0f;{1}", i, NL());
////                }
////            }
////            sb.AppendFormat("\t}}{0}{0}", NL());
////        }
////
////
////        public static string WriteDouble(double value)
////        {
////            if (double.IsNegativeInfinity(value))
////                return "double.NegativeInfinity";
////            if (double.IsPositiveInfinity(value))
////                return "double.PositiveInfinity";
////            if (double.IsNaN(value))
////                return "double.NaN";
////            return value.ToString(STR_DoubleFormat, oInfo);
////        }
////        private void WriteSetParameterValues(StringBuilder sb, int numReactions)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic void setParameterValues ()" + NL());
////            sb.Append("\t{" + NL());
////
////            for (int i = 0; i < globalParameterList.Count; i++)
////                sb.Append(String.Format("\t\t{0} = (double){1};{2}",
////                              convertSymbolToGP(globalParameterList[i].name),
////                              WriteDouble(globalParameterList[i].value),
////                              NL()));
////            // Initialize local parameter values
////            for (int i = 0; i < numReactions; i++)
////            {
////                for (int j = 0; j < localParameterList[i].Count; j++)
////                    sb.Append(String.Format("\t\t_lp[{0}][{1}] = (double){2};{3}",
////                                  i,
////                                  j,
////                                  WriteDouble(localParameterList[i][j].value),
////                                  NL()));
////            }
////
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteSetCompartmentVolumes(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic void setCompartmentVolumes ()" + NL());
////            sb.Append("\t{" + NL());
////            for (int i = 0; i < compartmentList.Count; i++)
////            {
////                sb.Append("\t\t" + convertSymbolToC(compartmentList[i].name) + " = (double)" +
////                          WriteDouble(compartmentList[i].value) + ";" + NL());
////
////                // at this point we also have to take care of all initial assignments for compartments as well as
////                // the assignment rules on compartments ... otherwise we are in trouble :)
////
////                Stack<string> initializations = NOM.GetMatchForSymbol(compartmentList[i].name);
////                while (initializations.Count > 0)
////                {
////                    sb.Append("\t\t" + substituteTerms(_NumReactions, "", initializations.Pop()) + ";" + NL());
////                }
////            }
////
////
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteSetBoundaryConditions(StringBuilder sb)
////        {
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic void setBoundaryConditions ()" + NL());
////            sb.Append("\t{" + NL());
////            for (int i = 0; i < boundarySpeciesList.Count; i++)
////            {
////                if (string.IsNullOrEmpty(boundarySpeciesList[i].formula))
////                    sb.Append("\t\t" + convertSpeciesToBc(boundarySpeciesList[i].name) + " = (double)" +
////                              WriteDouble(boundarySpeciesList[i].value) + ";" + NL());
////                else
////                    sb.Append("\t\t" + convertSpeciesToBc(boundarySpeciesList[i].name) + " = (double)" +
////                              boundarySpeciesList[i].formula + ";" + NL());
////            }
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private void WriteSetInitialConditions(StringBuilder sb, int numFloatingSpecies)
////        {
////            sb.Append("\tpublic void initializeInitialConditions ()" + NL());
////            sb.Append("\t{" + NL());
////            for (int i = 0; i < floatingSpeciesConcentrationList.Count; i++)
////            {
////                if (string.IsNullOrEmpty(floatingSpeciesConcentrationList[i].formula))
////                    sb.Append("\t\t_init" + convertSpeciesToY(floatingSpeciesConcentrationList[i].name) + " = (double)" +
////                              WriteDouble(floatingSpeciesConcentrationList[i].value) + ";" + NL());
////                else
////                    sb.Append("\t\t_init" + convertSpeciesToY(floatingSpeciesConcentrationList[i].name) + " = (double)" +
////                              floatingSpeciesConcentrationList[i].formula + ";" + NL());
////            }
////            sb.Append(NL());
////
////            sb.Append("\t}" + NL() + NL());
////
////            // ------------------------------------------------------------------------------
////            sb.Append("\tpublic void setInitialConditions ()" + NL());
////            sb.Append("\t{" + NL());
////
////            for (int i = 0; i < numFloatingSpecies; i++)
////            {
////                sb.Append("\t\t_y[" + i + "] =  _init_y[" + i + "];" + NL());
////                sb.Append("\t\t_amounts[" + i + "] = _y[" + i + "]*" +
////                          convertCompartmentToC(floatingSpeciesConcentrationList[i].compartmentName) + ";" + NL());
////            }
////            sb.Append(NL());
////
////            sb.Append("\t}" + NL() + NL());
////        }
////
////        private int ReadCompartments()
////        {
////            int numCompartments = NOM.getNumCompartments();
////            for (int i = 0; i < numCompartments; i++)
////            {
////                string sCompartmentId = NOM.getNthCompartmentId(i);
////                double value = NOM.getValue(sCompartmentId);
////                if (double.IsNaN(value)) value = 1;
////                compartmentList.Add(new Symbol(sCompartmentId, value));
////            }
////            return numCompartments;
////        }
////
////        private int ReadModifiableSpeciesReferences()
////        {
////            if (NOM.SbmlDocument.getLevel() < 3) return 0;
////            string id;
////            double value;
////            int numReactions = (int)NOM.SbmlModel.getNumReactions();
////            for (int i = 0; i < numReactions; i++)
////            {
////                var reaction = NOM.SbmlModel.getReaction(i);
////                for (int j = 0; j < reaction.getNumReactants(); j++)
////                {
////                    var reference = reaction.getReactant(j);
////                    id = reference.getId();
////                    if (string.IsNullOrEmpty(id)) continue;
////                    value = reference.getStoichiometry();
////                    if (double.IsNaN(value))
////                        value = 1;
////                    if (reference.isSetId())
////                    {
////                        ModifiableSpeciesReferenceList.Add(new Symbol(id, value));
////                    }
////                }
////                for (int j = 0; j < reaction.getNumProducts(); j++)
////                {
////                    var reference = reaction.getProduct(j);
////                    id = reference.getId();
////                    if (string.IsNullOrEmpty(id)) continue;
////                    value = reference.getStoichiometry();
////                    if (double.IsNaN(value))
////                        value = 1;
////                    if (reference.isSetId())
////                    {
////                        ModifiableSpeciesReferenceList.Add(new Symbol(id, value));
////                    }
////                }
////            }
////            return ModifiableSpeciesReferenceList.Count;
////        }
////        /// Generates the Model Code from the SBML string
////        public string generateModelCode(string sbmlStr)
////        {
////            string sASCII = Encoding.ASCII.GetString(Encoding.ASCII.GetBytes(sbmlStr));
////            Warnings = new List<string>();
////            var sb = new StringBuilder();
////            sASCII = NOM.convertTime(sASCII, "time");
////            NOM.loadSBML(sASCII, "time");
////
////            _ModelName = NOM.getModelName();
////            _NumReactions = NOM.getNumReactions();
////
////            globalParameterList = new SymbolList();
////            ModifiableSpeciesReferenceList = new SymbolList();
////
////            localParameterList = new SymbolList[_NumReactions];
////            reactionList = new SymbolList();
////            boundarySpeciesList = new SymbolList();
////            floatingSpeciesConcentrationList = new SymbolList();
////            floatingSpeciesAmountsList = new SymbolList();
////            compartmentList = new SymbolList();
////            conservationList = new SymbolList();
////            _functionNames = new ArrayList();
////            _functionParameters = new StringCollection();
////
////            StructAnalysis.LoadSBML(sASCII);
////
////            if (RoadRunner._bComputeAndAssignConservationLaws)
////            {
////                _NumIndependentSpecies = StructAnalysis.GetNumIndependentSpecies();
////                independentSpeciesList = StructAnalysis.GetIndependentSpeciesIds();
////                dependentSpeciesList = StructAnalysis.GetDependentSpeciesIds();
////            }
////            else
////            {
////                _NumIndependentSpecies = StructAnalysis.GetNumSpecies();
////                independentSpeciesList = StructAnalysis.GetSpeciesIds();
////                dependentSpeciesList = new string[0];
////            }
////
////            sb.Append("//************************************************************************** " + NL());
////
////            // Load the compartment array (name and value)
////            _NumCompartments = ReadCompartments();
////
////            // Read FloatingSpecies
////            _NumFloatingSpecies = ReadFloatingSpecies();
////
////            _NumDependentSpecies = _NumFloatingSpecies - _NumIndependentSpecies;
////
////            // Load the boundary species array (name and value)
////            _NumBoundarySpecies = ReadBoundarySpecies();
////
////            // Get all the parameters into a list, global and local
////            _NumGlobalParameters = ReadGlobalParameters();
////
////            _NumModifiableSpeciesReferences = ReadModifiableSpeciesReferences();
////
////            // Load up local parameters next
////            ReadLocalParameters(_NumReactions, out _LocalParameterDimensions, out _TotalLocalParmeters);
////
////            _NumEvents = NOM.getNumEvents();
////
////            // Get the L0 matrix
////            double[][] L0 = InitializeL0();
////
////
////            WriteClassHeader(sb);
////
////            WriteOutVariables(sb);
////
////            WriteOutSymbolTables(sb);
////
////            WriteResetEvents(sb, _NumEvents);
////
////            WriteSetConcentration(sb);
////
////            WriteGetConcentration(sb);
////
////            WriteConvertToAmounts(sb);
////
////            WriteConvertToConcentrations(sb);
////
////            WriteProperties(sb);
////
////            WriteAccessors(sb);
////
////            WriteUserDefinedFunctions(sb);
////
////            WriteSetInitialConditions(sb, _NumFloatingSpecies);
////
////            WriteSetBoundaryConditions(sb);
////
////            WriteSetCompartmentVolumes(sb);
////
////            WriteSetParameterValues(sb, _NumReactions);
////
////            WriteComputeConservedTotals(sb, _NumFloatingSpecies, _NumDependentSpecies);
////
////            WriteUpdateDependentSpecies(sb, _NumIndependentSpecies, _NumDependentSpecies, L0);
////
////            int numOfRules = WriteComputeRules(sb, _NumReactions);
////
////            WriteComputeAllRatesOfChange(sb, _NumIndependentSpecies, _NumDependentSpecies, L0);
////
////            WriteComputeReactionRates(sb, _NumReactions);
////
////            WriteEvalModel(sb, _NumReactions, _NumIndependentSpecies, _NumFloatingSpecies, numOfRules);
////
////            WriteEvalEvents(sb, _NumEvents, _NumFloatingSpecies);
////
////            WriteEventAssignments(sb, _NumReactions, _NumEvents);
////
////            WriteEvalInitialAssignments(sb, _NumReactions);
////
////            WriteTestConstraints(sb);
////
////            sb.AppendFormat("}}{0}{0}", NL());
////
////            return sb.ToString();
////        }
////    }
////}
