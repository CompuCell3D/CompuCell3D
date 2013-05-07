using System;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using SBW;
namespace edu_kgi_StructAnalysis { 
public class StructAnalysis  { 
static StructAnalysis ()
{
	SBWLowLevel.registerModuleShutdownListener(new LowLevel.SBWModuleListener(requestInstanceIfNeeded));
}
private static int _nModuleID = new Module("edu.kgi.StructAnalysis" ).ID;
private static int _nServiceID = SBWLowLevel.moduleFindServiceByName(_nModuleID, "StructAnalysis" );
private static int _nMethod0 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string loadSBML(string)");

///<summary>
///Initialization method, takes SBML as input
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string loadSBML(string var0)
{try{
DataBlockWriter oArguments = new DataBlockWriter();
oArguments.add(var0);
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod0, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod1 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string loadSBMLwithTests(string)");

///<summary>
///Initialization method, takes SBML as input
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string loadSBMLwithTests(string var0)
{try{
DataBlockWriter oArguments = new DataBlockWriter();
oArguments.add(var0);
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod1, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod2 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string analyzeWithLU()");

///<summary>
///Uses LU Decomposition for Conservation analysis
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string analyzeWithLU()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod2, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod3 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string analyzeWithLUandRunTests()");

///<summary>
///Uses LU Decomposition for Conservation analysis
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string analyzeWithLUandRunTests()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod3, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod4 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string analyzeWithFullyPivotedLU()");

///<summary>
///Uses fully pivoted LU Decomposition for Conservation analysis
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string analyzeWithFullyPivotedLU()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod4, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod5 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string analyzeWithFullyPivotedLUwithTests()");

///<summary>
///Uses fully pivoted LU Decomposition for Conservation analysis
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string analyzeWithFullyPivotedLUwithTests()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod5, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod6 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getL0Matrix()");

///<summary>
///Returns L0 Matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getL0Matrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod6, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod7 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getNrMatrix()");

///<summary>
///Returns Nr Matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getNrMatrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod7, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod8 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getN0Matrix()");

///<summary>
///Returns N0 Matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getN0Matrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod8, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod9 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getLinkMatrix()");

///<summary>
///Returns L, the Link Matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getLinkMatrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod9, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod10 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getK0Matrix()");

///<summary>
///Returns K0
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getK0Matrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod10, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod11 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getNullSpace()");

///<summary>
///Returns Nullspace
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getNullSpace()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod11, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod12 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getReorderedSpecies()");

///<summary>
///Returns the reordered list of species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getReorderedSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod12, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod13 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getReorderedSpeciesNamesList()");

///<summary>
///Returns the actual names of the reordered species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getReorderedSpeciesNamesList()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod13, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod14 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getIndependentSpecies()");

///<summary>
///Returns the list of independent species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getIndependentSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod14, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod15 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getIndependentSpeciesNamesList()");

///<summary>
///Returns the actual names of the independent species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getIndependentSpeciesNamesList()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod15, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod16 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getDependentSpecies()");

///<summary>
///Returns the list of dependent species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getDependentSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod16, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod17 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getDependentSpeciesNamesList()");

///<summary>
///Returns the actual names of the dependent species 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getDependentSpeciesNamesList()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod17, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod18 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getReactions()");

///<summary>
///Returns the list of Reactions 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getReactions()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod18, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod19 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getReactionsNamesList()");

///<summary>
///Returns actual names of the Reactions 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getReactionsNamesList()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod19, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod20 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getConservationLawArray()");

///<summary>
///Returns Gamma, the conservation law array 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getConservationLawArray()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod20, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod21 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] getConservedEntities()");

///<summary>
///Returns algebraic expressions for conserved cycles 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] getConservedEntities()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod21, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod22 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[] getConservedSums()");

///<summary>
///Returns values for conserved cycles using Initial conditions 
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[] getConservedSums()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod22, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod23 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "{} getInitialConditions()");

///<summary>
///Returns Initial Conditions used in the model
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static ArrayList getInitialConditions()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (ArrayList) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod23, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod24 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getStoichiometryMatrix()");

///<summary>
///Returns the original stoichiometry matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getStoichiometryMatrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod24, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod25 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double[][] getReorderedStoichiometryMatrix()");

///<summary>
///Returns reordered stoichiometry matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double[][] getReorderedStoichiometryMatrix()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double[][]) HighLevel.convertArray(SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod25, oArguments).getObject());
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod26 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string[] testConservationLaws()");

///<summary>
///Tests if conservation laws are correct
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string[] testConservationLaws()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string[]) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod26, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod27 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getTestDetails()");

///<summary>
///Return Details about conservation tests
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getTestDetails()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod27, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod28 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getModelName()");

///<summary>
///Returns the name of the model
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getModelName()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod28, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod29 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "int getNumSpecies()");

///<summary>
///Returns the total number of species
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static int getNumSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (int) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod29, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod30 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "int getNumIndSpecies()");

///<summary>
///Returns the number of independent species
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static int getNumIndSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (int) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod30, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod31 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "int getNumDepSpecies()");

///<summary>
///Returns the number of dependent species
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static int getNumDepSpecies()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (int) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod31, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod32 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "int getNumReactions()");

///<summary>
///Returns the total number of reactions
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static int getNumReactions()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (int) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod32, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod33 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "int getRank()");

///<summary>
///Returns rank of stoichiometry matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static int getRank()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (int) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod33, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod34 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "double getNmatrixSparsity()");

///<summary>
///Returns the number of nonzero values in Stoichiometry matrix
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static double getNmatrixSparsity()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (double) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod34, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod35 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "void setTolerance(double)");

///<summary>
///Set user specified tolerance
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static void setTolerance(double var0)
{try{
DataBlockWriter oArguments = new DataBlockWriter();
oArguments.add(var0);
SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod35, oArguments);
 return;
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod36 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getName()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getName()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod36, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod37 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getVersion()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getVersion()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod37, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod38 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getAuthor()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getAuthor()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod38, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod39 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getDescription()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getDescription()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod39, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod40 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getDisplayName()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getDisplayName()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod40, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod41 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getCopyright()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getCopyright()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod41, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}
private static int _nMethod42 = SBWLowLevel.serviceGetMethod( _nModuleID, _nServiceID, "string getURL()");

///<summary>
///1.06.9
///</summary>
[System.Diagnostics.DebuggerStepThrough(), System.Diagnostics.DebuggerHidden()]
public static string getURL()
{try{
DataBlockWriter oArguments = new DataBlockWriter();
return (string) SBWLowLevel.methodCall(_nModuleID, _nServiceID, _nMethod42, oArguments).getObject();
}
 catch(SBWException e) {
throw e;}}

	public static void requestInstanceIfNeeded(int nModuleId)
	{
		if (nModuleId == _nModuleID)
			_nModuleID = new Module("edu.kgi.StructAnalysis" ).ID;
	}

}
}