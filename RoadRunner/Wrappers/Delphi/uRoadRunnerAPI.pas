unit uRoadRunnerAPI;

{ Copyright 2012 Herbert M Sauro

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   In plain english this means:

   You CAN freely download and use this software, in whole or in part, for personal,
   company internal, or commercial purposes;

   You CAN use the software in packages or distributions that you create.

   You SHOULD include a copy of the license in any redistribution you may make;

   You are NOT required include the source of software, or of any modifications you may
   have made to it, in any redistribution you may assemble that includes it.

   YOU CANNOT:

   redistribute any piece of this software without proper attribution;
}

interface

Uses SysUtils, Classes, Windows, uMatrix, Generics.Collections, IOUtils, uRRList,
     uRRTypes, uSBWArray;

{
C_DECL_SPEC bool                    rrCallConv  setLogLevelFromString(const char* lvl);
C_DECL_SPEC bool                    rrCallConv  getLogLevel(int& lvl);
C_DECL_SPEC char*                   rrCallConv  getLogFileName();

C_DECL_SPEC char*                   rrCallConv  getBuildDate();
C_DECL_SPEC char*                   rrCallConv  getCopyright();
C_DECL_SPEC bool                    rrCallConv  setTempFolder(const char* folder);
C_DECL_SPEC char*                   rrCallConv  getTempFolder();
}


type
  TAnsiCharArray = array[0..100000] of AnsiChar;
  PAnsiCharArray = ^TAnsiCharArray;

  TAnsiCharArrayArray = array[0..100000] of PAnsiCharArray;  // Array of char*
  PAnsiCharArrayArray = ^TAnsiCharArrayArray;

  TRRStringArray = record
    count : integer;
    strList : PAnsiCharArrayArray;
  end;
  PRRStringArray = ^TRRStringArray;


  TPointerFunc = function : Pointer; stdcall;
  TVoidCharFunc = function : PAnsiChar; stdcall;   //char* func(void)
  THandleCharFunc = function (handle : Pointer) : PAnsiChar; stdcall;   //char* func(void)
  TVoidBoolFunc = function (rrHandle : Pointer) : boolean; stdcall; // bool func (void);
  TVoidIntFunc = function : integer; stdcall;
  THandleIntFunc = function (handle : Pointer) : integer; stdcall;
  TVoidDoubleFunc = function : double; stdcall;

  TBoolBoolFunc = function (var value : boolean) : boolean; stdcall;

  THandlePointerFunc = function (rrHandle : Pointer) : Pointer; stdcall; //void* func(void)
  TCharBoolFunc = function (rrHandle : Pointer; str : PAnsiChar) : bool; stdcall;  // bool func (char *)
  TDoubleBoolFunc = function (value : double) : bool; stdcall; // bool func (double)
  TIntBoolFunc = function (value : integer) : bool; stdcall;   // bool func (double)
  TVarIntBoolFunc = function (var value : integer) : bool; stdcall;   // bool func (double)
  THandleIntDoubleFunc = function (rrHandle : Pointer; index : integer) : double; stdcall;

  THandleStringListFunc = function(rrHandle : Pointer) : PRRStringArray; stdcall;

  TGetCopyright = TVoidCharFunc;
  TCreateRRInstance = TPointerFunc;
  TGetCCode = function : PRRCCodeHandle; stdcall;
  TSetTimeStart = function (rrHandle : Pointer; value : double) : bool; stdcall;
  TSetTimeEnd = function (rrHandle : Pointer; value : double) : bool; stdcall;
  TSetNumPoints = function (rrHandle : Pointer; value : integer) : bool; stdcall;
  TSimulateEx = function (rrHandle : Pointer; timeStart : double; timeEnd : double; numberOfPoints : integer) : PRRResultHandle; stdcall;
  TGetMatrix = function (rrHandle : Pointer) : PRRMatrixHandle; stdcall;
  TGetSetMatrix = function (mat : PRRMatrixHandle) : PRRMatrixHandle; stdcall;
  TFreeRRResult = function (ptr : PRRResultHandle) : boolean; stdcall;
  TFreeRRInstance = procedure (instance : Pointer); stdcall;
  THandleVectorFunc = function (rrHandle : Pointer) : PRRDoubleVectorHandle; stdcall;

  TSetSelectionList = function (rrHandle : Pointer; list : PAnsiChar) : bool; stdcall;
  TGetValue = function (rrHandle : Pointer; speciesId : PAnsiChar; var value : double) : boolean; stdcall;
  TSetValue = function (rrHandle : Pointer; speciesId : PAnsiChar; value : double) : bool; stdcall;
  TGetReactionIds = THandlePointerFunc;
  TReset = function : bool; stdcall;
  TFreeStringArray = function (handle : PRRStringArray) : boolean; stdcall;
  TFreeRRMatrix = function (matrix : PRRMatrixHandle) : boolean; stdcall;
  TFreeRRDoubleVector = function (vector : PRRDoubleVectorHandle) : boolean ; stdcall;
  TOneStep = function (rrHandle : Pointer; var currentTime : double; stepSize : double) : double; stdcall;
  TSteadyState = function (rrHandle : Pointer; var value : double) : boolean; stdcall;
  TGetMCA = function (rrHandle : Pointer; variable : PAnsiChar; parameter : PAnsiChar; var value : double) : boolean; stdcall;

var
   DLLLoaded : boolean;
   selectionList : AnsiString;
   loggingEnabled : boolean = false;
   loggingTmpFileName : AnsiString = '';
   rrHandle : Pointer;
   internalRRHandle : Pointer;

function loadIntoMatrix (matrix : T2DDoubleArray) : TMatrix;  overload;


function  hasError : boolean;
function  createRRInstance : Pointer;
function  createInternalRRInstance : Pointer;
procedure freeRRInstance; overload;
procedure freeRRInstance (myInstance : Pointer); overload;
function  getLastError : AnsiString;
function  getBuildDate : AnsiString;
function  getVersion : AnsiString;



{$REGION 'Documentation'}
///	<summary>
///	  Get copyright string
///	</summary>
///	<returns>
///	  Copyright string
///	</returns>
{$ENDREGION}
function  getCopyright : AnsiString;
function  getTempFolder : AnsiString;
function  enableLoggingToFile : boolean;
function  setLogLevel (debugLevel : AnsiString) : boolean;
function  getLogFileName : AnsiString;
function  setTempFolder (name : AnsiString) : boolean;

{$REGION 'Documentation'}
///	<summary>
///	  Returns the generated C Code for the model
///	</summary>
{$ENDREGION}
function  getCCode : TRRCCode;

function  loadSBML (sbmlStr : AnsiString) : boolean;
function  loadSBMLFromFile (fileName : AnsiString) : boolean;
function  getSBML : AnsiString;

function  getValue (Id : AnsiString) : double;
function  setValue (Id : AnsiString; value : double) : boolean;
function  reset : boolean;
function  setFloatingSpeciesInitialConcentrations (value : TDoubleArray) : boolean;

procedure setTimeStart (value : double);
procedure setTimeEnd (value : double);
procedure setNumberOfPoints (value : integer);

function  simulate : T2DDoubleArray;
function  mSimulate : TMatrix;
function  simulateEx (timeStart: double; timeEnd : double; numberOfPoints : integer)  : TMatrix;
function  oneStep (var currentTime : double; var stepSize : double) : double;
function  setTimeCourseSelectionList (strList : TStringList) : boolean;
function  getTimeCourseSelectionList: TRRList;
function  setCapabilities (str : AnsiString) : boolean;
function  getCapabilities : AnsiString;

function  evalModel : boolean;
function  getFullJacobian : T2DDoubleArray;
function  getReducedJacobian : T2DDoubleArray;
function  mGetReducedJacobian : TMatrix;

function  getStoichiometryMatrix : T2DDoubleArray;
function  mGetStoichiometryMatrix : TMatrix;
function  getLinkMatrix : T2DDoubleArray;
function  getNrMatrix : T2DDoubleArray;
function  getL0Matrix : T2DDoubleArray;
function  getConservationMatrix : T2DDoubleArray;
function  mGetConservationMatrix : TMatrix;

function  getReactionRates : TDoubleArray;
function  getRatesOfChange : TDoubleArray;

function  getCompartmentIds : TStringList;
function  getReactionIds : TStringList;
function  getBoundarySpeciesIds : TStringList;
function  getFloatingSpeciesIds : TStringList;
function  getGlobalParameterIds : TStringList;
function  getRatesOfChangeIds : TStringList;
function  getEigenvalueIds : TStringList;
function  getElasticityIds : TStringList;

function  getNumberOfReactions : integer;
function  getNumberOfBoundarySpecies : integer;
function  getNumberOfFloatingSpecies : integer;
function  getNumberOfGlobalParameters : integer;
function  getNumberOfCompartments : integer;

function  setCompartmentByIndex     (index : integer; value : double) : boolean;
function  setFloatingSpeciesByIndex (index : integer; value : double) : boolean;
function  setBoundarySpeciesByIndex (index : integer; value : double) : boolean;
function  setGlobalParameterByIndex (index : integer; value : double) : boolean;

function  getCompartmentByIndex     (index : integer) : double;
function  getFloatingSpeciesByIndex (index : integer) : double;
function  getBoundarySpeciesByIndex (index : integer) : double;
function  getGlobalParameterByIndex (index : integer) : double;

function  getFloatingSpeciesConcentrations : TDoubleArray;
function  getBoundarySpeciesConcentrations : TDoubleArray;

function  getNumberOfDependentSpecies : integer;
function  getNumberOfIndependentSpecies : integer;

function  steadyState : double;
function  computeSteadyStateValues : TDoubleArray;
function  setSteadyStateSelectionList (strList : TStringList) : boolean;

function  getEigenvalues : T2DDoubleArray;
function  mGetEigenvalues : TMatrix;
function  getEigenvaluesMatrix (m : T2DDoubleArray) : T2DDoubleArray;

function  getuCC (variable : AnsiString; parameter : AnsiString) : double;
function  getCC (variable : AnsiString; parameter : AnsiString) : double;
function  getuEE (variable : AnsiString; parameter : AnsiString) : double;
function  getEE (variable : AnsiString; parameter : AnsiString) : double;

function 	getUnscaledFluxControlCoefficientMatrix : T2DDoubleArray;
function  getScaledFluxControlCoefficientMatrix : T2DDoubleArray;

function  getAvailableSymbolsII : TRRList;
function  getAvailableTimeCourseSymbols : TRRList;
function  getAvailableSteadStateSymbols : TRRList;

function  setComputeAndAssignConservationLaws (value : boolean) : boolean;

procedure setRoadRunnerLibraryName (newLibName : AnsiString);
function  loadRoadRunner (var errMsg : AnsiString; methodList : TStringList) : boolean;
procedure releaseRoadRunnerLibrary;

procedure setMatrixElement ();

implementation

type
  TLibGetAvailableSymbols = function (rrHandle : Pointer) : PRRListRecordHandle; stdcall;
  TlibSetInitialConditions = function (rrHandle : Pointer; vec : PRRDoubleVectorHandle) : bool; stdcall;
  TlibComputeSteadyStateValues = function (rrHandle : Pointer): PRRDoubleVectorHandle;


var DLLHandle : Cardinal;
    libName : AnsiString = 'rrc_API.dll';
    instance : Pointer = nil;

    libLoadSBML : TCharBoolFunc;
    libLoadSBMLFromFile : TCharBoolFunc;
    libGetSBML : TVoidCharFunc;

    libHasError : TVoidBoolFunc;
    libGetLastError : TVoidCharFunc;
    libEnableLoggingToConsole : TVoidBoolFunc;
    libEnableLoggingToFile : TVoidBoolFunc;
    libSetLogLevel : function (str : PAnsiChar) : bool; stdcall;
    libGetLogFileName : function : PAnsiChar; stdcall;
    libSetTempFolder : function (rrHandle : Pointer; folder : PAnsiChar) : bool; stdcall;

    libGetBuildDate : TVoidCharFunc;
    libGetVersion : TVoidCharFunc;
    libGetCopyright : TGetCopyright;
    libGetTempFolder : TVoidCharFunc;
    libGetCCode : TGetCCode;

    libCreateRRInstance : TCreateRRInstance;
    libFreeRRInstance : TFreeRRInstance;
    libFreeResult : TFreeRRResult;

    libSimulate : THandlePointerFunc;
    libSimulateEx : TSimulateEx;
    libGetValue : TGetValue;
    libSetValue : TSetValue;
    libSetTimeCourseSelectionList : TSetSelectionList;
    libGetReactionIds : TGetReactionIds;
    libReset : TReset;
    libSetFloatingSpeciesInitialConcentrations : function (rrHandle : Pointer; value : Pointer) : boolean; stdcall;
    libGetCapabilities : THandleCharFunc;
    libSetCapabilities : TCharBoolFunc;
    libEvalModel : TVoidBoolFunc;
    libGetFullJacobian : function : PRRMatrixHandle;
    libGetReducedJacobian : function : PRRMatrixHandle;

    libSetTimeStart : TSetTimeStart;
    libSetTimeEnd : TSetTimeEnd;
    libSetNumberOfPoints : TSetNumPoints;

    libGetNumberOfReactions : THandleIntFunc;
    libGetNumberOfBoundarySpecies : THandleIntFunc;
    libGetNumberOfFloatingSpecies : THandleIntFunc;
    libGetNumberOfGlobalParameters : THandleIntFunc;
    libGetNumberOfCompartments : THandleIntFunc;

    libSetCompartmentByIndex     : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libSetFloatingSpeciesByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libSetBoundarySpeciesByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libSetGlobalParameterByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;

    libGetCompartmentByIndex     : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libGetGlobalParameterByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libGetFloatingSpeciesByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;
    libGetBoundarySpeciesByIndex : function (rrHandle : Pointer; var index : integer; var value : double) : boolean; stdcall;

    libGetFloatingSpeciesConcentrations : function (rrHandle : Pointer) : PRRDoubleVectorHandle; stdcall;
    libGetBoundarySpeciesConcentrations : function (rrHandle : Pointer) : PRRDoubleVectorHandle; stdcall;

    libSetFloatingSpeciesConcentrations : function (rrHandle : Pointer; values : PRRDoubleVectorHandle) : boolean; stdcall;
    libSetBoundarySpeciesConcentrations : function (rrHandle : Pointer; values : PRRDoubleVectorHandle) : boolean; stdcall;

    libGetNumberOfDependentSpecies : function (rrHandle : Pointer) : integer; stdcall;
    libGetNumberOfIndependentSpecies : function (rrHandle : Pointer) : integer; stdcall;

    libSteadyState : TSteadyState;
    libGetReactionRate : THandleIntDoubleFunc;
    libGetReactionRates : THandleVectorFunc;
    libGetRatesOfChange : THandleVectorFunc;
    libOneStep : TOneStep;

    libGetCompartmentIds     : THandleStringListFunc;
    libGetBoundarySpeciesIds : THandleStringListFunc;
    libGetFloatingSpeciesIds : THandleStringListFunc;
    libGetGlobalParameterIds : THandleStringListFunc;
    libGetRatesOfChangeIds   : THandleStringListFunc;
    libGetEigenvalueIds      : THandleStringListFunc;
    libGetElasticityIds      : THandleStringListFunc;

    libSetSteadyStateSelectionList : TCharBoolFunc;
    libGetSteadyStateSelectionList : function (internalRRHandle : Pointer) : PRRListRecordHandle; stdcall;
    libGetTimeCourseSelectionList  : function (internalRRHandle : Pointer) : PRRListRecordHandle; stdcall;
    libGetAvailableTimeCourseSymbols : TLibGetAvailableSymbols;

    libGetAvailableSteadyStateSymbols : TLibGetAvailableSymbols;
    libComputeSteadyStateValues : TlibComputeSteadyStateValues;
    libSetInitialConditions : TlibSetInitialConditions;
    libSetComputeAndAssignConservationLaws : TBoolBoolFunc;

    libGetStoichiometryMatrix : TGetMatrix;
    libGetLinkMatrix          : TGetMatrix;
    libGetNrMatrix            : TGetMatrix;
    libGetL0Matrix            : TGetMatrix;
    libGetConservationMatrix  : TGetMatrix;

    libgetuFCC                : TGetMatrix;
    libgetFCC                 : TGetMatrix;

    libgetuCC                 : TGetMCA;
    libgetuEE                 : TGetMCA;
    libgetCC                  : TGetMCA;
    libgetEE                  : TGetMCA;
    libGetEigenvalues         : TGetMatrix;
    libGetEigenvaluesMatrix   : TGetSetMatrix;


    libCreateVector : function (size : integer) : PRRDoubleVectorHandle;  stdcall;
    libCreateRRMatrix : function (row, col : integer) : PRRMatrixHandle; stdcall;
    libSetMatrixElement : function (m : PRRMatrixHandle; r, c : integer; value : double) : boolean; stdcall;
    libGetListItem : function (list : pointer; index : integer) : PRRListItemRecord; stdcall;


    libFreeStringArray : TFreeStringArray;
    libFreeMatrix : TFreeRRMatrix;
    libFreeText : TCharBoolFunc;
    libFreeDoubleVector : TFreeRRDoubleVector;


// Utility Routines
// --------------------------------------------------------------
function getArrayOfStrings (pList: PRRStringArray) : TStringList;
var nStrings : integer;
    i, j : integer;
    element : PAnsiCharArray;
    str : AnsiString;
begin
  nStrings := pList^.count;
  result := TStringList.Create;
  for i := 0 to nStrings - 1 do
      begin
      element := pList^.strList[i];
      j := 0; str := '';
      while element[j] <> #0 do
          begin
          str := str + element[j];
          inc (j);
          end;
      result.Add (str);
      end;
end;


function loadIntoMatrix (matrix : PRRMatrixHandle) : TMatrix; overload;
var nr, nc : integer;
    i, j : integer;
begin
  nr := matrix^.RSize;
  nc := matrix^.CSize;
  result := TMatrix.Create (nr, nc);
  for i := 0 to nr - 1 do
      for j := 0 to nc - 1 do
          result[i+1,j+1] := matrix^.data[i*nc + j];
end;


function loadIntoRRMatrix  (mat : T2DDoubleArray) : PRRMatrixHandle;
var i, j : integer;
    r, c : integer;
    str : AnsiString;
begin
  r := getRows (mat); c := getColumns(mat);
  result := libCreateRRMatrix (r, c);
  if result = nil then
     exit;

  for i := 0 to r - 1 do
      for j := 0 to c - 1 do
          if libSetMatrixElement (result, i, j, mat[i,j]) = False then
             begin
             str := getLastError();
             raise Exception.Create ('Error while calling setMatrixElement: ' + str);
             end;
end;

function loadInTo2DArray (matrix : PRRMatrixHandle) : T2DDoubleArray;
var nr, nc : integer;
    i, j : integer;
begin
  nr := matrix^.RSize;
  nc := matrix^.CSize;
  setLength (result, nr, nc);
  for i := 0 to nr - 1 do
      for j := 0 to nc - 1 do
          result[i,j] := matrix^.data[i*nc + j];
end;


function extractList (list : PRRListRecordHandle) : TRRList;
var i : integer;
    item : PRRListItemRecord;
begin
  result := TRRList.Create;
  for i := 0 to list^.count - 1 do
      begin
      item := libGetListItem (list, i);
      case item^.ItemType of
        litList : result.Add (TRRListItem.Create (extractList (item^.lValue)));
        litString : result.Add (TRRListItem.Create (AnsiString (item^.sValue)));
        litInteger : result.Add (TRRListItem.Create (item^.iValue));
        litDouble : result.Add (TRRListItem.Create (item^.dValue));
      end;
  end;
end;


function loadIntoMatrix (matrix : T2DDoubleArray) : TMatrix;
var nr, nc : integer;
    i, j : integer;
begin
  nr := getRows (matrix);
  nc := getColumns (matrix);
  result := TMatrix.Create (nr, nc);
  for i := 0 to nr - 1 do
      for j := 0 to nc - 1 do
          result[i+1,j+1] := matrix[i, j];
end;


// -----------------------------------------------------------------
// For doumentation, see the C API docs at:
//      http://code.google.com/p/roadrunnerwork/
// -----------------------------------------------------------------

function createRRInstance : Pointer;
begin
  result := libCreateRRInstance;
end;

function createInternalRRInstance : Pointer;
begin
  internalRRHandle := libCreateRRInstance;
end;


procedure freeRRInstance (myInstance : Pointer);
begin
  if myInstance <> nil then
     libFreeRRInstance (myInstance);
end;

procedure freeRRInstance;
begin
  if instance <> nil then
     libFreeRRInstance (instance);
end;


function getBuildDate : AnsiString;
begin
  result := libGetBuildDate;
end;


function getVersion : AnsiString;
begin
  result := libGetVersion;
end;


function getCopyright : AnsiString;
var p : PAnsiChar;
begin
  p := libGetCopyright();
  result := AnsiString (p);
end;


function  getTempFolder : AnsiString;
begin
  result := libGetTempFolder;
end;

function hasError : boolean;
begin
  result := libHasError (internalRRHandle);
end;


function getLastError : AnsiString;
begin
  result := libGetLastError;
end;

function enableLoggingToFile : boolean;
begin
  result := libEnableLoggingToFile (internalRRHandle);
  loggingEnabled := true;
end;


function setLogLevel (debugLevel : AnsiString) : boolean;
begin
  result := libSetLogLevel (PAnsiChar (debugLevel));
end;


function getLogFileName : AnsiString;
begin
  result := libGetLogFileName;
end;


function setTempFolder (name : AnsiString) : boolean;
begin
  result := libSetTempFolder (internalRRHandle, PAnsiChar (name));
end;


function getCCode : TRRCCode;
var p : PRRCCodeHandle;
begin
  p := libGetCCode;
  result.Header := p^.Header;
  result.Source := p^.Source;
end;


function setComputeAndAssignConservationLaws (value : boolean) : boolean;
begin
  result := libSetComputeAndAssignConservationLaws (value);
end;


function loadSBML (sbmlStr : AnsiString) : boolean;
begin
  result := libLoadSBML (internalRRHandle, PAnsiChar (sbmlStr));
end;


function loadSBMLFromFile (fileName : AnsiString) : boolean;
var str : AnsiString;
begin
  if FileExists (fileName) then
     begin
     result := libLoadSBMLFromFile (internalRRHandle, PAnsiChar (fileName));
     end
  else
     raise Exception.Create ('Unable to locate SBML file [' + fileName + ']');
end;


function getSBML : AnsiString;
begin
  result := libGetSBML;
end;


function getValue (Id : AnsiString) : double;
begin
  if not libGetValue (internalRRHandle, PAnsiChar (Id), result) then
     raise Exception.Create ('Error in getVlaue');
end;


function setValue (Id : AnsiString; value : double) : boolean;
begin
  result := libSetValue (internalRRHandle, PAnsiChar (Id), value);
end;


function reset : boolean;
begin
  result := libReset;
end;


function setFloatingSpeciesInitialConcentrations (value : TDoubleArray) : boolean;
var p : PRRDoubleVectorHandle;  i : integer;
begin
 p := libCreateVector (length (value));
 for i := 0 to length (value) - 1 do
     p^.data[i] := value[i];
 result := libSetFloatingSpeciesInitialConcentrations (internalRRHandle, p);
 libFreeDoubleVector (p);
end;


function getCapabilities : AnsiString;
begin
  result := libGetCapabilities (internalRRHandle);
end;


function setCapabilities (str : AnsiString) : boolean;
begin
  result := libSetCapabilities (internalRRHandle, PAnsiChar (str));
end;


function evalModel : boolean;
begin
  result := libEvalModel (internalRRHandle);
end;


function getFullJacobian : T2DDoubleArray;
var p : PRRMatrixHandle;
begin
  p := libGetFullJacobian;
  if p = nil then
     raise Exception.Create ('No Jacobian matrix');
  try
    result := loadInTo2DArray (p);
  finally
    libFreeMatrix (p);
  end;
end;


function getReducedJacobian : T2DDoubleArray;
var p : PRRMatrixHandle;
begin
  p := libGetReducedJacobian;
  try
    result := loadInTo2DArray (p);
  finally
    libFreeMatrix (p);
  end;
end;


function mGetReducedJacobian : TMatrix;
var p : PRRMatrixHandle;
begin
  p := libGetReducedJacobian;
  try
    result := loadIntoMatrix(p);
  finally
    libFreeMatrix (p);
  end;
end;

function setTimeCourseSelectionList (strList : TStringList) : boolean;
var i : integer;
begin
  if strList.Count = 0 then
     exit;

  selectionList := strList[0];
  for i := 1 to strList.Count - 1 do
      selectionList := selectionList + ' ' + strList[i];
  if not libSetTimeCourseSelectionList (internalRRHandle, PAnsiChar (selectionList)) then
     raise Exception.Create ('Error calling setSelectionList');
end;


function getTimeCourseSelectionList: TRRList;
var ptr : PRRListRecordHandle;
begin
  ptr := libGetTimeCourseSelectionList (internalRRHandle);
  result := extractList (ptr);
end;


procedure setTimeStart (value : double);
begin
  if not libSetTimeStart (internalRRHandle, value) then
     raise Exception.Create ('Error while calling setTimeStart');
end;


procedure setTimeEnd (value : double);
begin
  if not libSetTimeEnd (internalRRHandle, value) then
     raise Exception.Create ('Error while calling setTimeEnd');
end;


procedure setNumberOfPoints (value : integer);
begin
  if not libSetNumberOfPoints (internalRRHandle, value) then
     raise Exception.Create ('Error while calling setNumberOfPoints');
end;



function simulate : T2DDoubleArray;
var RRResult : PRRResultHandle;
    i, j : integer;
    nr, nc : integer;
begin
  RRResult := libSimulate (internalRRHandle);

  if RRResult = nil then
     raise Exception.Create (getLastError());
  try
     nr := RRResult^.RSize;
     nc := RRResult^.CSize;
     setLength (result, nr, nc);
     for i := 0 to nr - 1 do
         for j := 0 to nc - 1 do
             result[i,j] := RRResult^.data[i*nc + j];
  finally
    libFreeResult (RRResult);
  end;
end;


function mSimulate : TMatrix;
var RRResult : PRRResultHandle;
    i, j : integer;
    nr, nc : integer;
begin
  RRResult := libSimulate (internalRRHandle);

  if RRResult = nil then
     raise Exception.Create (getLastError());
  try
     nr := RRResult^.RSize;
     nc := RRResult^.CSize;
     result := TMatrix.Create (nr, nc);
     for i := 0 to nr - 1 do
         for j := 0 to nc - 1 do
             result[i+1,j+1] := RRResult^.data[i*nc + j];
  finally
    libFreeResult (RRResult);
  end;
end;


function simulateEx (timeStart: double; timeEnd : double; numberOfPoints : integer)  : TMatrix;
var RRResult : PRRResultHandle;
    i, j : integer;
    nr, nc : integer;
begin
  RRResult := libSimulateEx (internalRRHandle, timeStart, timeEnd, numberOfPoints);
  if RRResult = nil then
     raise Exception.Create (getLastError());
  try
     nr := RRResult^.RSize;
     nc := RRResult^.CSize;
     result := TMatrix.Create (nr, nc);
     for i := 0 to nr - 1 do
         for j := 0 to nc - 1 do
             result[i+1,j+1] := RRResult^.data[i*nc + j];
  finally
    libFreeResult (RRResult);
  end;
end;


function oneStep (var currentTime : double; var stepSize : double) : double;
begin
  result := libOneStep (internalRRHandle, currentTime, stepSize);
end;


function getReactionIds : TStringList;
var pList : PRRStringArray;
begin
  pList := libGetReactionIds (internalRRHandle);
  if pList <> nil then
     try
       result := getArrayOfStrings(pList);
     finally
       libFreeStringArray (pList);
     end
  else
     result := TStringList.Create;
end;


function getNumberOfReactions : integer;
begin
  result := libGetNumberOfReactions (internalRRHandle);
end;

function getNumberOfBoundarySpecies : integer;
begin
  result := libGetNumberOfBoundarySpecies (internalRRHandle);
end;


function getBoundarySpeciesIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetBoundarySpeciesIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings(p);
  finally
    libFreeStringArray (p);
  end;
end;


function getFloatingSpeciesIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetFloatingSpeciesIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings(p);
  finally
    libFreeStringArray (p);
  end;
end;


function getGlobalParameterIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetGlobalParameterIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings (p);
  finally
    libFreeStringArray (p);
  end;
end;


function getFloatingSpeciesConcentrations : TDoubleArray;
var p : PRRDoubleVectorHandle; i : integer;
begin
  p := libGetFloatingSpeciesConcentrations (internalRRHandle);
  try
    setLength (result, p^.count);
    for i := 0 to p^.count - 1 do
        result[i] := p^.data[i];
  finally
    libFreeDoubleVector (p);
  end;
end;


function getBoundarySpeciesConcentrations : TDoubleArray;
var p : PRRDoubleVectorHandle; i : integer;
begin
  p := libGetBoundarySpeciesConcentrations (internalRRHandle);
  try
    setLength (result, p^.count);
    for i := 0 to p^.count - 1 do
        result[i] := p^.data[i];
  finally
    if p^.count > 0 then
       libFreeDoubleVector (p);
  end;
end;


function getRatesOfChangeIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetRatesOfChangeIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings (p);
  finally
    libFreeStringArray (p);
  end;
end;


function getEigenvalueIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetEigenvalueIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings (p);
  finally
    libFreeStringArray (p);
  end;
end;


function getElasticityIds : TStringList;
var p : PRRStringArray;
begin
  p := libGetElasticityIds (internalRRHandle);
  try
    if p = nil then
       result := TStringList.Create
    else
       result := getArrayOfStrings (p);
  finally
    libFreeStringArray (p);
  end;
end;

function getNumberOfFloatingSpecies : integer;
begin
  result := libGetNumberOfFloatingSpecies (internalRRHandle);
end;


function getNumberOfGlobalParameters : integer;
begin
  result := libGetNumberOfGlobalParameters (internalRRHandle);
end;


function getNumberOfCompartments : integer;
begin
  result := libGetNumberOfCompartments (internalRRHandle);
end;


function getCompartmentIds : TStringList;
var pList : PRRStringArray;
begin
  pList := libGetCompartmentIds (internalRRHandle);
  if pList <> nil then
     try
       result := getArrayOfStrings(pList);
     finally
       libFreeStringArray (pList);
     end
  else
    result := TStringList.Create;
end;


function setCompartmentByIndex (index : integer; value : double) : boolean;
begin
  result := libSetCompartmentByIndex (internalRRHandle, index, value);
end;


function setFloatingSpeciesByIndex (index : integer; value : double) : boolean;
begin
  result := libSetFloatingSpeciesByIndex (internalRRHandle, index, value);
end;


function setBoundarySpeciesByIndex (index : integer; value : double) : boolean;
begin
  result := libSetBoundarySpeciesByIndex (internalRRHandle, index, value);
end;


function setGlobalParameterByIndex (index : integer; value : double) : boolean;
begin
  result := libSetGlobalParameterByIndex (internalRRHandle, index, value);
end;


function getCompartmentByIndex (index : integer) : double;
begin
  if not libGetCompartmentByIndex (internalRRHandle, index, result) then
     raise Exception.Create ('Index out of range in getCompartmentByIndex');
end;


function getFloatingSpeciesByIndex (index : integer) : double;
begin
  if not libGetFloatingSpeciesByIndex (internalRRHandle, index, result) then
     raise Exception.Create ('Index out of range in getFloatingSpeciesByIndex');
end;


function getBoundarySpeciesByIndex (index : integer) : double;
begin
  if not libGetBoundarySpeciesByIndex (internalRRHandle, index, result) then
     raise Exception.Create ('Index out of range in getBoundarySpeciesByIndex');
end;


function getGlobalParameterByIndex (index : integer) : double;
begin
  if not libGetGlobalParameterByIndex (internalRRHandle, index, result) then
     raise Exception.Create ('Index out of range in getGlobalParameterByIndex');
end;


function getNumberOfDependentSpecies : integer;
begin
  result := libGetNumberOfDependentSpecies (internalRRHandle);
end;

function getNumberOfIndependentSpecies : integer;
begin
  result := libGetNumberOfIndependentSpecies (internalRRHandle);
end;


function steadyState : double;
var errMsg : AnsiString;
begin
  if not libSteadyState (internalRRHandle, result) then
     begin
     errMsg := getLastError;
     raise Exception.Create (errMsg);
     end;
end;


function computeSteadyStateValues : TDoubleArray;
var p : PRRDoubleVectorHandle; i : integer;
begin
  p := libComputeSteadyStateValues (internalRRHandle);
  try
    setLength (result, p.count);
    for i := 0 to p.count - 1 do
        result[i] := p.data[i];
  finally
    libFreeDoubleVector (p);
  end;
end;


function getEigenvalues : T2DDoubleArray;
var p : PRRMatrixHandle;
begin
  //p := libGetEigenvalues (rrHandle);
  if p = nil then
     raise Exception.Create ('No Eigenvalue matrix');
  try
    result := loadInTo2DArray (p);
  finally
    libFreeMatrix (p);
  end;
end;


function mGetEigenvalues : TMatrix;
var p : PRRMatrixHandle;
begin
  //p := libGetEigenvalues (rrHandle);
  if p = nil then
     raise Exception.Create ('No Eigenvalue matrix');
  try
    result := loadInToMatrix (p);
  finally
    libFreeMatrix (p);
  end;
end;


function  getEigenvaluesMatrix (m : T2DDoubleArray) : T2DDoubleArray;
var p1, p2 : PRRMatrixHandle;
begin
  p1 := loadIntoRRMatrix (m);
  if p1 = nil then
     exit;

  p2 := libGetEigenvaluesMatrix (p1);
  if p2 = nil then
     exit;

  result := loadInTo2DArray (p2);
end;


function setSteadyStateSelectionList (strList : TStringList) : boolean;
var i : integer;
    str : AnsiString;
begin
  if strList.Count > 0 then
     begin
     str := strList[0];
     for i := 1 to strList.Count - 1 do
         str := ' ' + strList[i];
     libSetSteadyStateSelectionList (internalRRHandle, PAnsiChar (str));
     end;
  result := true;
end;


function getSteadyStateSelectionList: TRRList;
var ptr : PRRListRecordHandle;
begin
  ptr := libGetSteadyStateSelectionList (internalRRHandle);
  result := extractList (ptr);
end;


function getUnscaledFluxControlCoefficientMatrix : T2DDoubleArray;
var p1 : PRRMatrixHandle;
begin
  p1 := libGetuFCC (internalRRHandle);
  if p1 = nil then
     exit;

  result := loadInTo2DArray (p1);
end;


function  getScaledFluxControlCoefficientMatrix : T2DDoubleArray;
var p1 : PRRMatrixHandle;
begin
  p1 := libGetFCC (internalRRHandle);
  if p1 = nil then
     raise Exception.Create ('Error in FCC function' + getLastError);

  result := loadInTo2DArray (p1);
end;


function getuCC (variable : AnsiString; parameter : AnsiString) : double;
begin
  if not libgetuCC (internalRRHandle, PAnsiChar (variable), PAnsiChar (parameter), result) then
     raise Exception.Create ('Error in getCC function');
end;


function getCC (variable : AnsiString; parameter : AnsiString) : double;
begin
  if not libgetCC (internalRRHandle, PAnsiChar (variable), PAnsiChar (parameter), result) then
     raise Exception.Create ('Error in getCC function');
end;


function getuEE (variable : AnsiString; parameter : AnsiString) : double;
begin
  if not libgetuEE (internalRRHandle, PAnsiChar (variable), PAnsiChar (parameter), result) then
     raise Exception.Create ('Error in getCC function');
end;


function getEE (variable : AnsiString; parameter : AnsiString) : double;
begin
  if not libgetEE (internalRRHandle, PAnsiChar (variable), PAnsiChar (parameter), result) then
     raise Exception.Create ('Error in getCC function');
end;


function getAvailableTimeCourseSymbols : TRRList;
var ptr : PRRListRecordHandle;
begin
  ptr := libGetAvailableTimeCourseSymbols (internalRRHandle);
  result := extractList (ptr);
end;


function getAvailableSteadStateSymbols : TRRList;
var ptr : PRRListRecordHandle;
begin
  ptr := libGetAvailableSteadyStateSymbols (internalRRHandle);
  result := extractList (ptr);
end;


// Deprecated
function getAvailableSymbolsII : TRRList;
var subList : TRRList; st : TStringList;
    i : integer;
    ptr : PRRListRecordHandle;
begin
  ptr := libGetAvailableTimeCourseSymbols (internalRRHandle);
  result := extractList (ptr);
  exit;

  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('time'));
  result.Add (TRRListItem.Create (subList));

  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Floating Species'));
  st := getFloatingSpeciesIds ();
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;

  st := getBoundarySpeciesIds ();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Boundary Species'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;

  {st := getFloatingSpeciesAmountIds();
  subList := TRRList.Create;
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  {st := getBoundarySpeciesAmountIds();
  subList := TRRList.Create;
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  lresultist.Add (TRRListItem.Create (subList));
  st.Free;}

  st := getGlobalParameterIds ();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Global Parameter Ids'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;

  {st := getCompartmentIds();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Compartments'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  st := getReactionIds ();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Reaction Ids'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;

  {st := getRatesOfChangeIds();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Rate of Change Ids'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  {st := getElasticityCoefficientIds();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Elasticity Coefficients'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  {st := getUnscaledElasticityCoefficientIds();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Unscaled Elasticity Coefficients'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  {st := getEigenValueIds();
  subList := TRRList.Create;
  subList.Add (TRRListItem.Create ('Eigenvalues'));
  for i := 0 to st.Count - 1 do
      subList.Add (TRRListItem.Create (st[i]));
  result.Add (TRRListItem.Create (subList));
  st.Free;}

  {p := libGetAvailableSymbols;
  setLength (result, p^.count);
  for i := 0 to p^.count - 1 do
      begin
      result[i].labelStr := p^.list[i].labelStr;
      result[i].stringList := getArrayOfStrings  (@(p^.list[i]));
      end;}
end;


function getReactionRate (index : integer) : double;
begin
  result := libGetReactionRate (internalRRHandle, index);
end;


function getReactionRates : TDoubleArray;
var p : PRRDoubleVectorHandle; i : integer;
begin
  p := libGetReactionRates (internalRRHandle);
  try
    if p = nil then
       begin
       setLength (result, 0);
       exit;
       end;
    setLength (result, p^.count);
    for i := 0 to p^.count - 1 do
        result[i] := p^.data[i];
  finally
    libFreeDoubleVector (p);
  end;
end;


function getRatesOfChange : TDoubleArray;
var p : PRRDoubleVectorHandle; i : integer;
begin
  p := libGetRatesOfChange (internalRRHandle);
  try
    if p = nil then
       begin
       setLength (result, 0);
       exit;
       end;
    setLength (result, p^.count);
    for i := 0 to p^.count - 1 do
        result[i] := p^.data[i];
  finally
    libFreeDoubleVector (p);
  end;
end;


function getStoichiometryMatrix : T2DDoubleArray;
var st : PRRMatrixHandle;
    i : integer;
begin
  st := libGetStoichiometryMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       setLength (result, 0, 0);
       exit;
       end;

    result := loadInTo2DArray(st);
  finally
    libFreeMatrix (st);
  end;
end;


function mGetStoichiometryMatrix : TMatrix;
var st : PRRMatrixHandle;
    i : integer;
begin
  st := libGetStoichiometryMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       result := nil;
       exit;
       end;

    result := loadInToMatrix (st);
  finally
    libFreeMatrix (st);
  end;
end;


function getLinkMatrix : T2DDoubleArray;
var st : PRRMatrixHandle;
    i, j : integer;
begin
  st := libGetLinkMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       setLength (result, 0, 0);
       exit;
       end;

    result := loadInTo2DArray (st);
  finally
    libFreeMatrix (st);
  end;
end;


function getNrMatrix : T2DDoubleArray;
var st : PRRMatrixHandle;
begin
  st := libGetNrMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       setLength (result, 0, 0);
       exit;
       end;

    result := loadInTo2DArray (st);
  finally
    libFreeMatrix (st);
  end;
end;


function getL0Matrix : T2DDoubleArray;
var st : PRRMatrixHandle;
begin
  st := libGetL0Matrix (internalRRHandle);
  try
    if st = nil then
       begin
       setLength (result, 0, 0);
       exit;
       end;

    result := loadInTo2DArray (st);
  finally
    libFreeMatrix (st);
  end;
end;


function getConservationMatrix : T2DDoubleArray;
var st : PRRMatrixHandle;
begin
  st := libGetConservationMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       setLength (result, 0, 0);
       exit;
       end;

    result := loadInTo2DArray (st);
  finally
    libFreeMatrix (st);
  end;
end;


function mGetConservationMatrix : TMatrix;
var st : PRRMatrixHandle;
begin
  st := libGetConservationMatrix (internalRRHandle);
  try
    if st = nil then
       begin
       result := nil;
       exit;
       end;

    result := loadInToMatrix (st);
  finally
    libFreeMatrix (st);
  end;
end;


procedure setMatrixElement ();
begin
end;

// ---------------------------------------------------------------------

procedure setRoadRunnerLibraryName (newLibName : AnsiString);
begin
  libName := newLibName;
end;


function loadSingleMethod (methodName : string; var errMsg : AnsiString; var success : boolean; methodList : TStringList) : Pointer;
begin
   result := GetProcAddress(dllHandle, PChar (methodName));
   if not Assigned (result) then
      begin
      methodList.Add (methodName + ': ****************** FAILED');
      errMsg := 'Failed to load method: ' + methodName;
      success := false;
      end
   else
      methodList.Add (methodName + ': found');
end;


function loadMethods (var errMsg : AnsiString; methodList : TStringList) : boolean;
begin
   result := true;
   try
   @libGetBuildDate  := loadSingleMethod ('getBuildDate', errMsg, result, methodList);
   @libGetVersion    := loadSingleMethod ('getVersion', errMsg, result, methodList);
   @libHasError      := loadSingleMethod ('hasError', errMsg, result, methodList);
   @libGetLastError  := loadSingleMethod ('getLastError', errMsg, result, methodList);

   @libSetLogLevel   := loadSingleMethod ('setLogLevel', errMsg, result, methodList);
   @libEnableLoggingToConsole := loadSingleMethod ('enableLoggingToConsole', errMsg, result, methodList);
   @libEnableLoggingToFile := loadSingleMethod ('enableLoggingToFile', errMsg, result, methodList);
   libGetLogFileName := loadSingleMethod ('getLogFileName', errMsg, result, methodList);

   @libSetTempFolder := loadSingleMethod ('setTempFolder', errMsg, result, methodList);
   @libGetTempFolder := loadSingleMethod ('getTempFolder', errMsg, result, methodList);

   @libGetCCode      := loadSingleMethod ('getCCode', errMsg, result, methodList);
   @libGetCopyright  := loadSingleMethod ('getCopyright', errMsg, result, methodList);

   @libCreateRRInstance := loadSingleMethod ('createRRInstance', errMsg, result, methodList);

   @libSetComputeAndAssignConservationLaws := loadSingleMethod ('setComputeAndAssignConservationLaws', errMsg, result, methodList);

   @libLoadSBMLFromFile   := loadSingleMethod ('loadSBMLFromFile', errMsg, result, methodList);
   @libLoadSBML           := loadSingleMethod ('loadSBML', errMsg, result, methodList);
   @libGetSBML            := loadSingleMethod ('getSBML', errMsg, result, methodList);

   @libSetTimeStart       := loadSingleMethod ('setTimeStart', errMsg, result, methodList);
   @libSetTimeEnd         := loadSingleMethod ('setTimeEnd', errMsg, result, methodList);
   @libSetNumberOfPoints  := loadSingleMethod ('setNumPoints', errMsg, result, methodList);
   @libSimulate           := loadSingleMethod ('simulate', errMsg, result, methodList);
   @libSimulateEx         := loadSingleMethod ('simulateEx', errMsg, result, methodList);
   @libOneStep            := loadSingleMethod ('oneStep', errMsg, result, methodList);
   @libReset              := loadSingleMethod ('reset', errMsg, result, methodList);
   @libGetCapabilities    := loadSingleMethod ('getCapabilities', errMsg, result, methodList);
   @libSetCapabilities    := loadSingleMethod ('setCapabilities', errMsg, result, methodList);
   @libSetFloatingSpeciesInitialConcentrations := loadSingleMethod ('setFloatingSpeciesInitialConcentrations', errMsg, result, methodList);

   @libEvalModel          := loadSingleMethod ('evalModel', errMsg, result, methodList);
   @libGetFullJacobian    := loadSingleMethod('getFullJacobian', errMsg, result, methodList);
   @libGetReducedJacobian := loadSingleMethod('getReducedJacobian', errMsg, result, methodList);

   @libSetValue           := loadSingleMethod ('setValue', errMsg, result, methodList);
   @libGetValue           := loadSingleMethod ('getValue', errMsg, result, methodList);
   @libSetTimeCourseSelectionList  := loadSingleMethod ('setTimeCourseSelectionList', errMsg, result, methodList);
   //@libGetTimeCourseSelectionList  := loadSingleMethod ('getTimeCourseSelectionList', errMsg, result, methodList);

   @libGetNumberOfReactions        := loadSingleMethod ('getNumberOfReactions', errMsg, result, methodList);
   @libGetNumberOfBoundarySpecies  := loadSingleMethod ('getNumberOfBoundarySpecies', errMsg, result, methodList);
   @libGetNumberOfFloatingSpecies  := loadSingleMethod ('getNumberOfFloatingSpecies', errMsg, result, methodList);
   @libGetNumberOfGlobalParameters := loadSingleMethod ('getNumberOfGlobalParameters', errMsg, result, methodList);
   @libGetNumberOfCompartments     := loadSingleMethod ('getNumberOfCompartments', errMsg, result, methodList);

   @libSetCompartmentByIndex       := loadSingleMethod ('setCompartmentByIndex', errMsg, result, methodList);
   @libSetFloatingSpeciesByIndex   := loadSingleMethod ('setFloatingSpeciesByIndex', errMsg, result, methodList);
   @libSetBoundarySpeciesByIndex   := loadSingleMethod ('setBoundarySpeciesByIndex', errMsg, result, methodList);
   @libSetGlobalParameterByIndex   := loadSingleMethod ('setGlobalParameterByIndex', errMsg, result, methodList);

   @libGetCompartmentByIndex       := loadSingleMethod ('getCompartmentByIndex', errMsg, result, methodList);
   @libGetFloatingSpeciesByIndex   := loadSingleMethod ('getFloatingSpeciesByIndex', errMsg, result, methodList);
   @libGetBoundarySpeciesByIndex   := loadSingleMethod ('getBoundarySpeciesByIndex', errMsg, result, methodList);
   @libGetGlobalParameterByIndex   := loadSingleMethod ('getGlobalParameterByIndex', errMsg, result, methodList);

   @libGetFloatingSpeciesConcentrations := loadSingleMethod ('getFloatingSpeciesConcentrations', errMsg, result, methodList);
   @libGetBoundarySpeciesConcentrations := loadSingleMethod ('getBoundarySpeciesConcentrations', errMsg, result, methodList);

   @libGetNumberOfDependentSpecies   := loadSingleMethod ('getNumberOfDependentSpecies', errMsg, result, methodList);
   @libGetNumberOfIndependentSpecies := loadSingleMethod ('getNumberOfIndependentSpecies', errMsg, result, methodList);

   @libSteadyState                   := loadSingleMethod ('steadyState', errMsg, result, methodList);
   @libComputeSteadyStateValues      := loadSingleMethod ('computeSteadyStateValues', errMsg, result, methodList);
   @libSetSteadyStateSelectionList   := loadSingleMethod ('setSteadyStateSelectionList', errMsg, result, methodList);
   @libGetSteadyStateSelectionList   := loadSingleMethod ('getSteadyStateSelectionList', errMsg, result, methodList);

   @libGetReactionRate        := loadSingleMethod ('getReactionRate', errMsg, result, methodList);
   @libGetReactionRates       := loadSingleMethod ('getReactionRates', errMsg, result, methodList);
   @libGetRatesOfChange       := loadSingleMethod ('getRatesOfChange', errMsg, result, methodList);

   @libGetCompartmentIds      := loadSingleMethod ('getCompartmentIds', errMsg, result, methodList);
   @libGetReactionIds         := loadSingleMethod ('getReactionIds', errMsg, result, methodList);
   @libGetBoundarySpeciesIds  := loadSingleMethod ('getBoundarySpeciesIds', errMsg, result, methodList);
   @libGetFloatingSpeciesIds  := loadSingleMethod ('getFloatingSpeciesIds', errMsg, result, methodList);
   @libGetGlobalParameterIds  := loadSingleMethod ('getGlobalParameterIds', errMsg, result, methodList);
   @libGetRatesOfChangeIds    := loadSingleMethod ('getRatesOfChangeIds', errMsg, result, methodList);
   @libGetEigenvalueIds       := loadSingleMethod ('getEigenvalueIds', errMsg, result, methodList);
   @libGetElasticityIds       := loadSingleMethod ('getElasticityCoefficientIds', errMsg, result, methodList);

   @libGetAvailableTimeCourseSymbols  := loadSingleMethod ('getAvailableTimeCourseSymbols', errMsg, result, methodList);
   @libGetAvailableSteadyStateSymbols := loadSingleMethod ('getAvailableSteadyStateSymbols', errMsg, result, methodList);

   @libGetStoichiometryMatrix   := loadSingleMethod ('getStoichiometryMatrix', errMsg, result, methodList);
   @libGetLinkMatrix            := loadSingleMethod ('getLinkMatrix', errMsg, result, methodList);
   @libGetNrMatrix              := loadSingleMethod ('getNrMatrix', errMsg, result, methodList);
   @libGetL0Matrix              := loadSingleMethod ('getL0Matrix', errMsg, result, methodList);
   @libGetConservationMatrix    := loadSingleMethod ('getConservationMatrix', errMsg, result, methodList);

   @libgetuCC                   := loadSingleMethod ('getuCC', errMsg, result, methodList);
   @libgetuEE                   := loadSingleMethod ('getuEE', errMsg, result, methodList);
   @libgetCC                    := loadSingleMethod ('getCC', errMsg, result, methodList);
   @libgetEE                    := loadSingleMethod ('getEE', errMsg, result, methodList);
   @libGetuFCC                  := loadSingleMethod ('getUnscaledFluxControlCoefficientMatrix', errMsg, result, methodList);
   @libGetFCC                  :=  loadSingleMethod ('getScaledFluxControlCoefficientMatrix', errMsg, result, methodList);

   @libGetEigenvalues           := loadSingleMethod ('getEigenvalues', errMsg, result, methodList);

   @libGetEigenvaluesMatrix     := loadSingleMethod ('getEigenvaluesMatrix', errMsg, result, methodList);
   @libGetListItem              := loadSingleMethod ('getListItem', errMsg, result, methodList);

   @libFreeRRInstance   := loadSingleMethod ('freeRRInstance', errMsg, result, methodList);
   @libFreeResult       := loadSingleMethod ('freeResult', errMsg, result, methodList);
   @libFreeMatrix       := loadSingleMethod ('freeMatrix', errMsg, result, methodList);
   @libFreeText         := loadSingleMethod ('freeText', errMsg, result, methodList);
   @libFreeStringArray  := loadSingleMethod ('freeStringArray', errMsg, result, methodList);

   @libCreateVector     := loadSingleMethod ('createVector', errMsg, result, methodList);
   @libCreateRRMatrix   := loadSingleMethod ('createRRMatrix', errMsg, result, methodList);
   @libSetMatrixElement := loadSingleMethod ('setMatrixElement', errMsg, result, methodlist);
   @libFreeDoubleVector := loadSingleMethod ('freeVector', errMsg, result, methodList);
   except
     on E: Exception do
        begin
        errMsg := e.message;
        result := false;
        exit;
        end;
   end;
end;

function loadRoadRunner (var errMsg : AnsiString; methodList : TStringList) : boolean;
var errStr : string;
    tempString: WideString;
    aString: PChar;
    path : AnsiString;
begin
  DLLLoaded := false;
  path := ExtractFilePath(ParamStr(0)) + libName;
  if FileExists (path) then
     begin
     tempString := WideString (path);
     DllHandle := LoadLibrary (PWideChar(tempString));

     if DllHandle <> 0 then
         begin
         if loadMethods (errMsg, methodList) then
            begin
            DLLLoaded := True;
            result := true;
            end
         else
            result := false;
         end
     else
         begin
         errStr := SysErrorMessage(Windows.GetLastError);
         DLLLoaded := False;
         errMsg := 'Failed to load roadRunner at:[' + getCurrentDir + ']: ' + errStr;
         end;
     end
  else
     begin
     DLLLoaded := False;
     errMsg := 'Unable to locate roadRunner library at:[' + getCurrentDir + ']';
     end;
end;


procedure releaseRoadRunnerLibrary;
begin
  DLLLoaded := false;
  libFreeRRInstance (instance);  // <- should this be here?
  freeLibrary (DLLHandle);
end;

end.
