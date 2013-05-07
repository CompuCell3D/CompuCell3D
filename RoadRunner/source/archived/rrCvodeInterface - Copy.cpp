#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iomanip>
#include <math.h>
#include <map>
#include <algorithm>
#include "rrRoadRunner.h"
#include "rrIModel.h"
#include "rrCvodedll.h"
#include "rrException.h"
#include "rrModelState.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrCvodeInterface.h"
#include "rrCvodeDll.h"

//---------------------------------------------------------------------------


using namespace std;
namespace rr
{

//Static stuff...
double     CvodeInterface::lastTimeValue = 0;
int     CvodeInterface::mOneStepCount = 0;
int     CvodeInterface::mCount = 0;
int     CvodeInterface::errorFileCounter = 0;
string  CvodeInterface::tempPathstring = "c:\\";
IModel* CvodeInterface::model = NULL;
// -------------------------------------------------------------------------
// Constructor
// Model contains all the symbol tables associated with the model
// ev is the model function
// -------------------------------------------------------------------------

CvodeInterface::CvodeInterface(IModel *aModel)
:
//defaultReltol(1E-12),
//defaultAbsTol(1E-16),
defaultReltol(1E-6),
defaultAbsTol(1E-12),

defaultMaxNumSteps(10000),
//gdata(NULL),
_amounts(NULL),
cvodeLogFile("cvodeLogFile"),
followEvents(true),
mRandom(),
defaultMaxAdamsOrder(12),
defaultMaxBDFOrder(5),
MaxAdamsOrder(defaultMaxAdamsOrder),
MaxBDFOrder(defaultMaxBDFOrder),
InitStep(0.0),
MinStep(0.0),
MaxStep(0.0),
MaxNumSteps(defaultMaxNumSteps),
relTol(defaultReltol),
absTol(defaultAbsTol)
//errorFileCounter,
//_amounts),
//_rootsFound),
//abstolArray),
//modelDelegate(&CvodeInterface::ModelFcn)
{
    //relTol = 1.e-4;
    //absTol = 1.(defaultReltol),
//    absTol(defaultAbsTol)

    InitializeCVODEInterface(aModel);
}

CvodeInterface::~CvodeInterface()
{
    FreeCvode_Mem((void**) cvodeMem);
    FreeCvode_Vector(_amounts);
    FreeCvode_Vector(abstolArray);
    fileClose(fileHandle);
}

bool CvodeInterface::HaveVariables()
{
    return (numAdditionalRules + numIndependentVariables > 0);
}

void CvodeInterface::InitializeCVODEInterface(IModel *oModel)
{
    if(!oModel)
    {
        throw SBWApplicationException("Fatal Error while initializing CVODE");
    }

    try
    {
        model = oModel;
        numIndependentVariables = oModel->getNumIndependentVariables();
        numAdditionalRules = oModel->rateRules.size();

////                modelDelegate = new TCallBackModelFcn(ModelFcn);
////                eventDelegate = new TCallBackRootFcn(EventFcn);

//        modelDelegate = &ModelFcn;
//        eventDelegate = new TCallBackRootFcn(EventFcn);

        if (numAdditionalRules + numIndependentVariables > 0)
        {
            int allocatedMemory = numIndependentVariables + numAdditionalRules;
            _amounts =     NewCvode_Vector(allocatedMemory);
            abstolArray = NewCvode_Vector(allocatedMemory);
            for (int i = 0; i < allocatedMemory; i++)
            {
                Cvode_SetVector((N_Vector) abstolArray, i, defaultAbsTol);
            }

            AssignNewVector(oModel, true);

            cvodeMem = (void*) Create_BDF_NEWTON_CVode();
            SetMaxOrder(cvodeMem, MaxBDFOrder);
            CVodeSetInitStep(cvodeMem, InitStep);
            SetMinStep(cvodeMem, MinStep);
            SetMaxStep(cvodeMem, MaxStep);

            SetMaxNumSteps(cvodeMem, MaxNumSteps);

            fileHandle = fileOpen(tempPathstring + cvodeLogFile + ToString(errorFileCounter) + ".txt");
            SetErrFile(cvodeMem, fileHandle);
//            errCode = AllocateCvodeMem(cvodeMem, allocatedMemory, modelDelegate, 0.0, (N_Vector) _amounts, relTol, (N_Vector) abstolArray);
            errCode = AllocateCvodeMem(cvodeMem, allocatedMemory, ModelFcn, (cvode_precision) 0.0, (N_Vector) _amounts, relTol, (N_Vector) abstolArray);

            if (errCode < 0)
            {
                HandleCVODEError(errCode);
            }

            if (oModel->getNumEvents() > 0)
            {
                //errCode = CVRootInit(cvodeMem, oModel->getNumEvents(), eventDelegate, gdata);
            }

            errCode = CvDense(cvodeMem, allocatedMemory); // int = size of systems
            if (errCode < 0)
            {
                HandleCVODEError(errCode);
            }

            oModel->resetEvents();
        }
        else if (model->getNumEvents() > 0)
        {
            int allocated = 1;
            _amounts =  NewCvode_Vector(allocated);
            abstolArray =  NewCvode_Vector(allocated);
            Cvode_SetVector( (N_Vector) _amounts, 0, 10);
            Cvode_SetVector( (N_Vector) abstolArray, 0, defaultAbsTol);

            cvodeMem = (long*) Create_BDF_NEWTON_CVode();
            SetMaxOrder(cvodeMem, MaxBDFOrder);
            SetMaxNumSteps(cvodeMem, MaxNumSteps);

            fileHandle = fileOpen(tempPathstring + cvodeLogFile + ToString(errorFileCounter) + ".txt");
            SetErrFile(cvodeMem, fileHandle);

//            errCode = AllocateCvodeMem(cvodeMem, allocated, modelDelegate, 0.0, (N_Vector) _amounts, relTol, (N_Vector) abstolArray);
            errCode = AllocateCvodeMem(cvodeMem, allocated, ModelFcn, 0.0, (N_Vector) _amounts, relTol, (N_Vector) abstolArray);
            if (errCode < 0)
            {
                HandleCVODEError(errCode);
            }

            if (oModel->getNumEvents() > 0)
            {
                //errCode = CVRootInit(cvodeMem, oModel->getNumEvents(), eventDelegate, gdata);
            }

            errCode = CvDense(cvodeMem, allocated); // int = size of systems
            if (errCode < 0)
            {
                HandleCVODEError(errCode);
            }

            oModel->resetEvents();
        }
    }
    catch (RRException ex)
    {
        throw SBWApplicationException("Fatal Error while initializing CVODE");//, ex.mMessage);
    }
}

////    class CvodeInterface : IDisposable
////    {
////        /// <summary>
////        /// Point to the CVODE DLL to use
////        /// </summary>
////        const string CVODE = "cvodedll";
////
////        static int nOneStepCount;
////
////        #region Default CVODE Settings
////
////        const double defaultReltol = 1E-6;
////        const double defaultAbsTol = 1E-12;
////        const int defaultMaxNumSteps = 10000;
////        static int defaultMaxAdamsOrder = 12;
////        static int defaultMaxBDFOrder = 5;
////
////        static int MaxAdamsOrder = defaultMaxAdamsOrder;
////        static int MaxBDFOrder = defaultMaxBDFOrder;
////
////        static double InitStep = 0.0;
////        static double MinStep = 0.0;
////        static double MaxStep = 0.0;
////
////
////        //static double defaultReltol = 1E-15;
////        //static double defaultAbsTol = 1E-20;
////
////        static int MaxNumSteps = defaultMaxNumSteps;
////        static double relTol = defaultReltol;
////        static double absTol = defaultAbsTol;
////
////        #endregion
////
////        #region Class Variables
////
////        // Error codes
////        const int CV_ROOT_RETURN = 2;
////        const int CV_TSTOP_RETURN = 1;
////        const int CV_SUCCESS = 0;
////        const int CV_MEM_NULL = -1;
////        const int CV_ILL_INPUT = -2;
////        const int CV_NO_MALLOC = -3;
////        const int CV_TOO_MUCH_WORK = -4;
////        const int CV_TOO_MUCH_ACC = -5;
////        const int CV_ERR_FAILURE = -6;
////        const int CV_CONV_FAILURE = -7;
////        const int CV_LINIT_FAIL = -8;
////        const int CV_LSETUP_FAIL = -9;
////        const int CV_LSOLVE_FAIL = -10;
////
////        const int CV_MEM_FAIL = -11;
////
////        const int CV_RTFUNC_NULL = -12;
////        const int CV_NO_SLDET = -13;
////        const int CV_BAD_K = -14;
////        const int CV_BAD_T = -15;
////        const int CV_BAD_DKY = -16;
////
////        const int CV_PDATA_NULL = -17;
////        static readonly string tempPathstring = Path.GetTempPath();
////
////
////        static int errorFileCounter;
////        static double lastTimeValue;
////        readonly IntPtr gdata = IntPtr.Zero;
////        IntPtr _amounts;
////        IntPtr _rootsFound;
////        IntPtr abstolArray;
////        string cvodeLogFile = "cvodeLogFile";
////        IntPtr cvodeMem;
////        int errCode;
////
////        IntPtr fileHandle;
////        IModel model;
////        int numIndependentVariables;
////
////        #endregion
////
////        #region Library Imports
////
////        [DllImport(CVODE, EntryPoint = "fileOpen", ExactSpelling = false,
////            CharSet = CharSet.Ansi, SetLastError = true)]
////        static extern IntPtr fileOpen(string fileName);
////
////        [DllImport(CVODE, EntryPoint = "fileClose", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern void fileClose(IntPtr fp);
////
////
////        [DllImport(CVODE, EntryPoint = "NewCvode_Vector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern IntPtr NewCvode_Vector(int n);
////
////        [DllImport(CVODE, EntryPoint = "FreeCvode_Vector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern void FreeCvode_Vector(IntPtr vect);
////
////        [DllImport(CVODE, EntryPoint = "FreeCvode_Mem", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern void FreeCvode_Mem(IntPtr p);
////
////        // void *p
////
////        [DllImport(CVODE, EntryPoint = "Cvode_SetVector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern void Cvode_SetVector(IntPtr v, int Index, double Value);
////
////        [DllImport(CVODE, EntryPoint = "Cvode_GetVector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern double Cvode_GetVector(IntPtr v, int Index);
////
////        [DllImport(CVODE, EntryPoint = "Create_BDF_NEWTON_CVode", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern IntPtr Create_BDF_NEWTON_CVode();
////
////        [DllImport(CVODE, EntryPoint = "Create_ADAMS_FUNCTIONAL_CVode", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern IntPtr Create_ADAMS_FUNCTIONAL_CVode();
////
////        [DllImport(CVODE, EntryPoint = "AllocateCvodeMem", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int AllocateCvodeMem(IntPtr cvode_mem, int n, TCallBackModelFcn fcn, double t0, IntPtr y,
////                                                  double reltol, IntPtr abstol);
////
////        [DllImport(CVODE, EntryPoint = "CvDense", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int CvDense(IntPtr cvode_mem, int n);
////
////        // int = size of systems
////
////        [DllImport(CVODE, EntryPoint = "CVReInit", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int CVReInit(IntPtr cvode_mem, double t0, IntPtr y0, double reltol, IntPtr abstol);
////
////        [DllImport(CVODE, EntryPoint = "Run_Cvode")]
////        //static extern int  RunCvode (IntPtr cvode_mem, double tout, IntPtr  y, ref double t, string ErrMsg);
////        static extern int RunCvode(IntPtr cvode_mem, double tout, IntPtr y, ref double t);
////
////        //static extern int  RunCvode (IntPtr cvode_mem, double tout, IntPtr y, ref double t);  // t = double *
////
////
////        [DllImport(CVODE, EntryPoint = "CVGetRootInfo", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int CVGetRootInfo(IntPtr cvode_mem, IntPtr rootsFound);
////
////        [DllImport(CVODE, EntryPoint = "CVRootInit", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int CVRootInit(IntPtr cvode_mem, int numRoots, TCallBackRootFcn rootfcn, IntPtr gdata);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxNumSteps", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxNumSteps(IntPtr cvode_mem, int mxsteps);
////
////
////        [DllImport(CVODE, EntryPoint = "SetMinStep", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMinStep(IntPtr cvode_mem, double minStep);
////        [DllImport(CVODE, EntryPoint = "SetMaxStep", ExactSpelling = false,
////                    CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxStep(IntPtr cvode_mem, double maxStep);
////        [DllImport(CVODE, EntryPoint = "SetInitStep", ExactSpelling = false,
////                    CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetInitStep(IntPtr cvode_mem, double initStep);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxOrder", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxOrder(IntPtr cvode_mem, int mxorder);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxErrTestFails", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxErrTestFails(IntPtr cvode_mem, int maxnef);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxConvFails", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxConvFails(IntPtr cvode_mem, int maxncf);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxNonLinIters", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetMaxNonLinIters(IntPtr cvode_mem, int maxcor);
////
////        [DllImport(CVODE, EntryPoint = "SetErrFile", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        static extern int SetErrFile(IntPtr cvode_mem, IntPtr errfp);
////
////        #endregion
////
////        #region CVODE Callback functions
////
////        #region Delegates
////
////        delegate void TCallBackModelFcn(int n, double time, IntPtr y, IntPtr ydot, IntPtr fdata);
////
////        delegate void TCallBackRootFcn(double t, IntPtr y, IntPtr gdot, IntPtr gdata);
////
////        #endregion
////
////        static TCallBackModelFcn modelDelegate;
////        static TCallBackRootFcn eventDelegate;
////
////        static int nCount;
////        static int nRootCount;
////


////        public void ModelFcn(int n, double time, IntPtr y, IntPtr ydot, IntPtr fdata)
////        {
////            var oldState = new ModelState(model);
////
////            var dCVodeArgument = new double[model.amounts.Length + model.rateRules.Length];
////            Marshal.Copy(y, dCVodeArgument, 0, Math.Min(n, dCVodeArgument.Length));
////
////#if (PRINT_STEP_DEBUG)
////                    System.Diagnostics.Debug.Write("CVode In: (" + nCount + ")" );
////                    for (int i = 0; i < dCVodeArgument.Length; i++)
////                    {
////                        System.Diagnostics.Debug.Write(dCVodeArgument[i].ToString() + ", ");
////                    }
////                    System.Diagnostics.Debug.WriteLine("");
////#endif
////
////            model.evalModel(time, dCVodeArgument);
////
////            model.rateRules.CopyTo(dCVodeArgument, 0);
////            model.dydt.CopyTo(dCVodeArgument, model.rateRules.Length);
////
////#if (PRINT_STEP_DEBUG)
////                    System.Diagnostics.Debug.Write("CVode Out: (" + nCount + ")");
////                    for (int i = 0; i < dCVodeArgument.Length; i++)
////                    {
////                        System.Diagnostics.Debug.Write(dCVodeArgument[i].ToString() + ", ");
////                    }
////                    System.Diagnostics.Debug.WriteLine("");
////#endif
////
////            Marshal.Copy(dCVodeArgument, 0, ydot, Math.Min(dCVodeArgument.Length, n));
////
////            nCount++;
////
////            oldState.AssignToModel(model);
////        }

//void CvodeInterface::ModelFcn(int n, double time, IntPtr y, IntPtr ydot, IntPtr fdata)
void ModelFcn(int n, double time, cvode_precision* y, cvode_precision* ydot, void* fdata)
{
    IModel *model = CvodeInterface::model;
    ModelState oldState(*model);
    int size = model->amounts.size() + model->rateRules.size();
    vector<double> dCVodeArgument(size);//model->.amounts.Length + model.rateRules.Length];

//    Marshal.Copy(y, dCVodeArgument, 0, Math.Min(n, dCVodeArgument.Length));
    for(int i = 0; i < min((int) dCVodeArgument.size(), n); i++)
    {
        dCVodeArgument[i] = y[i];
    }

    stringstream msg;
    msg<<left<<setw(20)<<"" + ToString(CvodeInterface::mCount) ;

    for (u_int i = 0; i < dCVodeArgument.size(); i++)
    {
        msg<<left<<setw(20)<<setprecision (18)<<dCVodeArgument[i];
    }

    model->evalModel(time, dCVodeArgument);
//    model.rateRules.CopyTo(dCVodeArgument, 0);
    dCVodeArgument = model->rateRules;

    //    model.dydt.CopyTo(dCVodeArgument, model.rateRules.Length);
    for(u_int i = 0 ; i < model->GetdYdT().size(); i++)
    {
        dCVodeArgument.push_back(model->GetdYdT().at(i));
    }

    msg<<"\t"<<CvodeInterface::mCount << "\t" ;
    for (u_int i = 0; i < dCVodeArgument.size(); i++)
    {
        msg<<setw(10)<<left<<setprecision (18)<<dCVodeArgument[i];
    }
    Log(lDebug5)<<msg.str();

//    Marshal.Copy(dCVodeArgument, 0, ydot, Math.Min(dCVodeArgument.Length, n));

    for (int i = 0; i < min((int) dCVodeArgument.size(), n); i++)
    {
        ydot[i]= dCVodeArgument[i];
    }

    CvodeInterface::mCount++;
    oldState.AssignToModel(*model);
}

////        double[] CvodeInterface::GetCopy(double[] oVector)
////        {
////            var oResult = new double[oVector.Length];
////            oVector.CopyTo(oResult, 0);
////            return oResult;
////        }
////
////        bool[] CvodeInterface::GetCopy(bool[] oVector)
////        {
////            var oResult = new bool[oVector.Length];
////            oVector.CopyTo(oResult, 0);
////            return oResult;
////        }
////
////        void CvodeInterface::EventFcn(double time, IntPtr y, IntPtr gdot, IntPtr fdata)
////        {
////
////            var oldState = new ModelState(model);
////
////            model.evalModel(time, BuildEvalArgument());
////            AssignResultsToModel();
////            model.evalEvents(time, BuildEvalArgument());
////
////            Marshal.Copy(model.eventTests, 0, gdot, model.getNumEvents());
////
////#if (PRINT_EVENT_DEBUG)
////                    System.Diagnostics.Debug.Write("Rootfunction Out: t=" + time.ToString("F14") + " (" + nRootCount + "): ");
////                    for (int i = 0; i < model.eventTests.Length; i++)
////                    {
////                        System.Diagnostics.Debug.Write(model.eventTests[i].ToString() + " p=" + model.previousEventStatusArray[i] + " c=" + model.eventStatusArray[i] + ", ");
////                    }
////                    System.Diagnostics.Debug.WriteLine("");
////#endif
////
////            nRootCount++;
////
////            oldState.AssignToModel(model);
////        }
////
////        #endregion
////
////
////        #region ErrorHandling
////
////        static CvodeErrorCodes[] errorCodes = InitilizeErrorCodes();
////
////        static CvodeErrorCodes[] CvodeInterface::InitilizeErrorCodes()
////        {
////            var oErrorCodes = new CvodeErrorCodes[28];
////            oErrorCodes[0] = new CvodeErrorCodes(CV_SUCCESS, "Success");
////            oErrorCodes[1] = new CvodeErrorCodes(-1, "The solver took mxstep steps but could not reach tout.");
////            oErrorCodes[2] = new CvodeErrorCodes(-2,
////                                                 "The solver could not satisfy the accuracy demanded by the user for some step.");
////            oErrorCodes[3] = new CvodeErrorCodes(-3,
////                                                 "Error test failures occurred too many times during one time step or minimum step size was reached.");
////            oErrorCodes[4] = new CvodeErrorCodes(-4,
////                                                 "Convergence test failures occurred too many times during one time step or minimum step size was reached.");
////            oErrorCodes[5] = new CvodeErrorCodes(-5, "The linear solver's initialization function failed.");
////            oErrorCodes[6] = new CvodeErrorCodes(-6,
////                                                 "The linear solver's setup function failed in an unrecoverable manner.");
////            oErrorCodes[7] = new CvodeErrorCodes(-7,
////                                                 "The linear solver's solve function failed in an unrecoverable manner.");
////            oErrorCodes[8] = new CvodeErrorCodes(-8, "The right-hand side function failed in an unrecoverable manner.");
////            oErrorCodes[9] = new CvodeErrorCodes(-9, "The right-hand side function failed at the first call.");
////            oErrorCodes[10] = new CvodeErrorCodes(-10, "The right-hand side function had repetead recoverable errors.");
////            oErrorCodes[11] = new CvodeErrorCodes(-11,
////                                                  "The right-hand side function had a recoverable error, but no recovery is possible.");
////            oErrorCodes[12] = new CvodeErrorCodes(-12, "The rootýnding function failed in an unrecoverable manner.");
////            oErrorCodes[13] = new CvodeErrorCodes(-13, "");
////            oErrorCodes[14] = new CvodeErrorCodes(-14, "");
////            oErrorCodes[15] = new CvodeErrorCodes(-15, "");
////            oErrorCodes[16] = new CvodeErrorCodes(-16, "");
////            oErrorCodes[17] = new CvodeErrorCodes(-17, "");
////            oErrorCodes[18] = new CvodeErrorCodes(-18, "");
////            oErrorCodes[19] = new CvodeErrorCodes(-19, "");
////            oErrorCodes[20] = new CvodeErrorCodes(-20, "A memory allocation failed");
////            oErrorCodes[21] = new CvodeErrorCodes(-21, "The cvode mem argument was NULL.");
////            oErrorCodes[22] = new CvodeErrorCodes(-22, "One of the function inputs is illegal.");
////            oErrorCodes[23] = new CvodeErrorCodes(-23,
////                                                  "The cvode memory block was not allocated by a call to CVodeMalloc.");
////            oErrorCodes[24] = new CvodeErrorCodes(-24, "The derivative order k is larger than the order used.");
////            oErrorCodes[25] = new CvodeErrorCodes(-25, "The time t s outside the last step taken.");
////            oErrorCodes[26] = new CvodeErrorCodes(-26, "The output derivative vector is NULL.");
////            oErrorCodes[27] = new CvodeErrorCodes(-27, "The output and initial times are too close to each other.");
////            return oErrorCodes;
////        }
////

void CvodeInterface::HandleCVODEError(int errCode)
{
    if (errCode < 0)
    {
        string msg = "";
        string errorFile = tempPathstring + cvodeLogFile + ToString(errorFileCounter) + ".txt";

        // and open a new file handle
        errorFileCounter++;
        FILE* newHandle = fileOpen(tempPathstring + cvodeLogFile + ToString(errorFileCounter) + ".txt");
        if (newHandle != NULL && cvodeMem != NULL)
        {
            SetErrFile(cvodeMem, newHandle);
        }
        // close error file used by the cvode library
        if (fileHandle != NULL)
        {
            fileClose(fileHandle);
        }
        fileHandle = newHandle;

        try
        {
//            msg = File.ReadAllText(errorFile);    //Todo: enable this..
//            File.Delete(errorFile);
        }
        catch (Exception)
        {
            // actually we don't need this error message any more ...
            // the error code will automatically be converted by the erroCodes list
            // hence if we can't read the log file there is no reason to worry the user.
            //msg = " Unknown Error from CVODE (ff)";
        }

        //throw CvodeException("Error in RunCVode: " + errorCodes[-errCode].msg + msg);
        Log(lError)<<"Error in RunCVode: "<<errCode<<msg;
        throw CvodeException("Error in RunCVode: " + ToString(errCode) + msg);
    }
}

double CvodeInterface::OneStep(double timeStart, double hstep)
{
    Log(lDebug3)<<"---------------------------------------------------";
    Log(lDebug3)<<"--- O N E     S T E P      ( "<<mOneStepCount<< " ) ";
    Log(lDebug3)<<"---------------------------------------------------";

    mOneStepCount++;
    mCount = 0;

    double timeEnd = 0.0;
    double tout = timeStart + hstep;
    int strikes = 3;
    try
    {
        // here we stop for a too small timestep ... this seems troublesome to me ...
        while (tout - timeEnd > 1E-16)
        {
            if (hstep < 1E-16)
            {
                return tout;
            }

            // here we bail in case we have no ODEs set up with CVODE ... though we should
            // still at least evaluate the model function
            if (!HaveVariables() && model->getNumEvents() == 0)
            {
                model->convertToAmounts();
                vector<double> args = BuildEvalArgument();
                model->evalModel(tout, args);
                return tout;
            }

            if (lastTimeValue > timeStart)
            {
                reStart(timeStart, model);
            }

            double nextTargetEndTime = tout;
            if (assignmentTimes.size() > 0 && assignmentTimes[0] < nextTargetEndTime)
            {
                nextTargetEndTime = assignmentTimes[0];
                assignmentTimes.erase(assignmentTimes.begin());
            }

            //RR_DECLSPEC int          Run_Cvode (void *cvode_mem, double tout, N_Vector y, double *t, char *ErrMsg);
            
            int nResult = 0;
            //int nResult = Run_Cvode(cvodeMem, nextTargetEndTime,  _amounts, &timeEnd);//, err); // t = double *
            double ydot[2];
            vector<double> dCVodeArgument(2);//model->.amounts.Length + model.rateRules.Length];
            for(int i = 0; i < min((int) dCVodeArgument.size(), 2); i++)
            {
            dCVodeArgument[i] =  Cvode_GetVector (_amounts, i);
            }

            model->evalModel(timeStart, dCVodeArgument);
            dCVodeArgument.clear();

            for(u_int i = 0 ; i < model->GetdYdT().size(); i++) {
                //dCVodeArgument.push_back(model->GetdYdT().at(i));
                dCVodeArgument.push_back( model->m_dydt[i]);
            }
            for (int i=0; i<2; i++) {
                Cvode_SetVector(_amounts, i, Cvode_GetVector (_amounts, i) + dCVodeArgument[i]*hstep);
                double x1 = Cvode_GetVector (_amounts, i);
                Log(lInfo)<<"====================================\t"<<x1<<endl;
            }            
            timeEnd = tout;
            //timeEnd = timeEnd + hstep;


            if (nResult == CV_ROOT_RETURN && followEvents)
            {
                Log(lDebug3)<<("---------------------------------------------------");
                Log(lDebug3)<<"--- E V E N T      ( " << mOneStepCount << " ) ";
                Log(lDebug3)<<("---------------------------------------------------");

                //bool tooCloseToStart = Math.Abs(timeEnd - timeStart) > absTol;
                bool tooCloseToStart = fabs(timeEnd - lastEvent) > relTol;
                strikes = (tooCloseToStart) ? 3 : strikes--;

                if (tooCloseToStart || strikes > 0)
                {
                    HandleRootsFound(timeEnd, tout);
                    reStart(timeEnd, model);
                    lastEvent = timeEnd;
                }
            }
            else if (nResult == CV_SUCCESS || !followEvents)
            {
                //model->resetEvents();
                model->SetTime(tout);
                AssignResultsToModel();
            }
            else
            {
                HandleCVODEError(nResult);
            }

            lastTimeValue = timeEnd;

            try
            {
                model->testConstraints();
            }
            catch (Exception e)
            {
                model->Warnings.push_back("Constraint Violated at time = " + ToString(timeEnd) + "\n" + e.Message);
            }

            AssignPendingEvents(timeEnd, tout);

            if (tout - timeEnd > 1E-16)
            {
                timeStart = timeEnd;
            }
            Log(lDebug3)<<"tout: "<<tout<<tab<<"timeEnd: "<<timeEnd;
        }
        return (timeEnd);
    }
    catch (Exception)
    {
        InitializeCVODEInterface(model);
        throw;
    }
}

void CvodeInterface::AssignPendingEvents(const double& timeEnd, const double& tout)
{
    for (int i = assignments.size() - 1; i >= 0; i--)
    {
        if (timeEnd >= assignments[i].GetTime())
        {
            model->SetTime(tout);
            AssignResultsToModel();
            model->convertToConcentrations();
            model->updateDependentSpeciesValues(model->y);
            assignments[i].AssignToModel();

            if (!RoadRunner::mConservedTotalChanged)
            {
                 model->computeConservedTotals();
            }
            model->convertToAmounts();
            vector<double> args = BuildEvalArgument();
            model->evalModel(timeEnd, args);
            reStart(timeEnd, model);
            assignments.erase(assignments.begin() + i);
        }
    }
}

////        List<int> CvodeInterface::RetestEvents(double timeEnd, List<int> handledEvents)
////        {
////            return RetestEvents(timeEnd, handledEvents, false);
////        }
////
////        List<int> CvodeInterface::RetestEvents(double timeEnd, List<int> handledEvents, out List<int> removeEvents)
////        {
////            return RetestEvents(timeEnd, handledEvents, false, out removeEvents);
////        }
////
vector<int> CvodeInterface::RetestEvents(const double& timeEnd, vector<int>& handledEvents, const bool& assignOldState)
{
    vector<int> removeEvents;
    return RetestEvents(timeEnd, handledEvents, assignOldState, removeEvents);
}

vector<int> CvodeInterface::RetestEvents(const double& timeEnd, vector<int>& handledEvents, const bool& assignOldState, vector<int>& removeEvents)
{
    vector<int> result;// = new vector<int>();
//     vector<int> removeEvents;// = new vector<int>();    //Todo: this code was like this originally.. which removeEvents to use???

    if (!RoadRunner::mConservedTotalChanged)
    {
        model->computeConservedTotals();
    }

    model->convertToAmounts();
    vector<double> args = BuildEvalArgument();
    model->evalModel(timeEnd, args);

    ModelState *oldState = new ModelState(*model);

    args = BuildEvalArgument();
    model->evalEvents(timeEnd, args);

    for (int i = 0; i < model->getNumEvents(); i++)
    {
//        if (model->eventStatusArray[i] == true && oldState->mEventStatusArray[i] == false && !handledEvents.Contains(i))
        bool containsI = (std::find(handledEvents.begin(), handledEvents.end(), i) != handledEvents.end()) ? true : false;
        if (model->eventStatusArray[i] == true && oldState->mEventStatusArray[i] == false && !containsI)
        {
            result.push_back(i);
        }

        if (model->eventStatusArray[i] == false && oldState->mEventStatusArray[i] == true && !model->eventPersistentType[i])
        {
            removeEvents.push_back(i);
        }
    }

    if (assignOldState)
    {
        oldState->AssignToModel(*model);
    }

    delete oldState;
    return result;
}

void CvodeInterface::HandleRootsFound(double &timeEnd, const double& tout)
{
    vector<int> rootsFound;// = new int[model->getNumEvents];
    // Create some space for the CVGetRootInfo call
    //_rootsFound = Marshal.AllocHGlobal(model->getNumEvents*sizeof (Int32));
//    CVGetRootInfo(cvodeMem, _rootsFound);    //This is a DLL Call.. Todo: implement..
    //    Marshal.Copy(_rootsFound, rootsFound, 0, model->getNumEvents);
    //    Marshal.FreeHGlobal(_rootsFound); // Free space used by CVGetRootInfo

    HandleRootsForTime(timeEnd, rootsFound);
}

void CvodeInterface::TestRootsAtInitialTime()
{
    vector<int> dummy;
    vector<int> events = RetestEvents(0, dummy, true); //Todo: dummy is passed but is not used..?
    if (events.size() > 0)
    {
        vector<int> rootsFound(model->getNumEvents());//         = new int[model->getNumEvents];
        vector<int>::iterator iter;
        for(iter = rootsFound.begin(); iter != rootsFound.end(); iter ++) //int item in events)
        {
            (*iter) = 1;
        }
        HandleRootsForTime(0, rootsFound);
    }
}

void CvodeInterface::RemovePendingAssignmentForIndex(const int& eventIndex)
{
    for (int j = assignments.size() - 1; j >= 0; j--)
    {
        if (assignments[j].GetIndex() == eventIndex)
        {
            //assignments.RemoveAt(j);
            assignments.erase(assignments.begin() + j);
        }
    }
}

void CvodeInterface::SortEventsByPriority(vector<int>& firedEvents)
{
    if ((firedEvents.size() > 1))
    {
        model->computeEventPriorites();
        //Todo: do the sorting...
//        firedEvents.Sort(new Comparison<int>((index1, index2) =>
//                                                 {
//                                                     double priority1 = model->eventPriorities[index1];
//                                                     double priority2 = model->eventPriorities[index2];
//
//                                                     // random toss in case we have same priorites
//                                                     if (priority1 == priority2 && priority1 != 0 &&
//                                                         index1 != index2)
//                                                     {
//                                                         if (Random.NextDouble() > 0.5)
//                                                             return -1;
//                                                         else
//                                                             return 1;
//                                                     }
//
//                                                     return -1 * priority1.CompareTo(priority2);
//                                                 }));
    }
}


void CvodeInterface::HandleRootsForTime(const double& timeEnd, vector<int>& rootsFound)
{
    AssignResultsToModel();
    model->convertToConcentrations();
    model->updateDependentSpeciesValues(model->y);

    vector<double> args = BuildEvalArgument();
    model->evalEvents(timeEnd, args);


    vector<int> firedEvents;// = new List<int>();
    map<int, vector<double> > preComputedAssignments;// = new Dictionary<int, double[]>();


    for (int i = 0; i < model->getNumEvents(); i++)
    {
        // We only fire an event if we transition from false to true
        if (rootsFound[i] == 1)
        {
            if (model->eventStatusArray[i])
            {
                firedEvents.push_back(i);
                if (model->eventType[i])
                {
                    preComputedAssignments[i] = model->computeEventAssignments[i]();
                }
            }
        }
        else
        {
            // if the trigger condition is not supposed to be persistent, remove the event from the firedEvents list;
            if (!model->eventPersistentType[i])
            {
                RemovePendingAssignmentForIndex(i);
            }
        }
    }
    vector<int> handled;// = new List<int>();
    while (firedEvents.size() > 0)
    {
        SortEventsByPriority(firedEvents);
        // Call event assignment if the eventstatus flag for the particular event is false
        for (u_int i = 0; i < firedEvents.size(); i++)
        {
            int currentEvent = firedEvents[i];
            // We only fire an event if we transition from false to true

            model->previousEventStatusArray[currentEvent] = model->eventStatusArray[currentEvent];
            double eventDelay = model->eventDelay[currentEvent]();
            if (eventDelay == 0)
            {    //Todo: enable this...
//                if (model->eventType[currentEvent] && preComputedAssignments.ContainsKey(currentEvent))
//                    model->performEventAssignments[currentEvent](preComputedAssignments[currentEvent]);
//                else
//                    model->eventAssignments[currentEvent]();
//
//                handled.Add(currentEvent);
//                List<int> removeEvents;
//                var additionalEvents = RetestEvents(timeEnd, handled, out removeEvents);
//                firedEvents.AddRange(additionalEvents);
//
//                foreach (var newEvent in additionalEvents)
//                {
//                    if (model->eventType[newEvent])
//                        preComputedAssignments[newEvent] = model->computeEventAssignments[newEvent]();
//                }
//
//                model->eventStatusArray[currentEvent] = false;
//                firedEvents.RemoveAt(i);
//
//                foreach (var item in removeEvents)
//                {
//                    if (firedEvents.Contains(item))
//                    {
//                        firedEvents.Remove(item);
//                        RemovePendingAssignmentForIndex(item);
//                    }
//                }
//
                break;
            }
            else
            {
            //Todo: enable this...
//                if (!assignmentTimes.Contains(timeEnd + eventDelay))
//                    assignmentTimes.Add(timeEnd + eventDelay);
//
//
//                var pending = new PendingAssignment(
//                                                            timeEnd + eventDelay,
//                                                            model->computeEventAssignments[currentEvent],
//                                                            model->performEventAssignments[currentEvent],
//                                                            model->eventType[currentEvent], currentEvent);
//
//                if (model->eventType[currentEvent] && preComputedAssignments.ContainsKey(currentEvent))
//                    pending.ComputedValues = preComputedAssignments[currentEvent];
//
//                assignments.Add(pending);
//                model->eventStatusArray[currentEvent] = false;
//                firedEvents.RemoveAt(i);
                break;
            }

            Log(lDebug)<<"time: "<<model->time<<" Event "<<(i + 1);

        }
    }

    if (!RoadRunner::mConservedTotalChanged)
    {
        model->computeConservedTotals();
    }
    model->convertToAmounts();


    args = BuildEvalArgument();
    model->evalModel(timeEnd, args);

    vector<double> dCurrentValues = model->GetCurrentValues();
    for (int k = 0; k < numAdditionalRules; k++)
    {
        Cvode_SetVector((N_Vector) _amounts, k, dCurrentValues[k]);
    }

    for (int k = 0; k < numIndependentVariables; k++)
    {
        Cvode_SetVector((N_Vector) _amounts, k + numAdditionalRules, model->amounts[k]);
    }

    CVReInit(cvodeMem, timeEnd, _amounts, relTol, abstolArray);
//    assignmentTimes.Sort();    //Todo: enable sorting. somehow
}


void CvodeInterface::AssignResultsToModel()
{
    model->updateDependentSpeciesValues(model->y);
    vector<double> dTemp(numAdditionalRules);// = new double[numAdditionalRules];
    for (int i = 0; i < numAdditionalRules; i++)
    {
        dTemp[i] = Cvode_GetVector((_generic_N_Vector*) _amounts, i);

    }

    for (int i = 0; i < numIndependentVariables; i++) //
    {
//        model->amounts[i] = Cvode_GetVector((_generic_N_Vector*) _amounts, i + numAdditionalRules);
        double val = Cvode_GetVector((_generic_N_Vector*) _amounts, i + numAdditionalRules);
        model->amounts[i] = (val);
        Log(lDebug)<<"Amount "<<setprecision(16)<<val;
    }

    vector<double> args = BuildEvalArgument();
    model->computeRules(args);
    model->AssignRates(dTemp);
    model->computeAllRatesOfChange();
}

// Restart the simulation using a different initial condition
void CvodeInterface::AssignNewVector(IModel *oModel, bool bAssignNewTolerances)
{
    vector<double> dTemp = model->GetCurrentValues();
    double dMin = absTol;

    for (int i = 0; i < numAdditionalRules; i++)
    {
        if (dTemp[i] > 0 && dTemp[i]/1000. < dMin)
        {
            dMin = dTemp[i]/1000.0;
        }
    }

    for (int i = 0; i < numIndependentVariables; i++)
    {
        if (oModel->GetAmounts(i) > 0 && oModel->GetAmounts(i)/1000.0 < dMin)    //Todo: was calling oModel->amounts[i]  is this in fact GetAmountsForSpeciesNr(i) ??
        {
            dMin = oModel->amounts[i]/1000.0;
        }
    }

    for (int i = 0; i < numAdditionalRules; i++)
    {
        if (bAssignNewTolerances)
        {
            setAbsTolerance(i, dMin);
        }
        Cvode_SetVector(_amounts, i, dTemp[i]);
    }

    for (int i = 0; i < numIndependentVariables; i++)
    {
        if (bAssignNewTolerances)
        {
            setAbsTolerance(i + numAdditionalRules, dMin);
        }
        Cvode_SetVector(_amounts, i + numAdditionalRules, oModel->GetAmounts(i));
    }

    if (!HaveVariables() && model->getNumEvents() > 0)
    {
        if (bAssignNewTolerances)
        {
            setAbsTolerance(0, dMin);
        }
        Cvode_SetVector(_amounts, 0, 1.0);
    }

    if (bAssignNewTolerances)
    {
        Log(lInfo)<<Format("Set tolerance to: {0:G}", dMin);
    }
}


void CvodeInterface::AssignNewVector(IModel *model)
{
    AssignNewVector(model, false);
}

void CvodeInterface::setAbsTolerance(int index, double dValue)
{
    double dTolerance = dValue;
    if (dValue > 0 && absTol > dValue)
    {
        dTolerance = dValue;
    }
    else
    {
        dTolerance = absTol;
    }

    Cvode_SetVector(abstolArray, index, dTolerance);
}

int CvodeInterface::reStart(double timeStart, IModel* model)
{
    AssignNewVector(model);

    SetInitStep(cvodeMem, InitStep);
    SetMinStep(cvodeMem, MinStep);
    SetMaxStep(cvodeMem, MaxStep);

    return (CVReInit(cvodeMem, timeStart, _amounts, relTol, abstolArray));
}
////
////        double CvodeInterface::getValue(int index)
////        {
////            return (Cvode_GetVector(_amounts, index + numAdditionalRules));
////        }
////

////        internal double[] BuildEvalArgument()
////        {
////            var dResult = new double[model.amounts.Length + model.rateRules.Length];
////            double[] dCurrentValues = model.GetCurrentValues();
////            dCurrentValues.CopyTo(dResult, 0);
////            model.amounts.CopyTo(dResult, model.rateRules.Length);
////            return dResult;
////        }


vector<double> CvodeInterface::BuildEvalArgument()
{
    vector<double> dResult;
    vector<double> dCurrentValues = model->GetCurrentValues();
    dResult = dCurrentValues;
    dResult.insert(dResult.end(), model->amounts.begin(), model->amounts.end());
    Log(lDebug4)<<"Size of dResult in BuildEvalArgument: "<<dResult.size();
    return dResult;
}
////
////        void CvodeInterface::release()
////        {
////            FreeCvode_Mem(cvodeMem);
////            FreeCvode_Vector(_amounts);
////            fileClose(fileHandle);
////        }
////
////        /// <summary>
////        /// Random number generator used to implement the random choosing
////        /// of event priorities.
////        /// <remarks>ReInitialize with specific seed in order to produce
////        /// repeatable runs.</remarks>
////        /// </summary>
////        Random CvodeInterface::Random { get; set; }
////    }
////}



} //namespace rr


