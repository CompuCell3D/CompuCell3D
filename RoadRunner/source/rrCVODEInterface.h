#ifndef rrCvodeInterfaceH
#define rrCvodeInterfaceH
#include <string>
#include "rrObject.h"
#include "rrPendingAssignment.h"
#include "cvode/cvode.h"


namespace rr
{

using std::string;

class Event;
class ModelFromC;
class RoadRunner;

class RR_DECLSPEC CvodeInterface : public rrObject
{
    private:
        const double                mDefaultReltol;
        const double                mDefaultAbsTol;
        const int                   mDefaultMaxNumSteps;

        string        		        mTempPathstring;
        int                  		mErrorFileCounter;
        int                         mNumIndependentVariables;
        N_Vector                    mAmounts;
        N_Vector                    mAbstolArray;
        string                      mLogFile;
        void*                       mCVODE_Memory;
        int                         mNumAdditionalRules;
        vector<double>              mAssignmentTimes;
        int                         mDefaultMaxAdamsOrder;
        int                         mDefaultMaxBDFOrder;
        double                      mLastTimeValue;
        double                      mLastEvent;
        ModelFromC*					mTheModel;
        int                         mOneStepCount;
        bool                        mFollowEvents;
        RoadRunner				   *mRR;

        void                        handleCVODEError(const int& errCode);
        void                        assignPendingEvents(const double& timeEnd, const double& tout);
        vector<int>                 retestEvents(const double& timeEnd, vector<int>& handledEvents);
        vector<int>                 retestEvents(const double& timeEnd, const vector<int>& handledEvents, vector<int>& removeEvents);
        vector<int>                 retestEvents(const double& timeEnd, vector<int>& handledEvents, const bool& assignOldState);
        vector<int>                 retestEvents(const double& timeEnd, const vector<int>& handledEvents, const bool& assignOldState, vector<int>& removeEvents);
        void                        handleRootsFound(double &timeEnd, const double& tout);
        void                        removePendingAssignmentForIndex(const int& eventIndex);
        void                        sortEventsByPriority(vector<int>& firedEvents);
        void                        sortEventsByPriority(vector<Event>& firedEvents);
        void                        handleRootsForTime(const double& timeEnd, vector<int>& rootsFound);
	 	int         				rootInit (const int& numRoots);//, TRootCallBack callBack, void *gdata);
		int         				reInit (const double& t0);
		int          				allocateCvodeMem ();
        void                        initializeCVODEInterface(ModelFromC *oModel);
        void                        setAbsTolerance(int index, double dValue);


    public:	//Hide these later on...
        int                         mMaxAdamsOrder;
        int                         mMaxBDFOrder;
        double                      mInitStep;
        double                      mMinStep;
        double                      mMaxStep;
        int                         mMaxNumSteps;
        double                      mRelTol;
        double                      mAbsTol;
        vector<PendingAssignment>   mAssignments;
        int                  		mRootCount;
        int         		        mCount;

	public:
                                    CvodeInterface(RoadRunner* rr, ModelFromC* oModel, const double& abTol = 1.e-12, const double& relTol = 1.e-12);
                                   ~CvodeInterface();

		void 						setTolerances(const double& aTol, const double& rTol);
        void                        assignResultsToModel();
		ModelFromC*					getModel();
        void                        testRootsAtInitialTime();
        bool                        haveVariables();

        double                      oneStep(const double& timeStart, const double& hstep);
        vector<double>              buildEvalArgument();
        void                        assignNewVector(ModelFromC *model);
        void                        assignNewVector(ModelFromC *oModel, bool bAssignNewTolerances);

        							// Restart the simulation using a different initial condition
        void                        reStart(double timeStart, ModelFromC* model);

};
}

#endif

//////C#
//////#define PRINT_EVENT_DEBUG
////
////using System;
////using System.Collections.Generic;
////using System.Diagnostics;
////using System.IO;
////using System.Runtime.InteropServices;
////using SBW;
////
////
////namespace LibRoadRunner.Solvers
////{
////    public class CvodeInterface : IDisposable
////    {
////        /// <summary>
////        /// Point to the CVODE DLL to use
////        /// </summary>
////        private const string CVODE = "cvodedll";
////
////        private static int nOneStepCount;
////
////        #region Default CVODE Settings
////
////        private const double defaultReltol = 1E-6;
////        private const double defaultAbsTol = 1E-12;
////        private const int defaultMaxNumSteps = 10000;
////        public static int defaultMaxAdamsOrder = 12;
////        public static int defaultMaxBDFOrder = 5;
////
////        public static int MaxAdamsOrder = defaultMaxAdamsOrder;
////        public static int MaxBDFOrder = defaultMaxBDFOrder;
////
////        public static double InitStep = 0.0;
////        public static double MinStep = 0.0;
////        public static double MaxStep = 0.0;
////
////
////        //static public double defaultReltol = 1E-15;
////        //static public double defaultAbsTol = 1E-20;
////
////        public static int MaxNumSteps = defaultMaxNumSteps;
////        public static double relTol = defaultReltol;
////        public static double absTol = defaultAbsTol;
////
////        #endregion
////
////        #region Class Variables
////
////        // Error codes
////        private const int CV_ROOT_RETURN = 2;
////        private const int CV_TSTOP_RETURN = 1;
////        private const int CV_SUCCESS = 0;
////        private const int CV_MEM_NULL = -1;
////        private const int CV_ILL_INPUT = -2;
////        private const int CV_NO_MALLOC = -3;
////        private const int CV_TOO_MUCH_WORK = -4;
////        private const int CV_TOO_MUCH_ACC = -5;
////        private const int CV_ERR_FAILURE = -6;
////        private const int CV_CONV_FAILURE = -7;
////        private const int CV_LINIT_FAIL = -8;
////        private const int CV_LSETUP_FAIL = -9;
////        private const int CV_LSOLVE_FAIL = -10;
////
////        private const int CV_MEM_FAIL = -11;
////
////        private const int CV_RTFUNC_NULL = -12;
////        private const int CV_NO_SLDET = -13;
////        private const int CV_BAD_K = -14;
////        private const int CV_BAD_T = -15;
////        private const int CV_BAD_DKY = -16;
////
////        private const int CV_PDATA_NULL = -17;
////        private static readonly string tempPathstring = Path.GetTempPath();
////
////
////        private static int errorFileCounter;
////        public static double lastTimeValue;
////        private readonly IntPtr gdata = IntPtr.Zero;
////        private IntPtr _amounts;
////        private IntPtr _rootsFound;
////        private IntPtr abstolArray;
////        private string cvodeLogFile = "cvodeLogFile";
////        private IntPtr cvodeMem;
////        public int errCode;
////
////        private IntPtr fileHandle;
////        private IModel model;
////        private int numIndependentVariables;
////
////        #endregion
////
////        #region Library Imports
////
////        [DllImport(CVODE, EntryPoint = "fileOpen", ExactSpelling = false,
////            CharSet = CharSet.Ansi, SetLastError = true)]
////        public static extern IntPtr fileOpen(string fileName);
////
////        [DllImport(CVODE, EntryPoint = "fileClose", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern void fileClose(IntPtr fp);
////
////
////        [DllImport(CVODE, EntryPoint = "NewCvode_Vector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern IntPtr NewCvode_Vector(int n);
////
////        [DllImport(CVODE, EntryPoint = "FreeCvode_Vector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern void FreeCvode_Vector(IntPtr vect);
////
////        [DllImport(CVODE, EntryPoint = "FreeCvode_Mem", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern void FreeCvode_Mem(IntPtr p);
////
////        // void *p
////
////        [DllImport(CVODE, EntryPoint = "Cvode_SetVector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern void Cvode_SetVector(IntPtr v, int Index, double Value);
////
////        [DllImport(CVODE, EntryPoint = "Cvode_GetVector", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern double Cvode_GetVector(IntPtr v, int Index);
////
////        [DllImport(CVODE, EntryPoint = "Create_BDF_NEWTON_CVode", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern IntPtr Create_BDF_NEWTON_CVode();
////
////        [DllImport(CVODE, EntryPoint = "Create_ADAMS_FUNCTIONAL_CVode", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern IntPtr Create_ADAMS_FUNCTIONAL_CVode();
////
////        [DllImport(CVODE, EntryPoint = "AllocateCvodeMem", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int AllocateCvodeMem(IntPtr cvode_mem, int n, TCallBackModelFcn fcn, double t0, IntPtr y,
////                                                  double reltol, IntPtr abstol);
////
////        [DllImport(CVODE, EntryPoint = "CvDense", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int CvDense(IntPtr cvode_mem, int n);
////
////        // int = size of systems
////
////        [DllImport(CVODE, EntryPoint = "CVReInit", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int CVReInit(IntPtr cvode_mem, double t0, IntPtr y0, double reltol, IntPtr abstol);
////
////        [DllImport(CVODE, EntryPoint = "Run_Cvode")]
////        //public static extern int  RunCvode (IntPtr cvode_mem, double tout, IntPtr  y, ref double t, string ErrMsg);
////        public static extern int RunCvode(IntPtr cvode_mem, double tout, IntPtr y, ref double t);
////
////        //public static extern int  RunCvode (IntPtr cvode_mem, double tout, IntPtr y, ref double t);  // t = double *
////
////
////        [DllImport(CVODE, EntryPoint = "CVGetRootInfo", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int CVGetRootInfo(IntPtr cvode_mem, IntPtr rootsFound);
////
////        [DllImport(CVODE, EntryPoint = "CVRootInit", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int CVRootInit(IntPtr cvode_mem, int numRoots, TCallBackRootFcn rootfcn, IntPtr gdata);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxNumSteps", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxNumSteps(IntPtr cvode_mem, int mxsteps);
////
////
////        [DllImport(CVODE, EntryPoint = "SetMinStep", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMinStep(IntPtr cvode_mem, double minStep);
////        [DllImport(CVODE, EntryPoint = "SetMaxStep", ExactSpelling = false,
////                    CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxStep(IntPtr cvode_mem, double maxStep);
////        [DllImport(CVODE, EntryPoint = "SetInitStep", ExactSpelling = false,
////                    CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetInitStep(IntPtr cvode_mem, double initStep);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxOrder", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxOrder(IntPtr cvode_mem, int mxorder);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxErrTestFails", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxErrTestFails(IntPtr cvode_mem, int maxnef);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxConvFails", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxConvFails(IntPtr cvode_mem, int maxncf);
////
////        [DllImport(CVODE, EntryPoint = "SetMaxNonLinIters", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetMaxNonLinIters(IntPtr cvode_mem, int maxcor);
////
////        [DllImport(CVODE, EntryPoint = "SetErrFile", ExactSpelling = false,
////            CharSet = CharSet.Unicode, SetLastError = true)]
////        public static extern int SetErrFile(IntPtr cvode_mem, IntPtr errfp);
////
////        #endregion
////
////        #region CVODE Callback functions
////
////        #region Delegates
////
////        public delegate void TCallBackModelFcn(int n, double time, IntPtr y, IntPtr ydot, IntPtr fdata);
////
////        public delegate void TCallBackRootFcn(double t, IntPtr y, IntPtr gdot, IntPtr gdata);
////
////        #endregion
////
////        public static TCallBackModelFcn modelDelegate;
////        public static TCallBackRootFcn eventDelegate;
////
////        private static int nCount;
////        private static int nRootCount;
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
////
////        public double[] GetCopy(double[] oVector)
////        {
////            var oResult = new double[oVector.Length];
////            oVector.CopyTo(oResult, 0);
////            return oResult;
////        }
////
////        public bool[] GetCopy(bool[] oVector)
////        {
////            var oResult = new bool[oVector.Length];
////            oVector.CopyTo(oResult, 0);
////            return oResult;
////        }
////
////        public void EventFcn(double time, IntPtr y, IntPtr gdot, IntPtr fdata)
////        {
////
////            var oldState = new ModelState(model);
////
////            model.evalModel(time, BuildEvalArgument());
////            AssignResultsToModel();
////            model.evalEvents(time, BuildEvalArgument());
////
////            Marshal.Copy(model.eventTests, 0, gdot, model.getNumEvents);
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
////        #region Constructor & Initialization
////
////        // -------------------------------------------------------------------------
////        // Constructor
////        // Model contains all the symbol tables associated with the model
////        // ev is the model function
////        // -------------------------------------------------------------------------
////
////
////        private int numAdditionalRules;
////
////        public CvodeInterface(IModel oModel)
////        {
////            Random = new Random();
////            InitializeCVODEInterface(oModel);
////        }
////
////        public bool HaveVariables
////        {
////            get { return (numAdditionalRules + numIndependentVariables > 0); }
////        }
////
////        private void InitializeCVODEInterface(IModel oModel)
////        {
////            try
////            {
////                model = oModel;
////                numIndependentVariables = oModel.getNumIndependentVariables;
////                numAdditionalRules = oModel.rateRules.Length;
////                modelDelegate = new TCallBackModelFcn(ModelFcn);
////                eventDelegate = new TCallBackRootFcn(EventFcn);
////
////                if (numAdditionalRules + numIndependentVariables > 0)
////                {
////                    int allocatedMemory = numIndependentVariables + numAdditionalRules;
////                    _amounts = NewCvode_Vector(allocatedMemory);
////                    abstolArray = NewCvode_Vector(allocatedMemory);
////                    for (int i = 0; i < allocatedMemory; i++)
////                    {
////                        Cvode_SetVector(abstolArray, i, defaultAbsTol);
////                    }
////
////                    AssignNewVector(oModel, true);
////
////                    cvodeMem = Create_BDF_NEWTON_CVode();
////                    SetMaxOrder(cvodeMem, MaxBDFOrder);
////                    //cvodeMem = Create_ADAMS_FUNCTIONAL_CVode();
////                    //SetMaxOrder(cvodeMem, MaxAdamsOrder);
////                    SetInitStep(cvodeMem, InitStep);
////                    SetMinStep(cvodeMem, MinStep);
////                    SetMaxStep(cvodeMem, MaxStep);
////
////                    SetMaxNumSteps(cvodeMem, MaxNumSteps);
////                    fileHandle = fileOpen(tempPathstring + cvodeLogFile + errorFileCounter + ".txt");
////                    SetErrFile(cvodeMem, fileHandle);
////                    errCode = AllocateCvodeMem(cvodeMem, allocatedMemory, modelDelegate, 0.0, _amounts, relTol,
////                                               abstolArray);
////                    if (errCode < 0) HandleCVODEError(errCode);
////                    if (oModel.getNumEvents > 0)
////                        errCode = CVRootInit(cvodeMem, oModel.getNumEvents, eventDelegate, gdata);
////                    errCode = CvDense(cvodeMem, allocatedMemory); // int = size of systems
////                    if (errCode < 0) HandleCVODEError(errCode);
////
////                    oModel.resetEvents();
////                }
////                else if (model.getNumEvents > 0)
////                {
////                    int allocated = 1;
////                    _amounts = NewCvode_Vector(allocated);
////                    abstolArray = NewCvode_Vector(allocated);
////                    Cvode_SetVector(_amounts, 0, 10f);
////                    Cvode_SetVector(abstolArray, 0, defaultAbsTol);
////
////                    cvodeMem = Create_BDF_NEWTON_CVode();
////                    SetMaxOrder(cvodeMem, MaxBDFOrder);
////                    //cvodeMem = Create_ADAMS_FUNCTIONAL_CVode();
////                    //SetMaxOrder(cvodeMem, MaxAdamsOrder);
////                    SetMaxNumSteps(cvodeMem, MaxNumSteps);
////                    fileHandle = fileOpen(tempPathstring + cvodeLogFile + errorFileCounter + ".txt");
////                    SetErrFile(cvodeMem, fileHandle);
////                    errCode = AllocateCvodeMem(cvodeMem, allocated, modelDelegate, 0.0, _amounts, relTol, abstolArray);
////                    if (errCode < 0) HandleCVODEError(errCode);
////                    if (oModel.getNumEvents > 0)
////                        errCode = CVRootInit(cvodeMem, oModel.getNumEvents, eventDelegate, gdata);
////                    errCode = CvDense(cvodeMem, allocated); // int = size of systems
////                    if (errCode < 0) HandleCVODEError(errCode);
////
////                    oModel.resetEvents();
////                }
////            }
////            catch (Exception ex)
////            {
////                throw new SBWApplicationException("Fatal Error while initializing CVODE", ex.Message);
////            }
////        }
////
////        #endregion
////
////        #region ErrorHandling
////
////        internal static CvodeErrorCodes[] errorCodes = InitilizeErrorCodes();
////
////        internal static CvodeErrorCodes[] InitilizeErrorCodes()
////        {
////            var oErrorCodes = new CvodeErrorCodes[28];
////            oErrorCodes[0] = new CvodeErrorCodes(CV_SUCCESS, "Success");
////            oErrorCodes[1] = new CvodeErrorCodes(-1, "The solver took mxstep internal steps but could not reach tout.");
////            oErrorCodes[2] = new CvodeErrorCodes(-2,
////                                                 "The solver could not satisfy the accuracy demanded by the user for some internal step.");
////            oErrorCodes[3] = new CvodeErrorCodes(-3,
////                                                 "Error test failures occurred too many times during one internal time step or minimum step size was reached.");
////            oErrorCodes[4] = new CvodeErrorCodes(-4,
////                                                 "Convergence test failures occurred too many times during one internal time step or minimum step size was reached.");
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
////            oErrorCodes[12] = new CvodeErrorCodes(-12, "The root?nding function failed in an unrecoverable manner.");
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
////        [DebuggerHidden, DebuggerStepThrough]
////        private void HandleCVODEError(int errCode)
////        {
////            if (errCode < 0)
////            {
////                string msg = "";
////                string errorFile = tempPathstring + cvodeLogFile + errorFileCounter + ".txt";
////
////                // and open a new file handle
////                errorFileCounter++;
////                IntPtr newHandle = fileOpen(tempPathstring + cvodeLogFile + errorFileCounter + ".txt");
////                if (newHandle != IntPtr.Zero && cvodeMem != IntPtr.Zero) SetErrFile(cvodeMem, newHandle);
////                // close error file used by the cvode library
////                if (fileHandle != IntPtr.Zero) fileClose(fileHandle);
////                fileHandle = newHandle;
////
////                try
////                {
////                    msg = File.ReadAllText(errorFile);
////                    File.Delete(errorFile);
////                }
////                catch (Exception)
////                {
////                    // actually we don't need this error message any more ...
////                    // the error code will automatically be converted by the erroCodes list
////                    // hence if we can't read the log file there is no reason to worry the user.
////                    //msg = " Unknown Error from CVODE (ff)";
////                }
////
////                throw new CvodeException("Error in RunCVode: " + errorCodes[-errCode].msg + msg);
////            }
////        }
////
////        #endregion
////
////        internal List<double> assignmentTimes = new List<double>();
////        internal List<PendingAssignment> assignments = new List<PendingAssignment>();
////        private bool followEvents = true;
////
////        internal double lastEvent;
////        //internal List<double> eventOccurance = new List<double>();
////
////        #region IDisposable Members
////
////        public void Dispose()
////        {
////            release();
////        }
////
////        #endregion
////
////        public double OneStep(double timeStart, double hstep)
////        {
////#if (PRINT_DEBUG)
////                    System.Diagnostics.Debug.WriteLine("---------------------------------------------------");
////                    System.Diagnostics.Debug.WriteLine("--- O N E     S T E P      ( " + nOneStepCount + " ) ");
////                    System.Diagnostics.Debug.WriteLine("---------------------------------------------------");
////#endif
////            nOneStepCount++;
////            nCount = 0;
////
////            double timeEnd = 0.0;
////            double tout = timeStart + hstep;
////            int strikes = 3;
////            try
////            {
////                // here we stop for a too small timestep ... this seems troublesome to me ...
////                while (tout - timeEnd > 1E-16)
////                {
////                    if (hstep < 1E-16) return tout;
////
////                    // here we bail in case we have no ODEs set up with CVODE ... though we should
////                    // still at least evaluate the model function
////                    if (!HaveVariables && model.getNumEvents == 0)
////                    {
////                        model.convertToAmounts();
////                        model.evalModel(tout, BuildEvalArgument());
////                        return tout;
////                    }
////
////
////                    if (lastTimeValue > timeStart)
////                        reStart(timeStart, model);
////
////                    double nextTargetEndTime = tout;
////                    if (assignmentTimes.Count > 0 && assignmentTimes[0] < nextTargetEndTime)
////                    {
////                        nextTargetEndTime = assignmentTimes[0];
////                        assignmentTimes.RemoveAt(0);
////                    }
////
////                    int nResult = RunCvode(cvodeMem, nextTargetEndTime, _amounts, ref timeEnd); // t = double *
////
////                    if (nResult == CV_ROOT_RETURN && followEvents)
////                    {
////#if (PRINT_DEBUG)
////                                System.Diagnostics.Debug.WriteLine("---------------------------------------------------");
////                                System.Diagnostics.Debug.WriteLine("--- E V E N T      ( " + nOneStepCount + " ) ");
////                                System.Diagnostics.Debug.WriteLine("---------------------------------------------------");
////#endif
////
////                        //bool tooCloseToStart = Math.Abs(timeEnd - timeStart) > absTol;
////                        bool tooCloseToStart = Math.Abs(timeEnd - lastEvent) > relTol;
////                        if (tooCloseToStart)
////                            strikes = 3;
////                        else
////                            strikes--;
////
////                        if (tooCloseToStart || strikes > 0)
////                        {
////                            HandleRootsFound(ref timeEnd, tout);
////                            reStart(timeEnd, model);
////                            lastEvent = timeEnd;
////                        }
////                    }
////                    else if (nResult == CV_SUCCESS || !followEvents)
////                    {
////                        //model.resetEvents();
////                        model.time = tout;
////                        AssignResultsToModel();
////                    }
////                    else
////                    {
////                        HandleCVODEError(nResult);
////                    }
////
////                    lastTimeValue = timeEnd;
////
////                    try
////                    {
////                        model.testConstraints();
////                    }
////                    catch (Exception e)
////                    {
////                        model.Warnings.Add("Constraint Violated at time = " + timeEnd + "\n" +
////                                                             e.Message);
////                    }
////
////                    AssignPendingEvents(timeEnd, tout);
////
////                    if (tout - timeEnd > 1E-16)
////                        timeStart = timeEnd;
////                }
////                return (timeEnd);
////            }
////            catch (NullReferenceException ex)
////            {
////                throw new SBWApplicationException("Internal error, please reload the model", ex.StackTrace);
////            }
////            catch (Exception)
////            {
////                InitializeCVODEInterface(model);
////                throw;
////            }
////        }
////
////        private void AssignPendingEvents(double timeEnd, double tout)
////        {
////            for (int i = assignments.Count - 1; i >= 0; i--)
////            {
////                if (timeEnd >= assignments[i].Time)
////                {
////                    model.time = tout;
////                    AssignResultsToModel();
////                    model.convertToConcentrations();
////                    model.updateDependentSpeciesValues(model.y);
////                    assignments[i].AssignToModel();
////                    if (!RoadRunner._bConservedTotalChanged) model.computeConservedTotals();
////                    model.convertToAmounts();
////                    model.evalModel(timeEnd, BuildEvalArgument());
////
////                    reStart(timeEnd, model);
////
////                    assignments.RemoveAt(i);
////                }
////            }
////        }
////
////        private List<int> RetestEvents(double timeEnd, List<int> handledEvents)
////        {
////            return RetestEvents(timeEnd, handledEvents, false);
////        }
////
////        private List<int> RetestEvents(double timeEnd, List<int> handledEvents, out List<int> removeEvents)
////        {
////            return RetestEvents(timeEnd, handledEvents, false, out removeEvents);
////        }
////
////        private List<int> RetestEvents(double timeEnd, List<int> handledEvents, bool assignOldState)
////        {
////            List<int> removeEvents;
////            return RetestEvents(timeEnd, handledEvents, assignOldState, out removeEvents);
////        }
////
////        private List<int> RetestEvents(double timeEnd, List<int> handledEvents, bool assignOldState, out List<int> removeEvents)
////        {
////            var result = new List<int>();
////            removeEvents = new List<int>();
////
////            if (!RoadRunner._bConservedTotalChanged) model.computeConservedTotals();
////            model.convertToAmounts();
////            model.evalModel(timeEnd, BuildEvalArgument());
////
////            var oldState = new ModelState(model);
////            model.evalEvents(timeEnd, BuildEvalArgument());
////
////            for (int i = 0; i < model.getNumEvents; i++)
////            {
////                if (model.eventStatusArray[i] == true && oldState.EventStatusArray[i] == false && !handledEvents.Contains(i))
////                    result.Add(i);
////                if (model.eventStatusArray[i] == false && oldState.EventStatusArray[i] == true && !model.eventPersistentType[i])
////                {
////                    removeEvents.Add(i);
////                }
////            }
////            if (assignOldState)
////                oldState.AssignToModel(model);
////
////            return result;
////        }
////
////
////        private void HandleRootsFound(ref double timeEnd, double tout)
////        {
////            var rootsFound = new int[model.getNumEvents];
////            // Create some space for the CVGetRootInfo call
////            _rootsFound = Marshal.AllocHGlobal(model.getNumEvents*sizeof (Int32));
////            CVGetRootInfo(cvodeMem, _rootsFound);
////            Marshal.Copy(_rootsFound, rootsFound, 0, model.getNumEvents);
////            Marshal.FreeHGlobal(_rootsFound); // Free space used by CVGetRootInfo
////
////            HandleRootsForTime(timeEnd, rootsFound);
////
////        }
////
////        internal void TestRootsAtInitialTime()
////        {
////            var events = RetestEvents(0, new List<int>(), true);
////            if (events.Count > 0)
////            {
////                var rootsFound = new int[model.getNumEvents];
////                foreach (int item in events)
////                {
////                    rootsFound[item] = 1;
////                }
////                HandleRootsForTime(0, rootsFound);
////
////            }
////        }
////
////        private void RemovePendingAssignmentForIndex(int eventIndex)
////        {
////            for (int j = assignments.Count - 1; j >= 0; j--)
////            {
////                if (assignments[j].Index == eventIndex)
////                    assignments.RemoveAt(j);
////            }
////        }
////
////        private void SortEventsByPriority(List<int> firedEvents)
////        {
////            if ((firedEvents.Count > 1))
////            {
////                model.computeEventPriorites();
////                firedEvents.Sort(new Comparison<int>((index1, index2) =>
////                                                         {
////                                                             double priority1 = model.eventPriorities[index1];
////                                                             double priority2 = model.eventPriorities[index2];
////
////                                                             // random toss in case we have same priorites
////                                                             if (priority1 == priority2 && priority1 != 0 &&
////                                                                 index1 != index2)
////                                                             {
////                                                                 if (Random.NextDouble() > 0.5)
////                                                                     return -1;
////                                                                 else
////                                                                     return 1;
////                                                             }
////
////                                                             return -1 * priority1.CompareTo(priority2);
////                                                         }));
////            }
////        }
////        private void HandleRootsForTime(double timeEnd, int[] rootsFound)
////        {
////            AssignResultsToModel();
////            model.convertToConcentrations();
////            model.updateDependentSpeciesValues(model.y);
////
////            model.evalEvents(timeEnd, BuildEvalArgument());
////
////
////            var firedEvents = new List<int>();
////            var preComputedAssignments = new Dictionary<int, double[]>();
////
////
////            for (int i = 0; i < model.getNumEvents; i++)
////            {
////                // We only fire an event if we transition from false to true
////                if (rootsFound[i] == 1)
////                {
////                    if (model.eventStatusArray[i])
////                    {
////                        firedEvents.Add(i);
////                        if (model.eventType[i])
////                            preComputedAssignments[i] = model.computeEventAssignments[i]();
////                    }
////                }
////                else
////                {
////                    // if the trigger condition is not supposed to be persistent, remove the event from the firedEvents list;
////                    if (!model.eventPersistentType[i])
////                    {
////                        RemovePendingAssignmentForIndex(i);
////                    }
////                }
////            }
////            var handled = new List<int>();
////            while (firedEvents.Count > 0)
////            {
////                SortEventsByPriority(firedEvents);
////                // Call event assignment if the eventstatus flag for the particular event is false
////                for (int i = 0; i < firedEvents.Count; i++)
////                {
////                    var currentEvent = firedEvents[i];
////                    // We only fire an event if we transition from false to true
////
////                    model.previousEventStatusArray[currentEvent] = model.eventStatusArray[currentEvent];
////                    double eventDelay = model.eventDelay[currentEvent]();
////                    if (eventDelay == 0)
////                    {
////                        if (model.eventType[currentEvent] && preComputedAssignments.ContainsKey(currentEvent))
////                            model.performEventAssignments[currentEvent](preComputedAssignments[currentEvent]);
////                        else
////                            model.eventAssignments[currentEvent]();
////
////                        handled.Add(currentEvent);
////                        List<int> removeEvents;
////                        var additionalEvents = RetestEvents(timeEnd, handled, out removeEvents);
////                        firedEvents.AddRange(additionalEvents);
////
////                        foreach (var newEvent in additionalEvents)
////                        {
////                            if (model.eventType[newEvent])
////                                preComputedAssignments[newEvent] = model.computeEventAssignments[newEvent]();
////                        }
////
////                        model.eventStatusArray[currentEvent] = false;
////                        firedEvents.RemoveAt(i);
////
////                        foreach (var item in removeEvents)
////                        {
////                            if (firedEvents.Contains(item))
////                            {
////                                firedEvents.Remove(item);
////                                RemovePendingAssignmentForIndex(item);
////                            }
////                        }
////
////                        break;
////                    }
////                    else
////                    {
////                        if (!assignmentTimes.Contains(timeEnd + eventDelay))
////                            assignmentTimes.Add(timeEnd + eventDelay);
////
////
////                        var pending = new PendingAssignment(
////                                                                    timeEnd + eventDelay,
////                                                                    model.computeEventAssignments[currentEvent],
////                                                                    model.performEventAssignments[currentEvent],
////                                                                    model.eventType[currentEvent], currentEvent);
////
////                        if (model.eventType[currentEvent] && preComputedAssignments.ContainsKey(currentEvent))
////                            pending.ComputedValues = preComputedAssignments[currentEvent];
////
////                        assignments.Add(pending);
////                        model.eventStatusArray[currentEvent] = false;
////                        firedEvents.RemoveAt(i);
////                        break;
////                    }
////
////#if (PRINT_DEBUG)
////                            System.Diagnostics.Debug.WriteLine("time: " + model.time.ToString("F4") + " Event " + (i + 1).ToString());
////#endif
////                }
////            }
////
////            if (!RoadRunner._bConservedTotalChanged) model.computeConservedTotals();
////            model.convertToAmounts();
////
////
////            model.evalModel(timeEnd, BuildEvalArgument());
////            double[] dCurrentValues = model.GetCurrentValues();
////            for (int k = 0; k < numAdditionalRules; k++)
////                Cvode_SetVector(_amounts, k, dCurrentValues[k]);
////
////            for (int k = 0; k < numIndependentVariables; k++)
////                Cvode_SetVector(_amounts, k + numAdditionalRules, model.amounts[k]);
////
////            CVReInit(cvodeMem, timeEnd, _amounts, relTol, abstolArray);
////            assignmentTimes.Sort();
////        }
////
////
////        private void AssignResultsToModel()
////        {
////            model.updateDependentSpeciesValues(model.y);
////            var dTemp = new double[numAdditionalRules];
////            for (int i = 0; i < numAdditionalRules; i++)
////            {
////                dTemp[i] = Cvode_GetVector(_amounts, i);
////            }
////            for (int i = 0; i < numIndependentVariables; i++)
////                model.amounts[i] = Cvode_GetVector(_amounts, i + numAdditionalRules);
////
////            model.computeRules(BuildEvalArgument());
////            model.AssignRates(dTemp);
////            model.computeAllRatesOfChange();
////        }
////
////
////        // Restart the simulation using a different initial condition
////        public void AssignNewVector(IModel oModel, bool bAssignNewTolerances)
////        {
////            double[] dTemp = model.GetCurrentValues();
////            double dMin = absTol;
////
////            for (int i = 0; i < numAdditionalRules; i++)
////            {
////                if (dTemp[i] > 0 && dTemp[i]/1000f < dMin) dMin = dTemp[i]/1000f;
////            }
////            for (int i = 0; i < numIndependentVariables; i++)
////            {
////                if (oModel.amounts[i] > 0 && oModel.amounts[i]/1000f < dMin) dMin = oModel.amounts[i]/1000f;
////            }
////
////            for (int i = 0; i < numAdditionalRules; i++)
////            {
////                if (bAssignNewTolerances) setAbsTolerance(i, dMin);
////                Cvode_SetVector(_amounts, i, dTemp[i]);
////            }
////            for (int i = 0; i < numIndependentVariables; i++)
////            {
////                if (bAssignNewTolerances) setAbsTolerance(i + numAdditionalRules, dMin);
////                Cvode_SetVector(_amounts, i + numAdditionalRules, oModel.amounts[i]);
////            }
////
////
////            if (!HaveVariables && model.getNumEvents > 0)
////            {
////                if (bAssignNewTolerances) setAbsTolerance(0, dMin);
////                Cvode_SetVector(_amounts, 0, 1f);
////            }
////
////            //if (bAssignNewTolerances)
////            //{
////            //    System.Diagnostics.Debug.WriteLine(string.Format("Set tolerance to: {0:G}", dMin));
////            //}
////        }
////
////
////        private void AssignNewVector(IModel model)
////        {
////            AssignNewVector(model, false);
////        }
////
////        public void setAbsTolerance(int index, double dValue)
////        {
////            double dTolerance = dValue;
////            if (dValue > 0 && absTol > dValue)
////                dTolerance = dValue;
////            else
////                dTolerance = absTol;
////
////            Cvode_SetVector(abstolArray, index, dTolerance);
////        }
////
////        public int reStart(double timeStart, IModel model)
////        {
////            AssignNewVector(model);
////
////            SetInitStep(cvodeMem, InitStep);
////            SetMinStep(cvodeMem, MinStep);
////            SetMaxStep(cvodeMem, MaxStep);
////
////            return (CVReInit(cvodeMem, timeStart, _amounts, relTol, abstolArray));
////        }
////
////        public double getValue(int index)
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
////
////        public void release()
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
////        public Random Random { get; set; }
////    }
////}
