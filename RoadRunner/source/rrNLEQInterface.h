#ifndef rrNLEQInterfaceH
#define rrNLEQInterfaceH
#include <vector>
#include "rrObject.h"
#include "rrModelFromC.h"
#include "rrSteadyStateSolver.h"
using std::vector;

namespace rr
{

void ModelFunction(int* nx, double* y, double* fval, int* pErr);

class RR_DECLSPEC NLEQInterface : public ISteadyStateSolver
{
    protected:
        int                             nOpts;
        long                           *IWK;
        long                            LIWK;
        long                            LWRK;
        double                         *RWK;
        double                         *XScal;
        long                            ierr;
        long                           *iopt;
        static ModelFromC              *model;     // Model generated from the SBML. Static so we can access it from standalone function
        static long                     n;

    public:
        bool                            isAvailable();
        static ModelFromC*              getModel();
        static long						getN();
        int                             defaultMaxInterations;
        int                             maxIterations;
        double                          defaultTolerance;
        double                          relativeTolerance;

                                        /// <summary>
                                        /// Creates a new Instance of NLEQ for the given Model
                                        /// </summary>
                                        /// <param name="model">the model to create NLEQ for</param>
                                        NLEQInterface(ModelFromC *_model = NULL);

                                        /// <summary>
                                        /// Sets the Scaling Factors
                                        /// </summary>
                                        /// <param name="sx">Array of Scaling factors</param>
        void                            setScalingFactors(const vector<double>& sx);

                                        /// <summary>
                                        /// Returns the Number of Newton Iterations
                                        /// </summary>
                                        /// <returns>the Number of Newton Iterations</returns>
        int                             getNumberOfNewtonIterations();

                                        /// <summary>
                                        /// Returns the Number of Corrector steps
                                        /// </summary>
                                        /// <returns>Returns the Number of Corrector steps</returns>
        int                             getNumberOfCorrectorSteps();

                                        /// <summary>
                                        /// Returns the Number of Model Evaluations
                                        /// </summary>
                                        /// <returns>the Number of Model Evaluations</returns>
        int                             getNumberOfModelEvaluations();

                                        /// <summary>
                                        /// Returns the Number Of Jacobian Evaluations
                                        /// </summary>
                                        /// <returns>the Number Of Jacobian Evaluations</returns>
        int                             getNumberOfJacobianEvaluations();

                                        /// <summary>
                                        /// Returns the Number of Model Evaluations For Jacobian
                                        /// </summary>
                                        /// <returns>the Number of Model Evaluations For Jacobian</returns>
        int                             getNumberOfModelEvaluationsForJacobian();

                                        /// <summary>
                                        /// Thea actual solver routine making the call to NLEQ1
                                        /// </summary>
                                        /// <param name="yin">Array of Model variables</param>
                                        /// <returns>sums of squares </returns>
        double                          solve(const vector<double>& yin);
        double                          computeSumsOfSquares();
    };
}

#endif

////c#
////using System;
////using System.Diagnostics;
////using System.Runtime.InteropServices;
////using SBW;
////
////namespace LibRoadRunner.Solvers
////{
////    public interface ISteadyStateSolver
////    {
////        /// <summary>
////        /// Thea actual solver rourine making the call to NLEQ1
////        /// </summary>
////        /// <param name="yin">Array of Model variables</param>
////        /// <returns>sums of squares </returns>
////        double solve(double[] yin);
////    }
////    public class NLEQInterface : ISteadyStateSolver
////    {
////        #region Delegates
////
////        public delegate void TCallBackModelFcn(IntPtr nx, IntPtr y, IntPtr fval, IntPtr pErr);
////
////        #endregion
////
////        private static int nOpts = 50;
////        public static readonly double defaultTolerance = 1e-4;
////        public static readonly int defaultMaxInterations = 100;
////
////        internal static double _relativeTolerance = defaultTolerance;
////        internal static int _maxIterations = defaultMaxInterations;
////        public static TCallBackModelFcn fcn;
////
////        private int[] IWK;
////        private int LIWK;
////        private int LWRK;
////        private double[] RWK;
////        private double[] XScal;
////        private int ierr;
////        private int[] iopt = new int[nOpts];
////
////        private IModel model; // Model generated from the SBML
////        private int n;
////
////        /// <summary>
////        /// This function test Nleq by running it to see whether it would be working.
////        /// </summary>
////        public static bool IsAvailable
////        {
////            get
////            {
////                try
////                {
////                    var temp = new NLEQInterface();
////                    temp.getNumberOfModelEvaluations();
////                    return true;
////                }
////                catch
////                {
////                    return false;
////                }
////            }
////        }
////        private NLEQInterface()
////        {
////            n = 1;
////            var data = new double[n];
////            fcn = new TCallBackModelFcn(ModelFcn);
////
////            // Allocate space, see NLEQ docs for details
////            LWRK = (n + 2 + 15) * n + 61;
////            LIWK = n + 52;
////
////            XScal = new double[n];
////            IWK = new int[LIWK];
////            RWK = new double[LWRK];
////
////            // Set up default scaling factors
////            for (int i = 0; i < n; i++) XScal[i] = 1.0;
////
////            for (int i = 0; i < nOpts; i++) iopt[i] = 0;
////            iopt[31 - 1] = 4; // Set for Highly nonlinear problem
////
////            // Initialise all array elements to 0.0
////            for (int i = 0; i < LIWK; i++)
////                IWK[i] = 0;
////            IWK[31 - 1] = maxIterations; // Max iterations
////            for (int i = 0; i < LWRK; i++)
////                RWK[i] = 0.0;
////            RWK[22 - 1] = 1E-16; // Minimal allowed damping factor
////
////            ierr = 0;
////            IWK[31 - 1] = maxIterations; // Max iterations
////
////            // Set up default scaling factors
////            for (int i = 0; i < n; i++) XScal[i] = 1.0;
////
////            for (int i = 0; i < nOpts; i++) iopt[i] = 0;
////            iopt[31 - 1] = 3; // Set for Highly nonlinear problem
////
////            // Initialise all array elements to 0.0
////            for (int i = 0; i < LIWK; i++)
////                IWK[i] = 0;
////            IWK[31 - 1] = maxIterations; // Max iterations
////            for (int i = 0; i < LWRK; i++)
////                RWK[i] = 0.0;
////            RWK[22 - 1] = 1E-20; // Minimal allowed damping factor
////
////            // For some reason NLEQ modifies the tolerance value, use a local copy instead
////            double tmpTol = _relativeTolerance;
////
////            NLEQ1(ref n, fcn, null, data, XScal, ref tmpTol, iopt, ref ierr, ref LIWK, IWK, ref LWRK, RWK);
////        }
////
////        /// <summary>
////        /// Creates a new Instance of NLEQ for the given Model
////        /// </summary>
////        /// <param name="model">the model to create NLEQ for</param>
////        public NLEQInterface(IModel model)
////        {
////            try
////            {
////                this.model = model;
////                n = model.getNumIndependentVariables;
////
////                fcn = new TCallBackModelFcn(ModelFcn);
////
////                // Allocate space, see NLEQ docs for details
////                LWRK = (n + 2 + 15)*n + 61;
////                LIWK = n + 52;
////
////                XScal = new double[n];
////                IWK = new int[LIWK];
////                RWK = new double[LWRK];
////
////                // Set up default scaling factors
////                for (int i = 0; i < n; i++) XScal[i] = 1.0;
////
////                for (int i = 0; i < nOpts; i++) iopt[i] = 0;
////                iopt[31 - 1] = 4; // Set for Highly nonlinear problem
////
////                // Initialise all array elements to 0.0
////                for (int i = 0; i < LIWK; i++)
////                    IWK[i] = 0;
////                IWK[31 - 1] = maxIterations; // Max iterations
////                for (int i = 0; i < LWRK; i++)
////                    RWK[i] = 0.0;
////                RWK[22 - 1] = 1E-16; // Minimal allowed damping factor
////
////
////            }
////            catch (Exception e)
////            {
////                throw new SBWException("Error during NLEQ Obejct creation:", e.Message);
////            }
////        }
////
////        public static double relativeTolerance
////        {
////            get { return _relativeTolerance; // return the value from privte field.
////            }
////            set { _relativeTolerance = value; // save value into private field.
////            }
////        }
////
////        public static int maxIterations
////        {
////            get { return _maxIterations; // return the value from privte field.
////            }
////            set { _maxIterations = value; // save value into private field.
////            }
////        }
////
////        // NLEQ2 seems to have problems with some models so we drop back to NLEQ1 for now.
////
////        //        [DllImport ("nleq2", EntryPoint="NLEQ2", ExactSpelling=false,
////        //             CharSet=CharSet.Unicode, SetLastError=true, CallingConvention=CallingConvention.Cdecl
////        //             )]
////        [DllImport("NleqLib", EntryPoint = "NLEQ1")]
////        //         NLEQ is a FORTRAN routine, therefore everything must be a reference
////        public static extern IntPtr NLEQ1(
////            ref int n,
////            TCallBackModelFcn fcn,
////            [In, Out] double[,] Jacobian,
////            [In, Out] double[] x,
////            [In, Out] double[] xscal,
////            ref double rtol,
////            [In, Out] int[] iopt,
////            ref int ierr,
////            ref int LIWK,
////            [In, Out] int[] IWK,
////            ref int LRWK,
////            [In, Out] double[] RWK);
////
////        //                [DllImport ("nleq2", EntryPoint="NLEQ2")                     ]
////        ////         NLEQ is a FORTRAN routine, therefore everything must be a reference
////        //        public static extern IntPtr NLEQ2(
////        //            ref int n,
////        //            TCallBackModelFcn fcn,
////        //            [In, Out] double[,] Jacobian,
////        //            [In, Out] double[] x,
////        //            [In, Out] double[] xscal,
////        //            ref double rtol,
////        //            [In, Out] int[] iopt,
////        //            ref int ierr,
////        //            ref int LIWK,
////        //            [In, Out] int[] IWK,
////        //            ref int LRWK,
////        //            [In, Out] double[] RWK);
////
////
////        /// <summary>
////        /// This is the function that's called by NLEQ
////        /// </summary>
////        /// <param name="nx"></param>
////        /// <param name="y"></param>
////        /// <param name="fval"></param>
////        /// <param name="pErr"></param>
////        private void ModelFcn(IntPtr nx, IntPtr y, IntPtr fval, IntPtr pErr)
////        {
////            if (model == null)
////            {
////                var temp = new double[n];
////                Marshal.Copy(temp, 0, fval, n);
////                Marshal.WriteInt32(pErr, 0);
////                return;
////            }
////
////
////            try
////            {
////                Marshal.Copy(y, model.amounts, 0, n);
////                var dTemp = new double[model.amounts.Length + model.rateRules.Length];
////                model.rateRules.CopyTo(dTemp, 0);
////                model.amounts.CopyTo(dTemp, model.rateRules.Length);
////                model.evalModel(0.0, dTemp);
////                //                bool bError = false;
////
////                //                for (int i = 0; i < model.amounts.Length; i++)
////                //                    if (model.amounts[i] < 0)
////                //                    {
////                //                        bError = true;
////                //                        break;
////                //                    }
////                //
////
////                Marshal.Copy(model.dydt, 0, fval, n);
////                //                if (bError)
////                //                    Marshal.WriteInt32(pErr, -1);
////                //                else
////                Marshal.WriteInt32(pErr, 0);
////            }
////            catch (Exception)
////            {
////            }
////        }
////
////
////        /// <summary>
////        /// Sets the Scaling Factors
////        /// </summary>
////        /// <param name="sx">Array of Scaling factors</param>
////        public void setScalingFactors(double[] sx)
////        {
////            for (int i = 0; i < n; i++)
////                XScal[i] = sx[i];
////        }
////
////
////        /// <summary>
////        /// Returns the Number of Newton Iterations
////        /// </summary>
////        /// <returns>the Number of Newton Iterations</returns>
////        public int getNumberOfNewtonIterations()
////        {
////            return IWK[0];
////        }
////
////        /// <summary>
////        /// Returns the Number of Corrector steps
////        /// </summary>
////        /// <returns>Returns the Number of Corrector steps</returns>
////        public int getNumberOfCorrectorSteps()
////        {
////            return IWK[2];
////        }
////
////        /// <summary>
////        /// Returns the Number of Model Evaluations
////        /// </summary>
////        /// <returns>the Number of Model Evaluations</returns>
////        public int getNumberOfModelEvaluations()
////        {
////            return IWK[3];
////        }
////
////        /// <summary>
////        /// Returns the Number Of Jacobian Evaluations
////        /// </summary>
////        /// <returns>the Number Of Jacobian Evaluations</returns>
////        public int getNumberOfJacobianEvaluations()
////        {
////            return IWK[4];
////        }
////
////        /// <summary>
////        /// Returns the Number of Model Evaluations For Jacobian
////        /// </summary>
////        /// <returns>the Number of Model Evaluations For Jacobian</returns>
////        public int getNumberOfModelEvaluationsForJacobian()
////        {
////            return IWK[7];
////        }
////
////
////        static void Test(string fileName)
////        {
////            var rr = new RoadRunner();
////            rr.loadSBMLFromFile(fileName);
////            Debug.WriteLine (rr.steadyState());
////        }
////
////        /// <summary>
////        /// Thea actual solver rourine making the call to NLEQ1
////        /// </summary>
////        /// <param name="yin">Array of Model variables</param>
////        /// <returns>sums of squares </returns>
////        public double solve(double[] yin)
////        {
////            try
////            {
////                if (yin == null || yin.Length == 0) return 0;
////                // Set up a dummy Jacobian, actual Jacobian is computed
////                // by NLEQ using finite differences
////                var Jacobian = new double[1,1];
////
////                //for (int i=0; i<n; i++) model.amounts[i] = yin[i];
////
////                ierr = 0;
////                IWK[31 - 1] = maxIterations; // Max iterations
////
////                // Set up default scaling factors
////                for (int i = 0; i < n; i++) XScal[i] = 1.0;
////
////                for (int i = 0; i < nOpts; i++) iopt[i] = 0;
////                iopt[31 - 1] = 3; // Set for Highly nonlinear problem
////
////                // Initialise all array elements to 0.0
////                for (int i = 0; i < LIWK; i++)
////                    IWK[i] = 0;
////                IWK[31 - 1] = maxIterations; // Max iterations
////                for (int i = 0; i < LWRK; i++)
////                    RWK[i] = 0.0;
////                RWK[22 - 1] = 1E-20; // Minimal allowed damping factor
////
////                // For some reason NLEQ modifies the tolerance value, use a local copy instead
////                double tmpTol = _relativeTolerance;
////
////                NLEQ1(ref n, fcn, null, model.amounts, XScal, ref tmpTol, iopt, ref ierr, ref LIWK, IWK, ref LWRK, RWK);
////
////                if (ierr == 2) // retry
////                {
////                    for (int i = 0; i < nOpts; i++) iopt[i] = 0;
////                    iopt[31 - 1] = 3; // Set for Highly nonlinear problem
////                    iopt[0] = 1; // Try again but tell NLEQ not to reinitialize
////                    tmpTol = _relativeTolerance;
////                    NLEQ1(ref n, fcn, Jacobian, model.amounts, XScal, ref tmpTol, iopt, ref ierr, ref LIWK, IWK,
////                          ref LWRK, RWK);
////                    // If we get the same error then give up
////                }
////
////
////                ThrowErrorForStatus();
////                return ComputeSumsOfSquares();
////            }
////            catch (NLEQException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWException("Unexpected error from solve routine of NLEQ: ", e.Message);
////            }
////        }
////
////
////        private void ThrowErrorForStatus()
////        {
////            if (ierr > 0)
////            {
////                switch (ierr)
////                {
////                    case 1:
////                        throw new NLEQException("Jacobian matrix singular in NLEQ");
////                    case 2:
////                        throw new NLEQException("Maximum iterations exceeded");
////                    case 3:
////                        throw new NLEQException("Damping factor has became to small to continue");
////                    case 4:
////                        throw new NLEQException(
////                            "Warning: Superlinear or quadratic convergence slowed down near the solution");
////                    case 5:
////                        throw new NLEQException("Warning: Error Tolerance reached but solution is suspect");
////                    case 10:
////                        throw new NLEQException("Integer or real workspace too small in NLEQ");
////                    case 20:
////                        throw new NLEQException("Bad input to size of model parameter");
////                    case 21:
////                        throw new NLEQException("Nonpositive value for RTOL supplied to NLEQ");
////                    case 22:
////                        throw new NLEQException("Negative scaling value via vector XSCAL supplied");
////                    case 30:
////                        throw new NLEQException("One or more fields specified in IOPT are invalid (NLEQ)");
////                    case 80:
////                        throw new NLEQException("Error signalled by linear solver routine N1FACT, in NLEQ");
////                    case 81:
////                        throw new NLEQException("Error signalled by linear solver routine N1SOLV, in NLEQ");
////                    case 82:
////                        throw new NLEQException("Possible negative concentrations in solution (NLEQ)");
////                    case 83:
////                        throw new NLEQException("Error signalled by user routine JAC in NLEQ");
////                    default:
////                        throw new NLEQException(String.Format("Unknown error in NLEQ, errCode = {0}", ierr));
////                }
////            }
////        }
////
////        public double ComputeSumsOfSquares()
////        {
////            // Compute the sums of squares and return value to caller
////            double sum = 0;
////            var dTemp = new double[model.amounts.Length + model.rateRules.Length];
////            model.rateRules.CopyTo(dTemp, 0);
////            model.amounts.CopyTo(dTemp, model.rateRules.Length);
////            model.evalModel(0.0, dTemp);
////            for (int i = 0; i < n; i++)
////                sum = sum + Math.Pow(model.dydt[i], 2.0);
////            return Math.Sqrt(sum);
////        }
////    }
////}

