#ifndef rrCvodedllH
#define rrCvodedllH
#include <stdlib.h>
#include <string>
#include "rrExporter.h"
#include "cvode/cvode.h"

using namespace std;
namespace rr
{


//// N_Vector is a point to an N_Vector structure
//RR_DECLSPEC void         Cvode_SetVector (N_Vector v, int Index, double Value);
//RR_DECLSPEC double       Cvode_GetVector (N_Vector v, int Index);
//
//RR_DECLSPEC int          AllocateCvodeMem (        void *,
//                                                int n,
//                                                TModelCallBack what1,
//                                                double what2,
//                                                N_Vector whatIsIt,
//                                                double what3,
//                                                N_Vector whatIsThis);//, long int[], double[]);
//
////RR_DECLSPEC int         CvDense (void *, int);  // int = size of systems
//RR_DECLSPEC int         CVReInit (void *cvode_mem, double t0, N_Vector y0, double reltol, N_Vector abstol);
//RR_DECLSPEC int         Run_Cvode (void *cvode_mem, double tout, N_Vector y, double *t);
//RR_DECLSPEC int         CVGetRootInfo (void *cvode_mem, int *rootsFound);
//RR_DECLSPEC int         CVRootInit (void *cvode_mem, int numRoots, TRootCallBack callBack, void *gdata);
//RR_DECLSPEC int         SetMaxNumSteps(void *cvode_mem, int mxsteps);
//RR_DECLSPEC int         SetMaxOrder(void *cvode_mem, int mxorder);
//RR_DECLSPEC int         CVSetFData (void *cvode_mem, void *f_data);
//RR_DECLSPEC int         SetMaxErrTestFails(void *cvode_mem, int maxnef);
//RR_DECLSPEC int         SetMaxConvFails(void *cvode_mem, int maxncf);
//RR_DECLSPEC int         SetMaxNonLinIters (void *cvode_mem, int maxcor);
//RR_DECLSPEC int         SetErrFile (void *cvode_mem, FILE *errfp);
//RR_DECLSPEC int         SetErrHandler (void *cvode_mem, CVErrHandlerFn callback, void* user_data );
//RR_DECLSPEC int         SetMinStep(void *cvode_mem, double minStep);
//RR_DECLSPEC int         SetMaxStep(void *cvode_mem, double maxStep);
//RR_DECLSPEC int         SetInitStep(void *cvode_mem, double initStep);
//
//RR_DECLSPEC int         InternalFunctionCall(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *f_data);
//RR_DECLSPEC int         InternalRootCall (realtype t, N_Vector y, realtype *gout, void *g_data);
}

#endif

////#include <stdlib.h>
////#ifdef WIN32
////#include <windows.h>
////
//////BOOL APIENTRY DllMain( HANDLE hModule,
//////    DWORD  ul_reason_for_call,
//////    LPVOID lpReserved
//////    )
//////{
//////    return TRUE;
//////}
////
////
////
////// N_Vector is a point to an N_Vector structure
////
////DLLEXPORT void*     NewCvode_Vector(int);
////DLLEXPORT void         FreeCvode_Vector (N_Vector);
////DLLEXPORT void         FreeCvode_Mem (void **p);
////DLLEXPORT void         Cvode_SetVector (N_Vector v, int Index, double Value);
////DLLEXPORT double     Cvode_GetVector (N_Vector v, int Index);
////
////DLLEXPORT void*        Create_BDF_NEWTON_CVode();
////DLLEXPORT void*        Create_ADAMS_FUNCTIONAL_CVode();
////DLLEXPORT int          AllocateCvodeMem (void *, int n, TModelCallBack, double, N_Vector, double, N_Vector);//, long int[], double[]);
////DLLEXPORT int          CvDense (void *, int);  // int = size of systems
////DLLEXPORT int          CVReInit (void *cvode_mem, double t0, N_Vector y0, double reltol, N_Vector abstol);
////DLLEXPORT int          Run_Cvode (void *cvode_mem, double tout, N_Vector y, double *t, char *ErrMsg);
////DLLEXPORT int          CVGetRootInfo (void *cvode_mem, int *rootsFound);
////DLLEXPORT int          CVRootInit (void *cvode_mem, int numRoots, TRootCallBack callBack, void *gdata);
////
////DLLEXPORT int  SetMaxNumSteps(void *cvode_mem, int mxsteps);
////DLLEXPORT int  SetMaxOrder(void *cvode_mem, int mxorder);
////DLLEXPORT int  CVSetFData (void *cvode_mem, void *f_data);
////DLLEXPORT int  SetMaxErrTestFails(void *cvode_mem, int maxnef);
////DLLEXPORT int  SetMaxConvFails(void *cvode_mem, int maxncf);
////DLLEXPORT int  SetMaxNonLinIters (void *cvode_mem, int maxcor);
////DLLEXPORT int  SetErrFile (void *cvode_mem, FILE *errfp);
////DLLEXPORT int  SetErrHandler (void *cvode_mem, CVErrHandlerFn callback, void* user_data );
////DLLEXPORT int  SetMinStep(void *cvode_mem, double minStep);
////DLLEXPORT int  SetMaxStep(void *cvode_mem, double maxStep);
////DLLEXPORT int  SetInitStep(void *cvode_mem, double initStep);
////
////DLLEXPORT FILE *fileOpen (char *fileName);
////DLLEXPORT void  fileClose (FILE *fp);
////
////#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */
////
////// Declare the call back pointers
////TModelCallBack callBackModel;
////TRootCallBack  callBackRoot;
////
////
////// C File IO interface routines
////FILE *fileOpen (char *fileName) {
////    return fopen (fileName, "w");
////}
////
////
////void fileClose (FILE *fp) {
////    fclose (fp);
////}
////
////
////// ---------------------------------------------------------------------
////
////// Creates a new N_Vector object and returns a pointer to the caller
////void *NewCvode_Vector (int n)
////{
////    return N_VNew_Serial (n);
////}
////
////
////// Frees an N_Vector object
////void FreeCvode_Vector (N_Vector v)
////{
////    if (v != NULL)
////        N_VDestroy_Serial (v);
////}
////
////void FreeCvode_Mem (void **p)
////{
////    if (p != NULL)
////        CVodeFree (p);
////}
////
////// Sets the value of an element in a N_Vector object
////void Cvode_SetVector (N_Vector v, int Index, double Value)
////{
////    double *data = NV_DATA_S(v);
////    data[Index] = Value;
////}
////
////double Cvode_GetVector (N_Vector v, int Index) {
////    double *data = NV_DATA_S(v);
////    return data[Index];
////}
////
////
////// Cvode calls this to compute the dy/dts. This routine in turn calls the
////// model function which is located in the host application.
////static int InternalFunctionCall(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *f_data)
////{
////    // Calls the callBackModel here
////    callBackModel (NV_LENGTH_S(cv_y), t, NV_DATA_S (cv_y), NV_DATA_S(cv_ydot), f_data);
////    return CV_SUCCESS;
////}
////
////
//////int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
////// Cvode calls this to check for event changes
////static int InternalRootCall (realtype t, N_Vector y, realtype *gout, void *g_data)
////{
////    callBackRoot (t, NV_DATA_S (y), gout, g_data);
////    return CV_SUCCESS;
////}
////
////
////// Set for stiff systems
////void *Create_BDF_NEWTON_CVode() {
////    return CVodeCreate(CV_BDF, CV_NEWTON);
////}
////
////
////// Set for non-stiff systems
////void *Create_ADAMS_FUNCTIONAL_CVode () {
////    return CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
////}
////
////
////// CallBack is the host application function that computes the dy/dt terms
////int AllocateCvodeMem (void *cvode_mem, int n, TModelCallBack callBack, double t0, N_Vector y, double reltol, N_Vector abstol/*, long int iopt[], double ropt[]*/)
////{
////    int result;
////
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    result = CV_SUCCESS;
////    callBackModel = callBack;
////    result =  CVodeInit(cvode_mem, InternalFunctionCall, t0, y);
////    if (result != CV_SUCCESS) return result;
////    result = CVodeSVtolerances(cvode_mem, reltol, abstol);
////    //return CVodeMalloc(cvode_mem, InternalFunctionCall, t0, y, CV_SV, reltol, abstol);
////    return result;
////}
////
////
////int CVRootInit (void *cvode_mem, int numRoots, TRootCallBack callBack, void *gdata)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    callBackRoot = callBack;
////    return CVodeRootInit (cvode_mem, numRoots, InternalRootCall);
////}
////
////
////int CvDense (void *p, int n)
////{
////    if (p == NULL) return CV_SUCCESS;
////    return CVDense(p, n);
////}
////
////
////int Run_Cvode (void *cvode_mem, double tout, N_Vector y, double *t, char *ErrMsg)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVode (cvode_mem, tout, y, t, CV_NORMAL);
////}
////
////// Initialize cvode with a new set of initial conditions
////int CVReInit (void *cvode_mem, double t0, N_Vector y0, double reltol, N_Vector abstol)
////{
////    int result;
////
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    result = CVodeReInit(cvode_mem,  t0, y0);
////    if (result != CV_SUCCESS) return result;
////    result = CVodeSVtolerances(cvode_mem, reltol, abstol);
////    return result;
////
////}
////
////int CVGetRootInfo(void *cvode_mem, int *rootsFound)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeGetRootInfo(cvode_mem, rootsFound);
////}
////
////
////int CVSetFData (void *cvode_mem, void *f_data) {
////    return 0;
////}
////
////int SetMaxNumSteps(void *cvode_mem, int mxsteps) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMaxNumSteps (cvode_mem, mxsteps);
////}
////
////int SetMaxOrder (void *cvode_mem, int mxorder) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMaxOrd (cvode_mem, mxorder);
////}
////
////
////int SetMaxErrTestFails (void *cvode_mem, int maxnef) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return (CVodeSetMaxErrTestFails(cvode_mem, maxnef));
////}
////
////
////int SetMaxConvFails(void *cvode_mem, int maxncf) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMaxConvFails(cvode_mem, maxncf);
////}
////
////int SetMaxNonLinIters (void *cvode_mem, int maxcor) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMaxNonlinIters (cvode_mem, maxcor);
////}
////
////
////int    SetErrFile (void *cvode_mem, FILE *errfp) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetErrFile (cvode_mem, errfp);
////}
////
////int    SetErrHandler (void *cvode_mem, CVErrHandlerFn callback, void* user_data ) {
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetErrHandlerFn (cvode_mem,  callback, user_data);
////}
////
////int SetMinStep(void *cvode_mem, double minStep)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMinStep(cvode_mem, minStep);
////}
////
////int SetMaxStep(void *cvode_mem, double maxStep)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetMaxStep(cvode_mem, maxStep);
////}
////
////int SetInitStep(void *cvode_mem, double initStep)
////{
////    if (cvode_mem == NULL) return CV_SUCCESS;
////    return CVodeSetInitStep(cvode_mem, initStep);
////}
