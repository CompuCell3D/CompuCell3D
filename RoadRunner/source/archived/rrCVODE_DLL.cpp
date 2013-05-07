#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iostream>
#include "nvector/nvector_serial.h"
#include "cvode/cvode_dense.h"
#include "rrCVODE_DLL.h"

namespace rr
{

// Sets the value of an element in a N_Vector object
void Cvode_SetVector (N_Vector v, int Index, double Value)
{
    double *data = NV_DATA_S(v);
    data[Index] = Value;
}

double Cvode_GetVector (N_Vector v, int Index)
{
    double *data = NV_DATA_S(v);
    return data[Index];
}

// CallBack is the host application function that computes the dy/dt terms
int AllocateCvodeMem (void *cvode_mem, int n, TModelCallBack callBack, double t0, N_Vector y, double reltol, N_Vector abstol/*, long int iopt[], double ropt[]*/)
{
    int result;

    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }

    result = CV_SUCCESS;
    gCallBackModel = callBack;
    result =  CVodeInit(cvode_mem, InternalFunctionCall, t0, y);

    if (result != CV_SUCCESS)
    {
        return result;
    }
    result = CVodeSVtolerances(cvode_mem, reltol, abstol);
    return result;
}

// Cvode calls this to compute the dy/dts. This routine in turn calls the
// model function which is located in the host application.
int InternalFunctionCall(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *f_data)
{
    // Calls the callBackModel here
    gCallBackModel (NV_LENGTH_S(cv_y), t, NV_DATA_S (cv_y), NV_DATA_S(cv_ydot), f_data);
    return CV_SUCCESS;
}

//int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
// Cvode calls this to check for event changes
int InternalRootCall (realtype t, N_Vector y, realtype *gout, void *g_data)
{
    gCallBackRoot (t, NV_DATA_S (y), gout, g_data);
    return CV_SUCCESS;
}

int CVRootInit (void *cvode_mem, int numRoots, TRootCallBack callBack, void *gdata)
{
    if (cvode_mem == NULL)
    {
         return CV_SUCCESS;
    }

    gCallBackRoot = callBack;
    return CVodeRootInit (cvode_mem, numRoots, InternalRootCall);
}

int CvDense (void *p, int n)
{
    if (p == NULL)
    {
        return CV_SUCCESS; //???
    }
    return CVDense(p, n);
}

int Run_Cvode (void *cvode_mem, double tout, N_Vector y, double *t)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVode (cvode_mem, tout, y, t, CV_NORMAL);
}

// Initialize cvode with a new set of initial conditions
int CVReInit (void *cvode_mem, double t0, N_Vector y0, double reltol, N_Vector abstol)
{
    int result;

    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }

    result = CVodeReInit(cvode_mem,  t0, y0);

    if (result != CV_SUCCESS)
    {
        return result;
    }

    result = CVodeSVtolerances(cvode_mem, reltol, abstol);
    return result;
}

int CVGetRootInfo(void *cvode_mem, int *rootsFound)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeGetRootInfo(cvode_mem, rootsFound);
}

int CVSetFData (void *cvode_mem, void *f_data)
{
    return 0;
}

int SetMaxNumSteps(void *cvode_mem, int mxsteps)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMaxNumSteps (cvode_mem, mxsteps);
}

int SetMaxOrder (void *cvode_mem, int mxorder)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMaxOrd (cvode_mem, mxorder);
}

int SetMaxErrTestFails (void *cvode_mem, int maxnef)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return (CVodeSetMaxErrTestFails(cvode_mem, maxnef));
}

int SetMaxConvFails(void *cvode_mem, int maxncf)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMaxConvFails(cvode_mem, maxncf);
}

int SetMaxNonLinIters (void *cvode_mem, int maxcor)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMaxNonlinIters (cvode_mem, maxcor);
}

int    SetErrFile (void *cvode_mem, FILE *errfp)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetErrFile (cvode_mem, errfp);
}

int    SetErrHandler (void *cvode_mem, CVErrHandlerFn callback, void* user_data )
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetErrHandlerFn (cvode_mem,  callback, user_data);
}

int SetMinStep(void *cvode_mem, double minStep)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMinStep(cvode_mem, minStep);
}

int SetMaxStep(void *cvode_mem, double maxStep)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetMaxStep(cvode_mem, maxStep);
}

int SetInitStep(void *cvode_mem, double initStep)
{
    if (cvode_mem == NULL)
    {
        return CV_SUCCESS;
    }
    return CVodeSetInitStep(cvode_mem, initStep);
}

}
