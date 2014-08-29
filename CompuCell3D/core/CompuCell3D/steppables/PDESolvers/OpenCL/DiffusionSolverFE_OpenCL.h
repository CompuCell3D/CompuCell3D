#ifndef DIFFUSIONSOLVERFE_OPENCL_H
#define DIFFUSIONSOLVERFE_OPENCL_H

#include <CompuCell3D/CC3DEvents.h>

#include "../DiffusionSolverFE.h"

// 2012 Mitja:
// #include <CL/cl.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

//#include "windows.h"

//Ivan Komarov

//OpenCL version of Diffusion Solver

namespace CompuCell3D {
	class OpenCLHelper;

class PDESOLVERS_EXPORT DiffusionSolverFE_OpenCL :
	public DiffusionSolverFE<DiffusionSolverFE_OpenCL>, public DiffusableVectorCommon<float, Array3DCUDA>
{
	OpenCLHelper *oclHelper;
	cl_mem d_concentrationField;
	cl_mem d_cellTypes;
    cl_mem d_cellIds;    
	cl_mem d_scratchField;
	cl_mem d_solverParams;
    cl_mem d_bcSpecifier;
    
    cl_mem d_bcIndicator; // indicates which pixels are in the lattice  interior and whic at the boundary and which at the external boundary
    
    
    //secretionData
    cl_mem d_secretionData;
	//cl_mem d_fieldSize;
    
    //ptrs used to efficiently swap memory on the device
    cl_mem * concDevPtr; 
    cl_mem * scratchDevPtr;
    
    bool d_cellIdsAllocated;
    
    int concFieldArgPosition, scratchFieldArgPosition; //positions of concentration and scratch fields in kernel argument list
    int bcSpecifierArgPosition, bcIndicatorArgPosition; //positions of bcIndicator and bcSpecifier
    
    
	cl_mem d_nbhdConcShifts, d_nbhdDiffShifts;
	
	cl_int nbhdConcLen;
	cl_int nbhdDiffLen;

	int gpuDeviceIndex;//GPU device to use

//	double mutable totalSolveTime;
//	double mutable totalTransferTime;

//	LARGE_INTEGER fq;
		
public:
	typedef Array3DCUDA<float> ConcentrationField_t;//TODO: check if I can automate this type deduction
	DiffusionSolverFE_OpenCL(void);
	virtual ~DiffusionSolverFE_OpenCL(void);	

	virtual void handleEventLocal(CC3DEvent & _event);
    
	virtual void finish();

protected:
	//virtual void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData);
    
    
	virtual void initImpl();
	virtual void extraInitImpl();
	virtual void initCellTypesAndBoundariesImpl();
    virtual void boundaryConditionInit(int idx);
    virtual void stepImpl(const unsigned int _currentStep);
    virtual void boundaryConditionGPUSetup(int idx);
    virtual void diffuseSingleField(unsigned int idx);    
    
    virtual void secreteSingleField(unsigned int idx);

    virtual void secreteOnContactSingleField(unsigned int idx);

    virtual void secreteConstantConcentrationSingleField(unsigned int idx);    
    
	virtual void solverSpecific(CC3DXMLElement *_xmlData);//reading solver-specific information from XML file
    
    
    virtual Dim3D getInternalDim();
    
    virtual std::string toStringImpl();    
    void initSecretionData();
	
private:

	//for debugging
	//void CheckConcentrationField(float const *h_field)const;

	
	

	//TODO: remove parameter
	void gpuAlloc(size_t fieldLen);
	void fieldHostToDevice(float const *h_field);
	void fieldDeviceToHost(float *h_field)const;
	void CreateKernel();

	void SetConstKernelArguments();//set parameters that won't change during simualtion, like buffers' handles
	void SetSolverParams(DiffusionData  &diffData, SecretionData  &secrData);//set parameters that can be changed duting simulation, like a time step
    void prepSecreteOnContactSingleField(unsigned int idx); //creates and moves bufferes to GPU only when SecreteOnContact is requested
    void prepCellId(unsigned int idx); //initializes cell field boundaries  -  needed by SecreteOnContact - called once per field
    
    int iterationNumber; // this variable is important because other routines can sense if this is first or subsequent call to diffuse or secrete functions. Some work in this functions has to be done during initial call and skipped in others
    
    
    cl_kernel kernelUniDiff;
    cl_kernel kernelBoundaryConditionInit;
    cl_kernel kernelBoundaryConditionInitLatticeCorners;
    cl_kernel secreteSingleFieldKernel;
    cl_kernel secreteConstantConcentrationSingleFieldKernel;
    cl_kernel secreteOnContactSingleFieldKernel;
    // cl_kernel myKernel;
    
	cl_program program;
    
	size_t field_len;
	size_t localWorkSize[3];//block size
    size_t globalWorkSize[3];//fieldDim
};

}//CompuCell3D 

#endif
