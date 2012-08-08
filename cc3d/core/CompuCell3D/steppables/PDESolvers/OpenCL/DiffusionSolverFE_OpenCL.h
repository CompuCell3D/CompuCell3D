#ifndef DIFFUSIONSOLVERFE_OPENCL_H
#define DIFFUSIONSOLVERFE_OPENCL_H

#include "../DiffusionSolverFE.h"

// 2012 Mitja:
// #include <CL/cl.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif


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
	cl_mem d_scratchField;
	cl_mem d_solverParams;
	//cl_mem d_fieldSize;

	cl_mem d_nbhdConcShifts, d_nbhdDiffShifts;
	
	cl_int nbhdConcLen;
	cl_int nbhdDiffLen;

	int gpuDeviceIndex;//GPU device to use
		
public:
	typedef Array3DCUDA<float> ConcentrationField_t;//TODO: check if I can automate this type deduction
	DiffusionSolverFE_OpenCL(void);
	virtual ~DiffusionSolverFE_OpenCL(void);

	void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData const &diffData);

protected:
	//virtual void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData);
	virtual void initImpl();
	virtual void extraInitImpl();
	virtual void initCellTypesAndBoundariesImpl();
	virtual void solverSpecific(CC3DXMLElement *_xmlData);//reading solver-specific information from XML file
	
private:

	//for debugging
	void CheckConcentrationField(float const *h_field)const;

	//kernel's name selector
	std::string diffKernelName();

	//TODO: remove parameter
	void gpuAlloc(size_t fieldLen);
	void fieldHostToDevice(float const *h_field);
	void fieldDeviceToHost(float *h_field)const;
	void CreateKernel();

	void SetConstKernelArguments();//set parameters that won't change during simualtion, like buffers' handles
	void SetSolverParams(DiffusionData const &diffData);//set parameters that can be changed duting simulation, like a time step

    cl_kernel kernel;
	cl_program program;

	size_t field_len;
	size_t localWorkSize[3];//block size
};

}//CompuCell3D 

#endif
