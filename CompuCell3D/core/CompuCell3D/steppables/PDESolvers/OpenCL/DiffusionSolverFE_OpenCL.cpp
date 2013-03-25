#include "DiffusionSolverFE_OpenCL.h"

//Ivan Komarov

#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include "../GPUSolverParams.h"
#include "OpenCLHelper.h"
#include <XMLUtils/CC3DXMLElement.h>
#include <algorithm>
#include "OCLNeighbourIndsInfo.h"

#if defined(_WIN32)
    #undef max
    #undef min
#endif



using namespace CompuCell3D;

DiffusionSolverFE_OpenCL::DiffusionSolverFE_OpenCL(void):DiffusableVectorCommon<float, Array3DCUDA>(),oclHelper(NULL), d_cellTypes(NULL),
//	totalSolveTime(0.),
//	totalTransferTime(0.),
	gpuDeviceIndex(-1)
{
//	QueryPerformanceFrequency(&fq);
}


DiffusionSolverFE_OpenCL::~DiffusionSolverFE_OpenCL(void)
{
	if(oclHelper){
		cl_int res;
		oclHelper->Finish();

		res=clReleaseMemObject(d_nbhdConcShifts);
		ASSERT_OR_THROW("Can not release d_nbhdConcShifts", res==CL_SUCCESS);

		res=clReleaseMemObject(d_nbhdDiffShifts);
		ASSERT_OR_THROW("Can not release d_nbhdDiffShifts", res==CL_SUCCESS);

		res=clReleaseMemObject(d_cellTypes);
		ASSERT_OR_THROW("Can not release d_cellTypes", res==CL_SUCCESS);

		res=clReleaseMemObject(d_solverParams);
		ASSERT_OR_THROW("Can not release d_solverParams", res==CL_SUCCESS);

		res=clReleaseMemObject(d_scratchField);
		ASSERT_OR_THROW("Can not release d_scratchField", res==CL_SUCCESS);

		res=clReleaseMemObject(d_concentrationField);
		ASSERT_OR_THROW("Can not release d_concentrationField", res==CL_SUCCESS);

		res=clReleaseKernel(kernel);
		ASSERT_OR_THROW("Can not release kernel", res==CL_SUCCESS);

		res=clReleaseProgram(program);
		ASSERT_OR_THROW("Can not release program", res==CL_SUCCESS);

		delete oclHelper;
	}
}

void DiffusionSolverFE_OpenCL::diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData const &diffData)
{

	//cerr<<"diffuseSingleFieldImpl\n";

	ASSERT_OR_THROW("Coupling Terms are not supported yet", !haveCouplingTerms);
	ASSERT_OR_THROW("Box watcher is not supported yet", !diffData.useBoxWatcher);
	ASSERT_OR_THROW("Threshold is not supported yet",	!diffData.useThresholds);
	//ASSERT_OR_THROW("2D domains are not supported yet",fieldDim.x!=1&&fieldDim.y!=1&&fieldDim.z!=1);

	const size_t globalWorkSize[]={fieldDim.x, fieldDim.y, fieldDim.z};

	SetSolverParams(diffData);
	
	float *h_Field=concentrationField.getContainer();

	fieldHostToDevice(h_Field);
	//cerr<<"calling a kernel...\n";
	//cerr<<"Block size is: "<<localWorkSize[0]<<"x"<<localWorkSize[1]<<"x"<<localWorkSize[2]<<
	//	"; globalWorkSize is: "<<globalWorkSize[0]<<"x"<<globalWorkSize[1]<<"x"<<globalWorkSize[2]<<endl;
//	LARGE_INTEGER tb, te;
//	QueryPerformanceCounter(&tb);
	cl_int err = oclHelper->EnqueueNDRangeKernel(kernel, 3, globalWorkSize, localWorkSize);
	oclHelper->Finish();
//	QueryPerformanceCounter(&te);
//	totalSolveTime+=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;
	if(err!=CL_SUCCESS)
		cerr<<oclHelper->ErrorString(err)<<endl;
	ASSERT_OR_THROW("Kernel failed", err==CL_SUCCESS);
	fieldDeviceToHost(h_Field);
}

void DiffusionSolverFE_OpenCL::SetConstKernelArguments()
{
	int kArg=0;
	cl_int err= clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_concentrationField);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_cellTypes);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_solverParams);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_nbhdConcShifts);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_nbhdDiffShifts);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(cl_mem), &d_scratchField);
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(float)*(localWorkSize[0]+2)*(localWorkSize[1]+2)*(localWorkSize[2]+2), NULL);//local field
	err  = err | clSetKernelArg(kernel, kArg++, sizeof(unsigned char)*(localWorkSize[0]+2)*(localWorkSize[1]+2)*(localWorkSize[2]+2), NULL);//local cell type

	ASSERT_OR_THROW("Can not set kernel's arguments\n", err==CL_SUCCESS);
}

void DiffusionSolverFE_OpenCL::SetSolverParams(DiffusionData const &diffData)
{
	
	UniSolverParams_t  h_solverParams;
	for( int i=0; i<UCHAR_MAX; ++i){
		h_solverParams.diffCoef[i]=diffData.diffCoef[i];
		h_solverParams.decayCoef[i]=diffData.decayCoef[i];
	}
//	h_solverParams.dt=diffData.deltaT;
	h_solverParams.dx=diffData.deltaX;
	h_solverParams.hexLattice=(latticeType==HEXAGONAL_LATTICE);
	h_solverParams.nbhdConcLen=nbhdConcLen;
	h_solverParams.nbhdDiffLen=nbhdDiffLen;

	h_solverParams.xDim=fieldDim.x;
	h_solverParams.yDim=fieldDim.y;
	h_solverParams.zDim=fieldDim.z;

	//cerr<<"dt="<<h_solverParams.dt<<"; dx="<<h_solverParams.dx<<endl;

	oclHelper->WriteBuffer(d_solverParams, &h_solverParams, 1);

	float dt=diffData.deltaT;
	cl_int err  = clSetKernelArg(kernel, 8, sizeof(dt), &dt);//local cell type
	ASSERT_OR_THROW("Can't pass time step to prod kernel\n", err==CL_SUCCESS);

	//clSetKernelArg(kernel, 3, sizeof(float), &d_solverParams);
	//oclHelper->Finish();
	
}

void DiffusionSolverFE_OpenCL::solverSpecific(CC3DXMLElement *_xmlData){
	//getting requested GPU device index
	if(_xmlData->findElement("GPUDeviceIndex")){
		gpuDeviceIndex=_xmlData->getFirstElement("GPUDeviceIndex")->getInt();
		cerr<<"GPU device #"<<gpuDeviceIndex<<" requested\n";
	}else{
		cerr<<"No specific GPU requested, it will be selected automatically\n";
		gpuDeviceIndex=-1;
	}

}

void DiffusionSolverFE_OpenCL::initImpl(){

	if(gpuDeviceIndex==-1)
		gpuDeviceIndex=0;

	//cerr<<"Requested GPU device index is "<<gpuDeviceIndex<<endl;
	oclHelper=new OpenCLHelper(gpuDeviceIndex);

	localWorkSize[0]=BLOCK_SIZE;
	localWorkSize[1]=BLOCK_SIZE;
	//TODO: BLOCK size can be non-optimal in terms of maximum performance
	localWorkSize[2]=std::min(oclHelper->getMaxWorkGroupSize()/(BLOCK_SIZE*BLOCK_SIZE),  size_t(fieldDim.z));

	
	field_len=h_celltype_field->getArraySize();
	gpuAlloc(field_len);

	cerr<<"building OpenCL program"<<endl;
	const char *kernelSource[]={"lib/CompuCell3DSteppables/OpenCL/GPUSolverParams.h",
		"lib/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl"};

	if(!oclHelper->LoadProgram(kernelSource, 2, program)){
		ASSERT_OR_THROW("Can't load the OpenCL kernel", false);
	}

	
	
}

void DiffusionSolverFE_OpenCL::gpuAlloc(size_t fieldLen){
	cerr<<"Allocating GPU memory for the field of length "<<fieldLen<<"\n";
	cerr<<"Field dimensions are: "<<fieldDim.x<<" "<<fieldDim.y<<" "<<fieldDim.z<<"\n";

	size_t mem_size_field=fieldLen*sizeof(float);
	size_t mem_size_celltype_field=fieldLen*sizeof(unsigned char);

	d_concentrationField=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_field);
	d_cellTypes=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_celltype_field);
	d_scratchField=oclHelper->CreateBuffer(CL_MEM_WRITE_ONLY, mem_size_field);
	
	d_solverParams=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(UniSolverParams_t));
	
	//cl_int4 fieldDim4={fieldDim.x,fieldDim.y,fieldDim.z};//to be on a safe side I use the OpenCL 
	//oclHelper->WriteBuffer(d_fieldSize, &fieldDim4, 1);
		
}


void DiffusionSolverFE_OpenCL::extraInitImpl(){

	CreateKernel();
	int layers;

	try{
		OCLNeighbourIndsInfo onii=OCLNeighbourIndsInfo::Init(latticeType, fieldDim, getBoundaryStrategy(), hexOffsetArray, offsetVecCartesian);

		d_nbhdDiffShifts=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4)*onii.mh_nbhdDiffShifts.size());
		d_nbhdConcShifts=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4)*onii.mh_nbhdConcShifts.size());

		cl_int err= oclHelper->WriteBuffer(d_nbhdDiffShifts, &onii.mh_nbhdDiffShifts[0], onii.mh_nbhdDiffShifts.size());

		err = err | oclHelper->WriteBuffer(d_nbhdConcShifts, &onii.mh_nbhdConcShifts[0], onii.mh_nbhdConcShifts.size());
		ASSERT_OR_THROW("Can not initialize shifts", err==CL_SUCCESS);

		//set the arguements of our kernel 
		SetConstKernelArguments();
	}catch(...){
		ASSERT_OR_THROW("exception caught", false);
	}

	cerr<<"extraInitImpl finished\n";
	
}

//for debugging
/*void DiffusionSolverFE_OpenCL::CheckConcentrationField(float const *h_field)const{
	//size_t lim=(h_solverParamPtr->dimx+2)*(h_solverParamPtr->dimy+2)*(h_solverParamPtr->dimz+2);
	//cerr<<field_len<<" "<<lim<<endl;
	//for(size_t i=0; i<lim; ++i){
	//	h_field[i]=2.f;
	//}
	//cerr<<h_field[800]<<endl;
	double sum=0.f;
	float minVal=numeric_limits<float>::max();
	float maxVal=-numeric_limits<float>::max();
	for(int z=1; z<=fieldDim.z; ++z){
		for(int y=1; y<=fieldDim.y; ++y){
			for(int x=1; x<=fieldDim.x; ++x){
				float val=h_field[z*(fieldDim.x+2)*(fieldDim.y+2)+y*(fieldDim.x+2)+x];
#ifdef _WIN32
				if(!_finite(val)){
#else
				if(!finite(val)){
#endif
					cerr<<"NaN at position: "<<x<<"x"<<y<<"x"<<z<<endl;
					continue;
				}
				//if(val!=0) 
				//	cerr<<"f("<<x<<","<<y<<","<<z<<")="<<val<<"  ";
				sum+=val;
				minVal=std::min(val, minVal);
				maxVal=std::max(val, maxVal);
			}
		}
	}

	cerr<<"min: "<<minVal<<"; max: "<<maxVal<<" "<<sum<<endl;
}*/

void DiffusionSolverFE_OpenCL::fieldHostToDevice(float const *h_field){

	//cerr<<"before: ";
	//CheckConcentrationField(h_field);

	ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);
//	LARGE_INTEGER tb, te;
//	QueryPerformanceCounter(&tb);
	if(oclHelper->WriteBuffer(d_concentrationField, h_field, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not write to device buffer", false);
	}

	//to preserve boundary layers
	if(oclHelper->CopyBuffer<float>(d_concentrationField, d_scratchField, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not copy device buffer", false);
	}
//	QueryPerformanceCounter(&te);
//	totalTransferTime+=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;

}

void DiffusionSolverFE_OpenCL::fieldDeviceToHost(float *h_field)const{
	ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);

//	LARGE_INTEGER tb, te;
//	QueryPerformanceCounter(&tb);
	if(oclHelper->ReadBuffer(d_scratchField, h_field, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not read from device buffer", false);
	}
//	QueryPerformanceCounter(&te);
//	totalTransferTime+=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;

	//TODO: disable code
	//cerr<<"after: ";
	//CheckConcentrationField(h_field);
}

string DiffusionSolverFE_OpenCL::diffKernelName(){

	//cerr<<"latticeType="<<latticeType<<endl;
	/*if(latticeType==HEXAGONAL_LATTICE){
		if(fieldDim.z==1){
			return "diffHexagonal2D1";
		}else{
			ASSERT_OR_THROW("Domain dimensions along x and y axis must be > 1", fieldDim.x>1&&fieldDim.y>1);
			return "diffHexagonal3D";
		}
	}
	else{
		cerr<<"latticeType="<<latticeType<<endl;
		ASSERT_OR_THROW("Unknown type of lattice", latticeType==SQUARE_LATTICE)
		if(fieldDim.z==1){
			return "diffCartesian2D";
		}else{
			ASSERT_OR_THROW("Domain dimensions along x and y axis must be > 1", fieldDim.x>1&&fieldDim.y>1);
			return "diffCartesian3D";
		}
	}*/

	return "uniDiff";
			
}

void DiffusionSolverFE_OpenCL::CreateKernel(){
    //initialize our kernel from the program
	string kernelName=diffKernelName();
	cerr<<"kernel "<<kernelName<<" used"<<endl;
	//string kernelName="hexDiff";
	cl_int err;
	kernel = clCreateKernel(program, kernelName.c_str(), &err);
	printf("clCreateKernel for kernel %s: %s\n", kernelName.c_str(), oclHelper->ErrorString(err));
	ASSERT_OR_THROW("Can not create a kernel", err==CL_SUCCESS);
	
}

void DiffusionSolverFE_OpenCL::initCellTypesAndBoundariesImpl(){
	//cerr<<"Copying Cell Types array to GPU...\n";
	cl_int err=oclHelper->WriteBuffer(d_cellTypes, h_celltype_field->getContainer(), field_len); 
	ASSERT_OR_THROW("Can not copy Cell Type field to GPU", err==CL_SUCCESS);
}

void DiffusionSolverFE_OpenCL::finish(){
//	cerr<<totalTransferTime<<"s for memory transfer to and from device"<<endl;
//	cerr<<totalSolveTime<<"s for solving itself"<<endl;
}