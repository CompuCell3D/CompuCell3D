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

using namespace CompuCell3D;

DiffusionSolverFE_OpenCL::DiffusionSolverFE_OpenCL(void):DiffusableVectorCommon<float, Array3DCUDA>(),oclHelper(NULL), d_cellTypes(NULL)
{
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


int DivUp(int dividend, int divisor){
	if(dividend%divisor==0)
		return dividend/divisor;
	else
		return dividend/divisor+1;
}

void DiffusionSolverFE_OpenCL::diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData const &diffData)
{

	//cerr<<"diffuseSingleFieldImpl\n";

	ASSERT_OR_THROW("Coupling Terms are not supported yet", !haveCouplingTerms);
	ASSERT_OR_THROW("Box watcher is not supported yet", !diffData.useBoxWatcher);
	ASSERT_OR_THROW("Threshold is not supported yet",	!diffData.useThresholds);
	//ASSERT_OR_THROW("2D domains are not supported yet",fieldDim.x!=1&&fieldDim.y!=1&&fieldDim.z!=1);

	const size_t globalWorkSize[]={DivUp(fieldDim.x, localWorkSize[0])*localWorkSize[0], 
		DivUp(fieldDim.y, localWorkSize[1])*localWorkSize[1], 
		DivUp(fieldDim.z, localWorkSize[2])*localWorkSize[2]};

	SetSolverParams(diffData);
	
	float *h_Field=concentrationField.getContainer();

	fieldHostToDevice(h_Field);
	//cerr<<"calling a kernel...\n";
	//cerr<<"Block size is: "<<localWorkSize[0]<<"x"<<localWorkSize[1]<<"x"<<localWorkSize[2]<<
		//"; globalWorkSize is: "<<globalWorkSize[0]<<"x"<<globalWorkSize[1]<<"x"<<globalWorkSize[2]<<endl;
	cl_int err = oclHelper->EnqueueNDRangeKernel(kernel, 3, globalWorkSize, localWorkSize);
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
	h_solverParams.dt=diffData.deltaT;
	h_solverParams.dx=diffData.deltaX;
	h_solverParams.hexLattice=(latticeType==HEXAGONAL_LATTICE);
	h_solverParams.nbhdConcLen=nbhdConcLen;
	h_solverParams.nbhdDiffLen=nbhdDiffLen;

	h_solverParams.xDim=fieldDim.x;
	h_solverParams.yDim=fieldDim.y;
	h_solverParams.zDim=fieldDim.z;

	//cerr<<"dt="<<h_solverParams.dt<<"; dx="<<h_solverParams.dx<<endl;

	oclHelper->WriteBuffer(d_solverParams, &h_solverParams, 1);

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
		gpuDeviceIndex=0;
	}

}

void DiffusionSolverFE_OpenCL::initImpl(){

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

// 2012 Mitja - the "cl_int4" type is defined differently in OpenCL >= 1.1 than in OpenCL 1.0,
//   so to pass compilation the following code needs at least OpenCL 1.1:
#if defined (CL_VERSION_1_1)

	try{
	if(latticeType==HEXAGONAL_LATTICE){
		cerr<<"Hexagonal lattice used"<<endl;
		if(fieldDim.z==1){
			nbhdConcLen=6;
			nbhdDiffLen=3;
			layers=2;
		}else{
			nbhdConcLen=12;
			nbhdDiffLen=6;
			layers=6;
		}
	}else{
		cerr<<"Cartesian lattice used"<<endl;
		if(fieldDim.z==1){
			nbhdConcLen=4;
			nbhdDiffLen=2;
		}else{
			nbhdConcLen=6;
			nbhdDiffLen=3;
		}
	}

	std::vector<cl_int4> h_nbhdDiffShifts;
	std::vector<cl_int4> h_nbhdConcShifts;

	if(latticeType==HEXAGONAL_LATTICE)
	{
		cerr<<"fieldDim.z="<<fieldDim.z<<endl;
		
		//cerr<<"bhoa size is: "<<bhoa.size()<<endl;
		//for(size_t i=0; i<bhoa.size(); ++i){
		//	cerr<<"currSize: "<<bhoa[i].size()<<endl;
		//	for(int j=0; j<getBoundaryStrategy()->getMaxOffset(); ++j)
		//	{
		//		cerr<<bhoa[i][j]<<"   ";
		//	}
		//}

		//cerr<<"hexOffsetArray size is: "<<hexOffsetArray.size()<<endl;
		//for(size_t i=0; i<hexOffsetArray.size(); ++i){
		//	cerr<<"currSize: "<<hexOffsetArray[i].size()<<endl;
		//	for(size_t j=0; j<hexOffsetArray.size(); ++j)
		//	{
		//		cerr<<hexOffsetArray[i][j]<<"   ";
		//	}
		//}
		//cerr<<"diffHexOffsetLen="<<diffHexOffsetLen<<endl;
		//cerr<<"bndHexOffsetLen="<<bndHexOffsetLen<<endl;

		//cerr<<"hexOffsetArray.size()="<<hexOffsetArray.size()<<endl;
		//cerr<<"hexOffsetArray[0].size()="<<hexOffsetArray[0].size()<<endl;

		h_nbhdDiffShifts.resize(layers*nbhdDiffLen);
		h_nbhdConcShifts.resize(layers*nbhdConcLen);

		if(fieldDim.z!=1){
					
			std::vector<std::vector<Point3D> > bhoa;
			getBoundaryStrategy()->getHexOffsetArray(bhoa);
			for(int i=0; i<layers; ++i){
				int offset=nbhdDiffLen*i;
				for(size_t j=0; j<hexOffsetArray[i].size(); ++j)
				{
					ASSERT_OR_THROW("wrong index 1", i<hexOffsetArray.size());
					ASSERT_OR_THROW("wrong index 2", j<hexOffsetArray[i].size());
					cl_int4 shift={hexOffsetArray[i][j].x, hexOffsetArray[i][j].y, hexOffsetArray[i][j].z};

					ASSERT_OR_THROW("wrong index", (offset+j<h_nbhdDiffShifts.size()));
					h_nbhdDiffShifts[offset+j]=shift;
				}
			}
		
			//cerr<<"bhoa.size()="<<bhoa.size()<<endl;
			//cerr<<"bndMaxOffset="<<getBoundaryStrategy()->getMaxOffset()<<endl;
	
			for(int i=0; i<layers; ++i){
				int offset=nbhdConcLen*i;
				for(int j=0; j<getBoundaryStrategy()->getMaxOffset(); ++j)
				{
					ASSERT_OR_THROW("wrong index 1", i<bhoa.size());
					ASSERT_OR_THROW("wrong index 2", j<bhoa[i].size());
					cl_int4 shift={bhoa[i][j].x, bhoa[i][j].y, bhoa[i][j].z};
					ASSERT_OR_THROW("wrong index", (offset+j<h_nbhdConcShifts.size()));
					h_nbhdConcShifts[offset+j]=shift;
				}
			}
		}else
		{//2D
			int yShifts[12]={0,1,1,0,-1,-1,
				0,1,1,0,-1,-1};
			int xShifts[12]={-1, -1, 0, 1, 0,-1,
				-1,0,1,1,1,0};

			//cerr<<"qq4.1 "<<h_nbhdConcShifts.size()<<" "<<h_nbhdDiffShifts.size()<<endl;
			for(int i=0; i<2; ++i){
				for(int j=0; j<6; ++j)
				{
					//cerr<<"1 i="<<i<<"j="<<j<<endl;
					cl_int4 shift={xShifts[6*i+j], yShifts[6*i+j], 0, 0};
					h_nbhdConcShifts[6*i+j]=shift;
					//cerr<<"2 i="<<i<<"j="<<j<<endl;
					if(j<3)
						h_nbhdDiffShifts[3*i+j]=shift;
					//cerr<<"3 i="<<i<<"j="<<j<<endl;
				}
			}
		}
		//cerr<<"\n???\n";

		//cerr<<"maxNeighborIndex="<<getMaxNeighborIndex()<<endl;

		//cerr<<"*********"<<h_nbhdHexConcShifts.size()<<"*********"<<endl;
		//for(size_t i=0; i<6; ++i){
		//	int offset=bndHexOffsetLen*i;
		//	cerr<<endl;
		//	for(int j=0; j<bndHexOffsetLen; ++j){
		//		cerr<<h_nbhdHexConcShifts[offset+j].s[0]<<","<<
		//			h_nbhdHexConcShifts[offset+j].s[1]<<","<<
		//			h_nbhdHexConcShifts[offset+j].s[2]<<"   ";
		//	}
		//}

		//for(size_t i=0; i<6; ++i){
		//	int offset=diffHexOffsetLen*i;
		//	cerr<<endl;
		//	for(int j=0; j<diffHexOffsetLen; ++j){
		//		cerr<<h_nbhdHexDiffShifts[offset+j].s[0]<<","<<
		//			h_nbhdHexDiffShifts[offset+j].s[1]<<","<<
		//			h_nbhdHexDiffShifts[offset+j].s[2]<<"   ";
		//	}
		//}

	}//if(latticeType==HEXAGONAL_LATTICE)
	else{
		h_nbhdDiffShifts.resize(nbhdDiffLen);
		h_nbhdConcShifts.resize(nbhdConcLen);

		if(fieldDim.z==1){
			h_nbhdConcShifts[0].s[0]=1;  h_nbhdConcShifts[0].s[1]=0;  h_nbhdConcShifts[0].s[2]=0; h_nbhdConcShifts[0].s[3]=0;
			h_nbhdConcShifts[1].s[0]=0;  h_nbhdConcShifts[1].s[1]=1;  h_nbhdConcShifts[1].s[2]=0; h_nbhdConcShifts[1].s[3]=0;
			h_nbhdConcShifts[2].s[0]=-1; h_nbhdConcShifts[2].s[1]=0;  h_nbhdConcShifts[2].s[2]=0; h_nbhdConcShifts[2].s[3]=0;
			h_nbhdConcShifts[3].s[0]=0;  h_nbhdConcShifts[3].s[1]=-1; h_nbhdConcShifts[3].s[2]=0; h_nbhdConcShifts[3].s[3]=0;
					
			h_nbhdDiffShifts[0].s[0]=1;  h_nbhdDiffShifts[0].s[1]=0;  h_nbhdDiffShifts[0].s[2]=0; h_nbhdDiffShifts[0].s[3]=0;
			h_nbhdDiffShifts[1].s[0]=0;  h_nbhdDiffShifts[1].s[1]=1;  h_nbhdDiffShifts[1].s[2]=0; h_nbhdDiffShifts[1].s[3]=0;
		}
		else{

			h_nbhdConcShifts[0].s[0]=1;  h_nbhdConcShifts[0].s[1]=0;  h_nbhdConcShifts[0].s[2]=0;
			h_nbhdConcShifts[1].s[0]=0;  h_nbhdConcShifts[1].s[1]=1;  h_nbhdConcShifts[1].s[2]=0;
			h_nbhdConcShifts[2].s[0]=0;  h_nbhdConcShifts[2].s[1]=0;  h_nbhdConcShifts[2].s[2]=1;
			h_nbhdConcShifts[3].s[0]=-1; h_nbhdConcShifts[3].s[1]=0;  h_nbhdConcShifts[3].s[2]=0;
			h_nbhdConcShifts[4].s[0]=0;  h_nbhdConcShifts[4].s[1]=-1; h_nbhdConcShifts[4].s[2]=0;
			h_nbhdConcShifts[5].s[0]=0;  h_nbhdConcShifts[5].s[1]=0;  h_nbhdConcShifts[5].s[2]=-1;
		
			h_nbhdDiffShifts[0].s[0]=1;  h_nbhdDiffShifts[0].s[1]=0;  h_nbhdDiffShifts[0].s[2]=0;
			h_nbhdDiffShifts[1].s[0]=0;  h_nbhdDiffShifts[1].s[1]=1;  h_nbhdDiffShifts[1].s[2]=0;
			h_nbhdDiffShifts[2].s[0]=0;  h_nbhdDiffShifts[2].s[1]=0;  h_nbhdDiffShifts[2].s[2]=1;
		}
	}

	d_nbhdDiffShifts=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4)*h_nbhdDiffShifts.size());
	d_nbhdConcShifts=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4)*h_nbhdConcShifts.size());

	
	
	cl_int err= oclHelper->WriteBuffer(d_nbhdDiffShifts, &h_nbhdDiffShifts[0], h_nbhdDiffShifts.size());

	err = err | oclHelper->WriteBuffer(d_nbhdConcShifts, &h_nbhdConcShifts[0], h_nbhdConcShifts.size());
	ASSERT_OR_THROW("Can not initialize shifts", err==CL_SUCCESS);

	//set the arguements of our kernel 
	SetConstKernelArguments();
	}catch(...){
		ASSERT_OR_THROW("exception caught", false);
	}


#else
    printf("    -----------------------------------------------------------------     \n");
    printf("    -----------------------------------------------------------------     \n");
    printf("                                                                          \n");
    printf("             WARNING WARNING WARNING WARNING WARNING WARNING              \n");
    printf("                                                                          \n");
    printf("                                                                          \n");
    printf("                   OpenCL 1.0 currently not supported                     \n");
    printf("                                                                          \n");
    printf("                                                                          \n");
    printf("             WARNING WARNING WARNING WARNING WARNING WARNING              \n");
    printf("                                                                          \n");
    printf("    -----------------------------------------------------------------     \n");
    printf("    -----------------------------------------------------------------     \n");
#endif

	cerr<<"extraInitImpl finihed\n";
	
}

//for debugging
void DiffusionSolverFE_OpenCL::CheckConcentrationField(float const *h_field)const{
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
}

void DiffusionSolverFE_OpenCL::fieldHostToDevice(float const *h_field){

	//cerr<<"before: ";
	//CheckConcentrationField(h_field);

	ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);
	if(oclHelper->WriteBuffer(d_concentrationField, h_field, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not write to device buffer", false);
	}

	//to preserve boundary layers
	if(oclHelper->CopyBuffer<float>(d_concentrationField, d_scratchField, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not copy device buffer", false);
	}

}

void DiffusionSolverFE_OpenCL::fieldDeviceToHost(float *h_field)const{
	ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);
	if(oclHelper->ReadBuffer(d_scratchField, h_field, field_len)!=CL_SUCCESS){
		ASSERT_OR_THROW("Can not read from device buffer", false);
	}

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
