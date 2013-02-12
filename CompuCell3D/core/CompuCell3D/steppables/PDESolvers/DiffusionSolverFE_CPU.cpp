#include "DiffusionSolverFE_CPU.h"

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>
#define NOMINMAX

// 2012 Mitja:
//  #include <windows.h>
#if defined(_WIN32)
	#include <windows.h>
#endif


using namespace CompuCell3D;

DiffusionSolverFE_CPU::DiffusionSolverFE_CPU(void):DiffusableVectorCommon<float, Array3DContiguous>()
{
}

DiffusionSolverFE_CPU::~DiffusionSolverFE_CPU(void)
{
}

int flatInd(int x, int y, int z, Dim3D const&dim){
	return z*dim.x*dim.y+y*dim.x+x;
}

void DiffusionSolverFE_CPU::diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData)
{

	// OPTIMIZATIONS - Maciej Swat
	// In addition to using contiguous array with scratch area being interlaced with concentration vector further optimizations are possible
	// In the most innner loop iof the FE solver one can replace maxNeighborIndex with hard coded number. Also instead of 
	// Using boundary strategy to get offset array it is best to hard code offsets and access them directly
	// The downside is that in such a case one woudl have to write separate diffuseSingleField functions fdor 2D, 3D and for hex and square lattices. 
	// However speedups may be worth extra effort.   
	//cerr<<"shiftArray="<<concentrationField.getShiftArray()<<" shiftSwap="<<concentrationField.getShiftSwap()<<endl;
	//hard coded offsets for 3D square lattice
	//Point3D offsetArray[6];
	//offsetArray[0]=Point3D(0,0,1);
	//offsetArray[1]=Point3D(0,1,0);
	//offsetArray[2]=Point3D(1,0,0);
	//offsetArray[3]=Point3D(0,0,-1);
	//offsetArray[4]=Point3D(0,-1,0);
	//offsetArray[5]=Point3D(-1,0,0);


	/// 'n' denotes neighbor

	///this is the diffusion equation
	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
	///a - diffusivity - diffConst

	///Finite difference method:
	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
	///N - number of neighbors
	///will have to double check this formula



	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	//cerr<<"Diffusion step"<<endl;
	//DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	//float diffConst=diffConstVec[idx];
	//float decayConst=decayConstVec[idx];

	//if(diffConst==0.0 && decayConst==0.0){
	//	return; //skip solving of the equation if diffusion and decay constants are 0
	//}

	Automaton *automaton=potts->getAutomaton();


	ConcentrationField_t * concentrationFieldPtr = &concentrationField;

	
	std::set<unsigned char>::iterator end_sitr=diffData.avoidTypeIdSet.end();
	std::set<unsigned char>::iterator end_sitr_decay=diffData.avoidDecayInIdSet.end();

	bool avoidMedium=false;
	bool avoidDecayInMedium=false;
	//the assumption is that medium has type ID 0
	if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr){
		avoidMedium=true;
	}

	if(diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay){
		avoidDecayInMedium=true;
	}


	
	if(diffData.useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);


	}


	//managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
	pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);
#pragma omp parallel
	{	

		Point3D pt,n;		
		CellG * currentCellPtr=0,*nCell=0;
		short currentCellType=0;
		float currentConcentration=0;
		float updatedConcentration=0.0;
		float concentrationSum=0.0;
		float varDiffSumTerm=0.0;				
		float * diffCoef=diffData.diffCoef;
		float * decayCoef=diffData.decayCoef;
		float currentDiffCoef=0.0;
		bool variableDiffusionCoefficientFlag=diffData.getVariableDiffusionCoeeficientFlag();

		std::set<unsigned char>::iterator sitr;

		Array3DCUDA<unsigned char> & cellTypeArray= *h_celltype_field; ////

		int threadNumber=pUtils->getCurrentWorkNodeNumber(); ////

		Dim3D minDim;		
		Dim3D maxDim;

		if(diffData.useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}


		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){
					currentConcentration = concentrationField.getDirect(x,y,z);
					currentCellType=cellTypeArray.getDirect(x,y,z);
					currentDiffCoef=diffCoef[currentCellType];
					pt=Point3D(x-1,y-1,z-1);

					updatedConcentration=0.0;
					concentrationSum=0.0;
					varDiffSumTerm=0.0;


					//loop over nearest neighbors
					const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
					for (register int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){
						const Point3D & offset = offsetVecRef[i];

						concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);

					}

					concentrationSum -= (maxNeighborIndex+1)*currentConcentration;
					
					concentrationSum*=currentDiffCoef;
					//                                         cout << " diffCoef[currentCellType]: " << diffCoef[currentCellType] << endl;    


					
					//using forward first derivatives - cartesian lattice 3D
					if (variableDiffusionCoefficientFlag){

						const std::vector<Point3D> & offsetFDVecRef=getOffsetVec(pt); //offsets for forward derivatives


						for (register int i = 0  ; i<offsetFDVecRef.size() ; ++i ){
							const Point3D & offsetFD = offsetFDVecRef[i];
							varDiffSumTerm+=(diffCoef[cellTypeArray.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)]-currentDiffCoef)*(concentrationField.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)-concentrationField.getDirect(x,y,z));
						}

					}

					updatedConcentration=diffusionLatticeScalingFactor*(concentrationSum+varDiffSumTerm)+(1-decayCoef[currentCellType])*currentConcentration;

					

					//imposing artificial limits on allowed concentration
					if(diffData.useThresholds){
						if(updatedConcentration>diffData.maxConcentration){
							updatedConcentration=diffData.maxConcentration;
						}
						if(updatedConcentration<diffData.minConcentration){
							updatedConcentration=diffData.minConcentration;
						}
					}

					
					concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch

				}


				//float checkConc=0.0;
				//for (int z = minDim.z; z < maxDim.z; z++)
				//	for (int y = minDim.y; y < maxDim.y; y++)
				//		for (int x = minDim.x; x < maxDim.x; x++){
				//			checkConc+=concentrationField.getDirect(x,y,z);
				//			//if (x>25 && x<30 &&y>25 &&y<30){
				//			//	cerr<<"pt="<<pt<<endl;
				//			//	cerr<<"currentCellType="<<currentCellType<<" diffCoef[currentCellType]="<<diffCoef[currentCellType]<<endl;
				//			//	cerr<<"concentrationSum="<<concentrationSum<<endl;
				//			//}
				//		}
				//cerr<<"checkConc="<<checkConc<<endl;


	}

	concentrationField.swapArrays();
	/*
	// OPTIMIZATIONS - Maciej Swat
	// In addition to using contiguous array with scratch area being interlaced with concentration vector further optimizations are possible
	// In the most innner loop iof the FE solver one can replace maxNeighborIndex with hard coded number. Also instead of 
	// Using boundary strategy to get offset array it is best to hard code offsets and access them directly
	// The downside is that in such a case one woudl have to write separate diffuseSingleField functions fdor 2D, 3D and for hex and square lattices. 
	// However speedups may be worth extra effort.   
	//cerr<<"shiftArray="<<concentrationField.getShiftArray()<<" shiftSwap="<<concentrationField.getShiftSwap()<<endl;
	//hard coded offsets for 3D square lattice
	//Point3D offsetArray[6];
	//offsetArray[0]=Point3D(0,0,1);
	//offsetArray[1]=Point3D(0,1,0);
	//offsetArray[2]=Point3D(1,0,0);
	//offsetArray[3]=Point3D(0,0,-1);
	//offsetArray[4]=Point3D(0,-1,0);
	//offsetArray[5]=Point3D(-1,0,0);


	/// 'n' denotes neighbor

	///this is the diffusion equation
	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
	///a - diffusivity - diffConst

	///Finite difference method:
	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
	///N - number of neighbors
	///will have to double check this formula



	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	//cerr<<"Diffusion step"<<endl;
	//float diffConst=diffConstVec[idx];
	//float decayConst=decayConstVec[idx];

	float diffConst=diffData.diffConst;
	float decayConst=diffData.decayConst;
	
	float deltaT=diffData.deltaT;
	float deltaX=diffData.deltaX;
	float dt_dx2=deltaT/(deltaX*deltaX);

	if(diffConst==0.0 && decayConst==0.0){
		return; //skip solving of the equation if diffusion and decay constants are 0
	}

	Automaton *automaton=potts->getAutomaton();


	ConcentrationField_t * concentrationFieldPtr = &concentrationField;

	//boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions


	std::set<unsigned char>::iterator end_sitr=diffData.avoidTypeIdSet.end();
	std::set<unsigned char>::iterator end_sitr_decay=diffData.avoidDecayInIdSet.end();

	bool avoidMedium=false;
	bool avoidDecayInMedium=false;
	//the assumption is that medium has type ID 0
	if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr){
		avoidMedium=true;
	}

	if(diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay){
		avoidDecayInMedium=true;
	}



	if(diffData.useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);


	}


	//managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
	pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);
#pragma omp parallel
	{	
		Point3D pt,n;		
		CellG * currentCellPtr=0,*nCell=0;
		short currentCellType=0;
		float currentConcentration=0;
		float updatedConcentration=0.0;
		float concentrationSum=0.0;
		float varDiffSumTerm=0.0;
		short neighborCounter=0;
		CellG *neighborCellPtr=0;
        float * diffCoef=diffData.diffCoef;
        float * decayCoef=diffData.decayCoef;
		bool variableDiffusionCoefficientFlag=diffData.getVariableDiffusionCoeeficientFlag();

		std::set<unsigned char>::iterator sitr;

		Array3DCUDA<unsigned char> & cellTypeArray= *h_celltype_field; ////

		int threadNumber=pUtils->getCurrentWorkNodeNumber(); ////

		//#pragma omp critical
		//		{
		//			cerr<<"threadNumber="<<threadNumber<<endl;
		//		}

		Dim3D minDim;		
		Dim3D maxDim;

		if(diffData.useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;
		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}
		

		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){
					currentConcentration = concentrationField.getDirect(x,y,z);
					currentCellType=cellTypeArray.getDirect(x,y,z);

					pt=Point3D(x-1,y-1,z-1);

					updatedConcentration=0.0;
					concentrationSum=0.0;
					varDiffSumTerm=0.0;
					neighborCounter=0;

					//loop over nearest neighbors
					CellG *neighborCellPtr=0;


					const std::vector<Point3D> & offsetVecRef=getBoundaryStrategy()->getOffsetVec(pt);
					for (register unsigned int i = 0  ; i<=getMaxNeighborIndex()  ; ++i ){
						const Point3D & offset = offsetVecRef[i];

						concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);

					}

					concentrationSum -= (getMaxNeighborIndex()+1)*currentConcentration;

					concentrationSum*=diffCoef[currentCellType];
//                                         cout << " diffCoef[currentCellType]: " << diffCoef[currentCellType] << endl;    



					//using forward first derivatives - cartesian lattice 3D
					if (variableDiffusionCoefficientFlag){

						const std::vector<Point3D> & offsetFDVecRef=getOffsetVec(pt); //offsets for forward derivatives


						for (register size_t i = 0  ; i<offsetFDVecRef.size() ; ++i ){
							const Point3D & offsetFD = offsetFDVecRef[i];
                                                        varDiffSumTerm+=(diffCoef[cellTypeArray.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)]-diffCoef[currentCellType])*(concentrationField.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)-concentrationField.getDirect(x,y,z));
                                                }

					}

					updatedConcentration=dt_dx2*(concentrationSum+varDiffSumTerm)+(1-deltaT*decayCoef[currentCellType])*currentConcentration;
					//updatedConcentration=concentrationSum;
					//updatedConcentration=varDiffSumTerm;


					if(haveCouplingTerms){
						updatedConcentration+=couplingTerm(pt,diffData.couplingDataVec,currentConcentration);
					}

					//imposing artificial limits on allowed concentration
					if(diffData.useThresholds){
						if(updatedConcentration>diffData.maxConcentration){
							updatedConcentration=diffData.maxConcentration;
						}
						if(updatedConcentration<diffData.minConcentration){
							updatedConcentration=diffData.minConcentration;
						}
					}
					concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch
				}


		//float checkConc=0.0;
		//for (int z = minDim.z; z < maxDim.z; z++)
		//	for (int y = minDim.y; y < maxDim.y; y++)
		//		for (int x = minDim.x; x < maxDim.x; x++){
		//			checkConc+=concentrationField.getDirect(x,y,z);
		//			//if (x>25 && x<30 &&y>25 &&y<30){
		//			//	cerr<<"pt="<<pt<<endl;
		//			//	cerr<<"currentCellType="<<currentCellType<<" diffCoef[currentCellType]="<<diffCoef[currentCellType]<<endl;
		//			//	cerr<<"concentrationSum="<<concentrationSum<<endl;
		//			//}
		//		}
		//cerr<<"checkConc="<<checkConc<<endl;


	}

		
	//haveCouplingTerms flag is set only when user defines coupling terms AND does not use extraTimesPerMCS - haveCouplingTerms option is kept only for legacy reasons
	//it it best to start using ReactionDiffusionSolver instead
	if(!haveCouplingTerms){   

		concentrationField.swapArrays();
	}

	



	//CheckConcentrationField(concentrationField);*/
		
}

//for debugging
void DiffusionSolverFE_CPU::CheckConcentrationField(ConcentrationField_t &concentrationField)const{

	double sum=0.f;
	float minVal=numeric_limits<float>::max();
	float maxVal=-numeric_limits<float>::max();
	for(int z=1; z<=fieldDim.z; ++z){
		for(int y=1; y<=fieldDim.y; ++y){
			for(int x=1; x<=fieldDim.x; ++x){
				//float val=h_field[z*(fieldDim.x+2)*(fieldDim.y+2)+y*(fieldDim.x+2)+x];
				float val=concentrationField.getDirect(x,y,z);
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::initImpl(){
	//do nothing on CPU
}

void DiffusionSolverFE_CPU::solverSpecific(CC3DXMLElement *_xmlData){
	//do nothing on CPU
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::extraInitImpl(){
	//do nothing on CPU
}

void DiffusionSolverFE_CPU::initCellTypesAndBoundariesImpl(){
	//do nothing on CPU
}
