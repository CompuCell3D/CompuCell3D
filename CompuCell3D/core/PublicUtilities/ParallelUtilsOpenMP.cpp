#include  "ParallelUtilsOpenMP.h"
#include <algorithm>
#include <Logger/CC3DLogger.h>
using namespace std;
using namespace CompuCell3D;

//this array contains optimal division for quasi-2D lattice into subgrids. First index of the array is number of threads requested

int latticeGridPartition2D[][3]={
	{1,1,1},						//0 threads
	{1,1,1},						//1 threads
	{1,1,2},						//2 threads
	{1,1,3},						//3 threads
	{1,2,2},						//4 threads
	{1,1,5},						//5 threads
	{1,2,3},						//6 threads
	{1,1,7},						//7 threads
	{1,2,4},						//8 threads
	{1,3,3},						//9 threads
	{1,2,5},						//10 threads
	{1,1,11},						//11 threads
	{1,3,4},						//12 threads
	{1,1,13},						//13 threads
	{1,2,7},						//14 threads
	{1,3,5},						//15 threads
	{1,4,4},						//16 threads
	{1,1,17},						//17 threads
	{1,3,6},						//18 threads
	{1,1,19},						//19 threads
	{1,4,5},						//20 threads
	{1,3,7},						//21 threads
	{1,2,11},						//22 threads
	{1,1,23},						//23 threads
	{1,4,6},						//24 threads
	{1,5,5},						//25 threads
	{1,2,13},						//26 threads
	{1,3,9},						//27 threads
	{1,4,7},						//28 threads
	{1,1,29},						//29 threads
	{1,5,6},						//30 threads
	{1,1,31},						//31 threads
	{1,4,8} 						//32 threads
};


//this array contains optimal division for 3D lattice into subgrids. First indes of the array is number of threads requested
int latticeGridPartition3D[][3]={
	{1,1,1},						//0 threads
	{1,1,1},						//1 threads
	{1,1,2},						//2 threads
	{1,1,3},						//3 threads
	{1,2,2},						//4 threads
	{1,1,5},						//5 threads
	{1,2,3},						//6 threads
	{1,1,7},						//7 threads
	{2,2,2},						//8 threads
	{1,3,3},						//9 threads
	{1,2,5},						//10 threads
	{1,1,11},						//11 threads
	{2,2,3},						//12 threads
	{1,1,13},						//13 threads
	{1,2,7},						//14 threads
	{1,3,5},						//15 threads
	{2,2,4},						//16 threads
	{1,1,17},						//17 threads
	{2,3,3},						//18 threads
	{1,1,19},						//19 threads
	{2,2,5},						//20 threads
	{1,3,7},						//21 threads
	{1,2,11},						//22 threads
	{1,1,23},						//23 threads
	{2,3,4},						//24 threads
	{1,5,5},						//25 threads
	{1,2,13},						//26 threads
	{3,3,3},						//27 threads
	{2,2,7},						//28 threads
	{1,1,29},						//29 threads
	{2,3,5},						//30 threads
	{1,1,31},						//31 threads
	{2,4,4} 						//32 threads
};



ParallelUtilsOpenMP::ParallelUtilsOpenMP():
	numberOfWorkNodes(0), //by default this var is set to 0 . user can set manually number of work nodes  
	threadsPerWorkNode(1) //by default this var is set to 1 . user can set manually number threads per worknode   
{
	initLock(&pyWrapperGlobalLock);
}

ParallelUtilsOpenMP::~ParallelUtilsOpenMP(){
	destroyLock(&pyWrapperGlobalLock);
}


void ParallelUtilsOpenMP::init(const Dim3D &_dim){
	//numberOfWorkNodes=0;
	//threadsPerWorkNode=1;

	fieldDim=_dim;
	Dim3D minDim=Dim3D(1,1,1);
	Dim3D maxDim=fieldDim;
	maxDim.x++;
	maxDim.y++;
	maxDim.z++;

	//FE solvers partition
	feSolverPartitionVec.clear();
	feSolverPartitionVec.push_back(make_pair(minDim,maxDim));
	calculateFESolverPartition();

	//Potts partition
	pottsPartitionVec.clear();	

	pottsPartitionVec.assign(1,vector<pair<Dim3D,Dim3D> >(1,make_pair(minDim,maxDim)));


	calculatePottsPartition();

}

void ParallelUtilsOpenMP::setDim(const Dim3D &_dim){

	fieldDim=_dim;
	Dim3D minDim=Dim3D(1,1,1);
	Dim3D maxDim=fieldDim;
	maxDim.x++;
	maxDim.y++;
	maxDim.z++;

	feSolverPartitionVec.clear();
	feSolverPartitionVec.push_back(make_pair(minDim,maxDim));

	calculateFESolverPartition();

}


void  ParallelUtilsOpenMP::handleEvent(CC3DEvent & _event){

	if (_event.id==LATTICE_RESIZE){
        CC3DEventLatticeResize & ev = static_cast<CC3DEventLatticeResize&>(_event);
		init(ev.newDim);
		//setDim(ev.newDim);
	}else if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){// I am trying to gradually get rid of VPU's and in this version we set threadsPerWorkNode which means that, by default, number of nodes is equal to number of threads. Users still can change it
		CC3DEventChangeNumberOfWorkNodes & ev=static_cast<CC3DEventChangeNumberOfWorkNodes&>(_event);
		setNumberOfWorkNodes(ev.newNumberOfNodes);
		//threadsPerWorkNode=1; 
	}

    
    

}



Dim3D ParallelUtilsOpenMP::getDim(){return fieldDim;}

void ParallelUtilsOpenMP::initLock(OpenMPLock_t * _lock){
	omp_init_lock(_lock);
}
void ParallelUtilsOpenMP::destroyLock(OpenMPLock_t * _lock){
	omp_destroy_lock(_lock);
}
void ParallelUtilsOpenMP::setLock(OpenMPLock_t * _lock){
	omp_set_lock(_lock);
}
void ParallelUtilsOpenMP::unsetLock(OpenMPLock_t * _lock){
	omp_unset_lock(_lock);
}

void ParallelUtilsOpenMP::setPyWrapperLock(){
	setLock(&pyWrapperGlobalLock);
}
void ParallelUtilsOpenMP::unsetPyWrapperLock(){
	unsetLock(&pyWrapperGlobalLock);
}

void ParallelUtilsOpenMP::allowNestedParallelRegions(bool _flag){
	
#if (defined(__GNUC__) && (__GNUC__>=9)) || defined(__clang__)
        if(_flag) {
            omp_set_max_active_levels(5);
        }
#else
    omp_set_nested(_flag);
#endif
}

void ParallelUtilsOpenMP::setNumberOfWorkNodes(unsigned int _num){
	//if (_num>0 && _num<=getNumberOfProcessors()){ //this might be too restrictive
	if (_num>0){
		numberOfWorkNodes=_num;
    omp_set_dynamic(0);
    omp_set_num_threads(numberOfWorkNodes);
		calculateFESolverPartition();
		calculatePottsPartition();
	}

}

void ParallelUtilsOpenMP::setNumberOfWorkNodesAuto(unsigned int _requested_num_work_nodes) {

    if (_requested_num_work_nodes < 0) {
        _requested_num_work_nodes = omp_get_num_threads();
        // in case we get 0
        _requested_num_work_nodes = (_requested_num_work_nodes ? _requested_num_work_nodes : 1);
    }
    

        omp_set_dynamic(0);
        omp_set_num_threads(_requested_num_work_nodes);

}


void ParallelUtilsOpenMP::setVPUs(unsigned int _numberOfVPUs,unsigned int _threadsPerVPU){
	if(_threadsPerVPU>0)
		threadsPerWorkNode=_threadsPerVPU;

	if (_numberOfVPUs>0){ 
		numberOfWorkNodes=_numberOfVPUs;
		calculateFESolverPartition();
		calculatePottsPartition();
	}
}

unsigned int ParallelUtilsOpenMP::getNumberOfProcessors(){
	return omp_get_num_procs();
}

unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodes(){
	return numberOfWorkNodes;
}

unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodesFESolver(){

	return feSolverPartitionVec.size();
}

unsigned int ParallelUtilsOpenMP::getMaxNumberOfWorkNodesFESolver(){
	unsigned int numberOfProcessors=getNumberOfProcessors();
	//users may request more "processors" than there are CPU's on the system	
	return (numberOfProcessors>=numberOfWorkNodes ? threadsPerWorkNode*numberOfProcessors : threadsPerWorkNode*numberOfWorkNodes);

}


//void ParallelUtilsOpenMP::setNumberOfWorkNodesFESolver(){
//	//you cannot set this directly
//}

void ParallelUtilsOpenMP::prepareParallelRegionFESolvers(bool _useBoxWatcher){
	omp_set_dynamic(0);
	if(_useBoxWatcher){
		omp_set_num_threads(getNumberOfWorkNodesFESolverWithBoxWatcher());
	}else{
		omp_set_num_threads(getNumberOfWorkNodesFESolver());
	}
}



unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodesFESolverWithBoxWatcher(){
	return feSolverPartitionWithBoxWatcherVec.size();
}

std::pair<Dim3D,Dim3D> ParallelUtilsOpenMP::getFESolverPartitionWithBoxWatcher(unsigned int _workNodeNum){
	return feSolverPartitionWithBoxWatcherVec[_workNodeNum];
}

unsigned int ParallelUtilsOpenMP::getCurrentWorkNodeNumber(){
	return omp_get_thread_num();
}

void ParallelUtilsOpenMP::calculateFESolverPartitionWithBoxWatcher(const Dim3D & _boxMin,const Dim3D & _boxMax){
	//unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:getNumberOfProcessors());
	unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:1); //by default we will use 1 thread. Users have to explicitely request more processors

	unsigned int optimalNumberOfThreads=(numProcs==1 ? 1:threadsPerWorkNode*numProcs);

	//this is inefficient implementation butbulk of the computatioanl time is spen elswhere so it is not a big deal. 
	if((_boxMax.z-_boxMin.z)==1 || (_boxMax.z-_boxMin.z)<optimalNumberOfThreads){//2D case or quasi 2D case, we will divide thread workloads according fieldDim.y
		if((_boxMax.y-_boxMin.y)>optimalNumberOfThreads){ // fieldDim.y is large enough so that it can be divided
			unsigned int threadNum=optimalNumberOfThreads;
			unsigned int wL=(_boxMax.y-_boxMin.y)/(threadNum); //base workload for single processor
			unsigned int extraWL=(_boxMax.y-_boxMin.y) % threadNum ; //extra workload - will be distributed evenly among first processes


			feSolverPartitionWithBoxWatcherVec.clear();

			Dim3D minDim(_boxMin);
			Dim3D maxDim;

			for (int i  = 0 ; i < threadNum ; ++i){
				maxDim=Dim3D(_boxMax.x, minDim.y+wL,_boxMax.z);
				if(extraWL>0){
					maxDim.y++;
					extraWL--;
				}
				feSolverPartitionWithBoxWatcherVec.push_back(make_pair(minDim,maxDim));
				minDim.y=maxDim.y;
			}
		}else{
			unsigned int threadNum=(_boxMax.y-_boxMin.y); // number of threads in this case is the same as the box dimension in the y direction

			unsigned int wL=1; //base workload for single processor



			feSolverPartitionWithBoxWatcherVec.clear();

			Dim3D minDim(_boxMin);
			Dim3D maxDim;

			for (int i  = 0 ; i < threadNum ; ++i){
				maxDim=Dim3D(_boxMax.x, minDim.y+wL,_boxMax.z);
				feSolverPartitionWithBoxWatcherVec.push_back(make_pair(minDim,maxDim));
				minDim.y=maxDim.y;
			}


		}
		return ;

	}

	if((_boxMax.z-_boxMin.z)>=optimalNumberOfThreads){//z dimension is large enough so that we can divide lattice using optimal division scheme


		unsigned int threadNum=optimalNumberOfThreads;
		unsigned int wL=(_boxMax.z-_boxMin.z)/(threadNum); //base workload for single processor
		unsigned int extraWL=(_boxMax.z-_boxMin.z) % threadNum ; //extra workload - will be distributed evenly among first processes


		feSolverPartitionWithBoxWatcherVec.clear();

		Dim3D minDim(_boxMin);
		Dim3D maxDim;

		for (int i  = 0 ; i < threadNum ; ++i){
			maxDim=Dim3D(_boxMax.x,_boxMax.y, minDim.z+wL);
			if(extraWL>0){
				maxDim.z++;
				extraWL--;
			}
			feSolverPartitionWithBoxWatcherVec.push_back(make_pair(minDim,maxDim));
			minDim.z=maxDim.z;
		}

		return;
	}



}

std::pair<Dim3D,Dim3D> ParallelUtilsOpenMP::getFESolverPartition(unsigned int _workNodeNum){
	return feSolverPartitionVec[_workNodeNum];
}


void ParallelUtilsOpenMP::calculateFESolverPartition(){
	//unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:getNumberOfProcessors());
	unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:1); //by default we will use 1 thread. Users have to explicitely request more processors

	unsigned int optimalNumberOfThreads=(numProcs==1 ? 1:threadsPerWorkNode*numProcs);

	if(fieldDim.z==1 || fieldDim.z<optimalNumberOfThreads){//2D case or quasi 2D case, we will divide thread workloads according fieldDim.y
		if(fieldDim.y>optimalNumberOfThreads){ // fieldDim.y is large enough so that it can be divided
			unsigned int threadNum=optimalNumberOfThreads;
			unsigned int wL=fieldDim.y/(threadNum); //base workload for single processor
			unsigned int extraWL=fieldDim.y % threadNum ; //extra workload - will be distributed evenly among first processes


			feSolverPartitionVec.clear();

			Dim3D minDim(1,1,1);
			Dim3D maxDim;

			for (int i  = 0 ; i < threadNum ; ++i){
				maxDim=Dim3D(fieldDim.x+1, minDim.y+wL,fieldDim.z+1);
				if(extraWL>0){
					maxDim.y++;
					extraWL--;
				}
				feSolverPartitionVec.push_back(make_pair(minDim,maxDim));
				minDim.y=maxDim.y;
			}
		}else{
			unsigned int threadNum=fieldDim.y; // number of threads in this case is the same as the dimension in the y direction

			unsigned int wL=1; //base workload for single processor



			feSolverPartitionVec.clear();

			Dim3D minDim(1,1,1);
			Dim3D maxDim;

			for (int i  = 0 ; i < threadNum ; ++i){
				maxDim=Dim3D(fieldDim.x+1, minDim.y+wL,fieldDim.z+1);
				feSolverPartitionVec.push_back(make_pair(minDim,maxDim));
				minDim.y=maxDim.y;
			}


		}
		return ;
	}

	if(fieldDim.z>=optimalNumberOfThreads){//z dimension is large enough so that we can divide lattice using optimal division scheme


		unsigned int threadNum=optimalNumberOfThreads;
		unsigned int wL=fieldDim.z/(threadNum); //base workload for single processor
		unsigned int extraWL=fieldDim.z % threadNum ; //extra workload - will be distributed evenly among first processes


		feSolverPartitionVec.clear();

		Dim3D minDim(1,1,1);
		Dim3D maxDim;

		for (int i  = 0 ; i < threadNum ; ++i){
			maxDim=Dim3D(fieldDim.x+1,fieldDim.y+1, minDim.z+wL);
			if(extraWL>0){
				maxDim.z++;
				extraWL--;
			}
			feSolverPartitionVec.push_back(make_pair(minDim,maxDim));
			minDim.z=maxDim.z;
		}

		return;
	}

}


void ParallelUtilsOpenMP::calculateKernelSolverPartition(const Dim3D & _boxMin,const Dim3D & _boxMax){
	calculateFESolverPartitionWithBoxWatcher(_boxMin,_boxMax);
}

std::pair<Dim3D,Dim3D> ParallelUtilsOpenMP::getKernelSolverPartition(unsigned int _workNodeNum){
	return getFESolverPartitionWithBoxWatcher(_workNodeNum);
}

void ParallelUtilsOpenMP::prepareParallelRegionKernelSolvers(){
	prepareParallelRegionFESolvers(true);
}

unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodesKernelSolver(){
	return getNumberOfWorkNodesFESolverWithBoxWatcher();
}

//Potts partition

const std::vector<unsigned int> & ParallelUtilsOpenMP::getPottsDimensionsToDivide(){
	return pottsDimensionsToDivide;

}

unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodesPotts(){
	return pottsPartitionVec.size();
}

unsigned int ParallelUtilsOpenMP::getMaxNumberOfWorkNodesPotts(){
	return getMaxNumberOfWorkNodesFESolver();
}

unsigned int ParallelUtilsOpenMP::getMaxNumberOfWorkNodes(){
	return omp_get_max_threads();
}

unsigned int ParallelUtilsOpenMP::getNumberOfWorkNodesPottsWithBoxWatcher(){ //THIS HAS TO BE CORRECTED
	return pottsPartitionVec.size();
}

unsigned int ParallelUtilsOpenMP::getNumberOfSubgridSectionsPotts(){
	return pottsPartitionVec[0].size();
}

std::pair<Dim3D,Dim3D> ParallelUtilsOpenMP::getPottsSection(unsigned int _workNodeNum,unsigned int _subgridSectionNumber){
	return pottsPartitionVec[_workNodeNum][_subgridSectionNumber];
}

void ParallelUtilsOpenMP::prepareParallelRegionPotts(bool _useBoxWatcher){
	omp_set_dynamic(0);
	if(_useBoxWatcher){
		omp_set_num_threads(getNumberOfWorkNodesPottsWithBoxWatcher());
	}else{
		omp_set_num_threads(getNumberOfWorkNodesPotts());
	}
}


std::vector<unsigned int> ParallelUtilsOpenMP::calculatePartitioning(unsigned int _numberOfProcessors,bool _quasi2DFlag){
	//first we will factor number of processors into prime numbers - not particularly elegant method but for this application works OK
	unsigned numberOfProcessorsCopy=_numberOfProcessors;

	std::vector<unsigned int> primeFactors;
	unsigned int divisor=numberOfProcessorsCopy-1;
	unsigned int lastDivisor=numberOfProcessorsCopy;
	while (divisor>1){
		if (! (numberOfProcessorsCopy% divisor)){
			primeFactors.push_back(numberOfProcessorsCopy/divisor);
			numberOfProcessorsCopy=divisor;
			lastDivisor=divisor;
			divisor=numberOfProcessorsCopy-1;
		}else{
			--divisor;
		}
	}

	if (lastDivisor != 1){
		primeFactors.push_back(lastDivisor);
	}


//prime factors sohuld be sorted by construction but just in case I
//sort it again
sort(primeFactors.begin(),primeFactors.end());

vector<unsigned int> partitionVec(3,1); //we initialize it with 1's
unsigned int primeFactorsSize=primeFactors.size();

//generating partitioning by subsequent multiplications of of partitionVec elements (in order 3,2,1) by prime factors (in decreasing order)
//this way we guarantee that max dimension will have most partitions
//however this method is not the most optimal. I can do some tweaks but for now it shuold be OO to have something at least
if (_quasi2DFlag){
	for(int i  = 0 ;i < primeFactorsSize ; ++i ){
		partitionVec[2-i%2]*=primeFactors[primeFactorsSize-i-1]	;
	}

}else{
	for(int i  = 0 ;i < primeFactorsSize ; ++i ){
		partitionVec[2-i%3]*=primeFactors[primeFactorsSize-i-1]	;
	}

}
    
return partitionVec;


}

void ParallelUtilsOpenMP::generateLatticePartition(unsigned int _numberOfProcessors,bool _quasi2DFlag,std::vector<unsigned int> _dimIndexOrderedVec){
	
	unsigned int numArrayElements=sizeof latticeGridPartition2D/ sizeof (unsigned int[3]);
	CC3D_Log(LOG_DEBUG) << "_numberOfProcessors ="<<_numberOfProcessors <<" numArrayElements="<<numArrayElements;
	vector<unsigned int> partitionVec(3,1);
	if (_numberOfProcessors <= numArrayElements-1){ //requested less processors than max number of prepared partitions
		
	}else{//requested more processors than we expect - using algorothmic partitioning

	}



	if (_quasi2DFlag){
		vector<unsigned int> partitionVec(3,1);
		if (_numberOfProcessors <= numArrayElements-1){ //requested less processors than max number of prepared partitions
			partitionVec[0]=latticeGridPartition2D[_numberOfProcessors][0];
			partitionVec[1]=latticeGridPartition2D[_numberOfProcessors][1];
			partitionVec[2]=latticeGridPartition2D[_numberOfProcessors][2];
		}else{//requested more processors than we expect - using algorothmic partitioning
			partitionVec=calculatePartitioning(_numberOfProcessors,_quasi2DFlag);
		}
		CC3D_Log(LOG_TRACE) << "PARTITION 2D \n\n\n";
		CC3D_Log(LOG_TRACE) << "("<<partitionVec[0]<<","<<partitionVec[1]<<","<<partitionVec[2]<<")";
		partitionLattice(partitionVec[0],partitionVec[1], partitionVec[2],_dimIndexOrderedVec);
	}else{
		vector<unsigned int> partitionVec(3,1);
		if (_numberOfProcessors <= numArrayElements-1){ //requested less processors than max number of prepared partitions
			partitionVec[0]=latticeGridPartition3D[_numberOfProcessors][0];
			partitionVec[1]=latticeGridPartition3D[_numberOfProcessors][1];
			partitionVec[2]=latticeGridPartition3D[_numberOfProcessors][2];
		}else{//requested more processors than we expect - using algorothmic partitioning
			partitionVec=calculatePartitioning(_numberOfProcessors,_quasi2DFlag);
		}
		CC3D_Log(LOG_TRACE) <<  "PARTITION 3D\n\n\n";
		CC3D_Log(LOG_TRACE) <<  "("<<partitionVec[0]<<","<<partitionVec[1]<<","<<partitionVec[2]<<")";

		partitionLattice(partitionVec[0],partitionVec[1], partitionVec[2],_dimIndexOrderedVec);

	}
}

void ParallelUtilsOpenMP::calculatePottsPartition(){
	//unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:getNumberOfProcessors());
	unsigned int numProcs=(numberOfWorkNodes ? numberOfWorkNodes:1); //by default we will use 1 thread. Users have to explicitely request more processors
	unsigned int optimalNumberOfThreads=(numProcs==1 ? 1:threadsPerWorkNode*numProcs);	

	vector<unsigned short> dimVec;
	vector<unsigned int> dimIndexOrderedVec(3,0);//this 3 element vector stores indexes of coordinate vector in the order of increasing dimensions

	dimVec.push_back(fieldDim.x);
	dimVec.push_back(fieldDim.y);
	dimVec.push_back(fieldDim.z);


	vector<unsigned short> dimVecTmp=dimVec; //have to make a copy because dimVecTmp will be altered

	//first we determine minimum dimension and its position in the vector
	unsigned short minDimCoord=*min_element(dimVecTmp.begin(),dimVecTmp.end());
	unsigned int indexMin=distance(dimVecTmp.begin(),min_element(dimVecTmp.begin(),dimVecTmp.end()));
	dimVecTmp[indexMin]=0;  //before determining maximum dimensio we have to set previously found minimum to zero 
	//to ensure that max_element does not pick the same position for the max dimension as it did for the minimum element 
	//in case all 3 dimensions are the same



	unsigned short maxDimCoord=*max_element(dimVecTmp.begin(),dimVecTmp.end());
	unsigned int indexMax=distance(dimVecTmp.begin(),max_element(dimVecTmp.begin(),dimVecTmp.end()));



	//finding index of "middle" dimension
	unsigned int indexMiddle;
	unsigned short middleDimCoord;
	for(int i = 0 ; i < 3; ++i){
		if(i!=indexMin && i!=indexMax){
			indexMiddle=i;
			middleDimCoord=dimVecTmp[indexMiddle];
			break;
		}
	}

	//listing indexes in the order of increasing dimension (correspoiding to given index)
	dimIndexOrderedVec[0]=indexMin;
	dimIndexOrderedVec[1]=indexMiddle;
	dimIndexOrderedVec[2]=indexMax;

	//here we will determine if simulation lattice is "flat" in one dimension i.e. quasi 2D or if it is trully 3D simulation. the cirterion is quite arbitrary so we may need to change it later
	bool quasi2D=true;
	if (minDimCoord==1){
		quasi2D=true; //explicitely 2D simulation
	}else{
		if (maxDimCoord/minDimCoord>=4){
			quasi2D=true; // quasi 2D simulation min dimension is significantly smaller than max dimension
		}else{
			quasi2D=false; //"true" 3D simulation
		}
	}
	CC3D_Log(LOG_TRACE) << "minDimCoord="<<minDimCoord<<" indexMin="<<indexMin<<" maxDimCoord="<<maxDimCoord<<" indexMax="<<indexMax<<" middleDimCoord="<<middleDimCoord<<" indexMiddle="<<indexMiddle;

	generateLatticePartition(optimalNumberOfThreads,quasi2D,dimIndexOrderedVec);

}


void ParallelUtilsOpenMP::partitionLattice(unsigned int minDimGridPoints,unsigned int middleDimGridPoints, unsigned int maxDimGridPoints,std::vector<unsigned int> _dimIndexOrderedVec)
{

	unsigned int indexMin=_dimIndexOrderedVec[0];
	unsigned int indexMiddle=_dimIndexOrderedVec[1];
	unsigned int indexMax=_dimIndexOrderedVec[2];

	unsigned short minDimCoord=fieldDim[indexMin];
	unsigned short middleDimCoord=fieldDim[indexMiddle];
	unsigned short maxDimCoord=fieldDim[indexMax];

	//lattice is divided into minDimGridPoints x middleDimGridPoints x maxDimGridPoints quadrants with division lines passing through min, max middleDimension axes

	pottsPartitionVec.clear();

	vector<short> minDimDivisionVec;
	vector<short> middleDimDivisionVec;
	vector<short> maxDimDivisionVec;
	//handling special cas of single partition - or effectively no partitioning
	if(minDimGridPoints==1 && middleDimGridPoints==1 && maxDimGridPoints==1){
		pottsPartitionVec.clear();
		Dim3D minDim(0,0,0);
		Dim3D maxDim=fieldDim;		
		pottsPartitionVec.assign(1,vector<pair<Dim3D,Dim3D> >(1,make_pair(minDim,maxDim)));
		pottsDimensionsToDivide.clear() ; //no dimension will be divided - single core simulation
		CC3D_Log(LOG_DEBUG) << "SINGLE PROCESSOR RUN minDim="<<minDim<<" maxDim="<<maxDim;
		return;
	}


	for (int i  = 0 ; i < minDimGridPoints ; ++i){
		minDimDivisionVec.push_back(i*(minDimCoord/minDimGridPoints));
	}

	for (int i  = 0 ; i < middleDimGridPoints ; ++i){
		middleDimDivisionVec.push_back(i*(middleDimCoord/middleDimGridPoints));
	}

	for (int i  = 0 ; i < maxDimGridPoints ; ++i){
		maxDimDivisionVec.push_back(i*(maxDimCoord/maxDimGridPoints));
	}

	if(minDimGridPoints==1){
		//each grid will be divided into 4 subgrids in quasi-2D lattice division
		pottsPartitionVec.assign(middleDimDivisionVec.size()*maxDimDivisionVec.size(),vector<pair<Dim3D,Dim3D> >(8,make_pair(Dim3D(),Dim3D())));
	}else{
		//each grid will be divided into 8 subgrids in quasi-3D lattice division
		pottsPartitionVec.assign(minDimDivisionVec.size()*middleDimDivisionVec.size()*maxDimDivisionVec.size(),vector<pair<Dim3D,Dim3D> >(8,make_pair(Dim3D(),Dim3D())));
	}

	int gridId=0;
	for (int i = 0  ; i < minDimDivisionVec.size() ; ++i)
		for (int j = 0  ; j < middleDimDivisionVec.size() ; ++j)
			for (int k = 0  ; k < maxDimDivisionVec.size() ; ++k){

				Dim3D minDim;
				Dim3D maxDim;

				minDim[indexMin]=minDimDivisionVec[i];
				minDim[indexMiddle]=middleDimDivisionVec[j];
				minDim[indexMax]=maxDimDivisionVec[k];

				maxDim[indexMin]=(i<minDimDivisionVec.size()-1 ?  minDimDivisionVec[i+1]: fieldDim[indexMin] );
				maxDim[indexMiddle]=(j<middleDimDivisionVec.size()-1 ?  middleDimDivisionVec[j+1]: fieldDim[indexMiddle] );
				maxDim[indexMax]=(k<maxDimDivisionVec.size()-1 ?  maxDimDivisionVec[k+1]: fieldDim[indexMax] );


				pottsPartitionVec[gridId][0]=make_pair(minDim,maxDim);
				CC3D_Log(LOG_DEBUG) << "gridId="<<gridId<<" minDim="<<minDim<<" maxDim="<<maxDim;

				++gridId;
			}
			int gridIdMax=gridId;

			for (int gid=0 ; gid<gridIdMax ; ++gid){
				//Grid boundaries
				Dim3D minDim=pottsPartitionVec[gid][0].first;
				Dim3D maxDim=pottsPartitionVec[gid][0].second;


				//Each grid will be partitioned into 4 subgrids (or 8 if more threads are used in 3D)
				Dim3D minSubDim;
				Dim3D maxSubDim;

				unsigned int subgridId=0;

				unsigned minIndexSubgrids=(minDimGridPoints==1 ? 1 : 2 ); //to differentiate between quasi2D and 3D cases

				for (int i = 0 ; i < minIndexSubgrids ; ++i)//minIndex
					for (int j = 0 ; j < 2 ; ++j) //middleIndex
						for (int k = 0 ; k < 2 ; ++k){//maxIndex

							//partitioning along min Dimension
							if(minIndexSubgrids==1){
								minSubDim[indexMin]=minDim[indexMin];
								maxSubDim[indexMin]=maxDim[indexMin];
							}else{
								minSubDim[indexMin]=minDim[indexMin]+i*((maxDim[indexMin]-minDim[indexMin])/2);
								if(i<1){
									maxSubDim[indexMin]=minDim[indexMin]+(i+1)*((maxDim[indexMin]-minDim[indexMin])/2);
								}else{
									maxSubDim[indexMin]=maxDim[indexMin];
								}
							}

							//partitioning along middle Dimension
							minSubDim[indexMiddle]=minDim[indexMiddle]+j*((maxDim[indexMiddle]-minDim[indexMiddle])/2);
							if(j<1){
								maxSubDim[indexMiddle]=minDim[indexMiddle]+(j+1)*((maxDim[indexMiddle]-minDim[indexMiddle])/2);
							}else{
								maxSubDim[indexMiddle]=maxDim[indexMiddle];
							}

							//partitioning along max Dimension
							minSubDim[indexMax]=minDim[indexMax]+k*((maxDim[indexMax]-minDim[indexMax])/2);
							if(k<1){
								maxSubDim[indexMax]=minDim[indexMax]+(k+1)*((maxDim[indexMax]-minDim[indexMax])/2);
							}else{
								maxSubDim[indexMax]=maxDim[indexMax];
							}

							pottsPartitionVec[gid][subgridId]=make_pair(minSubDim,maxSubDim);
							CC3D_Log(LOG_DEBUG) << " GID="<<gid<<" subgridId="<<subgridId<<" minSubDim="<<minSubDim<<" maxSubDim="<<maxSubDim;
							++subgridId;

						}

			}

}
