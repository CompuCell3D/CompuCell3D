#ifndef PARALLELUTILSOPENMP_H
#define PARALLELUTILSOPENMP_H


// #include <CompuCell3D/FlexibleDiffusionSolverFE.h>
#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <limits>
#include <omp.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#undef max
#undef min

#include <CompuCell3D/SteerableObject.h>
#include <CompuCell3D/CC3DEvents.h>



namespace CompuCell3D {
    class Dim3D;

    class ParallelUtilsOpenMP{
		
        public:
			typedef omp_lock_t OpenMPLock_t;

			ParallelUtilsOpenMP();
			~ParallelUtilsOpenMP();
            void setDim(const Dim3D &_dim);
            Dim3D getDim();

            virtual void handleEvent(CC3DEvent & _ev);
            //locks			
			void initLock(OpenMPLock_t * _lock);
			void destroyLock(OpenMPLock_t * _lock);
			void setLock(OpenMPLock_t * _lock);
			void unsetLock(OpenMPLock_t * _lock);
			//global PyWrapperLock
			void setPyWrapperLock();
			void unsetPyWrapperLock();

			void allowNestedParallelRegions(bool _flag=false);
            void setNumberOfWorkNodes(unsigned int _num);
            void setNumberOfWorkNodesAuto(unsigned int _requested_num_work_nodes = -1);
			void setVPUs(unsigned int _numberOfVPUs,unsigned int _threadsPerVPU=0);
            unsigned int getNumberOfProcessors();
            unsigned int getNumberOfWorkNodes();

			unsigned int getMaxNumberOfWorkNodesFESolver();
			unsigned int getMaxNumberOfWorkNodesPotts();
			unsigned int getMaxNumberOfWorkNodes();

            unsigned int getNumberOfWorkNodesFESolver();
			unsigned int getNumberOfWorkNodesFESolverWithBoxWatcher();
			unsigned int getNumberOfWorkNodesKernelSolver();
			unsigned int getNumberOfWorkNodesPotts();
			unsigned int getNumberOfWorkNodesPottsWithBoxWatcher();

			unsigned int getNumberOfSubgridSectionsPotts();

			void init(const Dim3D &_dim);

			unsigned int getCurrentWorkNodeNumber();
            //void setNumberOfWorkNodesFESolver();


			
            
            std::pair<Dim3D,Dim3D> getFESolverPartition(unsigned int _workNodeNum);
			std::pair<Dim3D,Dim3D> getFESolverPartitionWithBoxWatcher(unsigned int _workNodeNum);

			std::pair<Dim3D,Dim3D> getPottsSection(unsigned int _workNodeNum,unsigned int _subgridSectionNumber);

			void calculateFESolverPartitionWithBoxWatcher(const Dim3D & _boxMin,const Dim3D & _boxMax);
			void calculateKernelSolverPartition(const Dim3D & _boxMin,const Dim3D & _boxMax);
			std::pair<Dim3D,Dim3D> getKernelSolverPartition(unsigned int _workNodeNum);
			void prepareParallelRegionFESolvers(bool _useBoxWatcher=false);			
			void prepareParallelRegionKernelSolvers();
			void prepareParallelRegionPotts(bool _useBoxWatcher=false);
			const std::vector<unsigned int> & getPottsDimensionsToDivide();
			
			

	protected:
		void calculateFESolverPartition();
		void calculatePottsPartition();
		//void partitionLatticeQuasi2D(unsigned int middleDimGridPoints, unsigned int maxDimGridPoints,std::vector<unsigned int> _dimIndexOrderedVec);
		std::vector<unsigned int> calculatePartitioning(unsigned int _numberOfProcessors,bool _quasi2DFlag);
		void partitionLattice(unsigned int minDimGridPoints,unsigned int middleDimGridPoints, unsigned int maxDimGridPoints,std::vector<unsigned int> _dimIndexOrderedVec);
		void generateLatticePartition(unsigned int _numberOfProcessors,bool _quasi2DFlag,std::vector<unsigned int> _dimIndexOrderedVec);


		std::vector<std::pair<Dim3D,Dim3D> > feSolverPartitionVec;
		std::vector<std::pair<Dim3D,Dim3D> > feSolverPartitionWithBoxWatcherVec;

		std::vector<std::vector<std::pair<Dim3D,Dim3D> > >  pottsPartitionVec;
		std::vector<std::vector<std::pair<Dim3D,Dim3D> > >  pottsPartitionWithBoxWatcherVec;
		std::vector<unsigned int> pottsDimensionsToDivide;
	
		Dim3D fieldDim;
		unsigned int numberOfWorkNodes;
		unsigned int threadsPerWorkNode;

		OpenMPLock_t pyWrapperGlobalLock;


    };




};

#endif
