#include "DiffusionSolverFE_CPU_Implicit.h"

#include <Eigen/IterativeLinearSolvers>

#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <Logger/CC3DLogger.h>

#if defined(_WIN32)
#undef max
#undef min
#endif


using Eigen::SparseMatrix;
using Eigen::DynamicSparseMatrix;
using Eigen::RowMajor;
using Eigen::ConjugateGradient;
using Eigen::Dynamic;

using namespace CompuCell3D;

//#define NOMINMAX
//#include "windows.h"//TODO: remove

DiffusionSolverFE_CPU_Implicit::DiffusionSolverFE_CPU_Implicit(void) {
}

DiffusionSolverFE_CPU_Implicit::~DiffusionSolverFE_CPU_Implicit(void) {
}

void DiffusionSolverFE_CPU_Implicit::Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant) {
}

void DiffusionSolverFE_CPU_Implicit::handleEventLocal(CC3DEvent &_event) {
    if (_event.id == LATTICE_RESIZE) {
        // CODE WHICH HANDLES CELL LATTICE RESIZE
    }
}


template<typename SparseMatrixT>
void CompareMatrices(SparseMatrixT const &m1, SparseMatrixT const &m2) {
    if (m1.nonZeros() != m2.nonZeros()) throw CC3DException("Number of non zeros must be equal");
    if (m1.outerSize() != m2.outerSize()) throw CC3DException("Outer sizes must be equal");

    for (int k = 0; k < m1.outerSize(); ++k)

        for (typename SparseMatrixT::InnerIterator it1(m1, k), it2(m2, k); it1; ++it1, ++it2) {
            if (it1.row() != it2.row()) throw CC3DException("Rows sizes must be equal");

            if (it1.value() != it2.value())
                CC3D_Log(LOG_DEBUG) << "Columns must be equal "<<it1.row()<<" "<<it2.row()<<"\t"<<it1.col()<<" "<<it2.col();
            if (it1.index() != it2.index()) throw CC3DException("Indices sizes must be equal");

            if (it1.value() != it2.value())
                CC3D_Log(LOG_DEBUG) << it1.row()<<" "<<it2.row()<<"\t"<<it1.col()<<" "<<it2.col()<<"\t"<<it1.value()<<" "<<it2.value();

        }
}

namespace {
    int flatExtInd(int x, int y, int z, Dim3D const &dim) {
        return z * (dim.x + 2) * (dim.y + 2) + y * (dim.x + 2) + x;
    }
}

void DiffusionSolverFE_CPU_Implicit::Implicit(ConcentrationField_t const &concentrationField, DiffusionData const &diffData, 
	EigenRealVector const &b, EigenRealVector &x){
	size_t totalSize=(fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2);
	size_t totalExtSize=h_celltype_field->getArraySize();
	CC3D_Log(LOG_DEBUG) << "Field size: "<<totalSize<<"; total size: "<<totalExtSize;


    //SparseMatrix<float, RowMajor> eigenM(totalSize,totalSize);
    //float s=1;

    bool const is2D = (fieldDim.z == 1);

    int neighbours = is2D ? 4 : 6;

    //LARGE_INTEGER tb, te, fq;
    //QueryPerformanceFrequency(&fq);

    EigenSparseMatrix eigenM_new(totalSize, totalSize);
    //QueryPerformanceCounter(&tb);
    eigenM_new.reserve(totalSize * is2D ? 5 : 7);
    CC3D_Log(LOG_DEBUG) << "Assembling matrix... ";
    int prevZ = -1;

    //TODO: unify this with OpenCL matrix assembling...
    for (size_t i = 0; i < totalSize; ++i)//TODO: 2D simultaions can be arranged more effective, I guess...
    {

        int z = i / ((fieldDim.x + 2) * (fieldDim.y + 2));
        int remz = i % ((fieldDim.x + 2) * (fieldDim.y + 2));

        eigenM_new.startVec(i);

        int y = remz / (fieldDim.x + 2);
        int x = remz % (fieldDim.x + 2);

        //unsigned char currentCellType=h_celltype_field->getDirect(x+1,y+1,z+1);
        unsigned char currentCellType = h_celltype_field->getByIndex(i);
        float currentDiffCoef = diffData.diffCoef[currentCellType];//*(diffData.deltaT)/(deltaX*deltaX);

        int lastJ = -1;
        if (!is2D && z > 0) {
            int j = flatExtInd(x, y, z - 1, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }
        if (y > 0) {
            int j = flatExtInd(x, y - 1, z, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }

        if (x > 0) {
            int j = flatExtInd(x - 1, y, z, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }

        lastJ = i;
        eigenM_new.insertBack(i, i) = 1 + neighbours * currentDiffCoef;

        if (x < fieldDim.x - 1) {
            int j = flatExtInd(x + 1, y, z, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }

        if (y < fieldDim.y - 1) {
            int j = flatExtInd(x, y + 1, z, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }

        if (!is2D && z < fieldDim.z - 1) {
            int j = flatExtInd(x, y, z + 1, fieldDim);
            lastJ = j;
            eigenM_new.insertBack(i, j) = -currentDiffCoef;
        }

    }

    CC3D_Log(LOG_DEBUG) << "done";
    eigenM_new.finalize();
    eigenM_new.makeCompressed();
//	QueryPerformanceCounter(&te);
    //CC3D_Log(LOG_DEBUG)<<"completelly done";

//	double time=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;
	CC3D_Log(LOG_TRACE) << "It took "<<time<<" s to assemble a matrix with new algorithm" << std::endl;

//	CompareMatrices(eigenM, eigenM_new);

//	SparseMatrix<float,RowMajor> eigenM(eigenDynM);


    ConjugateGradient <EigenSparseMatrix> cg;

    CC3D_Log(LOG_DEBUG) << "Preparing system... ";
//	QueryPerformanceCounter(&tb);
    cg.compute(eigenM_new);
//	QueryPerformanceCounter(&te);
    CC3D_Log(LOG_DEBUG) << "done";

//	double timePrep=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;

    CC3D_Log(LOG_DEBUG) << "Solving system... ";
//	QueryPerformanceCounter(&tb);
    x = cg.solve(b);
    //QueryPerformanceCounter(&te);
    CC3D_Log(LOG_DEBUG) << "done";

    //time=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;

    CC3D_Log(LOG_DEBUG) << "#iterations:     " << cg.iterations();
    CC3D_Log(LOG_DEBUG) << "estimated error: " << cg.error();
}


void DiffusionSolverFE_CPU_Implicit::step(const unsigned int _currentStep) {

    currentStep = _currentStep;

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_TRACE) << "scalingExtraMCSVec[i]="<<scalingExtraMCSVec[i];

        diffuseSingleField(i);

        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

        }
    }
    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }
}

void DiffusionSolverFE_CPU_Implicit::diffuseSingleFieldImpl(ConcentrationField_t &concentrationField,
                                                            DiffusionData &diffData) {

    // OPTIMIZATIONS - Maciej Swat
    // In addition to using contiguous array with scratch area being interlaced with concentration vector further optimizations are possible
    // In the most innner loop iof the FE solver one can replace maxNeighborIndex with hard coded number. Also instead of
    // Using boundary strategy to get offset array it is best to hard code offsets and access them directly
    // The downside is that in such a case one woudl have to write separate diffuseSingleField functions fdor 2D, 3D and for hex and square lattices.
    // However speedups may be worth extra effort.
    CC3D_Log(LOG_TRACE) << "shiftArray="<<concentrationField.getShiftArray()<<" shiftSwap="<<concentrationField.getShiftSwap();
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
    CC3D_Log(LOG_TRACE) << "Diffusion step";
    //DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
    //float diffConst=diffConstVec[idx];
    //float decayConst=decayConstVec[idx];

    //if(diffConst==0.0 && decayConst==0.0){
    //	return; //skip solving of the equation if diffusion and decay constants are 0
    //}

    Automaton *automaton = potts->getAutomaton();


    ConcentrationField_t *concentrationFieldPtr = &concentrationField;


    std::set<unsigned char>::iterator end_sitr = diffData.avoidTypeIdSet.end();
    std::set<unsigned char>::iterator end_sitr_decay = diffData.avoidDecayInIdSet.end();

    bool avoidMedium = false;
    bool avoidDecayInMedium = false;
    //the assumption is that medium has type ID 0
    if (diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr) {
        avoidMedium = true;
    }

    if (diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay) {
        avoidDecayInMedium = true;
    }

    if (diffData.useBoxWatcher) {

        unsigned x_min = 1, x_max = fieldDim.x + 1;
        unsigned y_min = 1, y_max = fieldDim.y + 1;
        unsigned z_min = 1, z_max = fieldDim.z + 1;

        Dim3D minDimBW;
        Dim3D maxDimBW;
        Point3D minCoordinates = *(boxWatcherSteppable->getMinCoordinatesPtr());
        Point3D maxCoordinates = *(boxWatcherSteppable->getMaxCoordinatesPtr());
        CC3D_Log(LOG_TRACE) << "FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates;
        x_min = minCoordinates.x + 1;
        x_max = maxCoordinates.x + 1;
        y_min = minCoordinates.y + 1;
        y_max = maxCoordinates.y + 1;
        z_min = minCoordinates.z + 1;
        z_max = maxCoordinates.z + 1;

        minDimBW = Dim3D(x_min, y_min, z_min);
        maxDimBW = Dim3D(x_max, y_max, z_max);
        pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW, maxDimBW);


    }



    //managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
    pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);

    Dim3D minDim;
    Dim3D maxDim;

    if (diffData.useBoxWatcher) {
        minDim = pUtils->getFESolverPartitionWithBoxWatcher(0).first;
        maxDim = pUtils->getFESolverPartitionWithBoxWatcher(0).second;

    } else {
        minDim = pUtils->getFESolverPartition(0).first;
        maxDim = pUtils->getFESolverPartition(0).second;
    }


    EigenRealVector implicitSolution(h_celltype_field->getArraySize()), b(h_celltype_field->getArraySize());
    for (int z = minDim.z; z < maxDim.z; z++)
        for (int y = minDim.y; y < maxDim.y; y++)
            for (int x = minDim.x; x < maxDim.x; x++) {
                b[flatExtInd(x, y, z, fieldDim)] = concentrationField.getDirect(x, y, z);
            }

    Implicit(concentrationField, diffData, b, implicitSolution);

    //Copying solution back to main concentration array
    for (int z = minDim.z; z < maxDim.z; z++)
        for (int y = minDim.y; y < maxDim.y; y++)
            for (int x = minDim.x; x < maxDim.x; x++) {
                float impl = std::max(Real_t(0.f), implicitSolution[flatExtInd(x, y, z, fieldDim)]);
                concentrationField.setDirectSwap(x, y, z, impl);
            }

    concentrationField.swapArrays();

    //CheckConcentrationField(concentrationField);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU_Implicit::initImpl() {
    //do nothing on CPU
}

void DiffusionSolverFE_CPU_Implicit::solverSpecific(CC3DXMLElement *_xmlData) {
    //do nothing on CPU
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU_Implicit::extraInitImpl() {
    //do nothing on CPU
}

void DiffusionSolverFE_CPU_Implicit::initCellTypesAndBoundariesImpl() {
    //do nothing on CPU
}

std::string DiffusionSolverFE_CPU_Implicit::toStringImpl() {
    return "DiffusionSolverFE_Implicit";
}

