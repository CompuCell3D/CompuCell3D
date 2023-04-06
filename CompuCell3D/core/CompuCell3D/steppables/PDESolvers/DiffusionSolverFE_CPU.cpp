#include "DiffusionSolverFE_CPU.h"

#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include "FluctuationCompensator.h"
#include <cfloat>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;

DiffusionSolverFE_CPU::DiffusionSolverFE_CPU(void) : DiffusableVectorCommon<float, Array3DContiguous>() {
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DiffusionSolverFE_CPU::~DiffusionSolverFE_CPU(void) {
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int flatInd(int x, int y, int z, Dim3D const &dim) {
    return z * dim.x * dim.y + y * dim.x + x;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_CPU::handleEventLocal(CC3DEvent &_event) {
    if (_event.id != LATTICE_RESIZE) {
        return;
    }

    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();

    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);
    for (size_t i = 0; i < concentrationFieldVector.size(); ++i) {
        concentrationFieldVector[i]->resizeAndShift(ev.newDim, ev.shiftVec);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::secreteSingleField(unsigned int idx) {

    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    float maxUptakeInMedium = 0.0;
    float relativeUptakeRateInMedium = 0.0;
    float secrConstMedium = 0.0;

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();
    std::map<unsigned char, UptakeData>::iterator mitrUptakeShared;
    std::map<unsigned char, UptakeData>::iterator end_mitrUptake = secrData.typeIdUptakeDataMap.end();

    ConcentrationField_t &concentrationField = *this->getConcentrationField(idx);

    bool doUptakeFlag = false;
    bool uptakeInMediumFlag = false;
    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));


    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        secrConstMedium = mitrShared->second;
    }

    //uptake for medium setup
    if (secrData.typeIdUptakeDataMap.size()) {
        doUptakeFlag = true;
    }
    //uptake for medium setup
    if (doUptakeFlag) {
        mitrUptakeShared = secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
        if (mitrUptakeShared != end_mitrUptake) {
            maxUptakeInMedium = mitrUptakeShared->second.maxUptake;
            relativeUptakeRateInMedium = mitrUptakeShared->second.relativeUptakeRate;
            uptakeInMediumFlag = true;

        }
    }

    //HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
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


    pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);


#pragma omp parallel
    {

        CellG *currentCellPtr;

        float currentConcentration;
        float secrConst;


        std::map<unsigned char, float>::iterator mitr;
        std::map<unsigned char, UptakeData>::iterator mitrUptake;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();


        bool hasExtraBndLayer = hasExtraLayer();

        Dim3D minDim;
        Dim3D maxDim;

        getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);


        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    if (hasExtraBndLayer)
                        pt = Point3D(x - 1, y - 1, z - 1);
                    else
                        pt = Point3D(x, y, z);
                    currentCellPtr = cellFieldG->getQuick(pt);

                    currentConcentration = concentrationField.getDirect(x, y, z);

                    if (secreteInMedium && !currentCellPtr) {
                        concentrationField.setDirect(x, y, z, currentConcentration + secrConstMedium);
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrConstMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {
                            secrConst = mitr->second;
                            concentrationField.setDirect(x, y, z, currentConcentration + secrConst);
                        }
                    }

                    if (doUptakeFlag) {

                        if (uptakeInMediumFlag && !currentCellPtr) {
                            if (currentConcentration * relativeUptakeRateInMedium > maxUptakeInMedium) {
                                concentrationField.setDirect(x, y, z,
                                                             concentrationField.getDirect(x, y, z) - maxUptakeInMedium);
                            } else {
                                concentrationField.setDirect(x, y, z, concentrationField.getDirect(x, y, z) -
                                                                      currentConcentration *
                                                                      relativeUptakeRateInMedium);
                            }
                        }
                        if (currentCellPtr) {

                            mitrUptake = secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
                            if (mitrUptake != end_mitrUptake) {
                                if (currentConcentration * mitrUptake->second.relativeUptakeRate >
                                    mitrUptake->second.maxUptake) {
                                    concentrationField.setDirect(x, y, z, concentrationField.getDirect(x, y, z) -
                                                                          mitrUptake->second.maxUptake);

                                } else {
                                    concentrationField.setDirect(x, y, z, concentrationField.getDirect(x, y, z) -
                                                                          currentConcentration *
                                                                          mitrUptake->second.relativeUptakeRate);

                                }
                            }
                        }
                    }
                }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::secreteOnContactSingleField(unsigned int idx) {
    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    std::map<unsigned char, SecretionOnContactData>::iterator mitrShared;
    std::map<unsigned char, SecretionOnContactData>::iterator end_mitr = secrData.typeIdSecrOnContactDataMap.end();

    ConcentrationField_t &concentrationField = *this->getConcentrationField(idx);

    std::map<unsigned char, float> *contactCellMapMediumPtr;
    std::map<unsigned char, float> *contactCellMapPtr;


    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));

    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        contactCellMapMediumPtr = &(mitrShared->second.contactCellMap);
    }


    //HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;

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

    pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);
#pragma omp parallel
    {

        std::map<unsigned char, SecretionOnContactData>::iterator mitr;
        std::map<unsigned char, float>::iterator mitrTypeConst;

        float currentConcentration;
        float secrConst;
        float secrConstMedium = 0.0;

        CellG *currentCellPtr;
        Point3D pt;
        Neighbor n;
        CellG *nCell = 0;
        WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
        unsigned char type;

        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        bool hasExtraBndLayer = hasExtraLayer();

        Dim3D minDim;
        Dim3D maxDim;

        getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);

        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {
                    if (hasExtraBndLayer)
                        pt = Point3D(x - 1, y - 1, z - 1);
                    else
                        pt = Point3D(x, y, z);

                    currentCellPtr = cellFieldG->getQuick(pt);

                    currentConcentration = concentrationField.getDirect(x, y, z);

                    if (secreteInMedium && !currentCellPtr) {
                        for (unsigned int i = 0; i <= this->getMaxNeighborIndex(); ++i) {
                            n = this->getBoundaryStrategy()->getNeighborDirect(pt, i);
                            if (!n.distance)//not a valid neighbor
                                continue;

                            nCell = fieldG->get(n.pt);

                            if (nCell)
                                type = nCell->type;
                            else
                                type = 0;

                            mitrTypeConst = contactCellMapMediumPtr->find(type);

                            if (mitrTypeConst != contactCellMapMediumPtr->end()) {//OK to secrete, contact detected
                                secrConstMedium = mitrTypeConst->second;

                                concentrationField.setDirect(x, y, z, currentConcentration + secrConstMedium);
                            }
                        }
                        continue;
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrOnContactDataMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {

                            contactCellMapPtr = &(mitr->second.contactCellMap);

                            for (unsigned int i = 0; i <= this->getMaxNeighborIndex(); ++i) {

                                n = this->getBoundaryStrategy()->getNeighborDirect(pt, i);
                                if (!n.distance)//not a valid neighbor
                                    continue;

                                nCell = fieldG->get(n.pt);

                                if (nCell)
                                    type = nCell->type;
                                else
                                    type = 0;

                                if (currentCellPtr == nCell)
                                    continue; //skip secretion in pixels belongin to the same cell

                                mitrTypeConst = contactCellMapPtr->find(type);
                                if (mitrTypeConst != contactCellMapPtr->end()) {//OK to secrete, contact detected
                                    secrConst = mitrTypeConst->second;

                                    concentrationField.setDirect(x, y, z, currentConcentration + secrConst);
                                }
                            }
                        }
                    }
                }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::secreteConstantConcentrationSingleField(unsigned int idx) {
    CC3D_Log(LOG_TRACE) << "***************here secreteConstantConcentrationSingleField***************" << std::endl;

    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstConstantConcentrationMap.end();


    float secrConstMedium = 0.0;

    ConcentrationField_t &concentrationField = *this->getConcentrationField(idx);

    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));

    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        secrConstMedium = mitrShared->second;
    }


    //HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
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

    pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);

#pragma omp parallel
    {

        CellG *currentCellPtr;

        float currentConcentration;
        float secrConst;

        std::map<unsigned char, float>::iterator mitr;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        bool hasExtraBndLayer = hasExtraLayer();

        Dim3D minDim;
        Dim3D maxDim;

        getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);

        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    if (hasExtraBndLayer)
                        pt = Point3D(x - 1, y - 1, z - 1);
                    else
                        pt = Point3D(x, y, z);
                    currentCellPtr = cellFieldG->getQuick(pt);

                    if (secreteInMedium && !currentCellPtr) {
                        concentrationField.setDirect(x, y, z, secrConstMedium);
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrConstConstantConcentrationMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {
                            secrConst = mitr->second;
                            concentrationField.setDirect(x, y, z, secrConst);
                        }
                    }
                }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_CPU::getMinMaxBox(bool useBoxWatcher, int threadNumber, Dim3D &minDim, Dim3D &maxDim) const {
    bool isMaxThread;
    if (useBoxWatcher) {
        minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
        maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

        isMaxThread = (threadNumber == pUtils->getNumberOfWorkNodesFESolverWithBoxWatcher() - 1);

    } else {
        minDim = pUtils->getFESolverPartition(threadNumber).first;
        maxDim = pUtils->getFESolverPartition(threadNumber).second;

        isMaxThread = (threadNumber == pUtils->getNumberOfWorkNodesFESolver() - 1);
    }

    if (!hasExtraLayer()) {
        if (threadNumber == 0) {
            minDim -= Dim3D(1, 1, 1);
        }

        if (isMaxThread) {
            maxDim -= Dim3D(1, 1, 1);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::boundaryConditionInit(int idx) {

    ConcentrationField_t &_array = *this->getConcentrationField(idx);

    bool detailedBCFlag = bcSpecFlagVec[idx];
    BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];
    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
    float deltaX = diffData.deltaX;

    //ConcentrationField_t & _array=*concentrationField;
    if (!detailedBCFlag) {
        //dealing with periodic boundary condition in x direction
        //have to use internalDim-1 in the for loop as max index because otherwise with extra shitf if we used internalDim we run outside the lattice
        if (fieldDim.x > 1) {
            if (periodicBoundaryCheckVector[0]) {//periodic boundary conditions were set in x direction
                                                 //x - periodic
                int x = 0;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(fieldDim.x, y, z));

                    }

                x = fieldDim.x + 1;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(1, y, z));

                    }
            }
            else {//noFlux BC
                int x = 0;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x + 1, y, z));

                    }

                x = fieldDim.x + 1;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x - 1, y, z));

                    }
            }
        }
        //dealing with periodic boundary condition in y direction
        if (fieldDim.y > 1) {
            if (periodicBoundaryCheckVector[1]) {//periodic boundary conditions were set in x direction
                int y = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, fieldDim.y, z));

                    }

                y = fieldDim.y + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, 1, z));

                    }
            }
            else {//NoFlux BC
                int y = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y + 1, z));

                    }

                y = fieldDim.y + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y - 1, z));

                    }
            }
        }
        if (fieldDim.z > 1) {
            //dealing with periodic boundary condition in z direction
            if (periodicBoundaryCheckVector[2]) {//periodic boundary conditions were set in x direction
                int z = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, fieldDim.z));

                    }

                z = fieldDim.z + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, 1));

                    }
            }
            else {//Noflux BC
                int z = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, z + 1));

                    }

                z = fieldDim.z + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, z - 1));

                    }
            }

        }
    }
    else {
        //detailed specification of boundary conditions
        // X axis
        if (fieldDim.x > 1) {
            if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::PERIODIC ||
                bcSpec.planePositions[1] == BoundaryConditionSpecifier::PERIODIC) {
                int x = 0;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(fieldDim.x, y, z));

                    }

                x = fieldDim.x + 1;
                for (int y = 0; y < workFieldDim.y - 1; ++y)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(1, y, z));

                    }

            }
            else {

                if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[0];
                    int x = 0;
                    for (int y = 0; y < workFieldDim.y - 1; ++y)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                    float cdValue = bcSpec.values[0];
                    int x = 0;

                    for (int y = 0; y < workFieldDim.y - 1; ++y)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, _array.getDirect(1, y, z) - cdValue * deltaX);
                        }

                }

                if (bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[1];
                    int x = fieldDim.x + 1;
                    for (int y = 0; y < workFieldDim.y - 1; ++y)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE && fieldDim.y > 1) {
                    float cdValue = bcSpec.values[1];
                    int x = fieldDim.x + 1;

                    for (int y = 0; y < workFieldDim.y - 1; ++y)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, _array.getDirect(x - 1, y, z) + cdValue * deltaX);
                        }

                }

            }
        }
        //detailed specification of boundary conditions
        // Y axis
        if (fieldDim.y > 1) {
            if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::PERIODIC ||
                bcSpec.planePositions[3] == BoundaryConditionSpecifier::PERIODIC) {
                int y = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, fieldDim.y, z));

                    }

                y = fieldDim.y + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int z = 0; z < workFieldDim.z - 1; ++z) {
                        _array.setDirect(x, y, z, _array.getDirect(x, 1, z));

                    }

            }
            else {

                if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[2];
                    int y = 0;
                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                    float cdValue = bcSpec.values[2];
                    int y = 0;

                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, _array.getDirect(x, 1, z) - cdValue * deltaX);
                        }

                }

                if (bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[3];
                    int y = fieldDim.y + 1;
                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                    float cdValue = bcSpec.values[3];
                    int y = fieldDim.y + 1;

                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int z = 0; z < workFieldDim.z - 1; ++z) {
                            _array.setDirect(x, y, z, _array.getDirect(x, y - 1, z) + cdValue * deltaX);
                        }
                }

            }
        }
        //detailed specification of boundary conditions
        // Z axis
        if (fieldDim.z > 1) {
            if (bcSpec.planePositions[4] == BoundaryConditionSpecifier::PERIODIC ||
                bcSpec.planePositions[5] == BoundaryConditionSpecifier::PERIODIC) {
                int z = 0;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, fieldDim.z));

                    }

                z = fieldDim.z + 1;
                for (int x = 0; x < workFieldDim.x - 1; ++x)
                    for (int y = 0; y < workFieldDim.y - 1; ++y) {
                        _array.setDirect(x, y, z, _array.getDirect(x, y, 1));

                    }

            }
            else {

                if (bcSpec.planePositions[4] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[4];
                    int z = 0;
                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int y = 0; y < workFieldDim.y - 1; ++y) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[4] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                    float cdValue = bcSpec.values[4];
                    int z = 0;

                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int y = 0; y < workFieldDim.y - 1; ++y) {
                            _array.setDirect(x, y, z, _array.getDirect(x, y, 1) - cdValue * deltaX);
                        }

                }

                if (bcSpec.planePositions[5] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                    float cValue = bcSpec.values[5];
                    int z = fieldDim.z + 1;
                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int y = 0; y < workFieldDim.y - 1; ++y) {
                            _array.setDirect(x, y, z, cValue);
                        }

                }
                else if (bcSpec.planePositions[5] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                    float cdValue = bcSpec.values[5];
                    int z = fieldDim.z + 1;

                    for (int x = 0; x < workFieldDim.x - 1; ++x)
                        for (int y = 0; y < workFieldDim.y - 1; ++y) {
                            _array.setDirect(x, y, z, _array.getDirect(x, y, z - 1) + cdValue * deltaX);
                        }
                }

            }
        }

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Dim3D DiffusionSolverFE_CPU::getInternalDim() {
    return getConcentrationField(0)->getInternalDim();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::stepImpl(const unsigned int _currentStep) {

    if (fluctuationCompensator) fluctuationCompensator->applyCorrections();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_TRACE) << "scalingExtraMCSVec[i]="<<scalingExtraMCSVec[i];
        prepCellTypeField(i); // here we initialize celltype array  boundaries - we do it once per  MCS

        if (scaleSecretion) {
            if (!scalingExtraMCSVec[i]) { //we do not call diffusion step but call secretion - this happens when diffusion const is 0 but we still want to have secretion
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

                }
            }

            for (int extraMCS = 0; extraMCS < scalingExtraMCSVec[i]; extraMCS++) {
                boundaryConditionInit(i);//initializing boundary conditions
                diffuseSingleField(i);
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
                }
            }
        } else { //solver behaves as FlexibleDiffusionSolver - i.e. secretion is done at once followed by multiple diffusive steps

            for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

            }

            for (int extraMCS = 0; extraMCS < scalingExtraMCSVec[i]; extraMCS++) {
                boundaryConditionInit(i);//initializing boundary conditions
                diffuseSingleField(i);
            }

        }
    }

    if (fluctuationCompensator) fluctuationCompensator->resetCorrections();

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void DiffusionSolverFE_CPU::diffuseSingleField(unsigned int idx) {
    // OPTIMIZATIONS - Maciej Swat
    // In addition to using contiguous array with scratch area being interlaced with concentration vector further optimizations are possible
    // In the most innner loop iof the FE solver one can replace maxNeighborIndex with hard coded number. Also instead of
    // Using boundary strategy to get offset array it is best to hard code offsets and access them directly
    // The downside is that in such a case one woudl have to write separate diffuseSingleField functions fdor 2D, 3D and for hex and square lattices.
    // However speedups may be worth extra effort.
    //hard coded offsets for 3D square lattice


    /// 'n' denotes neighbor

    ///this is the diffusion equation
    ///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
    ///a - diffusivity - diffConst

    ///Finite difference method:
    ///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
    ///N - number of neighbors
    ///will have to double check this formula





    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
    ConcentrationField_t &concentrationField = *this->getConcentrationField(idx);

    Array3DCUDA<signed char> &bcField = *bc_indicator_field;

    BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];

    Automaton *automaton = potts->getAutomaton();



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
#pragma omp parallel
    {

        Point3D pt, n;
        CellG *currentCellPtr = 0, *nCell = 0;
        short currentCellType = 0;
        float currentConcentration = 0;
        float updatedConcentration = 0.0;
        float concentrationSum = 0.0;
        float varDiffSumTerm = 0.0;
        float *diffCoef = diffData.diffCoef;
        float *decayCoef = diffData.decayCoef;
        float currentDiffCoef = 0.0;
        bool variableDiffusionCoefficientFlag = diffData.getVariableDiffusionCoeeficientFlag();
        bool diffusiveSite;

        std::set<unsigned char>::iterator sitr;

        Array3DCUDA<unsigned char> &cellTypeArray = *h_celltype_field; ////

        int threadNumber = pUtils->getCurrentWorkNodeNumber(); ////

        Dim3D minDim;
        Dim3D maxDim;

        if (diffData.useBoxWatcher) {
            minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
            maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

        } else {
            minDim = pUtils->getFESolverPartition(threadNumber).first;
            maxDim = pUtils->getFESolverPartition(threadNumber).second;
        }


        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    currentConcentration = concentrationField.getDirect(x, y, z);
                    currentCellType = cellTypeArray.getDirect(x, y, z);
                    currentDiffCoef = diffCoef[currentCellType];
                    pt = Point3D(x - 1, y - 1, z - 1);
                    // No diffusion where not diffusive
                    diffusiveSite = abs(currentDiffCoef) > FLT_EPSILON;

                    updatedConcentration = 0.0;
                    concentrationSum = 0.0;
                    varDiffSumTerm = 0.0;

                    unsigned int numNeighbors = maxNeighborIndex + 1;

                    if (bcField.getDirect(x, y, z) == BoundaryConditionSpecifier::INTERNAL) {//internal pixel

                        //loop over nearest neighbors
                        const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                        for (register int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/ ; ++i) {
                            const Point3D &offset = offsetVecRef[i];

                            int offX = x + offset.x;
                            int offY = y + offset.y;
                            int offZ = z + offset.z;

                            float diffCoef_offset = diffCoef[cellTypeArray.getDirect(offX, offY, offZ)];

                            // No diffusion where not diffusive
                            if (abs(diffCoef_offset) < FLT_EPSILON) {
                                --numNeighbors;
                                continue;
                            }

                            concentrationSum += concentrationField.getDirect(offX, offY, offZ);

                        }

                        concentrationSum -= numNeighbors * currentConcentration;

                        concentrationSum *= currentDiffCoef;


                        //using forward first derivatives - cartesian lattice 3D
                        if (variableDiffusionCoefficientFlag && diffusiveSite) {
                            concentrationSum /= 2.0;
                            //////float c0=concentrationField.getDirect(x,y,z);
                            //////varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x+1,y,z)]*(concentrationField.getDirect(x+1,y,z)-c0);
                            //////varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x-1,y,z)]*(concentrationField.getDirect(x-1,y,z)-c0);
                            //////varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x,y+1,z)]*(concentrationField.getDirect(x,y+1,z)-c0);
                            //////varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x,y-1,z)]*(concentrationField.getDirect(x,y-1,z)-c0);
                            //////varDiffSumTerm /=2.0;

                            //loop over nearest neighbors
                            const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                            for (register int i = 0; i <= maxNeighborIndex; ++i) {
                                const Point3D &offset = offsetVecRef[i];
                                varDiffSumTerm +=
                                        diffCoef[cellTypeArray.getDirect(x + offset.x, y + offset.y, z + offset.z)] *
                                        (concentrationField.getDirect(x + offset.x, y + offset.y, z + offset.z) -
                                         currentConcentration);
                            }
                            varDiffSumTerm /= 2.0;

                            //////const std::vector<Point3D> & offsetFDVecRef=getOffsetVec(pt); //offsets for forward derivatives


                            //////for (register int i = 0  ; i<offsetFDVecRef.size() ; ++i ){
                            //////    const Point3D & offsetFD = offsetFDVecRef[i];
                            //////    varDiffSumTerm+=(diffCoef[cellTypeArray.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)]-currentDiffCoef)*(concentrationField.getDirect(x+offsetFD.x,y+offsetFD.y,z+offsetFD.z)-concentrationField.getDirect(x,y,z));
                            //////}

                        }
                    } else { // BOUNDARY pixel - boundary pixel means belonging to the lattice but touching boundary

                        //loop over nearest neighbors
                        const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);
						CC3D_Log(LOG_TRACE) << "VISITING PIXEL="<<pt+Point3D(1,1,1);
                        for (register int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/ ; ++i) {

                            const Point3D &offset = offsetVecRef[i];
                            signed char nBcIndicator = bcField.getDirect(x + offset.x, y + offset.y, z + offset.z);
                            if (nBcIndicator == BoundaryConditionSpecifier::INTERNAL || nBcIndicator ==
                                                                                        BoundaryConditionSpecifier::BOUNDARY) { // for pixel neighbors which are internal or boundary pixels  calculations use default "internal pixel" algorithm. boundary pixel means belonging to the lattice but touching boundary
                                int offX = x + offset.x;
                                int offY = y + offset.y;
                                int offZ = z + offset.z;

                                float diffCoef_offset = diffCoef[cellTypeArray.getDirect(offX, offY, offZ)];

                                // No diffusion where not diffusive
                                if (abs(diffCoef_offset) < FLT_EPSILON) {
                                    --numNeighbors;
                                    continue;
                                }

                                concentrationSum += concentrationField.getDirect(offX, offY, offZ);

                            } else {
                                if (bcSpec.planePositions[nBcIndicator] ==
                                    BoundaryConditionSpecifier::PERIODIC) {// for pixel neighbors which are external pixels with periodic BC  calculations use default "internal pixel" algorithm
                                    int offX = x + offset.x;
                                    int offY = y + offset.y;
                                    int offZ = z + offset.z;

                                    float diffCoef_offset = diffCoef[cellTypeArray.getDirect(offX, offY, offZ)];

                                    // No diffusion where not diffusive
                                    if (abs(diffCoef_offset) < FLT_EPSILON) {
                                        --numNeighbors;
                                        continue;
                                    }

                                    concentrationSum += concentrationField.getDirect(offX, offY, offZ);
                                } else if (bcSpec.planePositions[nBcIndicator] ==
                                           BoundaryConditionSpecifier::CONSTANT_VALUE) {
                                    concentrationSum += bcSpec.values[nBcIndicator];
                                } else if (bcSpec.planePositions[nBcIndicator] ==
                                           BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                                    if (nBcIndicator == BoundaryConditionSpecifier::MIN_X ||
                                        nBcIndicator == BoundaryConditionSpecifier::MIN_Y || nBcIndicator ==
                                                                                             BoundaryConditionSpecifier::MIN_Z) { // for "left hand side" edges of the lattice the sign of the derivative expression is '-'
                                        concentrationSum += concentrationField.getDirect(x, y, z) -
                                                            bcSpec.values[nBcIndicator] *
                                                            deltaX; // notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                    } else { // for "left hand side" edges of the lattice the sign of  the derivative expression is '+'

                                        concentrationSum += concentrationField.getDirect(x, y, z) +
                                                            bcSpec.values[nBcIndicator] *
                                                            deltaX;// notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                    }
                                }
                            }
                        }

                        concentrationSum -= numNeighbors * currentConcentration;

                        concentrationSum *= currentDiffCoef;



                        //using forward first derivatives - cartesian lattice 3D
                        if (variableDiffusionCoefficientFlag && diffusiveSite) {
                            concentrationSum /= 2.0;

                            const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                            for (register int i = 0; i <= maxNeighborIndex; ++i) {
                                const Point3D &offset = offsetVecRef[i];
                                signed char nBcIndicator = bcField.getDirect(x + offset.x, y + offset.y, z + offset.z);
                                float c_offset = concentrationField.getDirect(x + offset.x, y + offset.y, z + offset.z);
                                //for pixels belonging to outside boundary we have to use boundary conditions to determine the value of the concentration at this pixel
                                if (!(nBcIndicator == BoundaryConditionSpecifier::INTERNAL) &&
                                    !(nBcIndicator == BoundaryConditionSpecifier::BOUNDARY)) {
                                    if (bcSpec.planePositions[nBcIndicator] == BoundaryConditionSpecifier::PERIODIC) {
                                        //for periodic BC we do nothing we simply use whatever is returned by concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z)
                                    } else if (bcSpec.planePositions[nBcIndicator] ==
                                               BoundaryConditionSpecifier::CONSTANT_VALUE) {
                                        c_offset = bcSpec.values[nBcIndicator];
                                    } else if (bcSpec.planePositions[nBcIndicator] ==
                                               BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                                        if (nBcIndicator == BoundaryConditionSpecifier::MIN_X ||
                                            nBcIndicator == BoundaryConditionSpecifier::MIN_Y || nBcIndicator ==
                                                                                                 BoundaryConditionSpecifier::MIN_Z) { // for "left hand side" edges of the lattice the sign of the derivative expression is '-'
                                            c_offset = currentConcentration - bcSpec.values[nBcIndicator] *
                                                                              deltaX; // notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                        } else { // for "left hand side" edges of the lattice the sign of  the derivative expression is '+'

                                            c_offset = currentConcentration + bcSpec.values[nBcIndicator] *
                                                                              deltaX;// notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                        }
                                    }
                                }

                                varDiffSumTerm +=
                                        diffCoef[cellTypeArray.getDirect(x + offset.x, y + offset.y, z + offset.z)] *
                                        (c_offset - currentConcentration);
                            }

                            varDiffSumTerm /= 2.0;

                        }


                    }
                    updatedConcentration = (concentrationSum + varDiffSumTerm) +
                                           (1 - decayCoef[currentCellType]) * currentConcentration;


                    //imposing artificial limits on allowed concentration
                    if (diffData.useThresholds) {
                        if (updatedConcentration > diffData.maxConcentration) {
                            updatedConcentration = diffData.maxConcentration;
                        }
                        if (updatedConcentration < diffData.minConcentration) {
                            updatedConcentration = diffData.minConcentration;
                        }
                    }


                    concentrationField.setDirectSwap(x, y, z, updatedConcentration);//updating scratch

                }


    }

    concentrationField.swapArrays();

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::initImpl() {
    //do nothing on CPU
}

void DiffusionSolverFE_CPU::solverSpecific(CC3DXMLElement *_xmlData) {
    //do nothing on CPU
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_CPU::extraInitImpl() {
    //do nothing on CPU
}

void DiffusionSolverFE_CPU::initCellTypesAndBoundariesImpl() {
    //do nothing on CPU
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::string DiffusionSolverFE_CPU::toStringImpl() {
    return "DiffusionSolverFE";
}


