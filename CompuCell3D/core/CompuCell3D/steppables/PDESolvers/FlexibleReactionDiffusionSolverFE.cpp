

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>

#include <PublicUtilities/StringUtils.h>
#include <muParser/muParser.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <Logger/CC3DLogger.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//
//
// }


using namespace CompuCell3D;
using namespace std;


#include "FlexibleReactionDiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverSerializer::serialize() {

    for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
        ostringstream outName;

        outName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "_" << currentStep << "."
                << serializedFileExtension;
        ofstream outStream(outName.str().c_str());
        solverPtr->outputField(outStream, solverPtr->concentrationFieldVector[i]);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverSerializer::readFromFile() {
    try {
        for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
            ostringstream inName;
            inName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "." << serializedFileExtension;

            solverPtr->readConcentrationField(inName.str().c_str(), solverPtr->concentrationFieldVector[i]);;
        }

    } catch (CC3DException &e) {
        CC3D_Log(LOG_DEBUG) << "COULD NOT FIND ONE OF THE FILES";
        throw CC3DException("Error in reading diffusion fields from file", e);
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FlexibleReactionDiffusionSolverFE::FlexibleReactionDiffusionSolverFE()
        : DiffusableVectorCommon<float, Array3DContiguous>(), deltaX(1.0), deltaT(1.0), extraTimesPerMCS(0) {
    serializerPtr = 0;
    serializeFlag = false;
    pUtils = 0;
    readFromFileFlag = false;
    haveCouplingTerms = false;
    serializeFrequency = 0;
    boxWatcherSteppable = 0;
    cellTypeVariableName = "CellType";
    useBoxWatcher = false;

    diffusionLatticeScalingFactor = 1.0;
    autoscaleDiffusion = false;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FlexibleReactionDiffusionSolverFE::~FlexibleReactionDiffusionSolverFE() {
    if (serializerPtr)
        delete serializerPtr;
    serializerPtr = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


    simPtr = _simulator;
    simulator = _simulator;
    potts = _simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    ///**

    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


	pUtils=simulator->getParallelUtils();
	CC3D_Log(LOG_DEBUG) << "INSIDE INIT";



    ///setting member function pointers
    diffusePtr = &FlexibleReactionDiffusionSolverFE::diffuse;
    secretePtr = &FlexibleReactionDiffusionSolverFE::secrete;


    update(_xmlData, true);

    //numberOfFields=diffSecrFieldTuppleVec.size();




	vector<string> concentrationFieldNameVectorTmp; //temporary vector for field names
	///assign vector of field names
	concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(),string(""));
	CC3D_Log(LOG_DEBUG) << "diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
        CC3D_Log(LOG_DEBUG) << " concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i];	}



	CC3D_Log(LOG_DEBUG) << "FIELDS THAT I HAVE";
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i];
    }




    ///allocate fields including scrartch field
    allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(), fieldDim);
    workFieldDim = concentrationFieldVector[0]->getInternalDim();


    //here I need to copy field names from concentrationFieldNameVectorTmp to concentrationFieldNameVector
    //because concentrationFieldNameVector is reallocated with default values once I call allocateDiffusableFieldVector


    for (unsigned int i = 0; i < concentrationFieldNameVectorTmp.size(); ++i) {
        concentrationFieldNameVector[i] = concentrationFieldNameVectorTmp[i];
    }


    //register fields once they have been allocated
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        simPtr->registerConcentrationField(concentrationFieldNameVector[i], concentrationFieldVector[i]);
        CC3D_Log(LOG_DEBUG) << "registring field: "<<concentrationFieldNameVector[i]<<" field address="<<concentrationFieldVector[i];
		}





    //    exit(0);

    periodicBoundaryCheckVector.assign(3, false);
    string boundaryName;
    boundaryName = potts->getBoundaryXName();
    changeToLower(boundaryName);
    if (boundaryName == "periodic") {
        periodicBoundaryCheckVector[0] = true;
    }
    boundaryName = potts->getBoundaryYName();
    changeToLower(boundaryName);
    if (boundaryName == "periodic") {
        periodicBoundaryCheckVector[1] = true;
    }

    boundaryName = potts->getBoundaryZName();
    changeToLower(boundaryName);
    if (boundaryName == "periodic") {
        periodicBoundaryCheckVector[2] = true;
    }

    //determining latticeType and setting diffusionLatticeScalingFactor
    //When you evaluate div asa flux through the surface divided bby volume those scaling factors appear automatically. On cartesian lattife everythink is one so this is easy to forget that on different lattices they are not1
    if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) {
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z ==
                                                  1) { //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
            diffusionLatticeScalingFactor = 1.0 / sqrt(3.0);// (2/3)/dL^2 dL=sqrt(2/sqrt(3)) so (2/3)/dL^2=1/sqrt(3)
        } else {//3D simulation
            diffusionLatticeScalingFactor = pow(2.0, -4.0 /
                                                     3.0); //(1/2)/dL^2 dL dL^2=2**(1/3) so (1/2)/dL^2=1/(2.0*2^(1/3))=2^(-4/3)
        }

    }

    //we only autoscale diffusion when user requests it explicitely
    if (!autoscaleDiffusion) {
        diffusionLatticeScalingFactor = 1.0;
    }


    simulator->registerSteerableObject(this);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::extraInit(Simulator *simulator) {

    if ((serializeFlag || readFromFileFlag) && !serializerPtr) {
        serializerPtr = new FlexibleReactionDiffusionSolverSerializer();
        serializerPtr->solverPtr = this;
    }

    if (serializeFlag) {
        simulator->registerSerializer(serializerPtr);
    }

    bool steppableAlreadyRegisteredFlag;
    if (useBoxWatcher) {
        boxWatcherSteppable = (BoxWatcher *) Simulator::steppableManager.get("BoxWatcher",
                                                                             &steppableAlreadyRegisteredFlag);
        if (!steppableAlreadyRegisteredFlag)
            boxWatcherSteppable->init(simulator);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::handleEvent(CC3DEvent &_event) {
    if (_event.id != LATTICE_RESIZE) {
        return;
    }

    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);

    for (size_t i = 0; i < concentrationFieldVector.size(); ++i) {
        concentrationFieldVector[i]->resizeAndShift(ev.newDim, ev.shiftVec);
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::start() {


    dt_dx2 = deltaT / (deltaX * deltaX);

    if (simPtr->getRestartEnabled()) {
        return;  // we will not initialize cells if restart flag is on
    }

    if (readFromFileFlag) {
        try {

            serializerPtr->readFromFile();

        } catch (CC3DException &e) {
            CC3D_Log(LOG_DEBUG) << "Going to fail-safe initialization";
            initializeConcentration(); //if there was error, initialize using failsafe defaults
        }

    } else {
        initializeConcentration();//Normal reading from User specified files
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleReactionDiffusionSolverFE::initializeConcentration() {
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
        CC3D_Log(LOG_DEBUG) << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName, concentrationFieldVector[i]);
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::step(const unsigned int _currentStep) {

    currentStep = _currentStep;

    (this->*secretePtr)();

    (this->*diffusePtr)();


    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleReactionDiffusionSolverFE::secreteOnContactSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    std::map<unsigned char, SecretionOnContactData>::iterator mitrShared;
    std::map<unsigned char, SecretionOnContactData>::iterator end_mitr = secrData.typeIdSecrOnContactDataMap.end();


    ConcentrationField_t &concentrationField = *concentrationFieldVector[idx];
    //Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();


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
        WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        unsigned char type;

        int threadNumber = pUtils->getCurrentWorkNodeNumber();

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
                    pt = Point3D(x - 1, y - 1, z - 1);
                    ///**
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    currentConcentration = concentrationField.getDirect(x, y, z);

                    if (secreteInMedium && !currentCellPtr) {
                        for (int i = 0; i <= maxNeighborIndex/*offsetVec.size()*/ ; ++i) {
                            n = boundaryStrategy->getNeighborDirect(pt, i);
                            if (!n.distance)//not a valid neighbor
                                continue;
                            ///**
                            nCell = fieldG->get(n.pt);
                            //                      nCell = fieldG->get(n.pt);
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

                            for (int i = 0; i <= maxNeighborIndex/*offsetVec.size() */; ++i) {

                                n = boundaryStrategy->getNeighborDirect(pt, i);
                                if (!n.distance)//not a valid neighbor
                                    continue;
                                ///**
                                nCell = fieldG->get(n.pt);
                                //                      nCell = fieldG->get(n.pt);
                                if (nCell)
                                    type = nCell->type;
                                else
                                    type = 0;

                                if (currentCellPtr == nCell)
                                    continue; //skip secretion in pixels belongin to the same cell

                                mitrTypeConst = contactCellMapPtr->find(type);
                                if (mitrTypeConst != contactCellMapPtr->end()) {//OK to secrete, contact detected
                                    secrConst = mitrTypeConst->second;
                                    //                         concentrationField->set(pt,currentConcentration+secrConst);
                                    concentrationField.setDirect(x, y, z, currentConcentration + secrConst);
                                }
                            }
                        }
                    }
                }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::secreteSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    float maxUptakeInMedium = 0.0;
    float relativeUptakeRateInMedium = 0.0;
    float secrConstMedium = 0.0;

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();
    std::map<unsigned char, UptakeData>::iterator mitrUptakeShared;
    std::map<unsigned char, UptakeData>::iterator end_mitrUptake = secrData.typeIdUptakeDataMap.end();



    //ConcentrationField_t & concentrationField=*concentrationFieldVector[idx];

    ConcentrationField_t &concentrationField = *concentrationFieldVector[idx];
    //Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();

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
        //Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
        float currentConcentration;
        float secrConst;


        std::map<unsigned char, float>::iterator mitr;
        std::map<unsigned char, UptakeData>::iterator mitrUptake;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

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

                    pt = Point3D(x - 1, y - 1, z - 1);
                    // CC3D_Log(LOG_DEBUG) << "pt="<<pt<<" is valid "<<cellFieldG->isValid(pt);
                    ///**
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    //			   CC3D_Log(LOG_DEBUG) << "THIS IS PTR="<<currentCellPtr;

                    //             if(currentCellPtr)
                    // 				  CC3D_Log(LOG_DEBUG) << "This is id="<<currentCellPtr->id;
                    //currentConcentration = concentrationField.getDirect(x,y,z);

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
                                    CC3D_Log(LOG_DEBUG) << " uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake;
                                } else {
                                    CC3D_Log(LOG_DEBUG) << "concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<currentConcentration*mitrUptake->second.relativeUptakeRate;
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
void FlexibleReactionDiffusionSolverFE::secreteConstantConcentrationSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstConstantConcentrationMap.end();


    float secrConstMedium = 0.0;

    ConcentrationField_t &concentrationField = *concentrationFieldVector[idx];
    //Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();

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
        //Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
        float currentConcentration;
        float secrConst;

        std::map<unsigned char, float>::iterator mitr;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

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

                    pt = Point3D(x - 1, y - 1, z - 1);
                    CC3D_Log(LOG_TRACE) << "pt="<<pt<<" is valid "<<cellFieldG->isValid(pt);
					///**
					currentCellPtr=cellFieldG->getQuick(pt);
					//             currentCellPtr=cellFieldG->get(pt);
					// 			CC3D_Log(LOG_TRACE) << "THIS IS PTR="<<currentCellPtr;

                    //             if(currentCellPtr)
                    // 				  CC3D_Log(LOG_TRACE) << "This is id="<<currentCellPtr->id;
                    //currentConcentration = concentrationArray[x][y][z];

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
void FlexibleReactionDiffusionSolverFE::secrete() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

        }


    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::boundaryConditionInit(int idx) {

    ConcentrationField_t &_array = *concentrationFieldVector[idx];
    bool detailedBCFlag = bcSpecFlagVec[idx];
    BoundaryConditionSpecifier &bcSpec = bcSpecVec[idx];
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
        //dealing with periodic boundary condition in z direction
        if (fieldDim.z > 1) {
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
    } else {
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
                else if (bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
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



void FlexibleReactionDiffusionSolverFE::solveRDEquations() {


    for (int idx = 0; idx < numberOfFields; ++idx) {
        solveRDEquationsSingleField(idx);
    }

    for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
        ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
        if (diffSecrFieldTuppleVec[fieldIdx].diffData.diffConst == 0.0 &&
            diffSecrFieldTuppleVec[fieldIdx].diffData.decayConst == 0.0 &&
            diffSecrFieldTuppleVec[fieldIdx].diffData.additionalTerm == "0.0") {
            continue; // do not swap arrays in such acase
        }

        concentrationField.swapArrays();
    }

}

void FlexibleReactionDiffusionSolverFE::solveRDEquationsSingleField(unsigned int idx) {

    /// 'n' denotes neighbor

    ///this is the diffusion equation
    ///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
    ///a - diffusivity - diffConst

    ///Finite difference method:
    ///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
    ///N - number of neighbors
    ///will have to double check this formula



    DiffusionData diffData = diffSecrFieldTuppleVec[idx].diffData;
    float diffConst = diffData.diffConst;

    bool useBoxWatcher = false;

    if (diffData.useBoxWatcher)
        useBoxWatcher = true;


    if (diffSecrFieldTuppleVec[idx].diffData.diffConst == 0.0 &&
        diffSecrFieldTuppleVec[idx].diffData.decayConst == 0.0 &&
        diffSecrFieldTuppleVec[idx].diffData.additionalTerm == "0.0") {
        return; // do not solve equation if diffusion, decay constants and additional term are 0
    }


    float dt_dx2 = deltaT / (deltaX * deltaX);


    std::set<unsigned char>::iterator sitr;


    Automaton *automaton = potts->getAutomaton();


    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    boundaryConditionInit(idx);//initializing boundary conditions
    //boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions



    //////vector<bool> avoidMediumVec(numberOfFields,false);
    bool avoidMedium = false;
    if (diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != diffData.avoidTypeIdSet.end()) {
        avoidMedium = true;
    }


    if (useBoxWatcher) {

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
    pUtils->prepareParallelRegionFESolvers(useBoxWatcher);
#pragma omp parallel
    {


        CellG *currentCellPtr = 0;
        Point3D pt, n;
        float currentConcentration = 0;
        float updatedConcentration = 0.0;
        float concentrationSum = 0.0;
        short neighborCounter = 0;
        CellG *neighborCellPtr = 0;

        try {


            int threadNumber = pUtils->getCurrentWorkNodeNumber();

            Dim3D minDim;
            Dim3D maxDim;

            if (useBoxWatcher) {
                minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
                maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

            } else {
                minDim = pUtils->getFESolverPartition(threadNumber).first;
                maxDim = pUtils->getFESolverPartition(threadNumber).second;
            }


            // finding concentrations at x,y,z at t+dt

            ConcentrationField_t &concentrationField = *concentrationFieldVector[idx];

            ExpressionEvaluator &ev = eedVec[idx][threadNumber];

            for (int z = minDim.z; z < maxDim.z; z++)
                for (int y = minDim.y; y < maxDim.y; y++)
                    for (int x = minDim.x; x < maxDim.x; x++) {

                        pt = Point3D(x - 1, y - 1, z - 1);
                        ///**
                        currentCellPtr = cellFieldG->getQuick(pt);
                        currentConcentration = concentrationField.getDirect(x, y, z);

                        if (currentCellPtr)
                            ev[0] = currentCellPtr->type; //0 is idx to cell type var in exp evaluator
                        else
                            ev[0] = 0;

                        //setting up x,y,z variables
                        ev[1] = pt.x; //x-variable
                        ev[2] = pt.y; //y-variable
                        ev[3] = pt.z; //z-variable

                        //getting concentrations at x,y,z for all the fields
                        for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
                            ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
                            ev[4 + fieldIdx] = concentrationField.getDirect(x, y, z);
                        }



                        //if (currentCellPtr)
                        //	variableCellTypeMu[threadNumber]=currentCellPtr->type;
                        //else
                        //	variableCellTypeMu[threadNumber]=0;

                        ////getting concentrations at x,y,z for all the fields
                        //for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
                        //	ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
                        //	variableConcentrationVecMu[threadNumber][fieldIdx]=concentrationField.getDirect(x,y,z);

                        //}


                        //DoNotDiffuseTo means do not solve RD equations for points occupied by this cell type
                        if (avoidMedium && !currentCellPtr) {//if medium is to be avoided
                            concentrationField.setDirectSwap(x, y, z, variableConcentrationVecMu[threadNumber][idx]);
                            continue;
                        }

                        if (currentCellPtr &&
                            diffData.avoidTypeIdSet.find(currentCellPtr->type) != diffData.avoidTypeIdSet.end()) {

                            concentrationField.setDirectSwap(x, y, z, variableConcentrationVecMu[threadNumber][idx]);
                            continue; // avoid user defined types
                        }

                        updatedConcentration = 0.0;
                        concentrationSum = 0.0;
                        neighborCounter = 0;

                        //loop over nearest neighbors
                        CellG *neighborCellPtr = 0;
                        const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);
                        CC3D_Log(LOG_DEBUG) << "maxNeighborIndex="<<maxNeighborIndex;

                        for (int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/ ; ++i) {
                            const Point3D &offset = offsetVecRef[i];


                            if (diffData.avoidTypeIdSet.size() ||
                                avoidMedium) { //means user defined types to avoid in terms of the diffusion

                                n = pt + offsetVecRef[i];
                                neighborCellPtr = cellFieldG->get(n);
                                if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                                if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) !=
                                                       diffData.avoidTypeIdSet.end())
                                    continue;//avoid user specified types
                            }
                            concentrationSum += concentrationField.getDirect(x + offset.x, y + offset.y, z + offset.z);

                            ++neighborCounter;

                        }


                        updatedConcentration = dt_dx2 * diffusionLatticeScalingFactor * diffConst *
                                               (concentrationSum - neighborCounter * currentConcentration) +
                                               currentConcentration;

                        //additionalTerm contributions


                        //updatedConcentration+=deltaT*parserVec[threadNumber][idx].Eval();
                        updatedConcentration += deltaT * ev.eval();


                        concentrationField.setDirectSwap(x, y, z, updatedConcentration);//updating scratch


                    }


        } catch (mu::Parser::exception_type &e) {
            CC3D_Log(LOG_DEBUG) << e.GetMsg();
            throw CC3DException(e.GetMsg());
        }
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool FlexibleReactionDiffusionSolverFE::isBoudaryRegion(int x, int y, int z, Dim3D dim) {
    if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3)
        return true;
    else
        return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::diffuse() {

    for (int idx = 0; idx < extraTimesPerMCS + 1; ++idx) {
        solveRDEquations();
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                              ConcentrationField_t *concentrationField) {
    //scratchField->switchContainersQuick(*(concentrationField));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
    Point3D pt;
    float tempValue;

    for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
            for (pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                tempValue = _concentrationField->get(pt);
                _out << pt.x << " " << pt.y << " " << pt.z << " " << tempValue << endl;
            }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleReactionDiffusionSolverFE::readConcentrationField(std::string fileName,
                                                               ConcentrationField_t *concentrationField) {

    std::string basePath = simulator->getBasePath();
    std::string fn = fileName;
    if (basePath != "") {
        fn = basePath + "/" + fileName;
    }

    ifstream in(fn.c_str());

    if (!in.is_open()) throw CC3DException(string("Could not open chemical concentration file '") + fn + "'!");

    Point3D pt;
    float c;
    //Zero entire field
    for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
            for (pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                concentrationField->set(pt, 0);
            }

    while (!in.eof()) {
        in >> pt.x >> pt.y >> pt.z >> c;
        if (!in.fail())
            concentrationField->set(pt, c);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleReactionDiffusionSolverFE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //if(potts->getDisplayUnitsFlag()){
    //	Unit diffConstUnit=powerUnit(potts->getLengthUnit(),2)/potts->getTimeUnit();
    //	Unit decayConstUnit=1/potts->getTimeUnit();
    //	Unit secretionConstUnit=1/potts->getTimeUnit();

    //	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units");
    //	if (!unitsElem){ //add Units element
    //		unitsElem=_xmlData->attachElement("Units");
    //	}

    //	if(unitsElem->getFirstElement("DiffusionConstantUnit")){
    //		unitsElem->getFirstElement("DiffusionConstantUnit")->updateElementValue(diffConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("DiffusionConstantUnit",diffConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("DecayConstantUnit")){
    //		unitsElem->getFirstElement("DecayConstantUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("DecayConstantUnit",decayConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("DeltaXUnit")){
    //		unitsElem->getFirstElement("DeltaXUnit")->updateElementValue(potts->getLengthUnit().toString());
    //	}else{
    //		unitsElem->attachElement("DeltaXUnit",potts->getLengthUnit().toString());
    //	}

    //	if(unitsElem->getFirstElement("DeltaTUnit")){
    //		unitsElem->getFirstElement("DeltaTUnit")->updateElementValue(potts->getTimeUnit().toString());
    //	}else{
    //		unitsElem->attachElement("DeltaTUnit",potts->getTimeUnit().toString());
    //	}

    //	if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
    //		unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
    //	}



    //	if(unitsElem->getFirstElement("SecretionUnit")){
    //		unitsElem->getFirstElement("SecretionUnit")->updateElementValue(secretionConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("SecretionUnit",secretionConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("SecretionOnContactUnit")){
    //		unitsElem->getFirstElement("SecretionOnContactUnit")->updateElementValue(secretionConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("SecretionOnContactUnit",secretionConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("ConstantConcentrationUnit")){
    //		unitsElem->getFirstElement("ConstantConcentrationUnit")->updateElementValue(secretionConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("ConstantConcentrationUnit",secretionConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("DecayConstantUnit")){
    //		unitsElem->getFirstElement("DecayConstantUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("DecayConstantUnit",decayConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("DeltaXUnit")){
    //		unitsElem->getFirstElement("DeltaXUnit")->updateElementValue(potts->getLengthUnit().toString());
    //	}else{
    //		unitsElem->attachElement("DeltaXUnit",potts->getLengthUnit().toString());
    //	}

    //	if(unitsElem->getFirstElement("DeltaTUnit")){
    //		unitsElem->getFirstElement("DeltaTUnit")->updateElementValue(potts->getTimeUnit().toString());
    //	}else{
    //		unitsElem->attachElement("DeltaTUnit",potts->getTimeUnit().toString());
    //	}

    //	if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
    //		unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("UptakeUnit")){
    //		unitsElem->getFirstElement("UptakeUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("UptakeUnit",decayConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("RelativeUptakeUnit")){
    //		unitsElem->getFirstElement("RelativeUptakeUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("RelativeUptakeUnit",decayConstUnit.toString());
    //	}

    //	if(unitsElem->getFirstElement("MaxUptakeUnit")){
    //		unitsElem->getFirstElement("MaxUptakeUnit")->updateElementValue(decayConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("MaxUptakeUnit",decayConstUnit.toString());
    //	}



    //}

    //notice, limited steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
    // Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running

    diffSecrFieldTuppleVec.clear();
    bcSpecVec.clear();
    bcSpecFlagVec.clear();

    if (_xmlData->findElement("DeltaX"))
        deltaX = _xmlData->getFirstElement("DeltaX")->getDouble();

    if (_xmlData->findElement("DeltaT"))
        deltaT = _xmlData->getFirstElement("DeltaT")->getDouble();

    if (_xmlData->findElement("ExtraTimesPerMCS"))
        extraTimesPerMCS = _xmlData->getFirstElement("ExtraTimesPerMCS")->getUInt();


    if (_xmlData->findElement("CellTypeVariableName"))
        cellTypeVariableName = _xmlData->getFirstElement("CellTypeVariableName")->getDouble();
    if (_xmlData->findElement("UseBoxWatcher"))
        useBoxWatcher = true;

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");
    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {
        diffSecrFieldTuppleVec.push_back(FlexibleDiffusionSecretionRDFieldTupple());
        DiffusionData &diffData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].diffData;
        SecretionData &secrData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].secrData;

        if (diffFieldXMLVec[i]->findAttribute("Name")) {
            diffData.fieldName = diffFieldXMLVec[i]->getAttribute("Name");
        }

        if (diffFieldXMLVec[i]->findElement("DiffusionData"))
            diffData.update(diffFieldXMLVec[i]->getFirstElement("DiffusionData"));

        if (diffFieldXMLVec[i]->findElement("SecretionData"))
            secrData.update(diffFieldXMLVec[i]->getFirstElement("SecretionData"));

        if (diffFieldXMLVec[i]->findElement("ReadFromFile"))
            readFromFileFlag = true;

        //boundary conditions parsing
        bcSpecFlagVec.push_back(false);
        bcSpecVec.push_back(BoundaryConditionSpecifier());

        if (diffFieldXMLVec[i]->findElement("BoundaryConditions")) {
            bcSpecFlagVec[bcSpecFlagVec.size() - 1] = true;
            BoundaryConditionSpecifier &bcSpec = bcSpecVec[bcSpecVec.size() - 1];

            CC3DXMLElement *bcSpecElem = diffFieldXMLVec[i]->getFirstElement("BoundaryConditions");
            CC3DXMLElementList planeVec = bcSpecElem->getElements("Plane");

            for (unsigned int ip = 0; ip < planeVec.size(); ++ip) {
                if (!planeVec[ip]->findAttribute("Axis"))
                    throw CC3DException("Boundary Condition specification Plane element is missing Axis attribute");
                string axisName = planeVec[ip]->getAttribute("Axis");
                int index = 0;
                if (axisName == "x" || axisName == "X") {
                    index = 0;
                }
                if (axisName == "y" || axisName == "Y") {
                    index = 2;
                }
                if (axisName == "z" || axisName == "Z") {
                    index = 4;
                }

                if (planeVec[ip]->findElement("Periodic")) {
                    bcSpec.planePositions[index] = BoundaryConditionSpecifier::PERIODIC;
                    bcSpec.planePositions[index + 1] = BoundaryConditionSpecifier::PERIODIC;
                } else {
                    //if (planeVec[ip]->findElement("ConstantValue")){
                    CC3DXMLElementList cvVec = planeVec[ip]->getElements("ConstantValue");
                    CC3DXMLElementList cdVec = planeVec[ip]->getElements("ConstantDerivative");

                    for (unsigned int v = 0; v < cvVec.size(); ++v) {
                        string planePos = cvVec[v]->getAttribute("PlanePosition");
                        double value = cvVec[v]->getAttributeAsDouble("Value");
                        changeToLower(planePos);
                        if (planePos == "min") {
                            bcSpec.planePositions[index] = BoundaryConditionSpecifier::CONSTANT_VALUE;
                            bcSpec.values[index] = value;

                        } else if (planePos == "max") {
                            bcSpec.planePositions[index + 1] = BoundaryConditionSpecifier::CONSTANT_VALUE;
                            bcSpec.values[index + 1] = value;
                        } else {
                            throw CC3DException("PlanePosition attribute has to be either max on min");
                        }

                    }
                    if (cvVec.size() <= 1) {
                        for (unsigned int d = 0; d < cdVec.size(); ++d) {
                            string planePos = cdVec[d]->getAttribute("PlanePosition");
                            double value = cdVec[d]->getAttributeAsDouble("Value");
                            changeToLower(planePos);
                            if (planePos == "min") {
                                bcSpec.planePositions[index] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                                bcSpec.values[index] = value;

                            } else if (planePos == "max") {
                                bcSpec.planePositions[index + 1] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                                bcSpec.values[index + 1] = value;
                            } else {
                                throw CC3DException("PlanePosition attribute has to be either max on min");
                            }

                        }
                    }

                }

            }

            if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) {
                // static_cast<Cruncher*>(this)->getBoundaryStrategy()->getLatticeType();
                if (bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Z] == BoundaryConditionSpecifier::PERIODIC ||
                    bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Z] == BoundaryConditionSpecifier::PERIODIC) {
                    if (fieldDim.z > 1 && fieldDim.z % 3) {
                        throw CC3DException(
                                "For Periodic Boundary Conditions On Hex Lattice the Z Dimension Has To Be Divisible By 3");
                    }
                }

                if (bcSpec.planePositions[BoundaryConditionSpecifier::MIN_X] == BoundaryConditionSpecifier::PERIODIC ||
                    bcSpec.planePositions[BoundaryConditionSpecifier::MAX_X] == BoundaryConditionSpecifier::PERIODIC) {
                    if (fieldDim.x % 2) {
                        throw CC3DException(
                                "For Periodic Boundary Conditions On Hex Lattice the X Dimension Has To Be Divisible By 2 ");
                    }
                }

                if (bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Y] == BoundaryConditionSpecifier::PERIODIC ||
                    bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Y] == BoundaryConditionSpecifier::PERIODIC) {
                    if (fieldDim.y % 2) {
                        throw CC3DException(
                                "For Periodic Boundary Conditions On Hex Lattice the Y Dimension Has To Be Divisible By 2 ");
                    }
                }
            }

        }


    }
    if (_xmlData->findElement("Serialize")) {

        serializeFlag = true;
        if (_xmlData->getFirstElement("Serialize")->findAttribute("Frequency")) {
            serializeFrequency = _xmlData->getFirstElement("Serialize")->getAttributeAsUInt("Frequency");
        }
        CC3D_Log(LOG_DEBUG) << "serialize Flag="<<serializeFlag;

    }

    if (_xmlData->findElement("ReadFromFile")) {
        readFromFileFlag = true;
        CC3D_Log(LOG_DEBUG) << "readFromFileFlag="<<readFromFileFlag;
    }


    for (int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        diffSecrFieldTuppleVec[i].diffData.setAutomaton(automaton);
        diffSecrFieldTuppleVec[i].secrData.setAutomaton(automaton);
        diffSecrFieldTuppleVec[i].diffData.initialize(automaton);
        diffSecrFieldTuppleVec[i].secrData.initialize(automaton);
    }

    ///assigning member method ptrs to the vector

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.assign(
                diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.size(), 0);
        unsigned int j = 0;
        for (set<string>::iterator sitr = diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.begin();
             sitr != diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.end(); ++sitr) {

            if ((*sitr) == "Secretion") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleReactionDiffusionSolverFE::secreteSingleField;
                ++j;
            } else if ((*sitr) == "SecretionOnContact") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleReactionDiffusionSolverFE::secreteOnContactSingleField;
                ++j;
            } else if ((*sitr) == "ConstantConcentration") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleReactionDiffusionSolverFE::secreteConstantConcentrationSingleField;
                ++j;
            }

        }
    }

    numberOfFields = diffSecrFieldTuppleVec.size();

    //specify names for variables used in eed
    vector <string> variableNames;
    variableNames.push_back(cellTypeVariableName);
    variableNames.push_back("x"); //x -coordinate var
    variableNames.push_back("y"); //y -coordinate var
    variableNames.push_back("z"); //z -coordinate var
    for (unsigned int j = 0; j < numberOfFields; ++j) {
        variableNames.push_back(diffSecrFieldTuppleVec[j].diffData.fieldName);
    }

    try {
        //allocate expression evaluator depot vector
        eedVec.assign(numberOfFields, ExpressionEvaluatorDepot());
        for (unsigned int i = 0; i < numberOfFields; ++i) {
            eedVec[i].allocateSize(pUtils->getMaxNumberOfWorkNodesFESolver());
            eedVec[i].addVariables(variableNames.begin(), variableNames.end());
            if (diffSecrFieldTuppleVec[i].diffData.additionalTerm == "") {
                diffSecrFieldTuppleVec[i].diffData.additionalTerm = "0.0"; //in case additonal term is set empty we will return 0.0
            }
            eedVec[i].setExpression(diffSecrFieldTuppleVec[i].diffData.additionalTerm);

        }


    } catch (mu::Parser::exception_type &e) {
        CC3D_Log(LOG_DEBUG) << e.GetMsg();
        throw CC3DException(e.GetMsg());
    }

    ////allocate vector of parsers
    //parserVec.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),vector<mu::Parser>(numberOfFields,mu::Parser()));

    //variableConcentrationVecMu.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),vector<double>(numberOfFields,0.0));
    //variableCellTypeMu.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),0.0);
    ////initialize parsers
    //try{
    //	for(unsigned int t = 0 ; t < pUtils->getMaxNumberOfWorkNodesFESolver(); ++t){
    //		for(unsigned int i = 0 ; i < numberOfFields ; ++i){
    //			for(unsigned int j = 0 ; j < numberOfFields ; ++j){
    //				parserVec[t][i].DefineVar(diffSecrFieldTuppleVec[j].diffData.fieldName, &variableConcentrationVecMu[t][j]);
    //			}
    //			parserVec[t][i].DefineVar(cellTypeVariableName,&variableCellTypeMu[t]);
    //			if (diffSecrFieldTuppleVec[i].diffData.additionalTerm==""){
    //				diffSecrFieldTuppleVec[i].diffData.additionalTerm="0.0"; //in case additonal term is set empty we will return 0.0
    //			}
    //			parserVec[t][i].SetExpr(diffSecrFieldTuppleVec[i].diffData.additionalTerm);
    //		}
    //	}

	//} catch (mu::Parser::exception_type &e)
	//{
		// CC3D_Log(LOG_TRACE) << e.GetMsg();
    //	ASSERT_OR_THROW(e.GetMsg(),0);
    //}













}

std::string FlexibleReactionDiffusionSolverFE::toString() {
    return "FlexibleReactionDiffusionSolverFE";
}


std::string FlexibleReactionDiffusionSolverFE::steerableName() {
    return toString();
}


