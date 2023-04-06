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
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <time.h>


#include "KernelDiffusionSolver.h"
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KernelDiffusionSolverSerializer::serialize() {

    for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
        ostringstream outName;

        outName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "_" << currentStep << "."
                << serializedFileExtension;
        ofstream outStream(outName.str().c_str());
        solverPtr->outputField(outStream, solverPtr->concentrationFieldVector[i]);
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KernelDiffusionSolverSerializer::readFromFile() {
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
KernelDiffusionSolver::KernelDiffusionSolver() {
    serializerPtr = 0;
    pUtils = 0;
    serializeFlag = false;
    readFromFileFlag = false;
    haveCouplingTerms = false;
    serializeFrequency = 0;
    boxWatcherSteppable = 0;
    //    useBoxWatcher=false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
KernelDiffusionSolver::~KernelDiffusionSolver() {
    if (serializerPtr)
        delete serializerPtr;
    serializerPtr = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KernelDiffusionSolver::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


    simPtr = simulator;
    this->simulator = simulator;
    potts = simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    pUtils = simulator->getParallelUtils();

    update(_xmlData, true);

    numberOfFields = diffSecrFieldTuppleVec.size();
    CC3D_Log(LOG_DEBUG) << "Number of Fields: " << numberOfFields;

    for (int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Field Name: " << diffSecrFieldTuppleVec[i].getDiffusionData()->fieldName;
    }

	vector<string> concentrationFieldNameVectorTmp; //temporary vector for field names
	///assign vector of field names
	concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(),string(""));
	CC3D_Log(LOG_DEBUG) << "diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
        CC3D_Log(LOG_DEBUG) << " concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i];
    }


	CC3D_Log(LOG_DEBUG) << "fieldDim.x: " << fieldDim.x << "  fieldDim.y: " << fieldDim.y << "  fieldDim.z: " << fieldDim.z;

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




    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();
    float max_kernel = 1;
    for (int q = 0; q < kernel.size(); q++) {
        if (max_kernel < kernel[q])
            max_kernel = kernel[q];

        BoundaryStrategy::getInstance()->prepareNeighborListsBasedOnNeighborOrder(kernel[q]);
        tempmaxNeighborIndex.push_back(
                BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(kernel[q]));
    }
    for (int q = 0; q < tempmaxNeighborIndex.size(); q++) {
        CC3D_Log(LOG_DEBUG) << tempmaxNeighborIndex[q];
	}
	CC3D_Log(LOG_DEBUG) << "Kernel: " << max_kernel;
    BoundaryStrategy::getInstance()->prepareNeighborListsBasedOnNeighborOrder(max_kernel);
    maxNeighborIndex = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(max_kernel);
    Neighbor neighbor;
    Point3D pt;
    pt.x = 0;
    pt.y = 0;
    pt.z = 0;
    int dis = 0;
    float tempDis = 0;

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);

        if (tempDis != neighbor.distance) {
            dis++;
        }
        neighborDistance[ceil(neighbor.distance * 1000)] = dis;
        tempDis = neighbor.distance;
    }

    initializeKernel(simulator);

}


void KernelDiffusionSolver::initializeKernel(Simulator *simulator) {
    numberOfFields = diffSecrFieldTuppleVec.size();
    CC3D_Log(LOG_DEBUG) << "Number of Fields: " << numberOfFields;
    float diffConst;
    Point3D pt;
    pt.x = (fieldDim.x > 1 ? fieldDim.x / 2 : 0);
    pt.y = (fieldDim.y > 1 ? fieldDim.y / 2 : 0);
    pt.z = (fieldDim.z > 1 ? fieldDim.z / 2 : 0);

    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();
    boundaryStrategy = BoundaryStrategy::getInstance();
    Neighbor neighbor;
    
    vector<float> Ker;
    Ker.assign(maxNeighborIndex + 2, 0.0);
    NKer.assign(numberOfFields, vector<float>(maxNeighborIndex + 3, 0.0));  //extra point for offset, extra point for <= issue, extra point for neighborhood offset
    int dimension = 0;
    dimension += (fieldDim.x > 1 ? 1 : 0);
    dimension += (fieldDim.y > 1 ? 1 : 0);
    dimension += (fieldDim.z > 1 ? 1 : 0);
    CC3D_Log(LOG_DEBUG) << "pt="<<pt;
    //new BEN
    for (int m = 0; m < numberOfFields; m++) {
        float sum = 0;
        float normalize = 0;

        diffConst = diffSecrFieldTuppleVec[m].diffData.diffConst;
        decayConst = diffSecrFieldTuppleVec[m].diffData.decayConst;
        float ld = sqrt(diffConst);
        CC3D_Log(LOG_DEBUG) <<  "Diffusion Constant: " << diffConst;
        CC3D_Log(LOG_DEBUG) << "Decay Constant: " << decayConst;
        CC3D_Log(LOG_DEBUG) << "Kernel: " << kernel[m];
        //       neighborIter;
        for (unsigned int nIdx = 0; nIdx <= tempmaxNeighborIndex[m]; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            CC3D_Log(LOG_DEBUG) << "n.pt="<<neighbor.pt<<" distance="<<neighbor.distance;
            float temp = exp(-1.0 * pow(neighbor.distance * coarseGrainFactorVec[m], 2) / (4.0 * ld * ld));
            sum += temp;
            Ker[nIdx + 1] = temp;

        }

        float temp = exp(-1.0 * pow(0.0, 2) / (4.0 * ld * ld));
        sum += temp;
        Ker[0] = temp;

        for (int i = 0; i < Ker.size(); i++) {
            NKer[m][i] = (Ker[i] / sum);
        }

		if (decayConst > 0) {
			CC3D_Log(LOG_DEBUG) << "Decay Const="<< exp(-decayConst);
			for(int i = 0; i < Ker.size(); i++) {
				NKer[m][i] = NKer[m][i]*exp(-decayConst);
			}
		}
		NKer[m][Ker.size()] = NKer[m][Ker.size()-1];
		CC3D_Log(LOG_DEBUG) << "maxNeighborIndex: " << maxNeighborIndex;
        CC3D_Log(LOG_DEBUG) << "Ker.size(): " << Ker.size();

        for (int i = 0; i < Ker.size(); i++) {
            CC3D_Log(LOG_DEBUG) << "NKer: " << NKer[m][i] << "  i: " << i;
        }
    }
    CC3D_Log(LOG_DEBUG) << "fieldDim.x: " << fieldDim.x << "  fieldDim.y: " << fieldDim.y << "  fieldDim.z: " << fieldDim.z;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KernelDiffusionSolver::extraInit(Simulator *simulator) {
    if ((serializeFlag || readFromFileFlag) && !serializerPtr) {
        serializerPtr = new KernelDiffusionSolverSerializer();
        serializerPtr->solverPtr = this;
    }

    if (serializeFlag) {
        simulator->registerSerializer(serializerPtr);
    }

}

void KernelDiffusionSolver::handleEvent(CC3DEvent &_event) {
    CC3D_Log(LOG_TRACE) << " THIS IS EVENT HANDLE FOR FAST DIFFUSION 2D FE";
	if (_event.id==LATTICE_RESIZE){
		throw CC3DException(
                "KernelDiffusionSolver works only with simulations with full periodic boundary conditions and lattice resizing is not supported for such simulations");
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KernelDiffusionSolver::start() {
	if (simPtr->getRestartEnabled()){
		return ;  // we will not initialize cells if restart flag is on
	}
	CC3D_Log(LOG_DEBUG) << "initialzieConcentration";
    initializeConcentration();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void KernelDiffusionSolver::initializeConcentration() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (!diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression.empty()) {
            initializeFieldUsingEquation(concentrationFieldVector[i],
                                         diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression);
            continue;
        }
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
        CC3D_Log(LOG_DEBUG) << "fail-safe initialization "<<diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName, concentrationFieldVector[i]);
    }

    // diffSecrFieldTuppleVec.size() = 1; concentrationFieldVector.size() = 2
    CC3D_Log(LOG_DEBUG) << "numberOfFields = " << numberOfFields << "\tdiffSecrFieldTuppleVec.size() = " << diffSecrFieldTuppleVec.size() << "\tconcentrationFieldVector.size() = " << concentrationFieldVector.size();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void KernelDiffusionSolver::secreteOnContactSingleField(unsigned int idx) {

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

//Kernel solvers do not use box watchers
    pUtils->prepareParallelRegionFESolvers();
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

        //secretion is partitioned in the same way as we do partitioning of FE solver without box watcher
        Dim3D minDim;
        Dim3D maxDim;

        minDim = pUtils->getFESolverPartition(threadNumber).first;
        maxDim = pUtils->getFESolverPartition(threadNumber).second;


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
void KernelDiffusionSolver::secreteSingleField(unsigned int idx) {

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

//Kernel solvers do not use box watchers
    pUtils->prepareParallelRegionFESolvers();
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

        //secretion is partitioned in the same way as we do partitioning of FE solver without box watcher
        Dim3D minDim;
        Dim3D maxDim;

        minDim = pUtils->getFESolverPartition(threadNumber).first;
        maxDim = pUtils->getFESolverPartition(threadNumber).second;


        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    pt = Point3D(x - 1, y - 1, z - 1);
                    CC3D_Log(LOG_TRACE) << "pt="<<pt<<" is valid "<<cellFieldG->isValid(pt);
                    ///**
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    CC3D_Log(LOG_TRACE) << "THIS IS PTR="<<currentCellPtr;

					//             if(currentCellPtr)
									// CC3D_Log(LOG_TRACE) << "This is id="<<currentCellPtr->id;
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
                                    CC3D_Log(LOG_TRACE) << " uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake;								}else{
									CC3D_Log(LOG_TRACE) << "concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<currentConcentration*mitrUptake->second.relativeUptakeRate;
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
void KernelDiffusionSolver::secreteConstantConcentrationSingleField(unsigned int idx) {
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

//Kernel solvers do not use box watchers
    pUtils->prepareParallelRegionFESolvers();;

#pragma omp parallel
    {

        CellG *currentCellPtr;
        //Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
        float currentConcentration;
        float secrConst;

        std::map<unsigned char, float>::iterator mitr;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        //secretion is partitioned in the same way as we do partitioning of FE solver without box watcher
        Dim3D minDim;
        Dim3D maxDim;

        minDim = pUtils->getFESolverPartition(threadNumber).first;
        maxDim = pUtils->getFESolverPartition(threadNumber).second;


        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    pt = Point3D(x - 1, y - 1, z - 1);
                    CC3D_Log(LOG_TRACE) << "pt="<<pt<<" is valid "<<cellFieldG->isValid(pt);
                    ///**
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    CC3D_Log(LOG_TRACE) << "THIS IS PTR="<<currentCellPtr;

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
void KernelDiffusionSolver::secrete() {
    CC3D_Log(LOG_TRACE) << "secreting ";
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_TRACE) << "secreting field= "<<" i="<<i;
        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

            //          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
        }


    }


}


void KernelDiffusionSolver::step(const unsigned int _currentStep) {

    secrete();
    diffuse();

    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }


}

void KernelDiffusionSolver::diffuse() {

    for (int idx = 0; idx < numberOfFields; idx++) {
        diffuseSingleField(idx);
    }

}

void KernelDiffusionSolver::diffuseSingleField(unsigned int idx) {


    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    boundaryStrategy = BoundaryStrategy::getInstance();


    if (diffSecrFieldTuppleVec[idx].diffData.diffConst == 0.0 &&
        diffSecrFieldTuppleVec[idx].diffData.decayConst == 0.0) {
        return; //skip solving of the equation if diffusion and decay constants are 0
    }

    //Here we temporarily set Dim in BoundaryStrategy to reduced size we have to reset it back to the original size once we are done the field
    unsigned int coarseGrainFactor = coarseGrainFactorVec[idx];

    Dim3D originalDim = fieldDim;
    Dim3D workFieldDimTmp;
    workFieldDimTmp.x = (fieldDim.x / coarseGrainFactor > 0 ? fieldDim.x / coarseGrainFactor : 1);
    workFieldDimTmp.y = (fieldDim.y / coarseGrainFactor > 0 ? fieldDim.y / coarseGrainFactor : 1);
    workFieldDimTmp.z = (fieldDim.z / coarseGrainFactor > 0 ? fieldDim.z / coarseGrainFactor : 1);

    //partition is calculated in the same way as for the boxwatcher diffusion. Because of the possible coarse grain factor we have to make sure
    // to calculate partition each time we solve the equation
    Dim3D minDimBW(0, 0, 0);
    Dim3D maxDimBW = workFieldDimTmp;
    pUtils->calculateKernelSolverPartition(minDimBW, maxDimBW);


    boundaryStrategy->setDim(workFieldDimTmp);


    ConcentrationField_t &concentrationField = *concentrationFieldVector[idx];
    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];

//managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
    pUtils->prepareParallelRegionKernelSolvers();
#pragma omp parallel
    {
        Point3D pt;
        Point3D ptCoarseGrained;
        Neighbor neighbor;


        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        Dim3D minDim;
        Dim3D maxDim;

        minDim = pUtils->getKernelSolverPartition(threadNumber).first;
        maxDim = pUtils->getKernelSolverPartition(threadNumber).second;

        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {


                    float value = 0.0;
                    float zero_val = 0.0;
                    pt = Point3D(x * coarseGrainFactor, y * coarseGrainFactor, z * coarseGrainFactor);
                    ptCoarseGrained = Point3D(x, y, z);

                    value += concentrationField.getDirect(pt.x + 1, pt.y + 1, pt.z + 1) * NKer[idx][0];


                    for (unsigned int nIdx = 0; nIdx <= tempmaxNeighborIndex[idx]; ++nIdx) {
                        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(ptCoarseGrained), nIdx);


                        if (!neighbor.distance) {
                            //if distance is 0 then the neighbor returned is invalid
                            continue;
                        }


                        value += concentrationField.getDirect(neighbor.pt.x * coarseGrainFactor + 1,
                                                              neighbor.pt.y * coarseGrainFactor + 1,
                                                              neighbor.pt.z * coarseGrainFactor + 1) *
                                 NKer[idx][nIdx + 1];

                    }
                    concentrationField.setDirectSwap(pt.x + 1, pt.y + 1, pt.z + 1, value + zero_val);

                    writePixelValue(Point3D(pt.x, pt.y, pt.z),
                                    concentrationField.getDirectSwap(pt.x + 1, pt.y + 1, pt.z + 1), coarseGrainFactor,
                                    concentrationField);

                }


    }

    concentrationField.swapArrays();
    //have to reset Dim in boundary Strategy to original value otherwise you will buggy simulation
    boundaryStrategy->setDim(originalDim);

}


void KernelDiffusionSolver::writePixelValue(Point3D pt, float value, unsigned int coarseGrainFactor,
                                            ConcentrationField_t &_concentrationField) {
    if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1) {//2D case
        if (fieldDim.x == 1) {
            for (unsigned int y = pt.y; y < pt.y + coarseGrainFactor; ++y)
                for (unsigned int z = pt.z; z < pt.z + coarseGrainFactor; ++z) {
                    _concentrationField.setDirectSwap(1, y + 1, z + 1, value);
                }

        } else if (fieldDim.y == 1) {
            for (unsigned int x = pt.x; x < pt.x + coarseGrainFactor; ++x)
                for (unsigned int z = pt.z; z < pt.z + coarseGrainFactor; ++z) {
                    _concentrationField.setDirectSwap(x + 1, 1, z + 1, value);
                }

        } else if (fieldDim.z == 1) {
            for (unsigned int x = pt.x; x < pt.x + coarseGrainFactor; ++x)
                for (unsigned int y = pt.y; y < pt.y + coarseGrainFactor; ++y) {
                    _concentrationField.setDirectSwap(x + 1, y + 1, 1, value);
                }
        }
    } else {//3D case
        for (unsigned int x = pt.x; x < pt.x + coarseGrainFactor; ++x)
            for (unsigned int y = pt.y; y < pt.y + coarseGrainFactor; ++y)
                for (unsigned int z = pt.z; z < pt.z + coarseGrainFactor; ++z) {

                    _concentrationField.setDirectSwap(x + 1, y + 1, z + 1, value);

                }
    }
}


void KernelDiffusionSolver::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {
    Point3D pt;

    pt.z = 0;
    pt.y = 0;
    pt.x = 0;
    concentrationField->set(pt, 0);
    CC3D_Log(LOG_DEBUG) << "In ReadConcentration:  " << "concentrationField: " << concentrationField;

    std::string basePath = simulator->getBasePath();
    std::string fn = fileName;
    if (basePath != "") {
        fn = basePath + "/" + fileName;
    }

    ifstream in(fn.c_str());

    if (!in.is_open()) throw CC3DException(string("Could not open chemical concentration file '") + fn + "'!");



    //    Point3D pt;
    float c;
    //Zero entire field
    for (pt.z = 0; pt.z < fieldDim.z; pt.z++) {
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++) {
            for (pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                CC3D_Log(LOG_TRACE) << "pt.x: " << pt.x << "  pt.y: " << pt.y << "  pt.z: " << pt.z;
                concentrationField->set(pt, 0);
                CC3D_Log(LOG_TRACE) << "pt.x: " << pt.x << "  pt.y: " << pt.y << "  pt.z: " << pt.z;
            }
        }
    }
    CC3D_Log(LOG_DEBUG) << "Begin Filling Concentration Field";
    while (!in.eof()) {
        in >> pt.x >> pt.y >> pt.z >> c;
        if (!in.fail())
            concentrationField->set(pt, c);
    }
    CC3D_Log(LOG_DEBUG) << "Exiting ReadConcentration";

}

void KernelDiffusionSolver::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
    Point3D pt;
    float tempValue;

    for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
            for (pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                tempValue = _concentrationField->get(pt);
                _out << pt.x << " " << pt.y << " " << pt.z << " " << tempValue << endl;
            }
}

void KernelDiffusionSolver::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                  ConcentrationField_t *concentrationField) {
    //scratchField->switchContainersQuick(*(concentrationField));

}


void KernelDiffusionSolver::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //notice, limited steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
    // Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running

    diffSecrFieldTuppleVec.clear();

    coarseGrainFactorVec.clear();
    coarseGrainMultiplicativeFactorVec.clear();

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");
    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {
        diffSecrFieldTuppleVec.push_back(DiffusionSecretionKernelFieldTupple());
        DiffusionData &diffData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].diffData;
        SecretionData &secrData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].secrData;

        if (diffFieldXMLVec[i]->findElement("Kernel")) {
            kernel.push_back(diffFieldXMLVec[i]->getFirstElement("Kernel")->getUInt());
        }

        if (diffFieldXMLVec[i]->findAttribute("Name")) {
            diffData.fieldName = diffFieldXMLVec[i]->getAttribute("Name");
        }

        if (diffFieldXMLVec[i]->findElement("DiffusionData"))
            diffData.update(diffFieldXMLVec[i]->getFirstElement("DiffusionData"));

        if (diffFieldXMLVec[i]->findElement("SecretionData"))
            secrData.update(diffFieldXMLVec[i]->getFirstElement("SecretionData"));

        if (diffFieldXMLVec[i]->findElement("ReadFromFile"))
            readFromFileFlag = true;


        if (diffFieldXMLVec[i]->findElement("CoarseGrainFactor")) {
            coarseGrainFactorVec.push_back(diffFieldXMLVec[i]->getFirstElement("CoarseGrainFactor")->getUInt());
        } else {
            coarseGrainFactorVec.push_back(1);
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
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &KernelDiffusionSolver::secreteSingleField;
                ++j;
            } else if ((*sitr) == "SecretionOnContact") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &KernelDiffusionSolver::secreteOnContactSingleField;
                ++j;
            } else if ((*sitr) == "ConstantConcentration") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &KernelDiffusionSolver::secreteConstantConcentrationSingleField;
                ++j;
            }

        }
    }

    coarseGrainMultiplicativeFactorVec = coarseGrainFactorVec;
    //coarseGraining check

    bool suitableForCoarseGrainingFlag = true;
    for (unsigned int i = 0; i < coarseGrainFactorVec.size(); ++i) {
        coarseGrainMultiplicativeFactorVec[i] = coarseGrainFactorVec[i] * coarseGrainFactorVec[i];
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1) {//2D case

            if (fieldDim.x != 1 && (fieldDim.x % coarseGrainFactorVec[i])) {
                suitableForCoarseGrainingFlag = false;
                break;
            }
            if (fieldDim.y != 1 && (fieldDim.y % coarseGrainFactorVec[i])) {
                suitableForCoarseGrainingFlag = false;
                break;
            }
            if (fieldDim.z != 1 && (fieldDim.z % coarseGrainFactorVec[i])) {
                suitableForCoarseGrainingFlag = false;
                break;
            }


        } else {//3D case
            coarseGrainMultiplicativeFactorVec[i] =
                    coarseGrainFactorVec[i] * coarseGrainFactorVec[i] * coarseGrainFactorVec[i];
            if ((fieldDim.z % coarseGrainFactorVec[i]) || (fieldDim.y % coarseGrainFactorVec[i]) ||
                (fieldDim.z % coarseGrainFactorVec[i])) {
                suitableForCoarseGrainingFlag = false;
                break;
            }
        }
    }

    if (!suitableForCoarseGrainingFlag)
        throw CC3DException("DIMENSIONS OF THE LATTICE ARE INCOMPATIBLE WITH COARSE GRAINING FACTOR");

}


std::string KernelDiffusionSolver::toString() {
    return "KernelDiffusionSolver";
}

std::string KernelDiffusionSolver::steerableName() {
    return toString();
}

