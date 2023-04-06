

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>

#include <PublicUtilities/StringUtils.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Logger/CC3DLogger.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//
//
// }


using namespace CompuCell3D;
using namespace std;


#include "FlexibleDiffusionSolverADE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FlexibleDiffusionSolverADE::FlexibleDiffusionSolverADE()
        : DiffusableVector<float>(), deltaX(1.0), deltaT(1.0) {
    serializerPtr = 0;
    serializeFlag = false;
    readFromFileFlag = false;
    haveCouplingTerms = false;
    serializeFrequency = 0;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FlexibleDiffusionSolverADE::~FlexibleDiffusionSolverADE() {
    if (serializerPtr)
        delete serializerPtr;
    serializerPtr = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


    simPtr = _simulator;
    simulator = _simulator;
    potts = _simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    ///**
    //   cellFieldG=potts->getCellFieldG();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    CC3D_Log(LOG_DEBUG) << "INSIDE INIT";



    ///setting member function pointers
    diffusePtr = &FlexibleDiffusionSolverADE::diffuse;
    secretePtr = &FlexibleDiffusionSolverADE::secrete;


    update(_xmlData, true);

    numberOfFields = diffSecrFieldTuppleVec.size();


    vector <string> concentrationFieldNameVectorTmp; //temporary vector for field names
    ///assign vector of field names
    concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(), string(""));

    CC3D_Log(LOG_DEBUG) << "diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
        CC3D_Log(LOG_DEBUG) << " concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i]<<endl;
	}

    //setting up couplingData - field-field interaction terms
    vector<CouplingData>::iterator pos;

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        pos = diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
        for (int j = 0; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size(); ++j) {

            for (int idx = 0; idx < concentrationFieldNameVectorTmp.size(); ++idx) {
                if (concentrationFieldNameVectorTmp[idx] ==
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].intrFieldName) {
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].fieldIdx = idx;
                    haveCouplingTerms = true; //if this is called at list once we have already coupling terms and need to proceed differently with scratch field initialization
                    break;
                }
                //this means that required interacting field name has not been found
                if (idx == concentrationFieldNameVectorTmp.size() - 1) {
                    //remove this interacting term
                    //                pos=&(diffDataVec[i].degradationDataVec[j]);
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec.erase(pos);
                }
            }
            ++pos;
        }
    }

	CC3D_Log(LOG_DEBUG) << "FIELDS THAT I HAVE";
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i];
    }

	CC3D_Log(LOG_DEBUG) << "FlexibleDiffusionSolverADE: extra Init in read XML";

    workFieldDim = Dim3D(fieldDim.x + 2, fieldDim.y + 2, fieldDim.z + 2);
    ///allocate fields including scrartch field
    if (!haveCouplingTerms) {
        allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size() + 1,
                                      workFieldDim); //+1 is for additional scratch field
    } else {
        allocateDiffusableFieldVector(2 * diffSecrFieldTuppleVec.size(),
                                      workFieldDim); //with coupling terms every field need to have its own scratch field
    }

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

    simulator->registerSteerableObject(this);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::extraInit(Simulator *simulator) {

    //	Serialization doesn't work. So I will switch it OFF
    /*

    if((fdspdPtr->serializeFlag || fdspdPtr->readFromFileFlag) && !serializerPtr){
    serializerPtr=new FlexibleDiffusionSolverSerializer();
    serializerPtr->solverPtr=this;
    }

    if(serializeFlag){
    simulator->registerSerializer(serializerPtr);
    }
    */
    //Turning off box watcher as well
    //bool useBoxWatcher=false;
    //for (int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
    //    if(diffSecrFieldTuppleVec[i].diffData.useBoxWatcher){
    //      useBoxWatcher=true;
    //      break;
    //   }
    //}
    //bool steppableAlreadyRegisteredFlag;
    //if(useBoxWatcher){
    //   boxWatcherSteppable=(BoxWatcher*)Simulator::steppableManager.get("BoxWatcher",&steppableAlreadyRegisteredFlag);
    //   if(!steppableAlreadyRegisteredFlag)
    //      boxWatcherSteppable->init(simulator);
    //}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::start() {
	//     if(diffConst> (1.0/6.0-0.05) ){ //hard coded condtion for stability of the solutions - assume dt=1 dx=dy=dz=1
	//		CC3D_Log(LOG_TRACE) << "CANNOT SOLVE DIFFUSION EQUATION: STABILITY PROBLEM - DIFFUSION CONSTANT TOO LARGE. EXITING...";
    //       exit(0);
    //
    //    }


    dt_dx2 = deltaT / (deltaX * deltaX);
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

void FlexibleDiffusionSolverADE::initializeConcentration() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
        CC3D_Log(LOG_DEBUG) << "fail-safe initialization "<<diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
		readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName,concentrationFieldVector[i]);
	}
	//for(unsigned int i = 0 ; i <fdspdPtr->diffSecrFieldTuppleVec.size() ; ++i){
	//   if(fdspdPtr->diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
	// 	 CC3D_Log(LOG_TRACE) << "fail-safe initialization "<<fdspdPtr->diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
    //   readConcentrationField(fdspdPtr->diffSecrFieldTuppleVec[i].diffData.concentrationFileName,concentrationFieldVector[i]);
    //}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::step(const unsigned int _currentStep) {

    currentStep = _currentStep;

    (this->*secretePtr)();

    (this->*diffusePtr)();


    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleDiffusionSolverADE::secreteOnContactSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;


    std::map<unsigned char, SecretionOnContactData>::iterator mitr;
    std::map<unsigned char, SecretionOnContactData>::iterator end_mitr = secrData.typeIdSecrOnContactDataMap.end();

    CellG *currentCellPtr;

    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    Array3D_t &concentrationArray = concentrationFieldPtr->getContainer();

    float currentConcentration;
    float secrConst;
    float secrConstMedium = 0.0;
    std::map<unsigned char, float> *contactCellMapMediumPtr;
    std::map<unsigned char, float> *contactCellMapPtr;
    std::map<unsigned char, float>::iterator mitrTypeConst;

    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0

    if (secrData.secretionOnContactTypeIds.find(automaton->getTypeId("Medium")) !=
        secrData.secretionOnContactTypeIds.end()) {
        mitr = secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));
        secreteInMedium = true;
        contactCellMapMediumPtr = &(mitr->second.contactCellMap);
    }


    Point3D pt;
    Neighbor n;
    CellG *nCell = 0;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    unsigned char type;


    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);
                ///**
                currentCellPtr = cellFieldG->get(pt);
                //             currentCellPtr=cellFieldG->get(pt);
                currentConcentration = concentrationArray[x][y][z];


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

                            concentrationArray[x][y][z] = currentConcentration + secrConstMedium;
                        }


                    }

                    continue;
                }

                if (currentCellPtr) {
                    if (secrData.secretionOnContactTypeIds.find(currentCellPtr->type) ==
                        secrData.secretionOnContactTypeIds.end()) {
                        continue;
                    }
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

                            if (currentCellPtr == nCell) continue; //skip secretion in pixels belongin to the same cell

                            mitrTypeConst = contactCellMapPtr->find(type);
                            if (mitrTypeConst != contactCellMapPtr->end()) {//OK to secrete, contact detected
                                secrConst = mitrTypeConst->second;
                                //                         concentrationField->set(pt,currentConcentration+secrConst);
                                concentrationArray[x][y][z] = currentConcentration + secrConst;
                            }

                        }


                    }
                }
            }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::secreteSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;


    std::map<unsigned char, float>::iterator mitr;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();
    std::map<unsigned char, UptakeData>::iterator mitrUptake;
    std::map<unsigned char, UptakeData>::iterator end_mitrUptake = secrData.typeIdUptakeDataMap.end();


    CellG *currentCellPtr;
    Field3D<float> *concentrationField = concentrationFieldVector[idx];
    float currentConcentration;
    float secrConst;
    float secrConstMedium = 0.0;
    float maxUptakeInMedium = 0.0;
    float relativeUptakeRateInMedium = 0.0;

    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    Array3D_t &concentrationArray = concentrationFieldPtr->getContainer();

    bool doUptakeFlag = false;
    bool uptakeInMediumFlag = false;
    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitr = secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));

    if (mitr != end_mitr) {
        secreteInMedium = true;
        secrConstMedium = mitr->second;
    }

    //uptake for medium setup
    if (secrData.typeIdUptakeDataMap.size()) {
        doUptakeFlag = true;
    }
    //uptake for medium setup
    if (doUptakeFlag) {
        mitrUptake = secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
        if (mitrUptake != end_mitrUptake) {
            maxUptakeInMedium = mitrUptake->second.maxUptake;
            relativeUptakeRateInMedium = mitrUptake->second.relativeUptakeRate;
            uptakeInMediumFlag = true;

        }
    }


    Point3D pt;
    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);
                currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);

                //             if(currentCellPtr)
                currentConcentration = concentrationArray[x][y][z];

                if (secreteInMedium && !currentCellPtr) {
                    concentrationArray[x][y][z] = currentConcentration + secrConstMedium;
                }

                if (currentCellPtr) {
                    mitr = secrData.typeIdSecrConstMap.find(currentCellPtr->type);
                    if (mitr != end_mitr) {
                        secrConst = mitr->second;
                        concentrationArray[x][y][z] = currentConcentration + secrConst;
                    }
                }

                if (doUptakeFlag) {
                    if (uptakeInMediumFlag && !currentCellPtr) {
                        if (currentConcentration > maxUptakeInMedium) {
                            concentrationArray[x][y][z] -= maxUptakeInMedium;
                        } else {
                            concentrationArray[x][y][z] -= currentConcentration * relativeUptakeRateInMedium;
                        }
                    }
                    if (currentCellPtr) {

                        mitrUptake = secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
                        if (mitrUptake != end_mitrUptake) {
                            if (currentConcentration > mitrUptake->second.maxUptake) {
                                concentrationArray[x][y][z] -= mitrUptake->second.maxUptake;
                                CC3D_Log(LOG_TRACE) << " uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake;
                            } else {
                                CC3D_Log(LOG_TRACE) << "concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<currentConcentration*mitrUptake->second.relativeUptakeRate;
                                concentrationArray[x][y][z] -=
                                        currentConcentration * mitrUptake->second.relativeUptakeRate;
                            }
                        }
                    }
                }
            }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::secreteConstantConcentrationSingleField(unsigned int idx) {


    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;


    std::map<unsigned char, float>::iterator mitr;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstConstantConcentrationMap.end();

    CellG *currentCellPtr;
    Field3D<float> *concentrationField = concentrationFieldVector[idx];
    float currentConcentration;
    float secrConst;
    float secrConstMedium = 0.0;

    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    Array3D_t &concentrationArray = concentrationFieldPtr->getContainer();


    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0

    if (secrData.constantConcentrationTypeIds.find(automaton->getTypeId("Medium")) !=
        secrData.constantConcentrationTypeIds.end()) {
        mitr = secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));
        secreteInMedium = true;
        secrConstMedium = mitr->second;

    }

// 	if( mitr != end_mitr){
// 		secreteInMedium=true;
// 		secrConstMedium=mitr->second;
// 	}

    Point3D pt;
    CC3D_Log(LOG_DEBUG) << "work workFieldDim="<<workFieldDim;
    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);

				///**
				currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);

				//             if(currentCellPtr)
				//currentConcentration = concentrationArray[x][y][z];

                if (secreteInMedium && !currentCellPtr) {
                    concentrationArray[x][y][z] = secrConstMedium;
                }

                if (currentCellPtr) {
                    if (secrData.constantConcentrationTypeIds.find(currentCellPtr->type) ==
                        secrData.constantConcentrationTypeIds.end()) {
                        continue;
                    }

                    mitr = secrData.typeIdSecrConstConstantConcentrationMap.find(currentCellPtr->type);

                    if (mitr != end_mitr) {
                        secrConst = mitr->second;
                        concentrationArray[x][y][z] = secrConst;
                    }
                }
            }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::secrete() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

            //          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
        }


    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float FlexibleDiffusionSolverADE::couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec,
                                               float _currentConcentration) {

    float couplingTerm = 0.0;
    float coupledConcentration;
    for (int i = 0; i < _couplDataVec.size(); ++i) {
        coupledConcentration = concentrationFieldVector[_couplDataVec[i].fieldIdx]->get(_pt);
        couplingTerm += _couplDataVec[i].couplingCoef * _currentConcentration * coupledConcentration;
    }

    return couplingTerm;


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::boundaryConditionInit(ConcentrationField_t *concentrationField) {
    Array3D_t &concentrationArray = concentrationField->getContainer();

    //dealing with periodic boundary condition in x direction
    if (periodicBoundaryCheckVector[0]) {//periodic boundary conditions were set in x direction
        int x = 0;
        for (int y = 0; y < workFieldDim.y; ++y)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[workFieldDim.x - 2][y][z];
            }
        x = workFieldDim.x - 1;
        for (int y = 0; y < workFieldDim.y; ++y)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[1][y][z];
            }
    } else {//noFlux BC
        int x = 0;
        for (int y = 0; y < workFieldDim.y; ++y)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x + 1][y][z];
            }
        x = workFieldDim.x - 1;
        for (int y = 0; y < workFieldDim.y; ++y)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x - 1][y][z];
            }

    }

    //dealing with periodic boundary condition in y direction
    if (periodicBoundaryCheckVector[1]) {//periodic boundary conditions were set in x direction
        int y = 0;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x][workFieldDim.y - 2][z];
            }
        y = workFieldDim.y - 1;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x][1][z];
            }
    } else {//NoFlux BC
        int y = 0;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x][y + 1][z];
            }
        y = workFieldDim.y - 1;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int z = 0; z < workFieldDim.z; ++z) {
                concentrationArray[x][y][z] = concentrationArray[x][y - 1][z];
            }
    }

    //dealing with periodic boundary condition in z direction
    if (periodicBoundaryCheckVector[2]) {//periodic boundary conditions were set in x direction
        int z = 0;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int y = 0; y < workFieldDim.y; ++y) {
                concentrationArray[x][y][z] = concentrationArray[x][y][workFieldDim.z - 2];
            }
        z = workFieldDim.z - 1;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int y = 0; y < workFieldDim.y; ++y) {
                concentrationArray[x][y][z] = concentrationArray[x][y][1];
            }
    } else {//Noflux BC
        int z = 0;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int y = 0; y < workFieldDim.y; ++y) {
                concentrationArray[x][y][z] = concentrationArray[x][y][z + 1];
            }
        z = workFieldDim.z - 1;
        for (int x = 0; x < workFieldDim.x; ++x)
            for (int y = 0; y < workFieldDim.y; ++y) {
                concentrationArray[x][y][z] = concentrationArray[x][y][z - 1];
            }

    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleDiffusionSolverADE::diffuseSingleField(unsigned int idx) {
    /// 'n' denotes neighbor

    ///this is the diffusion equation
    ///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
    ///a - diffusivity - diffConst

    ///Finite difference method:
    ///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
    ///N - number of neighbors
    ///will have to double check this formula


    Point3D pt, n;
    unsigned int token = 0;
    double distance;
    CellG *currentCellPtr = 0, *nCell = 0;

    short currentCellType = 0;
    float conSum = 0.0;
    float updatedCon = 0.0;


    float currentCon = 0.0;
    short neighborCounter = 0;

    // These assignments would be better to remove to some other place.
    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
    float diffConst = diffData.diffConst;
    float decayConst = diffData.decayConst;
    float deltaT = diffData.deltaT;
    float deltaX = diffData.deltaX;
    float dt_dx2 = deltaT / (deltaX * deltaX);
    float mu = diffConst * dt_dx2;

    int dim = 3;
    int offX = 2, offY = 2, offZ = 2;

    // Set values for A and B based on the dimension.
    if (fieldDim.x == 1) {
        dim -= 1;
        offX = 1;
    }

    if (fieldDim.y == 1) {
        dim -= 1;
        offY = 1;
    }

    if (fieldDim.z == 1) {
        dim -= 1;
        offZ = 1;
    }

    A = (1 - dim * mu) / (1 + dim * mu);
    B = mu / (1 + dim * mu);

    std::set<unsigned char>::iterator sitr;
    std::set<unsigned char>::iterator end_sitr = diffData.avoidTypeIdSet.end();
    std::set<unsigned char>::iterator end_sitr_decay = diffData.avoidDecayInIdSet.end();

    Automaton *automaton = potts->getAutomaton();
    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    ConcentrationField_t *scratchFieldPtr;

    if (!haveCouplingTerms)
        scratchFieldPtr = concentrationFieldVector[diffSecrFieldTuppleVec.size()];
    else
        scratchFieldPtr = concentrationFieldVector[diffSecrFieldTuppleVec.size() + idx];

    //	scratchFieldPtr=concentrationFieldVector[fdspdPtr->diffSecrFieldTuppleVec.size()];

    Array3D_t &conArray = concentrationFieldPtr->getContainer();
    Array3D_t &scratchArray = scratchFieldPtr->getContainer();

    boundaryConditionInit(concentrationFieldPtr); //initializing boundary conditions

    bool avoidMedium = false;
    bool avoidDecayInMedium = false;
    //the assumption is that medium has type ID 0
    if (diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr) {
        avoidMedium = true;
    }

    if (diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay) {
        avoidDecayInMedium = true;
    }

    vector < vector < vector < float > > >
    newU(workFieldDim.x, vector < vector < float > > (workFieldDim.y, vector<float>(workFieldDim.z, float(0))));

    // Copy operator in Array3DBorders<float>::ContainerType works badly. Need to be fixed.
    // Copying element by element can be additional overhead

    for (int i = 0; i < workFieldDim.x; i++)
        for (int j = 0; j < workFieldDim.y; j++)
            for (int k = 0; k < workFieldDim.z; k++)
                newU[i][j][k] = conArray[i][j][k];

    vector < vector < vector < float > > > newV;
    // Temporary variables for ADE scheme

    // Initialize the boundary using Forward Euler method. REFACTOR
    vector<int> tmp(2, 0);

    //X faces boundary
    tmp[0] = 1;
    tmp[1] = workFieldDim.x - 2;
    for (int bb = 0; bb < tmp.size(); bb++) {
        int i = tmp[bb];
        for (int j = 1; j < workFieldDim.y - 1; j++)
            for (int k = 1; k < workFieldDim.z - 1; k++) {
                pt = Point3D(i - 1, j - 1, k - 1);
                currentCon = conArray[i][j][k];
                conSum = 0.0;
                neighborCounter = 0;

                //loop over nearest neighbors
                CellG *neighborCellPtr = 0;
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                for (int m = 0; m <= maxNeighborIndex; ++m) {
                    const Point3D &offset = offsetVecRef[m];

                    if (diffData.avoidTypeIdSet.size() ||
                        avoidMedium) //means user defined types to avoid in terms of the diffusion
                    {
                        n = pt + offsetVecRef[m];
                        neighborCellPtr = cellFieldG->get(n);
                        if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                        if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) != end_sitr)
                            continue;//avoid user specified types
                    }

                    conSum += conArray[i + offset.x][j + offset.y][k + offset.z];
                    ++neighborCounter;
                }

                newU[i][j][k] = mu * (conSum - neighborCounter * currentCon) + currentCon;
            }
    }

    //Y faces boundary
    tmp[1] = workFieldDim.y - 2;
    for (int bb = 0; bb < tmp.size(); bb++) {
        int j = tmp[bb];
        for (int i = 1; i < workFieldDim.x - 1; i++)
            for (int k = 1; k < workFieldDim.z - 1; k++) {
                pt = Point3D(i - 1, j - 1, k - 1);
                currentCon = conArray[i][j][k];
                conSum = 0.0;
                neighborCounter = 0;

                //loop over nearest neighbors
                CellG *neighborCellPtr = 0;
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                for (int m = 0; m <= maxNeighborIndex; ++m) {
                    const Point3D &offset = offsetVecRef[m];

                    if (diffData.avoidTypeIdSet.size() ||
                        avoidMedium) //means user defined types to avoid in terms of the diffusion
                    {
                        n = pt + offsetVecRef[m];
                        neighborCellPtr = cellFieldG->get(n);
                        if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                        if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) != end_sitr)
                            continue;//avoid user specified types
                    }
                    conSum += conArray[i + offset.x][j + offset.y][k + offset.z];
                    ++neighborCounter;
                }
                newU[i][j][k] = mu * (conSum - neighborCounter * currentCon) + currentCon;
            }
    }

    //Z faces boundary
    tmp[1] = workFieldDim.z - 2;
    for (int bb = 0; bb < tmp.size(); bb++) {
        int k = tmp[bb];
        for (int i = 1; i < workFieldDim.x - 1; i++)
            for (int j = 1; j < workFieldDim.z - 1; j++) {
                pt = Point3D(i - 1, j - 1, k - 1);
                currentCon = conArray[i][j][k];
                conSum = 0.0;
                neighborCounter = 0;

                //loop over nearest neighbors
                CellG *neighborCellPtr = 0;
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                for (int m = 0; m <= maxNeighborIndex; ++m) {
                    const Point3D &offset = offsetVecRef[m];

                    if (diffData.avoidTypeIdSet.size() ||
                        avoidMedium) //means user defined types to avoid in terms of the diffusion
                    {
                        n = pt + offsetVecRef[m];
                        neighborCellPtr = cellFieldG->get(n);
                        if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                        if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) != end_sitr)
                            continue;//avoid user specified types
                    }
                    conSum += conArray[i + offset.x][j + offset.y][k + offset.z];
                    ++neighborCounter;
                }
                newU[i][j][k] = mu * (conSum - neighborCounter * currentCon) + currentCon;
            }
    }


	newV = newU; // Important!
	CC3D_Log(LOG_DEBUG) <<  "\tA = " << A << "\tB = " << B << std::endl;

    // Update matrices newV and newU separately!
    for (int i = offX; i < workFieldDim.x - offX; i++)
        for (int j = offY; j < workFieldDim.y - offY; j++)
            for (int k = offZ; k < workFieldDim.z - offZ; k++) {
                pt = Point3D(i - 1, j - 1, k - 1);
                currentCon = conArray[i][j][k];
                conSum = 0.0;
                neighborCounter = 0;

                //loop over nearest neighbors
                CellG *neighborCellPtr = 0;
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                for (int m = 0; m <= maxNeighborIndex; ++m) {
                    const Point3D &offset = offsetVecRef[m];

                    if (diffData.avoidTypeIdSet.size() ||
                        avoidMedium) //means user defined types to avoid in terms of the diffusion
                    {
                        n = pt + offsetVecRef[m];
                        neighborCellPtr = cellFieldG->get(n);
                        if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                        if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) != end_sitr)
                            continue;//avoid user specified types
                    }
                    conSum += newU[i + offset.x][j + offset.y][k + offset.z];
                    ++neighborCounter;
                }

                // Changes: Replaced U and V by extOldCon
                newU[i][j][k] = A * currentCon + B * conSum;
            }

    for (int i = workFieldDim.x - 1 - offX; i > offX - 1; i--)
        for (int j = workFieldDim.y - 1 - offY; j > offY - 1; j--)
            for (int k = workFieldDim.z - 1 - offZ; k > offZ - 1; k--) {
                pt = Point3D(i + 1, j + 1, k + 1);
                currentCon = conArray[i][j][k];
                conSum = 0.0;
                neighborCounter = 0;

                //loop over nearest neighbors
                CellG *neighborCellPtr = 0;
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                for (int m = 0; m <= maxNeighborIndex; ++m) {
                    const Point3D &offset = offsetVecRef[m];

                    if (diffData.avoidTypeIdSet.size() ||
                        avoidMedium) //means user defined types to avoid in terms of the diffusion
                    {
                        n = pt + offsetVecRef[m];
                        neighborCellPtr = cellFieldG->get(n);
                        if (avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
                        if (neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type) != end_sitr)
                            continue;//avoid user specified types
                    }
                    conSum += newV[i + offset.x][j + offset.y][k + offset.z];
                    ++neighborCounter;
                }

                // Changes: Replaced U and V by extOldCon
                newV[i][j][k] = A * currentCon + B * conSum;
            }
    CC3D_Log(LOG_DEBUG) << "offX = " << offX << "\toffY = " << offY << "\toffZ = " << offZ;

    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);
                //currentCon = conArray[i][j][k];
                currentCellPtr = cellFieldG->get(pt);
                currentCon = conArray[x][y][z];

                if (avoidMedium && !currentCellPtr) //if medium is to be avoided
                {
                    if (avoidDecayInMedium) {
                        scratchArray[x][y][z] = currentCon;
                    } else {
                        scratchArray[x][y][z] = currentCon - deltaT * (decayConst * currentCon);
                    }
                    continue;
                }

                if (currentCellPtr && diffData.avoidTypeIdSet.find(currentCellPtr->type) != end_sitr) {
                    if (diffData.avoidDecayInIdSet.find(currentCellPtr->type) != end_sitr_decay) {
                        scratchArray[x][y][z] = currentCon;
                    } else {
                        scratchArray[x][y][z] = currentCon - deltaT * (decayConst * currentCon);
                    }
                    continue; // avoid user defined types
                }

                updatedCon = 0.5 * (newU[x][y][z] + newV[x][y][z]);

                //processing decay dependent on type of the current cell
                if (currentCellPtr) {
                    if (diffData.avoidDecayInIdSet.find(currentCellPtr->type) !=
                        end_sitr_decay) { ;//decay in this type is forbidden
                    } else {
                        updatedCon -= deltaT * (decayConst * currentCon);//decay in this type is allowed
                    }
                } else {
                    if (avoidDecayInMedium) { ;//decay in Medium is forbidden
                    } else {
                        updatedCon -= deltaT * (decayConst * currentCon); //decay in Medium is allowed
                    }
                }

                if (haveCouplingTerms) {
                    updatedCon += couplingTerm(pt, diffData.couplingDataVec, currentCon);
                }

                //imposing artificial limits on allowed concentration
                if (diffData.useThresholds) {
                    if (updatedCon > diffData.maxConcentration) {
                        updatedCon = diffData.maxConcentration;
                    }
                    if (updatedCon < diffData.minConcentration) {
                        updatedCon = diffData.minConcentration;
                    }
                }
                scratchArray[x][y][z] = updatedCon;//updating scratch
            }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::diffuse() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        diffuseSingleField(i);
        if (!haveCouplingTerms) { //without coupling terms field swap takes place immediately aftera given field has been diffused
            ConcentrationField_t *concentrationField = concentrationFieldVector[i];
            ConcentrationField_t *scratchField = concentrationFieldVector[diffSecrFieldTuppleVec.size()];
            //copy updated values from scratch to concentration fiel
            scrarch2Concentration(scratchField, concentrationField);
        }

    }

    if (haveCouplingTerms) {  //with coupling terms we swap scratch and concentration field after all fields have been diffused
        for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
            ConcentrationField_t *concentrationField = concentrationFieldVector[i];
            ConcentrationField_t *scratchField = concentrationFieldVector[diffSecrFieldTuppleVec.size() + i];
            scrarch2Concentration(scratchField, concentrationField);

        }

    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                       ConcentrationField_t *concentrationField) {
    scratchField->switchContainersQuick(*(concentrationField));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
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
void
FlexibleDiffusionSolverADE::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {

    std::string basePath = simulator->getBasePath();
    std::string fn = fileName;
    if (basePath != "") {
        fn = basePath + "/" + fileName;
    }

    ifstream in(fn.c_str());

    if (in.is_open()) throw CC3DException(string("Could not open chemical concentration file '") + fn + "'!");


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


void FlexibleDiffusionSolverADE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {


    //if(potts->getDisplayUnitsFlag()){
    //	Unit diffConstUnit=powerUnit(potts->getLengthUnit(),2)/potts->getTimeUnit();
    //	Unit decayConstUnit=1/potts->getTimeUnit();
    //    Unit secretionConstUnit=1/potts->getTimeUnit();

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

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");
    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {

        diffSecrFieldTuppleVec.push_back(DiffusionSecretionADEFieldTupple());

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
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleDiffusionSolverADE::secreteSingleField;
                ++j;
            } else if ((*sitr) == "SecretionOnContact") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleDiffusionSolverADE::secreteOnContactSingleField;
                ++j;
            } else if ((*sitr) == "ConstantConcentration") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &FlexibleDiffusionSolverADE::secreteConstantConcentrationSingleField;
                ++j;
            }
        }
    }

}

std::string FlexibleDiffusionSolverADE::toString() {
    return "FlexibleDiffusionSolverADE";
}


std::string FlexibleDiffusionSolverADE::steerableName() {
    return toString();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverADE::finish() {
    CC3D_Log(LOG_DEBUG) << "CALLING FINISH FOR FlexibleDiffusionSolverADE";

}


