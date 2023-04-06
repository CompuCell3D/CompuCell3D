

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>


#include <PublicUtilities/StringUtils.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>


using namespace CompuCell3D;
using namespace std;

#include "AdvectionDiffusionSolverFE.h"
#include <Logger/CC3DLogger.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
AdvectionDiffusionSolverFE::AdvectionDiffusionSolverFE()
        : DiffusableGraph<float>() {
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
AdvectionDiffusionSolverFE::~AdvectionDiffusionSolverFE() {

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    simPtr = simulator;
    potts = simulator->getPotts();
    automaton = potts->getAutomaton();


    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    update(_xmlData, true);

    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("NeighborTracker",
                                                  &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    NeighborTrackerPlugin *neighborTrackerPlugin = (NeighborTrackerPlugin *) plugin;

    neighborTrackerAccessorPtr = neighborTrackerPlugin->getNeighborTrackerAccessorPtr();

    //NeighborTrackerPlugin * neighborTrackerPlugin=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker"));
    //neighborTrackerAccessorPtr= neighborTrackerPlugin->getNeighborTrackerAccessorPtr();

    ///setting member function pointers
    diffusePtr = &AdvectionDiffusionSolverFE::diffuse;
    secretePtr = &AdvectionDiffusionSolverFE::secrete;


    numberOfFields = diffSecrFieldTuppleVec.size();

    ///allocate fields including scrartch field
    workFieldDim = Dim3D(fieldDim.x + 2, fieldDim.y + 2, fieldDim.z + 2);

    allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size() + 1, workFieldDim); //+1 is for additional scratch field


    ///assign vector of field names
    concentrationFieldNameVector.assign(diffSecrFieldTuppleVec.size(), string(""));


    for (unsigned int i = 0; i < diffDataVec.size(); ++i) {
        concentrationFieldNameVector[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;

    }

    //register fields once they have been allocated
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        simPtr->registerConcentrationField(concentrationFieldNameVector[i], concentrationFieldVector[i]);
        CC3D_Log(LOG_DEBUG) << "registring field: "<<concentrationFieldNameVector[i]<<" field address="<<concentrationFieldVector[i];
    }


    ///assigning member method ptrs to the vector

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.assign(
                diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.size(), 0);
        unsigned int j = 0;
        for (set<string>::iterator sitr = diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.begin();
             sitr != diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.end(); ++sitr) {
            if ((*sitr) == "Secretion") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &AdvectionDiffusionSolverFE::secreteSingleField;
                ++j;
            }else if((*sitr)=="SecretionOnContact"){
             diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&AdvectionDiffusionSolverFE::secreteOnContactSingleField;
            ++j;
         }
      }
   }

   CC3D_Log(LOG_DEBUG) << "ALLOCATED ALL FIELDS";


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::extraInit(Simulator *simulator) {


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double AdvectionDiffusionSolverFE::computeAverageCellRadius() {
    double sumRadius = 0.0;
    double radical = 1.0 / 3.0;
    int numberOfCells = 0;
    CellG *cell;
    CellInventory::cellInventoryIterator cInvItr;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        sumRadius += pow(cell->volume, radical);
        ++numberOfCells;

    }

    return 1.33 * sumRadius / numberOfCells;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::updateCellInventories() {

    for (unsigned int i = 0; i <= diffDataVec.size(); ++i) {
        updateLocalCellInventory(i);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::updateLocalCellInventory(unsigned int idx) {

    CellG *cell;
    CellInventory::cellInventoryIterator cInvItr;

    map < CellG * , float > *concentrationField = concentrationFieldMapVector[idx];
    map<CellG *, float>::iterator concentrationItr;
    map<CellG *, float>::iterator concentrationEndItr = concentrationField->end();

    //first eliminate from concentration field all the cells that do not appear any longer in cell inventory
    for (concentrationItr = concentrationField->begin();
         concentrationItr != concentrationField->end(); /*++ concentrationItr*/ ) {
        cInvItr = cellInventoryPtr->find(concentrationItr->first);
        if (cInvItr == cellInventoryPtr->cellInventoryEnd()) {
            concentrationField->erase(concentrationItr++);//erasing cell from concentrationField
        } else {
            concentrationItr++;
        }
    }

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        concentrationItr = concentrationField->find(cell);
        if (concentrationItr == concentrationEndItr) {//no such cell addfress int he map
            concentrationField->insert(
                    make_pair(cell, 0.0));//inserting new cell with default concentration in the field 0
        }


    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::start() {

    updateCellInventories();
    CC3D_Log(LOG_DEBUG) << "GOT HERE BEFORE INITIALIZE FIELD";
   initializeConcentration();
   CC3D_Log(LOG_DEBUG) << "GOT HERE AFTER INITIALIZE FIELD";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void AdvectionDiffusionSolverFE::initializeConcentration() {


    for (unsigned int i = 0; i < diffDataVec.size(); ++i) {
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;

        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName, concentrationFieldVector[i]);
        field2CellMap(concentrationFieldVector[i], concentrationFieldMapVector[i]);
        cellMap2Field(concentrationFieldMapVector[i], concentrationFieldVector[i]);
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
AdvectionDiffusionSolverFE::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {

    std::string basePath = simulator->getBasePath();
    std::string fn = fileName;
    if (basePath != "") {
        fn = basePath + "/" + fileName;
    }

    ifstream in(fn.c_str());

    if (!in.is_open()) throw CC3DException(string("Could not open chemical concentration file '") + fn + "'!");


    Point3D pt;
    float c;
    while (!in.eof()) {
        in >> pt.x >> pt.y >> pt.z >> c;
        if (!in.fail())
            concentrationField->set(pt, c);
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::cellMap2Field(std::map<CellG *, float> *concentrationMapField,
                                               ConcentrationField_t *concentrationField) {

    CellG *currentCellPtr = 0;
    Point3D pt;
    float currentConcentration;
    std::map<CellG *, float>::iterator mitr;

    Array3D_t &concentrationArray = concentrationField->getContainer();

    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);
                currentCellPtr = cellFieldG->get(pt);

                mitr = concentrationMapField->find(currentCellPtr);
                if (mitr != concentrationMapField->end()) {
                    currentConcentration = mitr->second;

                } else {
                    currentConcentration = 0.0;
                }
                concentrationArray[x][y][z] = currentConcentration;
            }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::field2CellMap(ConcentrationField_t *concentrationField,
                                               std::map<CellG *, float> *concentrationMapField) {

    CellG *currentCellPtr = 0;
    Point3D pt;
    float currentConcentration;
    Array3D_t &concentrationArray = concentrationField->getContainer();

    std::map<CellG *, float>::iterator mitr;

    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);

                currentCellPtr = cellFieldG->get(pt);

                mitr = concentrationMapField->find(currentCellPtr);
                currentConcentration = concentrationArray[x][y][z];


                if (mitr != concentrationMapField->end()) {

                    mitr->second = currentConcentration;
                } else {

                }

            }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::step(const unsigned int _currentStep) {
    currentStep = _currentStep;
    updateCellInventories(); //updates cell inventories of particular fields
    cellMap2Field(concentrationFieldMapVector[0], concentrationFieldVector[0]);
    (this->*secretePtr)();
    (this->*diffusePtr)();


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::secreteSingleField(unsigned int idx) {

    SecretionDataFlexAD &secrData = diffSecrFieldTuppleVec[idx].secrData;
    map < CellG * , float > *concentrationField = concentrationFieldMapVector[idx];
    map<CellG *, float>::iterator concentrationItr;
    map<CellG *, float>::iterator concentrationEndItr = concentrationField->end();

    std::map<unsigned char, float>::iterator mitr;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();

    CC3D_Log(LOG_DEBUG) << "secrData.typeIdSecrConstMap.size()=" << secrData.typeIdSecrConstMap.size();
    float currentConcentration;
    float secrConst;

    CC3D_Log(LOG_DEBUG) <<  "secretion single field";

    //the assumption is that medium has type ID 0
    mitr = secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));


   CellG *cell;
   CellInventory::cellInventoryIterator cInvItr;
   
   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
      cell=cellInventoryPtr->getCell(cInvItr);   
      //cell=*cInvItr;
     CC3D_Log(LOG_DEBUG) << "cell=" << cell->id << " type=" <<(int) cell->type;
        concentrationItr = concentrationField->find(cell);

         
         mitr=secrData.typeIdSecrConstMap.find(cell->type);
         if(mitr!=end_mitr){
            
            secrConst=mitr->second;
         CC3D_Log(LOG_DEBUG) << "secrConst=" << secrConst;
            currentConcentration = concentrationItr->second;
            concentrationItr->second = currentConcentration + secrConst;

        }


    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void AdvectionDiffusionSolverFE::secreteOnContactSingleField(unsigned int idx) {

    SecretionDataFlexAD &secrData = diffSecrFieldTuppleVec[idx].secrData;
    map < CellG * , float > *concentrationField = concentrationFieldMapVector[idx];
    map<CellG *, float>::iterator concentrationItr;
    map<CellG *, float>::iterator concentrationEndItr = concentrationField->end();

    std::map<unsigned char, SecretionOnContactData>::iterator mitr;
    std::map<unsigned char, SecretionOnContactData>::iterator end_mitr = secrData.typeIdSecrOnContactDataMap.end();
    map<unsigned char, float> *contactCellMapPtr;
    std::map<unsigned char, float>::iterator mitrTypeConst;

    set<NeighborSurfaceData>::iterator neighborItr;
    NeighborTracker *neighborTracker;


    float currentConcentration;
    float secrConst;


    CellG *cell;
    CellG *nCellPtr;

    CellInventory::cellInventoryIterator cInvItr;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;

        concentrationItr = concentrationField->find(cell);


        mitr = secrData.typeIdSecrOnContactDataMap.find(cell->type);


        if (mitr != end_mitr) {
            contactCellMapPtr = &(mitr->second.contactCellMap);
            neighborTracker = neighborTrackerAccessorPtr->get(cell->extraAttribPtr);
            for (
                    neighborItr = neighborTracker->cellNeighbors.begin();
                    neighborItr != neighborTracker->cellNeighbors.end();
                    ++neighborItr
                    ) {
                nCellPtr = neighborItr->neighborAddress;
                if (!nCellPtr) continue;

                mitrTypeConst = contactCellMapPtr->find(nCellPtr->type);
                if (mitrTypeConst != contactCellMapPtr->end()) {//OK to secrete, contact detected
                    secrConst = mitrTypeConst->second;
                    currentConcentration = concentrationItr->second;
                    concentrationItr->second = currentConcentration + secrConst;
                }


            }

        }


    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::diffuseSingleField(unsigned int idx) {

    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
    map < CellG * , float > *concentrationField = concentrationFieldMapVector[idx];
    map<CellG *, float>::iterator concentrationItr;
    map<CellG *, float>::iterator nConcentrationItr;
    map<CellG *, float>::iterator scratchConcentrationItr;
    map<CellG *, float>::iterator concentrationEndItr = concentrationField->end();
    map < CellG * , float > *scratchField = concentrationFieldMapVector[diffDataVec.size()];

    set<unsigned char>::iterator sitr;
    set<unsigned char>::iterator end_sitr = diffData.avoidTypeIdSet.end();

    NeighborTracker *neighborTracker;
    set<NeighborSurfaceData>::iterator neighborItr;


    CellG *currentCellPtr;
    CellG *nCellPtr;

    short currentCellType = 0;
    float concentrationSum = 0.0;
    float updatedConcentration = 0.0;

    float diffConst = diffData.diffConst;
    float decayConst = diffData.decayConst;
    float deltaT = diffData.deltaT;
    float dt_dx2 = deltaT / (averageRadius * averageRadius);

    float currentConcentration = 0.0;
    short neighborCounter = 0;

    CellInventory::cellInventoryIterator cInvItr;

    for (concentrationItr = concentrationField->begin();
         concentrationItr != concentrationField->end(); ++concentrationItr) {

        currentCellPtr = concentrationItr->first;

        currentConcentration = concentrationItr->second;

        scratchConcentrationItr = scratchField->find(currentCellPtr);

        if (diffData.avoidTypeIdSet.find(currentCellPtr->type) != end_sitr) {
            scratchConcentrationItr->second = currentConcentration;
            continue;
        }


        //find concentrationSum
        concentrationSum = 0.0;
        neighborCounter = 0;


        neighborTracker = neighborTrackerAccessorPtr->get(currentCellPtr->extraAttribPtr);

        for (neighborItr = neighborTracker->cellNeighbors.begin();
             neighborItr != neighborTracker->cellNeighbors.end(); ++neighborItr) {

            nCellPtr = neighborItr->neighborAddress;
            //check if it is still in neighbor boundary

            if (nCellPtr && diffData.avoidTypeIdSet.find(nCellPtr->type) != end_sitr) {
                continue;
            }

         nConcentrationItr=concentrationField->find(nCellPtr);
         if(nConcentrationItr != concentrationEndItr){
            concentrationSum += nConcentrationItr->second;
            ++neighborCounter;
         }
      }
      updatedConcentration =  dt_dx2*diffConst*(concentrationSum - neighborCounter*currentConcentration)
                           -deltaT*(decayConst*currentConcentration)
                           +currentConcentration;
      CC3D_Log(LOG_DEBUG) <<  "updatedConcentration=" << updatedConcentration;

        scratchConcentrationItr->second = updatedConcentration;

    }

    scrarch2Concentration(scratchField, concentrationField);
    cellMap2Field(concentrationField, concentrationFieldVector[idx]);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::scrarch2Concentration(map<CellG *, float> *scratchField,
                                                       map<CellG *, float> *concentrationField) {

    map<CellG *, float>::iterator concentrationItr;
    map<CellG *, float>::iterator scratchConcentrationItr;

    for (concentrationItr = concentrationField->begin();
         concentrationItr != concentrationField->end(); ++concentrationItr) {
        scratchConcentrationItr = scratchField->find(concentrationItr->first);
        concentrationItr->second = scratchConcentrationItr->second;
        scratchConcentrationItr->second = 0.0;
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdvectionDiffusionSolverFE::diffuse() {
    averageRadius = computeAverageCellRadius();
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        diffuseSingleField(i);
    }


}


void AdvectionDiffusionSolverFE::secrete() {


    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
        }

    }


}


void AdvectionDiffusionSolverFE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //notice, limited steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
    // Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running
    diffSecrFieldTuppleVec.clear();

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");
    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {
        diffSecrFieldTuppleVec.push_back(DiffusionSecretionADFieldTupple());
        DiffusionData &diffData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].diffData;
        SecretionData &secrData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].secrData;

        if (diffFieldXMLVec[i]->findAttribute("Name")) {
            diffData.fieldName = diffFieldXMLVec[i]->getAttribute("Name");
        }

        if (diffFieldXMLVec[i]->findElement("DiffusionData"))
            diffData.update(diffFieldXMLVec[i]->getFirstElement("DiffusionData"));

        if (diffFieldXMLVec[i]->findElement("SecretionData"))
            secrData.update(diffFieldXMLVec[i]->getFirstElement("SecretionData"));


    }


}


std::string AdvectionDiffusionSolverFE::toString() {
    return "AdvectionDiffusionSolverFE";
}


std::string AdvectionDiffusionSolverFE::steerableName() {
    return toString();
}



