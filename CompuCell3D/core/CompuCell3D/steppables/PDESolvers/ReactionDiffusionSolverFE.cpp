#include <cfloat>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
#include <CompuCell3D/plugins/CellTypeMonitor/CellTypeMonitorPlugin.h>
#include "FluctuationCompensator.h"

#include <PublicUtilities/StringUtils.h>
#include <muParser/muParser.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <PublicUtilities/ParallelUtilsOpenMP.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//
//
// }


using namespace CompuCell3D;
using namespace std;


#include "ReactionDiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverSerializer::serialize() {

    for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
        ostringstream outName;

        outName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "_" << currentStep << "."
                << serializedFileExtension;
        ofstream outStream(outName.str().c_str());
        solverPtr->outputField(outStream, solverPtr->concentrationFieldVector[i]);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverSerializer::readFromFile() {
    try {
        for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
            ostringstream inName;
            inName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "." << serializedFileExtension;

            solverPtr->readConcentrationField(inName.str().c_str(), solverPtr->concentrationFieldVector[i]);;
        }

    }
    catch (CC3DException &e) {
       CC3D_Log(LOG_DEBUG) << "COULD NOT FIND ONE OF THE FILES";
        throw CC3DException("Error in reading diffusion fields from file", e);
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE::ReactionDiffusionSolverFE()
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
    scaleSecretion = true;
    maxNumberOfDiffusionCalls = 1;
    bc_indicator_field = 0;
    fluctuationCompensator = 0;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE::~ReactionDiffusionSolverFE() {

    if (bc_indicator_field) {
        delete bc_indicator_field;
        bc_indicator_field = 0;
    }

    if (serializerPtr)
        delete serializerPtr;
    serializerPtr = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant,
                                      std::vector<float> const &maxDecayConstVec) {
    //for RD solver we have to find max diff constant of all all diff constants for all fields
    //based on that we calculate number of calls to reaction diffusion solver system and we call diffusion system same numnr of times to ensure that scratch and concentration field are synchronized between extra diffusion calls

    if (!maxDiffConstVec.size()) { //we will pass empty vector from update function . At the time of calling the update function we have no knowledge of maxDiffConstVec, maxStableDiffConstant
        return;
    }

    //scaling of diffusion and secretion coeeficients
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); i++) {
        scalingExtraMCSVec[i] = max(ceil(maxDiffConstVec[i] / maxStableDiffConstant), ceil(maxDecayConstVec[i] /
                                                                                           maxStableDecayConstant)); //compute number of calls to diffusion solver for individual fuields
    }

    //calculate maximumNumber Of calls to the diffusion solver
    maxNumberOfDiffusionCalls = *max_element(scalingExtraMCSVec.begin(), scalingExtraMCSVec.end());

    if (maxNumberOfDiffusionCalls == 0)
        return;

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); i++) {

        //diffusion data
        for (int currentCellType = 0; currentCellType < UCHAR_MAX + 1; currentCellType++) {
            float diffConstTemp = diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType];
            float decayConstTemp = diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType];
            diffSecrFieldTuppleVec[i].diffData.extraTimesPerMCS = maxNumberOfDiffusionCalls;
            diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] = (diffConstTemp /
                                                                            maxNumberOfDiffusionCalls); //scale diffusion
            diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType] = (decayConstTemp /
                                                                             maxNumberOfDiffusionCalls); //scale decay
        }

        if (scaleSecretion) {
            //secretion data
            SecretionData &secrData = diffSecrFieldTuppleVec[i].secrData;
            for (std::map<unsigned char, float>::iterator mitr = secrData.typeIdSecrConstMap.begin();
                 mitr != secrData.typeIdSecrConstMap.end(); ++mitr) {
                mitr->second /= maxNumberOfDiffusionCalls;
            }

            // Notice we do not scale constant concentration secretion. When users use Constant concentration secretion they want to keep concentration at a given cell at the specified level
            // so no scaling

            for (std::map<unsigned char, SecretionOnContactData>::iterator mitr = secrData.typeIdSecrOnContactDataMap.begin();
                 mitr != secrData.typeIdSecrOnContactDataMap.end(); ++mitr) {
                SecretionOnContactData &secrOnContactData = mitr->second;
                for (std::map<unsigned char, float>::iterator cmitr = secrOnContactData.contactCellMap.begin();
                     cmitr != secrOnContactData.contactCellMap.end(); ++cmitr) {
                    cmitr->second /= maxNumberOfDiffusionCalls;
                }
            }

            //uptake data
            for (std::map<unsigned char, UptakeData>::iterator mitr = secrData.typeIdUptakeDataMap.begin();
                 mitr != secrData.typeIdUptakeDataMap.end(); ++mitr) {
                mitr->second.maxUptake /= maxNumberOfDiffusionCalls;
                mitr->second.relativeUptakeRate /= maxNumberOfDiffusionCalls;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


    simPtr = _simulator;
    simulator = _simulator;
    potts = _simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    ///**

    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    pUtils = simulator->getParallelUtils();
   CC3D_Log(LOG_DEBUG) << "INSIDE INIT";

    ///setting member function pointers
    diffusePtr = &ReactionDiffusionSolverFE::diffuse;
    secretePtr = &ReactionDiffusionSolverFE::secrete;


    //determining max stable diffusion constant has to be done before calling update
    maxStableDiffConstant = 0.23;
    if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) {
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z ==
                                                  1) { //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
            maxStableDiffConstant = 0.16f;
        } else {//3D
            maxStableDiffConstant = 0.08f;
        }
    } else {//Square lattice
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z ==
                                                  1) { //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
            maxStableDiffConstant = 0.23f;
        } else {//3D
            maxStableDiffConstant = 0.14f;
        }
    }
    //setting max stable decay coefficient
    maxStableDecayConstant = 1.0 - FLT_MIN;


    //determining latticeType and setting diffusionLatticeScalingFactor
    //When you evaluate div as a flux through the surface divided bby volume those scaling factors appear automatically. On cartesian lattife everythink is one so this is easy to forget that on different lattices they are not1
    if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) {
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z ==
                                                  1) { //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
            diffusionLatticeScalingFactor = 1.0 / sqrt(3.0);// (2/3)/dL^2 dL=sqrt(2/sqrt(3)) so (2/3)/dL^2=1/sqrt(3)
        } else {//3D simulation
            diffusionLatticeScalingFactor = pow(2.0, -4.0 /
                                                     3.0); //(1/2)/dL^2 dL dL^2=2**(1/3) so (1/2)/dL^2=1/(2.0*2^(1/3))=2^(-4/3)
        }

    }


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

    update(_xmlData, true);

    latticeType = this->getBoundaryStrategy()->getLatticeType();
    //if (latticeType==HEXAGONAL_LATTICE){
    {
        bc_indicator_field = new Array3DCUDA<signed char>(fieldDim,
                                                          BoundaryConditionSpecifier::INTERNAL);// BoundaryConditionSpecifier::INTERNAL= -1
        boundaryConditionIndicatorInit(); // initializing the array which will be used to guide solver when to use BC values in the diffusion algorithm and when to use generic algorithm (i.e. the one for "internal pixels")
    }

    //numberOfFields=diffSecrFieldTuppleVec.size();




    vector <string> concentrationFieldNameVectorTmp; //temporary vector for field names
    ///assign vector of field names
    concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(), string(""));
   CC3D_Log(LOG_DEBUG) << "diffSecrFieldTuppleVec.size()=" << diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
       CC3D_Log(LOG_DEBUG) << " concentrationFieldNameVector[i]=" << concentrationFieldNameVectorTmp[i];
    }



   CC3D_Log(LOG_DEBUG) << "FIELDS THAT I HAVE";
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
       CC3D_Log(LOG_DEBUG) << "Field " << i << " name: " << concentrationFieldNameVectorTmp[i];
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
       CC3D_Log(LOG_DEBUG) << "registring field: " << concentrationFieldNameVector[i] << " field address="
             << concentrationFieldVector[i];
    }

    //we only autoscale diffusion when user requests it explicitely
    if (!autoscaleDiffusion) {
        diffusionLatticeScalingFactor = 1.0;
    }


    bool pluginAlreadyRegisteredFlag;
    cellTypeMonitorPlugin = (CellTypeMonitorPlugin *) Simulator::pluginManager.get("CellTypeMonitor",
                                                                                   &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag) {
        cellTypeMonitorPlugin->init(simulator);
    }

    h_celltype_field = cellTypeMonitorPlugin->getCellTypeArray();
    h_cellid_field = cellTypeMonitorPlugin->getCellIdArray();

    if (_xmlData->findElement("FluctuationCompensator")) {

        fluctuationCompensator = new FluctuationCompensator(simPtr);

        for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i)
            fluctuationCompensator->loadFieldName(concentrationFieldNameVectorTmp[i]);

        fluctuationCompensator->loadFields();

    }

    simulator->registerSteerableObject(this);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::init_cell_type_and_id_arrays() {
    Point3D pt;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
                CellG * cell = cellFieldG->getQuick(pt);
                if (cell) {
                    h_celltype_field->set(pt, cell->type);
                    h_cellid_field->set(pt, cell->id);
                } else {
                    h_celltype_field->set(pt, 0);
                    h_cellid_field->set(pt, 0);
                }
            }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::boundaryConditionIndicatorInit() {

    // bool detailedBCFlag=bcSpecFlagVec[idx];
    // BoundaryConditionSpecifier & bcSpec=bcSpecVec[idx];
    Array3DCUDA<signed char> &bcField = *bc_indicator_field;


    if (fieldDim.z > 2) {// if z dimension is "flat" we do not mark bc
        // Z axis  - external boundary layer
        for (int x = 0; x < fieldDim.x + 2; ++x)
            for (int y = 0; y < fieldDim.y + 2; ++y) {
                bcField.setDirect(x, y, 0, BoundaryConditionSpecifier::MIN_Z);
            }

        // Z axis  - external boundary layer
        for (int x = 0; x < fieldDim.x + 2; ++x)
            for (int y = 0; y < fieldDim.y + 2; ++y) {
                bcField.setDirect(x, y, fieldDim.z + 1, BoundaryConditionSpecifier::MAX_Z);
            }

        // Z axis  - internal boundary layer    
        for (int x = 1; x < fieldDim.x + 1; ++x)
            for (int y = 1; y < fieldDim.y + 1; ++y) {
                bcField.setDirect(x, y, 1, BoundaryConditionSpecifier::BOUNDARY);
            }

        // Z axis  - internal boundary layer    
        for (int x = 1; x < fieldDim.x + 1; ++x)
            for (int y = 1; y < fieldDim.y + 1; ++y) {
                bcField.setDirect(x, y, fieldDim.z, BoundaryConditionSpecifier::BOUNDARY);
            }

    }

    //BC a long x axis will be set second  - meaning all corner pixels will ge set according to x or y axis depending on location

    if (fieldDim.y > 2) {// if y dimension is "flat" we do not mark bc 
        // Y axis - external boundary layer                
        for (int x = 0; x < fieldDim.x + 2; ++x)
            for (int z = 0; z < fieldDim.z + 2; ++z) {
                bcField.setDirect(x, 0, z, BoundaryConditionSpecifier::MIN_Y);

            }
        // Y axis - external boundary layer                
        for (int x = 0; x < fieldDim.x + 2; ++x)
            for (int z = 0; z < fieldDim.z + 2; ++z) {
                bcField.setDirect(x, fieldDim.y + 1, z, BoundaryConditionSpecifier::MAX_Y);
            }

        // Y axis - internal boundary layer                
        for (int x = 1; x < fieldDim.x + 1; ++x)
            for (int z = 1; z < fieldDim.z + 1; ++z) {
                bcField.setDirect(x, 1, z, BoundaryConditionSpecifier::BOUNDARY);

            }

        // Y axis - internal boundary layer                
        for (int x = 1; x < fieldDim.x + 1; ++x)
            for (int z = 1; z < fieldDim.z + 1; ++z) {
                bcField.setDirect(x, fieldDim.y, z, BoundaryConditionSpecifier::BOUNDARY);
            }

    }

    //BC a long x axis will be set last  - meaning all corner pixels will ge set according to x axis
    if (fieldDim.x > 2) { // if x dimension is "flat" we do not mark bc 

        // X axis - external boundary layer                
        for (int y = 0; y < fieldDim.y + 2; ++y)
            for (int z = 0; z < fieldDim.z + 2; ++z) {
                bcField.setDirect(0, y, z, BoundaryConditionSpecifier::MIN_X);
            }
        // X axis - external boundary layer                    
        for (int y = 0; y < fieldDim.y + 2; ++y)
            for (int z = 0; z < fieldDim.z + 2; ++z) {
                bcField.setDirect(fieldDim.x + 1, y, z, BoundaryConditionSpecifier::MAX_X);
            }


        // X axis - internal boundary layer                
        for (int y = 1; y < fieldDim.y + 1; ++y)
            for (int z = 1; z < fieldDim.z + 1; ++z) {
                bcField.setDirect(1, y, z, BoundaryConditionSpecifier::BOUNDARY);
            }

        // X axis - internal boundary layer                    
        for (int y = 1; y < fieldDim.y + 1; ++y)
            for (int z = 1; z < fieldDim.z + 1; ++z) {
                bcField.setDirect(fieldDim.x, y, z, BoundaryConditionSpecifier::BOUNDARY);
            }

    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::extraInit(Simulator *simulator) {

    if ((serializeFlag || readFromFileFlag) && !serializerPtr) {
        serializerPtr = new ReactionDiffusionSolverSerializer();
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

    prepareForwardDerivativeOffsets();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE::prepareForwardDerivativeOffsets() {
    latticeType = this->getBoundaryStrategy()->getLatticeType();


    unsigned int maxHexArraySize = 6;

    hexOffsetArray.assign(maxHexArraySize, vector<Point3D>());


    if (latticeType == HEXAGONAL_LATTICE) {//2D case

        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1) {

            hexOffsetArray[0].push_back(Point3D(0, 1, 0));
            hexOffsetArray[0].push_back(Point3D(1, 1, 0));
            hexOffsetArray[0].push_back(Point3D(1, 0, 0));

            hexOffsetArray[1].push_back(Point3D(0, 1, 0));
            hexOffsetArray[1].push_back(Point3D(-1, 1, 0));
            hexOffsetArray[1].push_back(Point3D(1, 0, 0));

            hexOffsetArray[2] = hexOffsetArray[0];
            hexOffsetArray[4] = hexOffsetArray[0];

            hexOffsetArray[3] = hexOffsetArray[1];
            hexOffsetArray[5] = hexOffsetArray[1];

        } else { //3D case - we assume that forward derivatives are calculated using 3 sides with z=1 and 3 sides which have same offsets os for 2D case (with z=0)
            hexOffsetArray.assign(maxHexArraySize, vector<Point3D>(6));

            //y%2=0 and z%3=0						
            hexOffsetArray[0][0] = Point3D(0, 1, 0);
            hexOffsetArray[0][1] = Point3D(1, 1, 0);
            hexOffsetArray[0][2] = Point3D(1, 0, 0);
            hexOffsetArray[0][3] = Point3D(0, -1, 1);
            hexOffsetArray[0][4] = Point3D(0, 0, 1);
            hexOffsetArray[0][5] = Point3D(1, 0, 1);

            //y%2=1 and z%3=0						
            hexOffsetArray[1][0] = Point3D(-1, 1, 0);
            hexOffsetArray[1][1] = Point3D(0, 1, 0);
            hexOffsetArray[1][2] = Point3D(1, 0, 0);
            hexOffsetArray[1][3] = Point3D(-1, 0, 1);
            hexOffsetArray[1][4] = Point3D(0, 0, 1);
            hexOffsetArray[1][5] = Point3D(0, -1, 1);

            //y%2=0 and z%3=1						
            hexOffsetArray[2][0] = Point3D(-1, 1, 0);
            hexOffsetArray[2][1] = Point3D(0, 1, 0);
            hexOffsetArray[2][2] = Point3D(1, 0, 0);
            hexOffsetArray[2][3] = Point3D(0, 1, 1);
            hexOffsetArray[2][4] = Point3D(0, 0, 1);
            hexOffsetArray[2][5] = Point3D(-1, 1, 1);

            //y%2=1 and z%3=1						
            hexOffsetArray[3][0] = Point3D(0, 1, 0);
            hexOffsetArray[3][1] = Point3D(1, 1, 0);
            hexOffsetArray[3][2] = Point3D(1, 0, 0);
            hexOffsetArray[3][3] = Point3D(0, 1, 1);
            hexOffsetArray[3][4] = Point3D(0, 0, 1);
            hexOffsetArray[3][5] = Point3D(1, 1, 1);

            //y%2=0 and z%3=2						
            hexOffsetArray[4][0] = Point3D(-1, 1, 0);
            hexOffsetArray[4][1] = Point3D(0, 1, 0);
            hexOffsetArray[4][2] = Point3D(1, 0, 0);;
            hexOffsetArray[4][3] = Point3D(-1, 0, 1);
            hexOffsetArray[4][4] = Point3D(0, 0, 1);
            hexOffsetArray[4][5] = Point3D(0, -1, 1);

            //y%2=1 and z%3=2						
            hexOffsetArray[5][0] = Point3D(0, 1, 0);
            hexOffsetArray[5][1] = Point3D(1, 1, 0);
            hexOffsetArray[5][2] = Point3D(1, 0, 0);
            hexOffsetArray[5][3] = Point3D(0, -1, 1);
            hexOffsetArray[5][4] = Point3D(0, 0, 1);
            hexOffsetArray[5][5] = Point3D(1, 0, 1);
        }
    } else {
        Point3D pt(fieldDim.x / 2, fieldDim.y / 2, fieldDim.z / 2); // pick point in the middle of the lattice

        const std::vector <Point3D> &offsetVecRef = this->getBoundaryStrategy()->getOffsetVec(pt);
        for (unsigned int i = 0; i <= this->getMaxNeighborIndex(); ++i) {

            const Point3D &offset = offsetVecRef[i];
            //we use only those offset vectors which have only positive coordinates
            if (offset.x >= 0 && offset.y >= 0 && offset.z >= 0) {
                offsetVecCartesian.push_back(offset);
            }
        }

    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::handleEvent(CC3DEvent &_event) {
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
void ReactionDiffusionSolverFE::start() {


    dt_dx2 = deltaT / (deltaX * deltaX);

    if (simPtr->getRestartEnabled()) {
        return;  // we will not initialize cells if restart flag is on
    }

    if (readFromFileFlag) {
        try {

            serializerPtr->readFromFile();

        }
        catch (CC3DException &e) {
           CC3D_Log(LOG_DEBUG) << "Going to fail-safe initialization";
            initializeConcentration(); //if there was error, initialize using failsafe defaults
        }

    } else {
        initializeConcentration();//Normal reading from User specified files
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::initializeConcentration() {
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
       CC3D_Log(LOG_DEBUG) << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName, concentrationFieldVector[i]);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Dim3D ReactionDiffusionSolverFE::getInternalDim() {
    return getConcentrationField(0)->getInternalDim();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::prepCellTypeField(int idx) {

    // here we set up cellTypeArray boundaries
    Array3DCUDA<unsigned char> &cellTypeArray = *h_celltype_field;
    BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];
    int numberOfIters = 1;

    if (boundaryStrategy->getLatticeType() ==
        HEXAGONAL_LATTICE) { //for hex lattice we need two passes to correctly initialize lattice corners
        numberOfIters = 2;
    }

    Dim3D workFieldDimInternal = getInternalDim();

    for (int iter = 0; iter < numberOfIters; ++iter) {
        if (periodicBoundaryCheckVector[0] || bcSpec.planePositions[0] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[1] == BoundaryConditionSpecifier::PERIODIC) {
            //x - periodic
            int x = 0;
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {

                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(fieldDim.x, y, z));
                }

            x = fieldDim.x + 1;
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {

                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(1, y, z));
                }

        } else {
            int x = 0;
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x + 1, y, z));
                }

            x = fieldDim.x + 1;
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x - 1, y, z));
                }

        }

        if (periodicBoundaryCheckVector[1] || bcSpec.planePositions[2] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[3] == BoundaryConditionSpecifier::PERIODIC) {
            int y = 0;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, fieldDim.y, z));
                }

            y = fieldDim.y + 1;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, 1, z));
                }
        } else {
            int y = 0;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y + 1, z));
                }

            y = fieldDim.y + 1;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y - 1, z));
                }
        }

        if (periodicBoundaryCheckVector[2] || bcSpec.planePositions[4] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[5] == BoundaryConditionSpecifier::PERIODIC) {

            int z = 0;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, fieldDim.z));
                }

            z = fieldDim.z + 1;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, 1));
                }

        } else {

            int z = 0;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, z + 1));
                }

            z = fieldDim.z + 1;
            for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
                for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, z - 1));
                }


        }
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::step(const unsigned int _currentStep) {

    if (fluctuationCompensator) fluctuationCompensator->applyCorrections();

    init_cell_type_and_id_arrays();

    if (scaleSecretion) {
        for (int callIdx = 0; callIdx < maxNumberOfDiffusionCalls; ++callIdx) {

            for (int idx = 0; idx < numberOfFields; ++idx) {
                if (callIdx == 0)
                    prepCellTypeField(idx); // here we initialize celltype array  boundaries - we do it once per  MCS
                boundaryConditionInit(idx);//initializing boundary conditions
                //secretion
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[idx].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[idx].secrData.secretionFcnPtrVec[j])(idx);
                }

            }
        }

        for (int callIdx = 0; callIdx < maxNumberOfDiffusionCalls; ++callIdx) {

            for (int idx = 0; idx < numberOfFields; ++idx) {
                boundaryConditionInit(idx);//initializing boundary conditions
                solveRDEquationsSingleField(idx); //reaction-diffusion
            }

            for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
                ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
                concentrationField.swapArrays();
            }

        }

    } else { //solver behaves as FlexiblereactionDiffusionSolver - i.e. secretion is done at once followed by multiple diffusive steps

        //secretion
        for (int idx = 0; idx < numberOfFields; ++idx) {
            for (unsigned int j = 0; j < diffSecrFieldTuppleVec[idx].secrData.secretionFcnPtrVec.size(); ++j) {
                (this->*diffSecrFieldTuppleVec[idx].secrData.secretionFcnPtrVec[j])(idx);
            }
        }

        for (int callIdx = 0; callIdx < maxNumberOfDiffusionCalls; ++callIdx) {

            //reaction-diffusive steps
            for (int idx = 0; idx < numberOfFields; ++idx) {
                if (callIdx == 0)
                    prepCellTypeField(idx); // here we initialize celltype array  boundaries - we do it once per  MCS
                boundaryConditionInit(idx);//initializing boundary conditions
                solveRDEquationsSingleField(idx); //reaction-diffusion
            }

            for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
                ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
                concentrationField.swapArrays();
            }
        }

    }

    if (fluctuationCompensator) fluctuationCompensator->resetCorrections();

    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE::secreteOnContactSingleField(unsigned int idx) {


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
        // CC3D_Log(LOG_TRACE) << "FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates;
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
                        for (int i = 0; i <= maxNeighborIndex/*offsetVec.size()*/; ++i) {
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
void ReactionDiffusionSolverFE::secreteSingleField(unsigned int idx) {


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
                                    CC3D_Log(LOG_TRACE) << " uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake;
                                } else {
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
void ReactionDiffusionSolverFE::secreteConstantConcentrationSingleField(unsigned int idx) {


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
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    CC3D_Log(LOG_TRACE) << "THIS IS PTR="<<currentCellPtr;

                    //             if(currentCellPtr)
                    // CC3D_Log(LOG_TRACE) << "This is id="<<currentCellPtr->id;
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
void ReactionDiffusionSolverFE::secrete() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

        }


    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::boundaryConditionInit(int idx) {

    ConcentrationField_t &_array = *concentrationFieldVector[idx];
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
void ReactionDiffusionSolverFE::solveRDEquations() {

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

void ReactionDiffusionSolverFE::solveRDEquationsSingleField(unsigned int idx) {

    /// 'n' denotes neighbor

    ///this is the diffusion equation
    ///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
    ///a - diffusivity - diffConst

    ///Finite difference method:
    ///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
    ///N - number of neighbors
    ///will have to double check this formula



    DiffusionData diffData = diffSecrFieldTuppleVec[idx].diffData;
    //float diffConst=diffData.diffConst;

    bool useBoxWatcher = false;

    if (diffData.useBoxWatcher)
        useBoxWatcher = true;

    //for this version we do not skip diffusion step even if diffusion const is 0
    //if(diffSecrFieldTuppleVec[idx].diffData.diffConst==0.0 && diffSecrFieldTuppleVec[idx].diffData.decayConst==0.0 && diffSecrFieldTuppleVec[idx].diffData.additionalTerm=="0.0" ){
    //	return; // do not solve equation if diffusion, decay constants and additional term are 0 
    //}


    float dt_dx2 = deltaT / (deltaX * deltaX);


    float dt = 1.0 / (float) maxNumberOfDiffusionCalls;

    std::set<unsigned char>::iterator sitr;


    Automaton *automaton = potts->getAutomaton();


    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];

    Array3DCUDA<signed char> &bcField = *bc_indicator_field;

    BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];


    //boundaryConditionInit(idx);//initializing boundary conditions this gets called in the step fcn
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

        short currentCellType = 0;
        float varDiffSumTerm = 0.0;
        float *diffCoef = diffData.diffCoef;
        float *decayCoef = diffData.decayCoef;
        float currentDiffCoef = 0.0;
        bool variableDiffusionCoefficientFlag = diffData.getVariableDiffusionCoeeficientFlag();
        bool diffusiveSite;

        Array3DCUDA<unsigned char> &cellTypeArray = *h_celltype_field;

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

            if (latticeType == HEXAGONAL_LATTICE) {
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

                            ev[0] = currentCellType;
                            //setting up x,y,z variables
                            ev[1] = pt.x; //x-variable
                            ev[2] = pt.y; //y-variable
                            ev[3] = pt.z; //z-variable

                            //getting concentrations at x,y,z for all the fields	
                            for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
                                ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
                                ev[4 + fieldIdx] = concentrationField.getDirect(x, y, z);
                            }

                            //loop over nearest neighbors
                            if (bcField.getDirect(x, y, z) == BoundaryConditionSpecifier::INTERNAL) {//internal pixel

                                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);
                                for (register int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/; ++i) {
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
                                    //loop over nearest neighbors
                                    const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                                    for (register int i = 0; i <= maxNeighborIndex; ++i) {
                                        const Point3D &offset = offsetVecRef[i];
                                        varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x + offset.x, y + offset.y,
                                                                                           z + offset.z)] *
                                                          (concentrationField.getDirect(x + offset.x, y + offset.y,
                                                                                        z + offset.z) -
                                                           currentConcentration);
                                    }
                                    varDiffSumTerm /= 2.0;

                                }
                            } else {

                                //loop over nearest neighbors
                                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                                for (register int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/; ++i) {

                                    const Point3D &offset = offsetVecRef[i];
                                    signed char nBcIndicator = bcField.getDirect(x + offset.x, y + offset.y,
                                                                                 z + offset.z);
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


                                if (variableDiffusionCoefficientFlag && diffusiveSite) {
                                    concentrationSum /= 2.0;

                                    const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                                    for (register int i = 0; i <= maxNeighborIndex; ++i) {
                                        const Point3D &offset = offsetVecRef[i];
                                        signed char nBcIndicator = bcField.getDirect(x + offset.x, y + offset.y,
                                                                                     z + offset.z);
                                        float c_offset = concentrationField.getDirect(x + offset.x, y + offset.y,
                                                                                      z + offset.z);
                                        //for pixels belonging to outside boundary we have to use boundary conditions to determine the value of the concentration at this pixel
                                        if (!(nBcIndicator == BoundaryConditionSpecifier::INTERNAL) &&
                                            !(nBcIndicator == BoundaryConditionSpecifier::BOUNDARY)) {
                                            if (bcSpec.planePositions[nBcIndicator] ==
                                                BoundaryConditionSpecifier::PERIODIC) {
                                                //for periodic BC we do nothing we simply use whatever is returned by concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z)
                                            } else if (bcSpec.planePositions[nBcIndicator] ==
                                                       BoundaryConditionSpecifier::CONSTANT_VALUE) {
                                                c_offset = bcSpec.values[nBcIndicator];
                                            }
                                            else if (bcSpec.planePositions[nBcIndicator] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                                                if (nBcIndicator == BoundaryConditionSpecifier::MIN_X || nBcIndicator == BoundaryConditionSpecifier::MIN_Y || nBcIndicator == BoundaryConditionSpecifier::MIN_Z) { // for "left hand side" edges of the lattice the sign of the derivative expression is '-'
                                                        CC3D_Log(LOG_TRACE) << " x,y,z =" <<x+offset.x<<","<<y+offset.y<<","<<z+offset.z<<" c="<<concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z)<<" bcSpec.values[nBcIndicator]="<<bcSpec.values[nBcIndicator]<<" nBcIndicator="<<(int)nBcIndicator;
                                                        CC3D_Log(LOG_TRACE) << " MIN (x,y,z)="<<x<<","<<y<<","<<z<<" c="<<concentrationField.getDirect(x,y,z)<<" nnBcIndicator="<<(int)nBcIndicator<<" bcSpec.values[nBcIndicator]="<<bcSpec.values[nBcIndicator]<<" concentrationField.getDirect(x,y,z)-bcSpec.values[nBcIndicator]*deltaX="<<concentrationField.getDirect(x,y,z)-bcSpec.values[nBcIndicator]*deltaX;
                                                    c_offset = currentConcentration - bcSpec.values[nBcIndicator] * deltaX; // notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                                }
                                                else { // for "left hand side" edges of the lattice the sign of  the derivative expression is '+'
                                                CC3D_Log(LOG_TRACE) << " MAX (x,y,z)="<<x<<","<<y<<","<<z<<" c="<<concentrationField.getDirect(x,y,z)<<" nnBcIndicator="<<(int)nBcIndicator<<" bcSpec.values[nBcIndicator]="<<bcSpec.values[nBcIndicator]<<" concentrationField.getDirect(x,y,z)-bcSpec.values[nBcIndicator]*deltaX="<<concentrationField.getDirect(x,y,z)+bcSpec.values[nBcIndicator]*deltaX;
                                                    c_offset = currentConcentration + bcSpec.values[nBcIndicator] * deltaX;// notice we use values of the field of the central pixel not of the neiighbor. The neighbor is outside of the lattice and for non cartesian lattice with non periodic BC cannot be trusted to hold appropriate value
                                                }
                                            }
                                        }

                                        varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x + offset.x, y + offset.y,
                                                                                           z + offset.z)] *
                                                          (c_offset - currentConcentration);
                                    }

                                    varDiffSumTerm /= 2.0;

                                }


                            }


                            updatedConcentration = (concentrationSum + varDiffSumTerm) +
                                                   (1 - decayCoef[currentCellType]) * currentConcentration;

                            //reaction terms are scaled here all other constants e.g. diffusion or secretion constants are scaled in the Scale function
                            updatedConcentration += dt * ev.eval();
                            concentrationField.setDirectSwap(x, y, z, updatedConcentration);//updating scratch


                        }
            } else {

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

                            ev[0] = currentCellType;
                            //setting up x,y,z variables
                            ev[1] = pt.x; //x-variable
                            ev[2] = pt.y; //y-variable
                            ev[3] = pt.z; //z-variable

                            //getting concentrations at x,y,z for all the fields	
                            for (int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) {
                                ConcentrationField_t &concentrationField = *concentrationFieldVector[fieldIdx];
                                ev[4 + fieldIdx] = concentrationField.getDirect(x, y, z);
                            }

                            //loop over nearest neighbors
                            if (diffusiveSite) {
                                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);
                                for (register int i = 0; i <= maxNeighborIndex /*offsetVec.size()*/; ++i) {
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
                            }

                            //using forward first derivatives - cartesian lattice 3D
                            if (variableDiffusionCoefficientFlag && diffusiveSite) {
                                concentrationSum /= 2.0;
                                //loop over nearest neighbors
                                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);

                                for (register int i = 0; i <= maxNeighborIndex; ++i) {
                                    const Point3D &offset = offsetVecRef[i];
                                    varDiffSumTerm += diffCoef[cellTypeArray.getDirect(x + offset.x, y + offset.y,
                                                                                       z + offset.z)] *
                                                      (concentrationField.getDirect(x + offset.x, y + offset.y,
                                                                                    z + offset.z) -
                                                       currentConcentration);
                                }
                                varDiffSumTerm /= 2.0;

                            }

                            updatedConcentration = (concentrationSum + varDiffSumTerm) +
                                                   (1 - decayCoef[currentCellType]) * currentConcentration;

                            //reaction terms are scaled here all other constants e.g. diffusion or secretion constants are scaled in the Scale function
                            updatedConcentration += dt * ev.eval();
                            concentrationField.setDirectSwap(x, y, z,
                                                             updatedConcentration);//updating scratch

                        }


            }


        }
        catch (mu::Parser::exception_type &e) {
           CC3D_Log(LOG_DEBUG) << e.GetMsg();
            throw CC3DException(e.GetMsg());
        }
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ReactionDiffusionSolverFE::isBoudaryRegion(int x, int y, int z, Dim3D dim) {
    if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3)
        return true;
    else
        return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::diffuse() {

    for (int idx = 0; idx < extraTimesPerMCS + 1; ++idx) {
        solveRDEquations();
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                      ConcentrationField_t *concentrationField) {
    //scratchField->switchContainersQuick(*(concentrationField));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
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
void ReactionDiffusionSolverFE::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {

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

void ReactionDiffusionSolverFE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

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
        diffSecrFieldTuppleVec.push_back(DiffusionSecretionRDFieldTupple());
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
            BoundaryConditionSpecifier & bcSpec = bcSpecVec[bcSpecVec.size() - 1];

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

        } else { //translating default  BCs defined in Potts into BoundaryConditionSpecifier structure

            bcSpecFlagVec[bcSpecFlagVec.size() - 1] = true;
            BoundaryConditionSpecifier & bcSpec = bcSpecVec[bcSpecVec.size() - 1];

            if (periodicBoundaryCheckVector[0]) {//periodic boundary conditions were set in x direction	
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_X] = BoundaryConditionSpecifier::PERIODIC;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_X] = BoundaryConditionSpecifier::PERIODIC;
            } else {//no-flux boundary conditions were set in x direction
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_X] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_X] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.values[BoundaryConditionSpecifier::MIN_X] = 0.0;
                bcSpec.values[BoundaryConditionSpecifier::MAX_X] = 0.0;
            }

            if (periodicBoundaryCheckVector[1]) {//periodic boundary conditions were set in x direction	
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Y] = BoundaryConditionSpecifier::PERIODIC;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Y] = BoundaryConditionSpecifier::PERIODIC;
            } else {//no-flux boundary conditions were set in x direction
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Y] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Y] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.values[BoundaryConditionSpecifier::MIN_Y] = 0.0;
                bcSpec.values[BoundaryConditionSpecifier::MAX_Y] = 0.0;
            }

            if (periodicBoundaryCheckVector[2]) {//periodic boundary conditions were set in x direction	
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Z] = BoundaryConditionSpecifier::PERIODIC;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Z] = BoundaryConditionSpecifier::PERIODIC;
            } else {//no-flux boundary conditions were set in x direction
                bcSpec.planePositions[BoundaryConditionSpecifier::MIN_Z] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.planePositions[BoundaryConditionSpecifier::MAX_Z] = BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
                bcSpec.values[BoundaryConditionSpecifier::MIN_Z] = 0.0;
                bcSpec.values[BoundaryConditionSpecifier::MAX_Z] = 0.0;
            }


        }


    }

    if (_xmlData->findElement("Serialize")) {

        serializeFlag = true;
        if (_xmlData->getFirstElement("Serialize")->findAttribute("Frequency")) {
            serializeFrequency = _xmlData->getFirstElement("Serialize")->getAttributeAsUInt("Frequency");
        }
        CC3D_Log(LOG_DEBUG) << "serialize Flag=" << serializeFlag;

    }

    if (_xmlData->findElement("ReadFromFile")) {
        readFromFileFlag = true;
        CC3D_Log(LOG_DEBUG) << "readFromFileFlag=" << readFromFileFlag;
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
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &ReactionDiffusionSolverFE::secreteSingleField;
                ++j;
            } else if ((*sitr) == "SecretionOnContact") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &ReactionDiffusionSolverFE::secreteOnContactSingleField;
                ++j;
            } else if ((*sitr) == "ConstantConcentration") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &ReactionDiffusionSolverFE::secreteConstantConcentrationSingleField;
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


    }
    catch (mu::Parser::exception_type &e) {
       CC3D_Log(LOG_DEBUG) << e.GetMsg();
        throw CC3DException(e.GetMsg());
    }

    scalingExtraMCSVec.assign(diffSecrFieldTuppleVec.size(), 0);
    maxDiffConstVec.assign(diffSecrFieldTuppleVec.size(), 0.0);
    maxDecayConstVec.assign(diffSecrFieldTuppleVec.size(), 0.0);

    //finding maximum diffusion coefficients for each field
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        for (int currentCellType = 0; currentCellType < UCHAR_MAX + 1; currentCellType++) {
            maxDiffConstVec[i] = (maxDiffConstVec[i] < diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType])
                                 ? diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] : maxDiffConstVec[i];
            maxDecayConstVec[i] = (maxDecayConstVec[i] < diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType])
                                  ? diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType] : maxDecayConstVec[i];
        }
    }


    Scale(maxDiffConstVec, maxStableDiffConstant, maxDecayConstVec);//TODO: remove for implicit solvers?

}

std::string ReactionDiffusionSolverFE::toString() {
    return "ReactionDiffusionSolverFE";
}


std::string ReactionDiffusionSolverFE::steerableName() {
    return toString();
}


