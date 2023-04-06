#include "DiffusionSolverFE.h"
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
#include <PublicUtilities/Vector3.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>

#include "DiffusionSolverFE_CPU.h"
#include "DiffusionSolverFE_CPU_Implicit.h"
#include "GPUEnabled.h"

#include "MyTime.h"
#include <cfloat>

#if OPENCL_ENABLED == 1
#include "OpenCL/DiffusionSolverFE_OpenCL.h"
//#include "OpenCL/DiffusionSolverFE_OpenCL_Implicit.h"
#include "OpenCL/ReactionDiffusionSolverFE_OpenCL_Implicit.h"
#endif

#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;
using namespace std;


#include "DiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverSerializer<Cruncher>::serialize() {

    for (size_t i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
        ostringstream outName;

        outName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "_" << currentStep << "."
                << serializedFileExtension;
        ofstream outStream(outName.str().c_str());
        solverPtr->outputField(outStream,
                               static_cast<Cruncher *>(solverPtr)->getConcentrationField(i));

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverSerializer<Cruncher>::readFromFile() {
    try {
        for (size_t i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
            ostringstream inName;
            inName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "." << serializedFileExtension;

            solverPtr->readConcentrationField(inName.str().c_str(),
                                              static_cast<Cruncher *>(solverPtr)->getConcentrationField(i));
        }
    } catch (CC3DException &e) {
        CC3D_Log(LOG_DEBUG) << "COULD NOT FIND ONE OF THE FILES";
        throw CC3DException("Error in reading diffusion fields from file", e);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
DiffusionSolverFE<Cruncher>::DiffusionSolverFE()
        :deltaX(1.0), deltaT(1.0), latticeType(SQUARE_LATTICE) {
    potts = 0;
    serializerPtr = 0;
    pUtils = 0;
    serializeFlag = false;
    readFromFileFlag = false;
    haveCouplingTerms = false;
    serializeFrequency = 0;
    boxWatcherSteppable = 0;
    diffusionLatticeScalingFactor = 1.0;
    autoscaleDiffusion = false;
    scaleSecretion = true;
    cellTypeMonitorPlugin = 0;
    maxStableDiffConstant = 0.23;
    automaton = 0;
    cellFieldG = 0;
    cellInventoryPtr = 0;
    currentStep = -1;
    decayConst = 0.0;
    diffConst = 0.0;
    diffusePtr = 0;
    dt_dx2 = 0.0;
    h_celltype_field = 0;
    simPtr = 0;
    secretePtr = 0;
    m_RDTime = 0.0;
    maxDiffusionZ = -1;
    numberOfFields = 0;
    scalingExtraMCS = -1;
    bc_indicator_field = 0;
    fluctuationCompensator = 0;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
DiffusionSolverFE<Cruncher>::~DiffusionSolverFE() {
    if (bc_indicator_field) {
        delete bc_indicator_field;
        bc_indicator_field = 0;
    }

    if (serializerPtr) {
        delete serializerPtr;
        serializerPtr = 0;
    }

    if (fluctuationCompensator) {
        delete fluctuationCompensator;
        fluctuationCompensator = 0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant,
                                        std::vector<float> const &maxDecayConstVec) {
    if (!maxDiffConstVec.size()) { //we will pass empty vector from update function . At the time of calling the update function we have no knowledge of maxDiffConstVec, maxStableDiffConstant
        return;
    }

    //scaling of diffusion and secretion coeeficients
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); i++) {
        scalingExtraMCSVec[i] = max(ceil(maxDiffConstVec[i] / maxStableDiffConstant), ceil(maxDecayConstVec[i] /
                                                                                           maxStableDecayConstant)); //compute number of calls to diffusion solver
        if (scalingExtraMCSVec[i] == 0)
            continue;

        //diffusion data
        for (int currentCellType = 0; currentCellType < UCHAR_MAX + 1; currentCellType++) {
            float diffConstTemp = diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType];
            float decayConstTemp = diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType];
            diffSecrFieldTuppleVec[i].diffData.extraTimesPerMCS = scalingExtraMCSVec[i];
            diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] = (diffConstTemp /
                                                                            scalingExtraMCSVec[i]); //scale diffusion
            diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType] = (decayConstTemp /
                                                                             scalingExtraMCSVec[i]); //scale decay

        }

        if (scaleSecretion) {
            //secretion data
            SecretionData &secrData = diffSecrFieldTuppleVec[i].secrData;
            for (std::map<unsigned char, float>::iterator mitr = secrData.typeIdSecrConstMap.begin();
                 mitr != secrData.typeIdSecrConstMap.end(); ++mitr) {
                mitr->second /= scalingExtraMCSVec[i];
            }

            // Notice we do not scale constant concentration secretion. When users use Constant concentration secretion they want to keep concentration at a given cell at the specified level
            // so no scaling


            for (std::map<unsigned char, SecretionOnContactData>::iterator mitr = secrData.typeIdSecrOnContactDataMap.begin();
                 mitr != secrData.typeIdSecrOnContactDataMap.end(); ++mitr) {
                SecretionOnContactData &secrOnContactData = mitr->second;
                for (std::map<unsigned char, float>::iterator cmitr = secrOnContactData.contactCellMap.begin();
                     cmitr != secrOnContactData.contactCellMap.end(); ++cmitr) {
                    cmitr->second /= scalingExtraMCSVec[i];
                }
            }

            //uptake data
            for (std::map<unsigned char, UptakeData>::iterator mitr = secrData.typeIdUptakeDataMap.begin();
                 mitr != secrData.typeIdUptakeDataMap.end(); ++mitr) {
                mitr->second.maxUptake /= scalingExtraMCSVec[i];
                mitr->second.relativeUptakeRate /= scalingExtraMCSVec[i];
            }

        }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
bool DiffusionSolverFE<Cruncher>::hasAdditionalTerms() const {

    for (size_t i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        DiffusionData const &diffData = diffSecrFieldTuppleVec[i].diffData;
        if (!diffData.additionalTerm.empty())
            return true;
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

    simPtr = _simulator;
    simulator = _simulator;
    potts = _simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    ///**
    //   cellFieldG=potts->getCellFieldG();
    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    pUtils = simulator->getParallelUtils();

    // pUtils=simulator->getParallelUtilsSingleThread();

    ///setting member function pointers
    diffusePtr = &DiffusionSolverFE::diffuse;
    secretePtr = &DiffusionSolverFE::secrete;

    //determinign max stable diffusion constant has to be done before calling update
    maxStableDiffConstant = 0.23;
    if (static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType() == HEXAGONAL_LATTICE) {
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
    diffusionLatticeScalingFactor = 1.0;
    if (static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType() == HEXAGONAL_LATTICE) {
        if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z ==
                                                  1) { //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
            diffusionLatticeScalingFactor = 1.0f / sqrt(3.0f);// (2/3)/dL^2 dL=sqrt(2/sqrt(3)) so (2/3)/dL^2=1/sqrt(3)
        } else {//3D simulation
            diffusionLatticeScalingFactor = pow(2.0f, -4.0f /
                                                      3.0f); //(1/2)/dL^2 dL dL^2=2**(1/3) so (1/2)/dL^2=1/(2.0*2^(1/3))=2^(-4/3)
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

    latticeType = static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType();
    // if (latticeType==HEXAGONAL_LATTICE){
    {
        bc_indicator_field = new Array3DCUDA<signed char>(fieldDim,
                                                          BoundaryConditionSpecifier::INTERNAL);// BoundaryConditionSpecifier::INTERNAL= -1
        boundaryConditionIndicatorInit(); // initializing the array which will be used to guide solver when to use BC values in the diffusion algorithm and when to use generic algorithm (i.e. the one for "internal pixels")
    }


    numberOfFields = diffSecrFieldTuppleVec.size();

	vector<string> concentrationFieldNameVectorTmp; //temporary vector for field names
	///assign vector of field names
	concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(),string(""));
	CC3D_Log(LOG_DEBUG) << "diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
		concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
		CC3D_Log(LOG_DEBUG) << " concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i];
    }

    //setting up couplingData - field-field interaction terms
    vector<CouplingData>::iterator pos;

    float maxDiffConst = 0.0;
    scalingExtraMCS = 0;


    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        pos = diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
        for (size_t j = 0; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size(); ++j) {

            for (size_t idx = 0; idx < concentrationFieldNameVectorTmp.size(); ++idx) {
                if (concentrationFieldNameVectorTmp[idx] ==
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].intrFieldName) {
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].fieldIdx = idx;
                    haveCouplingTerms = true; //if this is called at list once we have already coupling terms and need to proceed differently with scratch field initialization
                    break;
                }
                //this means that required interacting field name has not been found
                if (idx == concentrationFieldNameVectorTmp.size() - 1) {
                    //remove this interacting term
                    diffSecrFieldTuppleVec[i].diffData.couplingDataVec.erase(pos);
                }
            }
            ++pos;
        }

        for (int currentCellType = 0; currentCellType < UCHAR_MAX + 1; currentCellType++) {
            maxDiffConstVec[i] = (maxDiffConstVec[i] < diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType])
                                 ? diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] : maxDiffConstVec[i];
        }
    }

	CC3D_Log(LOG_DEBUG) << "FIELDS THAT I HAVE";
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i];
	}
	CC3D_Log(LOG_DEBUG) << "DiffusionSolverFE: extra Init in read XML";

    ///allocate fields including scrartch field
    static_cast<Cruncher *>(this)->allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(), fieldDim);
    workFieldDim = static_cast<Cruncher *>(this)->getConcentrationField(0)->getInternalDim();

    for (unsigned int i = 0; i < concentrationFieldNameVectorTmp.size(); ++i) {
        static_cast<Cruncher *>(this)->setConcentrationFieldName(i, concentrationFieldNameVectorTmp[i]);
    }

    //register fields once they have been allocated
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        simPtr->registerConcentrationField(
                static_cast<Cruncher *>(this)->getConcentrationFieldName(i),
                static_cast<Cruncher *>(this)->getConcentrationField(i));
        CC3D_Log(LOG_DEBUG) << "registring field: "<<
			static_cast<Cruncher*>(this)->getConcentrationFieldName(i)<<" field address="<<
			static_cast<Cruncher*>(this)->getConcentrationField(i);
    }

    float extraCheck;

    // //check diffusion constant and scale extraTimesPerMCS

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

    //platform-specific initialization
    initImpl();
}

template<class Cruncher>
void DiffusionSolverFE<Cruncher>::init_cell_type_and_id_arrays() {
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
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::boundaryConditionIndicatorInit() {

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
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::extraInit(Simulator *simulator) {

    if ((serializeFlag || readFromFileFlag) && !serializerPtr) {
        serializerPtr = new DiffusionSolverSerializer<Cruncher>();
        serializerPtr->solverPtr = this;
    }

    if (serializeFlag) {
        simulator->registerSerializer(serializerPtr);
    }

    bool useBoxWatcher = false;
    for (size_t i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (diffSecrFieldTuppleVec[i].diffData.useBoxWatcher) {
            useBoxWatcher = true;
            break;
        }
    }
    bool steppableAlreadyRegisteredFlag;
    if (useBoxWatcher) {
        boxWatcherSteppable = (BoxWatcher *) Simulator::steppableManager.get("BoxWatcher",
                                                                             &steppableAlreadyRegisteredFlag);
        if (!steppableAlreadyRegisteredFlag)
            boxWatcherSteppable->init(simulator);
    }

    prepareForwardDerivativeOffsets();

    //platform-specific initialization
    extraInitImpl();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::handleEvent(CC3DEvent &_event) {

    if (_event.id != LATTICE_RESIZE) {
        return;
    }


    static_cast<Cruncher *>(this)->handleEventLocal(_event);

    h_celltype_field = cellTypeMonitorPlugin->getCellTypeArray();

    fieldDim = cellFieldG->getDim();
    workFieldDim = static_cast<Cruncher *>(this)->getConcentrationField(0)->getInternalDim();

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
bool DiffusionSolverFE<Cruncher>::checkIfOffsetInArray(Point3D _pt, vector <Point3D> &_array) {
    for (size_t i = 0; i < _array.size(); ++i) {
        if (_array[i] == _pt)
            return true;
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::prepareForwardDerivativeOffsets() {
    latticeType = static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType();


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

        const std::vector <Point3D> &offsetVecRef = static_cast<Cruncher *>(this)->getBoundaryStrategy()->getOffsetVec(
                pt);
        for (unsigned int i = 0; i <= static_cast<Cruncher *>(this)->getMaxNeighborIndex(); ++i) {

            const Point3D &offset = offsetVecRef[i];
            //we use only those offset vectors which have only positive coordinates
            if (offset.x >= 0 && offset.y >= 0 && offset.z >= 0) {
                offsetVecCartesian.push_back(offset);
            }
        }

    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::start() {

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

    m_RDTime = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::finish() {
    CC3D_Log(LOG_DEBUG) << m_RDTime<<" ms spent in solving "<<(hasAdditionalTerms()?"reaction-":"")<<"diffusion problem";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::initializeConcentration() {
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        if (!diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression.empty()) {
            static_cast<Cruncher *>(this)->initializeFieldUsingEquation(
                    static_cast<Cruncher *>(this)->getConcentrationField(i),
                    diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression);
            continue;
        }
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
        CC3D_Log(LOG_DEBUG) << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName,
                               static_cast<Cruncher *>(this)->getConcentrationField(i));

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::stepImpl(const unsigned int _currentStep) {
    // implemented separately in GPU and CPU code
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::step(const unsigned int _currentStep) {

    currentStep = _currentStep;

    MyTime::Time_t stepBT = MyTime::CTime();
    // updating cell type and  cell id arrays
    init_cell_type_and_id_arrays();
    stepImpl(_currentStep);
    m_RDTime += MyTime::ElapsedTime(stepBT, MyTime::CTime());



    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteOnContactSingleField(unsigned int idx) {
    // implemented separately in GPU and CPU code
}


//Default implementation. Most of the solvers have it.
template<class Cruncher>
bool DiffusionSolverFE<Cruncher>::hasExtraLayer() const {
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteSingleField(unsigned int idx) {
    // implemented separately in GPU and CPU code
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteConstantConcentrationSingleField(unsigned int idx) {
    // implemented separately in GPU and CPU code
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::secrete() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
            (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

            //          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
        }

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
float DiffusionSolverFE<Cruncher>::couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec,
                                                float _currentConcentration) {

    float couplingTerm = 0.0;
    float coupledConcentration;
    for (size_t i = 0; i < _couplDataVec.size(); ++i) {
        coupledConcentration = static_cast<Cruncher *>(this)->getConcentrationField(_couplDataVec[i].fieldIdx)->get(
                _pt);
        couplingTerm += _couplDataVec[i].couplingCoef * _currentConcentration * coupledConcentration;
    }

    return couplingTerm;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::boundaryConditionInit(int idx) {

    // implemented separately in GPU and CPU code
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::diffuseSingleField(unsigned int idx) {
    // implemented separately in GPU and CPU code
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
bool DiffusionSolverFE<Cruncher>::isBoudaryRegion(int x, int y, int z, Dim3D dim) {
    if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3)
        return true;
    else
        return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
Dim3D DiffusionSolverFE<Cruncher>::getInternalDim() {
    return Dim3D();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::prepCellTypeField(int idx) {

    // here we set up cellTypeArray boundaries
    Array3DCUDA<unsigned char> &cellTypeArray = *h_celltype_field;
    BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];
    int numberOfIters = 1;



    if (static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType() ==
        HEXAGONAL_LATTICE) { //for hex lattice we need two passes to correctly initialize lattice corners
        numberOfIters = 2;
    }

    Dim3D workFieldDimInternal = getInternalDim();



    for (int iter = 0; iter < numberOfIters; ++iter) {
        if (periodicBoundaryCheckVector[0] || bcSpec.planePositions[0] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[1] == BoundaryConditionSpecifier::PERIODIC) {
            //x - periodic
            int x = 0;

            for (int y = 0; y < fieldDim.y + 2; ++y)
                for (int z = 0; z < fieldDim.z + 2; ++z) {

                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(fieldDim.x, y, z));
                }

            x = fieldDim.x + 1;
            for (int y = 0; y < fieldDim.y + 2; ++y)
                for (int z = 0; z < fieldDim.z + 2; ++z) {

                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(1, y, z));
                }

        } else {
            int x = 0;

            for (int y = 0; y < fieldDim.y + 2; ++y)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x + 1, y, z));
                }

            x = fieldDim.x + 1;
            for (int y = 0; y < fieldDim.y + 2; ++y)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x - 1, y, z));
                }

        }

        if (periodicBoundaryCheckVector[1] || bcSpec.planePositions[2] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[3] == BoundaryConditionSpecifier::PERIODIC) {
            int y = 0;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, fieldDim.y, z));
                }

            y = fieldDim.y + 1;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, 1, z));
                }
        } else {
            int y = 0;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    // for(int x=0 ; x< fieldDim.x+2; ++x)
                    // for(int z=0 ; z<fieldDim.z+2 ; ++z){
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y + 1, z));
                }

            y = fieldDim.y + 1;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int z = 0; z < fieldDim.z + 2; ++z) {
                    // for(int x=0 ; x< fieldDim.x+2; ++x)
                    // for(int z=0 ; z<fieldDim.z+2 ; ++z){
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y - 1, z));
                }
        }

        if (periodicBoundaryCheckVector[2] || bcSpec.planePositions[4] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[5] == BoundaryConditionSpecifier::PERIODIC) {

            int z = 0;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int y = 0; y < fieldDim.y + 2; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, fieldDim.z));
                }

            z = fieldDim.z + 1;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int y = 0; y < fieldDim.y + 2; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, 1));
                }

        } else {

            int z = 0;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int y = 0; y < fieldDim.y + 2; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, z + 1));
                }

            z = fieldDim.z + 1;
            for (int x = 0; x < fieldDim.x + 2; ++x)
                for (int y = 0; y < fieldDim.y + 2; ++y) {
                    cellTypeArray.setDirect(x, y, z, cellTypeArray.getDirect(x, y, z - 1));
                }


        }
    }



}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::diffuse() {
// implemented separately in GPU and CPU code

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
template<class ConcentrationField_t>
void DiffusionSolverFE<Cruncher>::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                        ConcentrationField_t *concentrationField) {
    //scratchField->switchContainersQuick(*(concentrationField));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
template<class ConcentrationField_t>
void DiffusionSolverFE<Cruncher>::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
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
template<class Cruncher>
template<class ConcentrationField_t>
void
DiffusionSolverFE<Cruncher>::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {

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
                concentrationField->set(pt,0);
			}

    while (!in.eof()) {
        in >> pt.x >> pt.y >> pt.z >> c;
        if (!in.fail())
            concentrationField->set(pt, c);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
void DiffusionSolverFE<Cruncher>::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //notice, limited steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
	// Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running
	CC3D_Log(LOG_DEBUG) << "INSIDE UPDATE XML";

    solverSpecific(_xmlData);
    diffSecrFieldTuppleVec.clear();
    bcSpecVec.clear();
    bcSpecFlagVec.clear();

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");


    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {
        diffSecrFieldTuppleVec.push_back(DiffusionSecretionDiffusionFEFieldTupple<Cruncher>());
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

            if (static_cast<Cruncher *>(this)->getBoundaryStrategy()->getLatticeType() == HEXAGONAL_LATTICE) {
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
        CC3D_Log(LOG_DEBUG) << "serialize Flag="<<serializeFlag;
	}

    if (_xmlData->findElement("ReadFromFile")) {
        readFromFileFlag = true;
        CC3D_Log(LOG_DEBUG) << "readFromFileFlag="<<readFromFileFlag;
    }

    for (size_t i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
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
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &DiffusionSolverFE::secreteSingleField;
                ++j;
            } else if ((*sitr) == "SecretionOnContact") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &DiffusionSolverFE::secreteOnContactSingleField;
                ++j;
            } else if ((*sitr) == "ConstantConcentration") {
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &DiffusionSolverFE::secreteConstantConcentrationSingleField;
                ++j;
            }
        }
    }

    //allocating  maxDiffConstVec and  scalingExtraMCSVec   
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
std::string DiffusionSolverFE<Cruncher>::toString() { //TODO: overload in cruncher?

    return toStringImpl();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Cruncher>
std::string DiffusionSolverFE<Cruncher>::steerableName() {
    return toString();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//The explicit instantiation part.
//Add new solvers here
template
class CompuCell3D::DiffusionSolverFE<DiffusionSolverFE_CPU>;

template
class CompuCell3D::DiffusionSolverFE<DiffusionSolverFE_CPU_Implicit>;

#if OPENCL_ENABLED == 1
template class CompuCell3D::DiffusionSolverFE<DiffusionSolverFE_OpenCL>;
//template class CompuCell3D::DiffusionSolverFE<DiffusionSolverFE_OpenCL_Implicit>;
template class CompuCell3D::DiffusionSolverFE<ReactionDiffusionSolverFE_OpenCL_Implicit>;
#endif
