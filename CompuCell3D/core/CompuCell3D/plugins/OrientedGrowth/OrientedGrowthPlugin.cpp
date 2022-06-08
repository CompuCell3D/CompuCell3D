
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "OrientedGrowthPlugin.h"


OrientedGrowthPlugin::OrientedGrowthPlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        boundaryStrategy(0) {}

OrientedGrowthPlugin::~OrientedGrowthPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void OrientedGrowthPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    update(xmlData, true);

    potts->getCellFactoryGroupPtr()->registerClass(&orientedGrowthDataAccessor);
    potts->registerEnergyFunctionWithName(this, "OrientedGrowth");

    potts->registerStepper(this);

    simulator->registerSteerableObject(this);
}

void OrientedGrowthPlugin::extraInit(Simulator *simulator) {
}

void OrientedGrowthPlugin::step() {
    //Put your code here - it will be invoked after every successful
    // pixel copy and after all lattice monitor finished running
}

void OrientedGrowthPlugin::setConstraintWidth(CellG *Cell, float _constraint) {
    orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_targetWidth = _constraint;
}

void OrientedGrowthPlugin::setElongationAxis(CellG *Cell, float _elongX, float _elongY) {
    float magnitude = sqrt(pow(_elongX, 2) + pow(_elongY, 2));
    if (magnitude == 0) {
        orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_enabled = false;
    } else {
        orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_x = (_elongX / magnitude);
        orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_y = (_elongY / magnitude);
    }
}

void OrientedGrowthPlugin::setElongationEnabled(CellG *Cell, bool _enabled) {
    orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_enabled = _enabled;
}

float OrientedGrowthPlugin::getElongationAxis_X(CellG *Cell) {
    return orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_x;
}

float OrientedGrowthPlugin::getElongationAxis_Y(CellG *Cell) {
    return orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_y;
}

bool OrientedGrowthPlugin::getElongationEnabled(CellG *Cell) {
    return orientedGrowthDataAccessor.get(Cell->extraAttribPtr)->elong_enabled;
}

double OrientedGrowthPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    double energy = 0;

    if (oldCell) {
        bool cell_enabled = orientedGrowthDataAccessor.get(oldCell->extraAttribPtr)->elong_enabled;
        if (cell_enabled == true) {
            float elongNormalX = orientedGrowthDataAccessor.get(oldCell->extraAttribPtr)->elong_y;
            float elongNormalY = orientedGrowthDataAccessor.get(oldCell->extraAttribPtr)->elong_x * (-1);
            float changeVecX = oldCell->xCOM - pt.x;
            float changeVecY = oldCell->yCOM - pt.y;
            float dotProduct = abs((changeVecX * elongNormalX) + (changeVecY * elongNormalY));
            float myTargetWidth = orientedGrowthDataAccessor.get(oldCell->extraAttribPtr)->elong_targetWidth;

            if (dotProduct > myTargetWidth) {
                float offset = dotProduct - myTargetWidth;
                energy -= xml_energy_penalty;
                energy -= pow(offset, 2) / xml_energy_falloff;
            }
        }
    }

    if (newCell) {
        bool cell_enabled = orientedGrowthDataAccessor.get(newCell->extraAttribPtr)->elong_enabled;
        if (cell_enabled == true) {
            float elongNormalX = orientedGrowthDataAccessor.get(newCell->extraAttribPtr)->elong_y;
            float elongNormalY = orientedGrowthDataAccessor.get(newCell->extraAttribPtr)->elong_x * (-1);
            float changeVecX = newCell->xCOM - pt.x;
            float changeVecY = newCell->yCOM - pt.y;
            float dotProduct = abs((changeVecX * elongNormalX) + (changeVecY * elongNormalY));
            int myTargetWidth = orientedGrowthDataAccessor.get(newCell->extraAttribPtr)->elong_targetWidth;

            if (dotProduct > myTargetWidth) {
                float offset = dotProduct - myTargetWidth;
                energy += xml_energy_penalty;
                energy += pow(offset, 2) / xml_energy_falloff;
            }
        }
    }

    return energy;
}


void OrientedGrowthPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    set<unsigned char> cellTypesSet;

    CC3DXMLElement *myElementOne = xmlData->getFirstElement("Penalty");
    if (myElementOne) {
        xml_energy_penalty = myElementOne->getDouble();
    } else {
        xml_energy_penalty = 99999;
    }

    CC3DXMLElement *myElementTwo = xmlData->getFirstElement("Falloff");
    if (myElementTwo) {
        xml_energy_falloff = myElementTwo->getDouble();
    } else {
        xml_energy_falloff = 1;
    }

    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();
}


std::string OrientedGrowthPlugin::toString() {
    return "OrientedGrowth";
}


std::string OrientedGrowthPlugin::steerableName() {
    return toString();
}
