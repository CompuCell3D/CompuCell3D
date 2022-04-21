

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3D.h>

#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace CompuCell3D;
using namespace std;


#include "ReactionDiffusionSolverFE_SavHog.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE_SavHog::ReactionDiffusionSolverFE_SavHog()
        : DiffusableVector<float>(), deltaX(1.0), deltaT(1.0) {
    imposeDiffusionBox = false;
    dumpFrequency = 0;
//    C1=20;
//    C2=3;
//    C3=15;
//    c1=0.0065;
//    c2=0.841;
//    a=0.15;
//    k=3.5;
//    b=0.35;
//    eps1=0.5;
//    eps2=0.0589;
//    eps3=0.5;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE_SavHog::~ReactionDiffusionSolverFE_SavHog() {

}

void ReactionDiffusionSolverFE_SavHog::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

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

    //	if(unitsElem->getFirstElement("MaxDiffusionZ")){
    //		unitsElem->getFirstElement("MaxDiffusionZ")->updateElementValue(potts->getLengthUnit().toString());
    //	}else{
    //		unitsElem->attachElement("MaxDiffusionZ",potts->getLengthUnit().toString());
    //	}


    //	if(unitsElem->getFirstElement("DeltaTUnit")){
    //		unitsElem->getFirstElement("DeltaTUnit")->updateElementValue(potts->getTimeUnit().toString());
    //	}else{
    //		unitsElem->attachElement("DeltaTUnit",potts->getTimeUnit().toString());
    //	}

    //}

    fieldNameVector.clear();

    if (_xmlData->findElement("DeltaX"))
        deltaX = _xmlData->getFirstElement("DeltaX")->getDouble();

    if (_xmlData->findElement("DeltaT"))
        deltaT = _xmlData->getFirstElement("DeltaT")->getDouble();

    if (_xmlData->findElement("DiffusionConstant"))
        diffConst = _xmlData->getFirstElement("DiffusionConstant")->getDouble();

    if (_xmlData->findElement("DecayConstant"))
        decayConst = _xmlData->getFirstElement("DecayConstant")->getDouble();

    if (_xmlData->findElement("MaxDiffusionZ"))
        maxDiffusionZ = _xmlData->getFirstElement("MaxDiffusionZ")->getUInt();

    if (_xmlData->findElement("DumpResults"))
        dumpFrequency = _xmlData->getFirstElement("DumpResults")->getAttributeAsUInt("Frequency");

    if (_xmlData->findElement("fFunctionParameters")) {
        C1 = _xmlData->getFirstElement("fFunctionParameters")->getAttributeAsDouble("C1");
        C2 = _xmlData->getFirstElement("fFunctionParameters")->getAttributeAsDouble("C2");
        C3 = _xmlData->getFirstElement("fFunctionParameters")->getAttributeAsDouble("C3");
        a = _xmlData->getFirstElement("fFunctionParameters")->getAttributeAsDouble("a");
    }

    if (_xmlData->findElement("epsFunctionParameters")) {
        eps1 = _xmlData->getFirstElement("epsFunctionParameters")->getAttributeAsDouble("eps1");
        eps2 = _xmlData->getFirstElement("epsFunctionParameters")->getAttributeAsDouble("eps2");
        eps3 = _xmlData->getFirstElement("epsFunctionParameters")->getAttributeAsDouble("eps3");
    }

    if (_xmlData->findElement("IntervalParameters")) {
        c1 = _xmlData->getFirstElement("IntervalParameters")->getAttributeAsDouble("c1");
        c2 = _xmlData->getFirstElement("IntervalParameters")->getAttributeAsDouble("c2");
    }

    if (_xmlData->findElement("RefractorinessParameters")) {
        k = _xmlData->getFirstElement("RefractorinessParameters")->getAttributeAsDouble("k");
        b = _xmlData->getFirstElement("RefractorinessParameters")->getAttributeAsDouble("b");
    }

    if (_xmlData->findElement("MinDiffusionBoxCorner")) {
        minDiffusionBoxCorner.x = _xmlData->getFirstElement("MinDiffusionBoxCorner")->getAttributeAsUInt("x");
        minDiffusionBoxCorner.y = _xmlData->getFirstElement("MinDiffusionBoxCorner")->getAttributeAsUInt("y");
        minDiffusionBoxCorner.z = _xmlData->getFirstElement("MinDiffusionBoxCorner")->getAttributeAsUInt("z");
    }
    if (_xmlData->findElement("MaxDiffusionBoxCorner")) {
        maxDiffusionBoxCorner.x = _xmlData->getFirstElement("MaxDiffusionBoxCorner")->getAttributeAsUInt("x");
        maxDiffusionBoxCorner.y = _xmlData->getFirstElement("MaxDiffusionBoxCorner")->getAttributeAsUInt("y");
        maxDiffusionBoxCorner.z = _xmlData->getFirstElement("MaxDiffusionBoxCorner")->getAttributeAsUInt("z");
    }
    if (_xmlData->findElement("NumberOfFields")) {
        numberOfFields = _xmlData->getFirstElement("NumberOfFields")->getUInt();
        numberOfFieldsDeclared = true;
    }

    CC3DXMLElementList fieldNameXMLVec = _xmlData->getElements("FieldName");
    for (unsigned i = 0; i < fieldNameXMLVec.size(); ++i) {
        fieldNameVector.push_back(fieldNameXMLVec[i]->getText());
    }


    if (fieldNameVector.size() > numberOfFields)
        throw CC3DException("You are trying to define more field that you declared!");
    if (!numberOfFieldsDeclared)
        throw CC3DException(
                "You need to declare how many fields you will be using.\n Use <NumberOfFields>N</NumberOfFields> syntax before listing any fields.\n N denotes number of fields");

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    //Here i record how many fields were requested initially. NOtice that in update fcn I check to make sure the number of fields in parse data structure is the same



    //numberOfFields=rdspdPtr->numberOfFields;
    //numberOfFieldsDeclared=rdspdPtr->numberOfFieldsDeclared;

    simPtr = simulator;
    potts = simulator->getPotts();
    automaton = potts->getAutomaton();

    update(_xmlData, true);


    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    workFieldDim = Dim3D(fieldDim.x + 2, fieldDim.y + 2, fieldDim.z + 2);
    allocateDiffusableFieldVector(3, workFieldDim);/// c,r, scratch fields
    for (int i = 0; i < fieldNameVector.size(); ++i) {

        concentrationFieldNameVector[i] = fieldNameVector[i];
        ///register concentration field to make it accesible from external viewers etc.

        simPtr->registerConcentrationField(concentrationFieldNameVector[i], concentrationFieldVector[i]);
    }
    if (minDiffusionBoxCorner == maxDiffusionBoxCorner)
        imposeDiffusionBox = false;
    else
        imposeDiffusionBox = true;

    ///setting member function pointers
    diffusePtr = &ReactionDiffusionSolverFE_SavHog::diffuse;
    simulator->registerSteerableObject(this);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::start() {

    dt_dx2 = deltaT / (deltaX * deltaX);
    initializeConcentration();


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE_SavHog::initializeConcentration() {


    RandomNumberGenerator *rand = simPtr->getRandomNumberGeneratorInstance();


    CellInventory::cellInventoryIterator cInvItr;

    CellG *cell;
    map < CellG * , float > concentrationMap;
    map < CellG * , float > refractorinessMap;

    float concentrationMin = 0.5;
    float concentrationMax = 1.0;

    float refractorinessMin = 1.0;
    float refractorinessMax = 2.0;

    float randomConcentration;
    float randomRefractoriness;


    float x, y, z;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {

        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        if (cell->type != automaton->getTypeId("Ground") && cell->type != automaton->getTypeId("Wall")) {
            // do not put anything in the groundCell or wall
            randomConcentration = fabs(concentrationMax - concentrationMin) * rand->getRatio() + concentrationMin;
            randomRefractoriness = fabs(refractorinessMax - refractorinessMin) * rand->getRatio() + refractorinessMin;

            concentrationMap.insert(make_pair(cell, randomConcentration));
            refractorinessMap.insert(make_pair(cell, randomRefractoriness));
        }

    }
    Point3D pt;
    CellG *currentCellPtr;
    map<CellG *, float>::iterator concentrationMapItr;
    map<CellG *, float>::iterator refractorinessMapItr;

    Array3D_t &concentrationArray0 = concentrationFieldVector[0]->getContainer();
    Array3D_t &concentrationArray1 = concentrationFieldVector[1]->getContainer();


    //once we have decided about the initial values we need to initialize lattice
    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {

                pt = Point3D(x - 1, y - 1, z - 1);
                //currentCellPtr=cellFieldG->getQuick(pt);
                currentCellPtr = cellFieldG->get(pt);
                if (!currentCellPtr) continue;///medium does not participate in FN equations

                //only autocycling cells will have concentration and refractoriness initialized
                if (currentCellPtr->type != automaton->getTypeId("Autocycling"))continue;

                concentrationMapItr = concentrationMap.find(currentCellPtr);
                if (concentrationMapItr !=
                    concentrationMap.end()) {///setting all pixels belonging to given cell to same concentr.
                    concentrationArray0[x][y][z] = concentrationMapItr->second;
                }

                refractorinessMapItr = refractorinessMap.find(currentCellPtr);
                if (refractorinessMapItr !=
                    refractorinessMap.end()) {///setting all pixels belonging to given cell to same refract.
                    concentrationArray1[x][y][z] = refractorinessMapItr->second;
                }


            }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::step(const unsigned int _currentStep) {
    currentStep = _currentStep;

    (this->*diffusePtr)();

    if (dumpFrequency && !(_currentStep % dumpFrequency)) {
        dumpResults(_currentStep);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::diffuse() {
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
    float concentrationSum = 0.0;
    float updatedConcentration = 0.0;

    float currentRefract = 0.0;//refractoriness
    float updatedRefract = 0.0;//refractoriness

    float currentConcentration = 0.0;
    short neighborCounter = 0;

    unsigned char autocyclingType = automaton->getTypeId("Autocycling");
    unsigned char presporeType = automaton->getTypeId("Prespore");
    unsigned char prestalkType = automaton->getTypeId("Prestalk");
    unsigned char wallType = automaton->getTypeId("Wall");


    float F;/// F=a*\delta \tau / (\delta x)^2
    /// in the simplest case (\delta*\tau=1 \delta x=1) F=a;

    Array3D_t &concentrationArray0 = concentrationFieldVector[0]->getContainer();
    Array3D_t &concentrationArray1 = concentrationFieldVector[1]->getContainer();
    Array3D_t &scratchArray = concentrationFieldVector[2]->getContainer();


    F = diffConst;

    for (int z = 1; z < workFieldDim.z - 1; z++)
        for (int y = 1; y < workFieldDim.y - 1; y++)
            for (int x = 1; x < workFieldDim.x - 1; x++) {
                pt = Point3D(x - 1, y - 1, z - 1);
                //currentCellPtr=cellFieldG->getQuick(pt);
                currentCellPtr = cellFieldG->get(pt);
                /// go to the next iteration in case you get Medium cell - medium does not participate in the diffusion
                ///(model dependent may be changed if necessary
                if (!currentCellPtr) {

                    if (imposeDiffusionBox)
                        if (!insideDiffusionBox(pt)) continue;

                }
                updatedConcentration = 0.0;
                updatedRefract = 0.0;
                concentrationSum = 0.0;
                neighborCounter = 0;

                ///loop over nearest neighbors
                const std::vector <Point3D> &offsetVecRef = boundaryStrategy->getOffsetVec(pt);
                for (int i = 0; i <= maxNeighborIndex/*offsetVec.size()*/ ; ++i) {
                    const Point3D &offset = offsetVecRef[i];
                    n = pt + offset;
                    //nCell = cellFieldG->getQuick(n); //even if n is "out of the lattice" the returned pointer will be zero which is
                    //fine in this case

                    nCell = cellFieldG->get(n);

                    if (imposeDiffusionBox)
                        if (!insideDiffusionBox(n) && !nCell) continue;

                    concentrationSum += concentrationArray0[x + offset.x][y + offset.y][z + offset.z];
                    ++neighborCounter;

                }

                currentConcentration = concentrationArray0[x][y][z];
                currentRefract = concentrationArray1[x][y][z];

                if (
                        currentCellPtr &&
                        (
                                currentCellPtr->type == autocyclingType ||
                                currentCellPtr->type == presporeType ||
                                currentCellPtr->type == prestalkType
                        )

                        ) {

                    updatedConcentration = dt_dx2 * F * (concentrationSum - neighborCounter * currentConcentration)
                                           - deltaT * (f(currentConcentration) + currentRefract)
                                           + currentConcentration;
                } else {
                    if (currentCellPtr && currentCellPtr->type == wallType) {
                        updatedConcentration = currentConcentration;
                    } else {

                        updatedConcentration = dt_dx2 * F * (concentrationSum - neighborCounter * currentConcentration)
                                               + currentConcentration;
                    }

                }


                scratchArray[x][y][z] = updatedConcentration;///updating scratch


                ///Refractoriness

                if (currentCellPtr && currentCellPtr->type == autocyclingType) {
                    updatedRefract =
                            deltaT * eps(currentConcentration) * (k * currentConcentration - b - currentRefract) +
                            currentRefract;
                    concentrationArray1[x][y][z] = updatedRefract;///updating Refractoriness
                } else if (currentCellPtr &&
                           (currentCellPtr->type == prestalkType || currentCellPtr->type == presporeType)) {
                    updatedRefract = deltaT * eps(currentConcentration) * (k * currentConcentration - currentRefract) +
                                     currentRefract;///remove -b
                    concentrationArray1[x][y][z] = updatedRefract;///updating Refractoriness
                } else {

                }

            }
    ///copy updated values from scratch to concentration field
    scrarch2Concentration(concentrationFieldVector[2], concentrationFieldVector[0]);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::scrarch2Concentration(ConcentrationField_t *scratchField,
                                                             ConcentrationField_t *concentrationField) {

    scratchField->switchContainersQuick(*(concentrationField));


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE_SavHog::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
    Point3D pt;
    float tempValue;

    for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
            for (pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                tempValue = _concentrationField->get(pt);
                _out << pt << " " << tempValue << endl;
            }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE_SavHog::dumpResults(unsigned int _step) {
    ofstream out;
    ostringstream fileNameActivator;
    fileNameActivator << "Activator" << _step << ".txt";
    out.open(fileNameActivator.str().c_str());

    outputField(out, concentrationFieldVector[0]);
    out.close();

    ostringstream fileNameInhibitor;
    fileNameInhibitor << "Inhibitor" << _step << ".txt";
    out.open(fileNameInhibitor.str().c_str());

    outputField(out, concentrationFieldVector[1]);

    out.close();

}

std::string ReactionDiffusionSolverFE_SavHog::toString() {
    return "ReactionDiffusionSolverFE_SavHog";
}

std::string ReactionDiffusionSolverFE_SavHog::steerableName() {
    return toString();
}

