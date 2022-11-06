

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


#include <time.h>


#include "SteadyStateDiffusionSolver2D.h"

#include "hpppdesolvers.h" //have to put this header last to avoid STL header clash on linux
#include <Logger/CC3DLogger.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//    
//    
// }


using namespace CompuCell3D;
using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2DSerializer::serialize() {

    for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
        ostringstream outName;

        outName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "_" << currentStep << "."
                << serializedFileExtension;
        ofstream outStream(outName.str().c_str());
        solverPtr->outputField(outStream, solverPtr->concentrationFieldVector[i]);
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2DSerializer::readFromFile() {
    try {
        for (int i = 0; i < solverPtr->diffSecrFieldTuppleVec.size(); ++i) {
            ostringstream inName;
            inName << solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName << "." << serializedFileExtension;

            solverPtr->readConcentrationField(inName.str().c_str(), solverPtr->concentrationFieldVector[i]);;
        }

    } catch (CC3DException &e) {
        CC3D_Log(LOG_DEBUG) <<"COULD NOT FIND ONE OF THE FILES";
        throw CC3DException("Error in reading diffusion fields from file", e);
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SteadyStateDiffusionSolver2D::SteadyStateDiffusionSolver2D()
        : DiffusableVectorFortran<Array2DLinearFortranField3DAdapter>(), deltaX(1.0), deltaT(1.0) {
    serializerPtr = 0;
    serializeFlag = false;
    readFromFileFlag = false;
    //haveCouplingTerms=false;
    serializeFrequency = 0;
    //boxWatcherSteppable=0;
    //    useBoxWatcher=false;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SteadyStateDiffusionSolver2D::~SteadyStateDiffusionSolver2D() {

    if (serializerPtr)
        delete serializerPtr;
    serializerPtr = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


    simPtr = simulator;
    potts = simulator->getPotts();
    automaton = potts->getAutomaton();

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();

    ///getting field ptr from Potts3D
    cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    update(_xmlData, true);



    ///setting member function pointers
    diffusePtr = &SteadyStateDiffusionSolver2D::diffuse;
    secretePtr = &SteadyStateDiffusionSolver2D::secrete;


    numberOfFields = diffSecrFieldTuppleVec.size();


	vector<string> concentrationFieldNameVectorTmp; //temporary vector for field names
	///assign vector of field names
	concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(),string(""));
	CC3D_Log(LOG_DEBUG) <<"diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size();

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
        CC3D_Log(LOG_DEBUG) <<" concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i];
    }

    // //setting up couplingData - field-field interaction terms
    // vector<CouplingData>::iterator pos;

    // for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
    // pos=diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
    // for(int j = 0 ; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size() ; ++j){

    // for(int idx=0; idx<concentrationFieldNameVectorTmp.size() ; ++idx){
    // if( concentrationFieldNameVectorTmp[idx] == diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].intrFieldName ){
    // diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].fieldIdx=idx;
    // haveCouplingTerms=true; //if this is called at list once we have already coupling terms and need to proceed differently with scratch field initialization
    // break;
    // }
    // //this means that required interacting field name has not been found
    // if( idx == concentrationFieldNameVectorTmp.size()-1 ){
    // //remove this interacting term
    // //                pos=&(diffDataVec[i].degradationDataVec[j]);
    // diffSecrFieldTuppleVec[i].diffData.couplingDataVec.erase(pos);
    // }
    // }
    // ++pos;
    // }
    // }

	CC3D_Log(LOG_DEBUG) <<"FIELDS THAT I HAVE";
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		CC3D_Log(LOG_DEBUG) <<"Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i];
    }

	CC3D_Log(LOG_DEBUG) <<"FlexibleDiffusionSolverFE: extra Init in read XML";

    workFieldDim = Dim3D(fieldDim.x + 1, fieldDim.y + 1, 1);
    ///allocate fields including scrartch field
    allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(), fieldDim);
    //scratch vecor
    scratchVec.assign(
            4 * (fieldDim.y + 1 + 1) + (13 + (int) (log(fieldDim.y + 1 + 1.0) / log(2.0))) * (fieldDim.x + 1 + 1), 0.0);
    scratch = &(scratchVec[0]);

    CC3D_Log(LOG_DEBUG) <<"fieldDim="<<fieldDim;
	//vectors used to specify boundary conditions
	bdaVec.assign(fieldDim.y+1,0.0);
	bdbVec.assign(fieldDim.y+1,0.0);
	bdcVec.assign(fieldDim.x+1,0.0);
	bddVec.assign(fieldDim.x+1,0.0);


    // if(!haveCouplingTerms){
    // allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size()+1,workFieldDim); //+1 is for additional scratch field
    // }else{
    // allocateDiffusableFieldVector(2*diffSecrFieldTuppleVec.size(),workFieldDim); //with coupling terms every field need to have its own scratch field
    // }

    //here I need to copy field names from concentrationFieldNameVectorTmp to concentrationFieldNameVector
    //because concentrationFieldNameVector is reallocated with default values once I call allocateDiffusableFieldVector


    for (unsigned int i = 0; i < concentrationFieldNameVectorTmp.size(); ++i) {
        concentrationFieldNameVector[i] = concentrationFieldNameVectorTmp[i];
    }


    //register fields once they have been allocated
    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        simPtr->registerConcentrationField(concentrationFieldNameVector[i], concentrationFieldVector[i]);
        CC3D_Log(LOG_DEBUG) <<"registring field: "<<concentrationFieldNameVector[i]<<" field address="<<concentrationFieldVector[i];
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
void SteadyStateDiffusionSolver2D::extraInit(Simulator *simulator) {

    if ((serializeFlag || readFromFileFlag) && !serializerPtr) {
        serializerPtr = new SteadyStateDiffusionSolver2DSerializer();
        serializerPtr->solverPtr = this;
    }

    if (serializeFlag) {
        simulator->registerSerializer(serializerPtr);
    }

    //checking if box watcher is necessary at all


    bool pluginAlreadyRegisteredFlag;

    Plugin *centerOfMassPlugin = Simulator::pluginManager.get("CenterOfMass", &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        centerOfMassPlugin->init(simulator);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::handleEvent(CC3DEvent &_event) {

    if (_event.id == LATTICE_RESIZE) {
        cellFieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();

        CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);

        for (size_t i = 0; i < concentrationFieldVector.size(); ++i) {
            concentrationFieldVector[i]->resizeAndShift(ev.newDim, ev.shiftVec);
        }


        fieldDim = cellFieldG->getDim();
        workFieldDim = Dim3D(fieldDim.x + 1, fieldDim.y + 1, 1);

        scratchVec.assign(
                4 * (fieldDim.y + 1 + 1) + (13 + (int) (log(fieldDim.y + 1 + 1.0) / log(2.0))) * (fieldDim.x + 1 + 1),
                0.0);
        scratch = &(scratchVec[0]);


        //vectors used to specify boundary conditions
        bdaVec.assign(fieldDim.y + 1, 0.0);
        bdbVec.assign(fieldDim.y + 1, 0.0);
        bdcVec.assign(fieldDim.x + 1, 0.0);
        bddVec.assign(fieldDim.x + 1, 0.0);


    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::start() {
	//     if(diffConst> (1.0/6.0-0.05) ){ //hard coded condtion for stability of the solutions - assume dt=1 dx=dy=dz=1
	// 		 CC3D_Log(LOG_TRACE) <<"CANNOT SOLVE DIFFUSION EQUATION: STABILITY PROBLEM - DIFFUSION CONSTANT TOO LARGE. EXITING...";
    //       exit(0);
    //
    //    }
    if (simPtr->getRestartEnabled()) {
        return;  // we will not initialize cells if restart flag is on
    }


    dt_dx2 = deltaT / (deltaX * deltaX);
    if (readFromFileFlag) {
        try {

            //          serializerPtr->readFromFile();

        } catch (CC3DException &e) {
            CC3D_Log(LOG_DEBUG) <<"Going to fail-safe initialization";
            initializeConcentration(); //if there was error, initialize using failsafe defaults
        }

    } else {
        initializeConcentration();//Normal reading from User specified files
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SteadyStateDiffusionSolver2D::initializeConcentration() {


    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        if (!diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression.empty()) {
            initializeFieldUsingEquation(concentrationFieldVector[i],
                                         diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression);
            continue;
        }
        if (diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
        CC3D_Log(LOG_DEBUG) <<"fail-safe initialization "<<diffSecrFieldTuppleVec[i].diffData.concentrationFileName;
        readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName, concentrationFieldVector[i]);
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::step(const unsigned int _currentStep) {

    currentStep = _currentStep;

    //secrete function resets field to 0. If there is no user-specified secretion we have to explicitely reset the field
    (this->*secretePtr)();

    (this->*diffusePtr)();


    if (serializeFrequency > 0 && serializeFlag && !(_currentStep % serializeFrequency)) {
        serializerPtr->setCurrentStep(currentStep);
        serializerPtr->serialize();
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::secreteSingleField(unsigned int idx) {

    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    vector<double> &containerRef = concentrationFieldPtr->getContainerRef();

    //containerRef.assign(containerRef.size(),0.0); //zero concentration vector

    CellInventory *cellInventoryPtr = &(potts->getCellInventory());

    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;


    Point3D pt;


    double secrConst;
    SecretionData &secrData = diffSecrFieldTuppleVec[idx].secrData;
    std::map<unsigned char, float>::iterator mitr;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();
    std::map<unsigned char, UptakeData>::iterator mitrUptakeShared;
    std::map<unsigned char, UptakeData>::iterator end_mitrUptake = secrData.typeIdUptakeDataMap.end();
    std::map<unsigned char, UptakeData>::iterator mitrUptake;

    float maxUptakeInMedium = 0.0;
    float relativeUptakeRateInMedium = 0.0;
    float mmCoefInMedium = 0.0;
    float secrConstMedium = 0.0;

    float currentConcentration = 0.0;

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
        mitrUptakeShared = secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
        if (mitrUptakeShared != end_mitrUptake) {
            maxUptakeInMedium = mitrUptakeShared->second.maxUptake;
            relativeUptakeRateInMedium = mitrUptakeShared->second.relativeUptakeRate;
            mmCoefInMedium = mitrUptakeShared->second.mmCoef;
            uptakeInMediumFlag = true;

        }
    }


    //int c=0;

    for (int x = 0; x < fieldDim.x; x++)
        for (int y = 0; y < fieldDim.y; y++) {
            pt.x = x;
            pt.y = y;
            pt.z = 0;
            CC3D_Log(LOG_TRACE) <<"pt="<<pt<<"index="<<concentrationFieldPtr->index(x,y);
            bool pixelUntouched = true;

            cell = cellFieldG->get(pt);

            currentConcentration = concentrationFieldPtr->get(pt);

            if (secreteInMedium && !cell) {
                concentrationFieldPtr->set(pt,
                                           -secrConstMedium);//minus sign is added because of the convention implied by pde solving fcn - deltaT scaling is done in diffuseSingleField function
                pixelUntouched = false;
            }

            if (cell) { //do not secrete in medium
                mitr = secrData.typeIdSecrConstMap.find(cell->type);
                if (mitr != end_mitr) {
                    secrConst = -mitr->second;    //minus sign is added because of the convention implied by pde solving fcn - deltaT scaling is done in diffuseSingleField function
                    concentrationFieldPtr->set(pt, secrConst);
                    pixelUntouched = false;

                }
            }


            if (doUptakeFlag && currentConcentration > 0.0) {

                if (uptakeInMediumFlag && !cell) {
                    if (relativeUptakeRateInMedium) { //doing relative uptake

                        if (currentConcentration * relativeUptakeRateInMedium > maxUptakeInMedium) {
                            concentrationFieldPtr->set(pt, maxUptakeInMedium);//positive value here means uptake
                            pixelUntouched = false;
                        } else {
                            concentrationFieldPtr->set(pt, currentConcentration *
                                                           relativeUptakeRateInMedium); //positive value here means uptake
                            pixelUntouched = false;
                        }
                    } else { //doing  MichaelisMenten coef-based type of uptake

                        concentrationFieldPtr->set(pt, maxUptakeInMedium * (currentConcentration /
                                                                            (currentConcentration +
                                                                             mmCoefInMedium)));//positive value here means uptake
                        pixelUntouched = false;

                    }
                }

                if (cell) {

                    mitrUptake = secrData.typeIdUptakeDataMap.find(cell->type);
                    if (mitrUptake != end_mitrUptake) {
                        if (mitrUptake->second.relativeUptakeRate) { //doing relative uptake
                            if (currentConcentration * mitrUptake->second.relativeUptakeRate >
                                mitrUptake->second.maxUptake) {//positive value here means uptake
                                concentrationFieldPtr->set(pt, mitrUptake->second.maxUptake);
                                pixelUntouched = false;
                            } else {
                                concentrationFieldPtr->set(pt, currentConcentration *
                                                               mitrUptake->second.relativeUptakeRate);//positive value here means uptake
                                pixelUntouched = false;
                                CC3D_Log(LOG_TRACE) <<"concentration="<< concentrationFieldPtr->getQuick(x,y)- currentConcentration*mitrUptake->second.relativeUptakeRate;
                            }
                        } else {//doing  MichaelisMenten coef-based type of uptake
                            concentrationFieldPtr->set(pt, mitrUptake->second.maxUptake * (currentConcentration /
                                                                                           (currentConcentration +
                                                                                            mitrUptake->second.mmCoef)));//positive value here means uptake
                            pixelUntouched = false;
                        }
                    }
                }
            }
            if (pixelUntouched) {
                concentrationFieldPtr->set(pt, 0.0);
            }

        }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::secrete() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {

        if (manageSecretionInPythonVec[i]) {
            //do nothing here we do all manipulation in Python        
        } else {
            if (diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size()) {
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
                }
            } else {
                //secrete function resets field to 0. If there is no user-specified secretion we have to explicitely reset the field
                ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[i];
                vector<double> &containerRef = concentrationFieldPtr->getContainerRef();

                containerRef.assign(containerRef.size(), 0.0); //zero concentration vector

            }
        }
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::boundaryConditionInit(ConcentrationField_t *concentrationField) {
    // Array2D_t & concentrationArray=concentrationField->getContainer();


    // if(periodicBoundaryCheckVector[0]){//periodic BC along X
    // for(int y=0 ; y<workFieldDim.y ; ++y){
    // concentrationArray[0][y]=concentrationArray[workFieldDim.x-2][y];
    // concentrationArray[workFieldDim.x-1][y]=concentrationArray[1][y];
    // }


    // }else{//noFlux BC along X
    // for(int y=0 ; y<workFieldDim.y ; ++y){
    // concentrationArray[0][y]=concentrationArray[1][y];
    // concentrationArray[workFieldDim.x-1][y]=concentrationArray[workFieldDim.x-2][y];
    // }


    // }

    // if(periodicBoundaryCheckVector[1]){//periodic BC along Y
    // for(int x=0 ; x<workFieldDim.x ; ++x){
    // concentrationArray[x][0]=concentrationArray[x][workFieldDim.y-2];
    // concentrationArray[x][workFieldDim.y-1]=concentrationArray[x][1];
    // }

    // }else{//noFlux BC along Y
    // for(int x=0 ; x<workFieldDim.x ; ++x){
    // concentrationArray[x][0]=concentrationArray[x][1];
    // concentrationArray[x][workFieldDim.y-1]=concentrationArray[x][workFieldDim.y-2];
    // }

    // }

    // //    if(periodicBoundaryCheckVector[0] || periodicBoundaryCheckVector[1]){
    // //       for(int y=0 ; y<workFieldDim.y ; ++y){
    // //             concentrationArray[0][y]=concentrationArray[workFieldDim.x-2][y];
    // //             concentrationArray[workFieldDim.x-1][y]=concentrationArray[1][y];
    // //       }
    // //
    // //       for(int x=0 ; x<workFieldDim.x ; ++x){
    // //             concentrationArray[x][0]=concentrationArray[x][workFieldDim.y-2];
    // //             concentrationArray[x][workFieldDim.y-1]=concentrationArray[x][1];
    // //       }
    // //
    // //
    // //    }else{
    // //       for(int y=0 ; y<workFieldDim.y ; ++y){
    // //             concentrationArray[0][y]=concentrationArray[1][y];
    // //             concentrationArray[workFieldDim.x-1][y]=concentrationArray[workFieldDim.x-2][y];
    // //       }
    // //
    // //       for(int x=0 ; x<workFieldDim.x ; ++x){
    // //             concentrationArray[x][0]=concentrationArray[x][1];
    // //             concentrationArray[x][workFieldDim.y-1]=concentrationArray[x][workFieldDim.y-2];
    // //       }
    // //
    // //
    // //    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SteadyStateDiffusionSolver2D::diffuseSingleField(unsigned int idx) {

    double x_min, x_max;
    integer m, mbdcnd;
    double *bda, *bdb, y_min, y_max;
    integer n, nbdcnd;
    double *bdc, *bdd, elmbda, *f;
    integer idimf;
    double pertrb;
    integer ierror;
    double *w;

    DiffusionData &diffData = diffSecrFieldTuppleVec[idx].diffData;
    double diffConst = diffData.diffConst;
    double decayConst = diffData.decayConst;
    double deltaT = diffData.deltaT;
    double deltaX = diffData.deltaX;
    double dt_dx2 = deltaT / (deltaX * deltaX);


    //dimensions here we need to use m,n euqal to corresopnding dimension -1
    x_min = 0;
    x_max = fieldDim.x;
    m = (int) x_max - 1;

    y_min = 0;
    y_max = fieldDim.y;
    n = (int) y_max - 1;
    //boundary conditions vectors
    bda = &(bdaVec[0]);
    bdb = &(bdbVec[0]);

    bdc = &(bdcVec[0]);
    bdd = &(bddVec[0]);

    ConcentrationField_t *concentrationFieldPtr = concentrationFieldVector[idx];
    vector<double> &containerRef = concentrationFieldPtr->getContainerRef();

    if (idx < bcSpecVec.size()) {

        BoundaryConditionSpecifier & bcSpec = bcSpecVec[idx];

        // X Axis
        if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[1] == BoundaryConditionSpecifier::PERIODIC) {
            mbdcnd = 0; //periodic boundary conditions -x
        } else {
            //the solution is specified at X = A and X = B
            Point3D pt;
            if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_VALUE &&
                bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                mbdcnd = 1; //constant value boundary conditions -xMin, xMax

                for (int y = 0; y < fieldDim.y; y++) {
                    pt.x = 0;
                    pt.y = y;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[0] * dt_dx2 * diffConst);
                    pt.x = fieldDim.x - 1;
                    concentrationFieldPtr->set(pt, bcSpec.values[1] * dt_dx2 * diffConst);
                }


            } else if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_VALUE &&
                       bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                mbdcnd = 2; //constant value boundary conditions -xMin, constant derivative boundary conditions xMax
                for (int y = 0; y < fieldDim.y; y++) {
                    pt.x = 0;
                    pt.y = y;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[0] * dt_dx2 * diffConst);
                }

                for (unsigned int i = 0; i < bdbVec.size(); ++i) {
                    bdbVec[i] = bcSpec.values[1];
                }

            } else if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE &&
                       bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                mbdcnd = 3;    //constant derivative boundary conditions -xMin, constant derivative boundary conditions xMax
                for (unsigned int i = 0; i < bdaVec.size(); ++i) {
                    bdaVec[i] = bcSpec.values[0];
                }

                for (unsigned int i = 0; i < bdbVec.size(); ++i) {
                    bdbVec[i] = bcSpec.values[1];
                }

            } else if (bcSpec.planePositions[0] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE &&
                       bcSpec.planePositions[1] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                mbdcnd = 4; //constant derivative boundary conditions -xMin, constant value boundary conditions xMax

                for (unsigned int i = 0; i < bdaVec.size(); ++i) {
                    bdaVec[i] = bcSpec.values[0];
                }

                for (int y = 0; y < fieldDim.y; y++) {
                    pt.x = fieldDim.x - 1;
                    pt.y = y;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[1] * dt_dx2 * diffConst);
                }

            }

        }

        // Y Axis
        if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::PERIODIC ||
            bcSpec.planePositions[3] == BoundaryConditionSpecifier::PERIODIC) {
            nbdcnd = 0; //periodic boundary conditions -y
        } else {
            //the solution is specified at X = A and X = B
            Point3D pt;
            if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_VALUE &&
                bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                nbdcnd = 1; //constant value boundary conditions -yMin, constant value boundary conditions yMax

                for (int x = 0; x < fieldDim.x; x++) {
                    pt.x = x;
                    pt.y = 0;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[2] * dt_dx2 * diffConst);
                    pt.y = fieldDim.y - 1;
                    concentrationFieldPtr->set(pt, bcSpec.values[3] * dt_dx2 * diffConst);
                }


            } else if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_VALUE &&
                       bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                nbdcnd = 2; //constant value boundary conditions -yMin, constant derivative boundary conditions yMax
                for (int x = 0; x < fieldDim.x; x++) {
                    pt.x = x;
                    pt.y = 0;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[2] * dt_dx2 * diffConst);
                }

                for (unsigned int i = 0; i < bddVec.size(); ++i) {
                    bddVec[i] = bcSpec.values[3];
                }

            } else if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE &&
                       bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE) {
                nbdcnd = 3; //constant derivative boundary conditions -yMin, constant derivative boundary conditions yMax
                for (unsigned int i = 0; i < bdcVec.size(); ++i) {
                    bdcVec[i] = bcSpec.values[2];
                }

                for (unsigned int i = 0; i < bddVec.size(); ++i) {
                    bddVec[i] = bcSpec.values[3];
                }

            } else if (bcSpec.planePositions[2] == BoundaryConditionSpecifier::CONSTANT_DERIVATIVE &&
                       bcSpec.planePositions[3] == BoundaryConditionSpecifier::CONSTANT_VALUE) {
                nbdcnd = 4; //constant derivative boundary conditions -yMin, constant value boundary conditions yMax

                for (unsigned int i = 0; i < bdcVec.size(); ++i) {
                    bdcVec[i] = bcSpec.values[2];
                }

                for (int x = 0; x < fieldDim.x; x++) {
                    pt.x = x;
                    pt.y = fieldDim.y - 1;
                    pt.z = 0;
                    concentrationFieldPtr->set(pt, bcSpec.values[3] * dt_dx2 * diffConst);
                }

            }

        }

    } else {
        mbdcnd = 0;//periodic boundary conditions -x
        nbdcnd = 0; //periodic boundary conditions -y
    }

    //THIS IS THE EQUATION THAT IS BEING SOLVED
    /*         (d/dX)(dU/dX) + (d/dY)(dU/dY) + (d/dZ)(dU/dZ) */

    /*                    + LAMBDA*U = F(X,Y,Z) . */



    //now have to do rescaling to put equation into a form required by PDE solving function
    elmbda = -decayConst * deltaT / (dt_dx2 *
                                     diffConst); // although this could be simplified I leave it this for to show where all the scaling factors come from




    for (int i = 0; i < containerRef.size(); ++i) {
        containerRef[i] /= dt_dx2 * diffConst;

    }

    idimf = x_max + 1; //this is required by solving function

    f = concentrationFieldPtr->getContainerArrayPtr();

    w = scratch;

    //call steady state diffusion equation solver
    hwscrt_(&x_min, &x_max, &m, &mbdcnd, bda, bdb,
            &y_min, &y_max, &n, &nbdcnd, bdc, bdd,
            &elmbda, f,
            &idimf,
            &pertrb, &ierror, w);

    Point3D pt;


    if (ierror) {
        CC3D_Log(LOG_DEBUG) <<"solution has a problem. Error code: "<<ierror;
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::diffuse() {

    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        diffuseSingleField(i);
        //if(!haveCouplingTerms){ //without coupling terms field swap takes place immediately aftera given field has been diffused
        //	ConcentrationField_t * concentrationField=concentrationFieldVector[i];
        //	ConcentrationField_t * scratchField=concentrationFieldVector[diffSecrFieldTuppleVec.size()];
        //	//copy updated values from scratch to concentration field
        //	// scrarch2Concentration(scratchField,concentrationField);
        //}

    }


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void SteadyStateDiffusionSolver2D::scrarch2Concentration( ConcentrationField_t *scratchField, ConcentrationField_t *concentrationField){
// scratchField->switchContainersQuick(*(concentrationField));

// }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SteadyStateDiffusionSolver2D::outputField(std::ostream &_out, ConcentrationField_t *_concentrationField) {
    Point3D pt;
    double tempValue;


    for (pt.x = 0; pt.x < fieldDim.x; pt.x++)
        for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
            for (pt.z = 0; pt.z < fieldDim.z; pt.z++) {
                tempValue = _concentrationField->get(pt);
                _out << pt.x << " " << pt.y << " " << pt.z << " " << tempValue << endl;
            }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
SteadyStateDiffusionSolver2D::readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField) {

    std::string basePath = simulator->getBasePath();
    std::string fn = fileName;
    if (basePath != "") {
        fn = basePath + "/" + fileName;
    }

    ifstream in(fn.c_str());

    if (!in.is_open()) throw CC3DException(string("Could not open chemical concentration file '") + fn + "'!");

    Point3D pt;
    double c;
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


void SteadyStateDiffusionSolver2D::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {


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

    //	// if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
    //	// unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
    //	// }else{
    //	// unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
    //	// }



    //	if(unitsElem->getFirstElement("SecretionUnit")){
    //		unitsElem->getFirstElement("SecretionUnit")->updateElementValue(secretionConstUnit.toString());
    //	}else{
    //		unitsElem->attachElement("SecretionUnit",secretionConstUnit.toString());
    //	}

    //	// if(unitsElem->getFirstElement("SecretionOnContactUnit")){
    //	// unitsElem->getFirstElement("SecretionOnContactUnit")->updateElementValue(secretionConstUnit.toString());
    //	// }else{
    //	// unitsElem->attachElement("SecretionOnContactUnit",secretionConstUnit.toString());
    //	// }

    //	// if(unitsElem->getFirstElement("ConstantConcentrationUnit")){
    //	// unitsElem->getFirstElement("ConstantConcentrationUnit")->updateElementValue(secretionConstUnit.toString());
    //	// }else{
    //	// unitsElem->attachElement("ConstantConcentrationUnit",secretionConstUnit.toString());
    //	// }

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

    //	// if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
    //	// unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
    //	// }else{
    //	// unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
    //	// }

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

    manageSecretionInPythonVec.clear();

    CC3DXMLElementList diffFieldXMLVec = _xmlData->getElements("DiffusionField");
    for (unsigned int i = 0; i < diffFieldXMLVec.size(); ++i) {

        manageSecretionInPythonVec.push_back(false); // we set manage secretion in Python flag to False by default

        diffSecrFieldTuppleVec.push_back(SteadyStateDiffusionSecretionFieldTupple());
        DiffusionData &diffData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].diffData;
        SecretionData &secrData = diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size() - 1].secrData;

        if (diffFieldXMLVec[i]->findElement("ManageSecretionInPython")) {
            manageSecretionInPythonVec[i] = true;
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

        //boundary conditions parsing
        if (diffFieldXMLVec[i]->findElement("BoundaryConditions")) {
            bcSpecVec.push_back(BoundaryConditionSpecifier());
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

        }
    }
    if (_xmlData->findElement("Serialize")) {

        serializeFlag = true;
        if (_xmlData->getFirstElement("Serialize")->findAttribute("Frequency")) {
            serializeFrequency = _xmlData->getFirstElement("Serialize")->getAttributeAsUInt("Frequency");
        }
        CC3D_Log(LOG_DEBUG) <<"serialize Flag="<<serializeFlag;

    }

    if (_xmlData->findElement("ReadFromFile")) {
        readFromFileFlag = true;
        CC3D_Log(LOG_DEBUG) <<"readFromFileFlag="<<readFromFileFlag;
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
                diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j] = &SteadyStateDiffusionSolver2D::secreteSingleField;
                ++j;
            }
            // else if((*sitr)=="SecretionOnContact"){
            // diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&SteadyStateDiffusionSolver2D::secreteOnContactSingleField;
            // ++j;
            // }
            // else if((*sitr)=="ConstantConcentration"){
            // diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&SteadyStateDiffusionSolver2D::secreteConstantConcentrationSingleField;
            // ++j;
            // }

        }
    }
}

std::string SteadyStateDiffusionSolver2D::toString() {
    return "SteadyStateDiffusionSolver2D";
}


std::string SteadyStateDiffusionSolver2D::steerableName() {
    return toString();
}

