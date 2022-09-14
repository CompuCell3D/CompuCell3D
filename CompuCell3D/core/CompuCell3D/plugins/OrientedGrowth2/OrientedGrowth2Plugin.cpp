
#include <CompuCell3D/CC3D.h>        
using namespace CompuCell3D;

#include "OrientedGrowth2Plugin.h"
#include <math.h>

OrientedGrowth2Plugin::OrientedGrowth2Plugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0)
{}

OrientedGrowth2Plugin::~OrientedGrowth2Plugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
}

void OrientedGrowth2Plugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
    update(xmlData,true);
   
    potts->getCellFactoryGroupPtr()->registerClass(&orientedGrowth2DataAccessor);
    potts->registerEnergyFunctionWithName(this,"OrientedGrowth2");
        
    potts->registerStepper(this);
    
    simulator->registerSteerableObject(this);
}

void OrientedGrowth2Plugin::extraInit(Simulator *simulator){    
}

void OrientedGrowth2Plugin::step() {
    //Put your code here - it will be invoked after every succesful pixel copy and after all lattice monitor finished running    	
}

void OrientedGrowth2Plugin::setConstraintWidth(CellG *Cell, float _constraint){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_targetWidth = _constraint;
}
void OrientedGrowth2Plugin::setConstraintLength(CellG *Cell, float _constraint){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_targetLength = _constraint;
}
void OrientedGrowth2Plugin::setConstraintVolume(CellG *Cell, int _constraint){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_volume = _constraint;
}
void OrientedGrowth2Plugin::setApicalRadius(CellG *Cell, float _constraint){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_apicalRadius = _constraint;
}
void OrientedGrowth2Plugin::setBasalRadius(CellG *Cell, float _constraint){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_basalRadius = _constraint;
}
void OrientedGrowth2Plugin::setElongationAxis(CellG *Cell, float _elongX, float _elongY, float _elongZ){
    float magnitude = sqrt(pow(_elongX,2)+pow(_elongY,2)+pow(_elongZ,2));
    if (magnitude == 0){
        orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_enabled = false;
    }
    else{
        orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_x = (_elongX/magnitude);
        orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_y = (_elongY/magnitude);
        orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_z = (_elongZ/magnitude);
    }
}
void OrientedGrowth2Plugin::setElongationCOM(CellG *Cell, float _elongXCOM, float _elongYCOM, float _elongZCOM){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_xCOM = _elongXCOM;
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_yCOM = _elongYCOM;
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_zCOM = _elongZCOM;
}
void OrientedGrowth2Plugin::setElongationEnabled(CellG *Cell, bool _enabled){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_enabled = _enabled;
}
void OrientedGrowth2Plugin::setConstrictionEnabled(CellG *Cell, bool _enabled){
    orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_constricted = _enabled;
}

void OrientedGrowth2Plugin::updateElongationAxis(CellG *ogCell){
    //This plugin function updates the ElongationAxis and compartment targetVolume if apically constricted.
    float elongNormalX = 0, elongNormalY = 0, elongNormalZ = 0;
    float deltaX = 0, deltaY = 0, deltaZ = 0;
    float myTargetRadius = 0, myTargetHeight = 0;
    float api_rad = 0, bas_rad = 0, volumeRatio = 0, currentH = 0;
    float dotProduct = 0, deltaMag = 0, deltaVecAngle = 0;
    bool cell_enabled = false, constrict_enabled = false, apical_end = false, basal_end = false;
    CC3DCellList compartments;

    if (ogCell) { //quit if ogCell not set
        cell_enabled = getElongationEnabled(ogCell);
        constrict_enabled = getConstrictionEnabled(ogCell);
        if (cell_enabled) {
            compartments = potts->getCellInventory().getClusterInventory().getClusterCells(ogCell->clusterId);                
            if (constrict_enabled && compartments.size()==2 && compartments[0]->volume>0 && compartments[1]->volume>0) {
                //vector between cluster COMs
                elongNormalX = getElongationAxis_X(ogCell);
                elongNormalY = getElongationAxis_Y(ogCell);
                elongNormalZ = getElongationAxis_Z(ogCell);
                deltaX = compartments[0]->xCOM - compartments[1]->xCOM;
                deltaY = compartments[0]->yCOM - compartments[1]->yCOM;
                deltaZ = compartments[0]->zCOM - compartments[1]->zCOM;
                deltaMag = sqrt(pow(deltaX,2)+pow(deltaY,2)+pow(deltaZ,2));
                if (deltaMag > 0) {
                    //compare elong and delta unit vectors to find out which compartment is apical.
                    deltaX /= deltaMag;
                    deltaY /= deltaMag;
                    deltaZ /= deltaMag;
                    dotProduct = ((deltaX * elongNormalX) + (deltaY * elongNormalY) + (deltaZ * elongNormalZ));
                    deltaVecAngle = acos(dotProduct);
                     //std::cout <<  "\ndeltaMag = " << deltaMag << "  deltaVecAngle = " << deltaVecAngle*180.0/M_PI << "\n";
                    if (deltaVecAngle < M_PI_2) {
                        //Aligned
                        //std::cout <<  "AlignedAngle = " << deltaVecAngle*180.0/M_PI << "\n";
                        if (compartments[0]->id == ogCell->id) {
                            //std::cout <<  "\033[3;95mApical\033[0m";
                            apical_end = true;
                        } else if (compartments[1]->id == ogCell->id) {
                            //std::cout <<  "\033[3;36mBasal\033[0m";
                            basal_end = true;
                        } else {
                            std::cout <<  "\033[2;33mUnknownCompartment\033[0m\n";
                        }
                    } else {
                        //Compartments flipped from elongation vector
                        //std::cout <<  "\033[3;31mFlipped Angle = " << deltaVecAngle*180.0/M_PI << "\033[0m\n";
                        if (compartments[0]->id == ogCell->id) {
                            //std::cout <<  "\033[3;95mBasal\033[0m";
                            basal_end = true;
                        } else if (compartments[1]->id == ogCell->id) {
                            //std::cout <<  "\033[3;36mApical\033[0m";
                            apical_end = true;
                        } else {
                            std::cout <<  "\033[3;33mUnknownCompartment\033[0m\n";
                        }
                        deltaX = -deltaX;
                        deltaY = -deltaY;
                        deltaZ = -deltaZ;
                    }
                    if (apical_end || basal_end) {
                        //Parameters of cluster
                        myTargetHeight = getConstraintLength(ogCell)/2.0;

                        //get last basal cell COM, whether processing apical or basal cell, used for dampening changes    
                        api_rad = getApicalRadius(ogCell);
                        bas_rad = getBasalRadius(ogCell);
                        float mid_rad = (api_rad + bas_rad)/2.0;

                        if (apical_end) {
                            bas_rad = mid_rad;
                        } else {
                            api_rad = mid_rad;
                        }

                        setElongationAxis(ogCell, deltaX, deltaY, deltaZ);
                        float myVolume = (M_PI*myTargetHeight/3)*(pow(bas_rad,2)+bas_rad*api_rad+pow(api_rad,2));
                        setConstraintVolume(ogCell, (int)myVolume);
                        //To check for exploding volumes observed
                        //if (myVolume > 1000 || myVolume <= 0) {
                        //    std::cout <<  "\033[3;35mdeltaMag " << deltaMag << " myTargetHeight " << myTargetHeight << " myVolume " << myVolume << "\033[0m\n";
                        //    std::cout <<  "\033[3;35mbas_rad " << bas_rad << " api_rad " << api_rad << "\033[0m\n";
                        //}
                    }
                }
            }
        }
    }
}


float OrientedGrowth2Plugin::getConstraintWidth(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_targetWidth;
}
float OrientedGrowth2Plugin::getConstraintLength(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_targetLength;
}
int OrientedGrowth2Plugin::getConstraintVolume(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_volume;
}
float OrientedGrowth2Plugin::getApicalRadius(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_apicalRadius;
}
float OrientedGrowth2Plugin::getBasalRadius(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_basalRadius;
}
float OrientedGrowth2Plugin::getElongationAxis_X(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_x;
}
float OrientedGrowth2Plugin::getElongationAxis_Y(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_y;
}
float OrientedGrowth2Plugin::getElongationAxis_Z(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_z;
}
float OrientedGrowth2Plugin::getElongationCOM_X(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_xCOM;
}
float OrientedGrowth2Plugin::getElongationCOM_Y(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_yCOM;
}
float OrientedGrowth2Plugin::getElongationCOM_Z(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_zCOM;
}
bool OrientedGrowth2Plugin::getElongationEnabled(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_enabled;
}
bool OrientedGrowth2Plugin::getConstrictionEnabled(CellG *Cell){
    return orientedGrowth2DataAccessor.get(Cell->extraAttribPtr)->elong_constricted;
}

double OrientedGrowth2Plugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	
    //This plugin function atypically uses a stored COM so that changes to the cell COM from the COM
    //plugin do not change the reference for all the cell's pixels this MCS.
    double energy = 0, energyExp = 0, energyFalloff = 0, offset = 0;
    float elongNormalX = 0, elongNormalY = 0, elongNormalZ = 0;
    float changeVecX = 0, changeVecY = 0, changeVecZ = 0;
    float deltaX = 0, deltaY = 0, deltaZ = 0;
    float myXCOM = 0, myYCOM = 0, myZCOM = 0;
    float myTargetRadius = 0, myTargetHeight = 0;
    float api_rad = 0, bas_rad = 0, apiCOM =0, basCOM = 0, myCOM = 0;
    float dotProduct = 0, deltaMag = 0, deltaVecAngle = 0;
    bool cell_enabled = false, constrict_enabled = false, apical_end = false, basal_end = false;
    CC3DCellList compartments;
    CellG *ogCell, *refCell;

    if ((oldCell && newCell) || (!oldCell && !newCell)) { //quit if both or neither pointer set
        return 0.0;
    } else {
        if (oldCell) {
            ogCell = const_cast<CellG*>(oldCell);
        } else {
            ogCell = const_cast<CellG*>(newCell);
        }
        cell_enabled = getElongationEnabled(ogCell);
        if (cell_enabled) {
            elongNormalX = getElongationAxis_X(ogCell);
            elongNormalY = getElongationAxis_Y(ogCell);
            elongNormalZ = getElongationAxis_Z(ogCell);
            myXCOM = ogCell->xCOM;
            myYCOM = ogCell->yCOM;
            myZCOM = ogCell->zCOM;
            constrict_enabled = getConstrictionEnabled(ogCell);
            compartments = potts->getCellInventory().getClusterInventory().getClusterCells(ogCell->clusterId);                
            if (constrict_enabled && compartments.size()==2 && compartments[0]->volume>0 && compartments[1]->volume>0) {
            	//get orientation vector between cluster COMs because of uknown compartment order and for shaping
                deltaX = compartments[0]->xCOM - compartments[1]->xCOM;
                deltaY = compartments[0]->yCOM - compartments[1]->yCOM;
                deltaZ = compartments[0]->zCOM - compartments[1]->zCOM;
                deltaMag = sqrt(pow(deltaX,2)+pow(deltaY,2)+pow(deltaZ,2));
                if (deltaMag > 0) {
                	//compare elong and delta unit vectors to find out which compartment is apical.
                	deltaX /= deltaMag;
                    deltaY /= deltaMag;
                    deltaZ /= deltaMag;
                    dotProduct = ((deltaX * elongNormalX) + (deltaY * elongNormalY) + (deltaZ * elongNormalZ));
                	deltaVecAngle = acos(dotProduct);
                    //std::cout <<  "\ndeltaMag = " << deltaMag << "  deltaVecAngle = " << deltaVecAngle*180.0/M_PI << "\n";
                	if (deltaVecAngle < M_PI_2) {
                        //Aligned
                        //std::cout <<  "AlignedAngle = " << deltaVecAngle*180.0/M_PI << "\n";
                        if (compartments[0]->id == ogCell->id) {
                            //std::cout <<  "\033[3;95mApical\033[0m";
                            apical_end = true;
                        } else if (compartments[1]->id == ogCell->id) {
                            //std::cout <<  "\033[3;36mBasal\033[0m";
                            basal_end = true;
                        } else {
                            std::cout <<  "\033[2;33mUnknownCompartment\033[0m\n";
                        }
                        refCell = compartments[1];
                    } else {
                        //Compartments flipped from elongation vector
                        //std::cout <<  "\033[3;31mFlipped Angle = " << deltaVecAngle*180.0/M_PI << "\033[0m\n";
                        if (compartments[0]->id == ogCell->id) {
                            //std::cout <<  "\033[3;95mBasal\033[0m";
                            basal_end = true;
                        } else if (compartments[1]->id == ogCell->id) {
                            //std::cout <<  "\033[3;36mApical\033[0m";
                            apical_end = true;
                        } else {
                            std::cout <<  "\033[3;33mUnknownCompartment\033[0m\n";
                        }
                        deltaX = -deltaX;
                        deltaY = -deltaY;
                        deltaZ = -deltaZ;
                        refCell = compartments[0];
                    }
                    if (apical_end || basal_end) {
                        //Parameters of cluster, frustrum of right circular cone COM
						myTargetHeight = getConstraintLength(ogCell)/2.0;
                        api_rad = getApicalRadius(ogCell);
                        bas_rad = getBasalRadius(ogCell);
                        float mid_rad = (api_rad + bas_rad)/2.0;
                        apiCOM = (myTargetHeight/4)*(pow(mid_rad,2)+2*mid_rad*api_rad+3*pow(api_rad,2))/(pow(mid_rad,2)+mid_rad*api_rad+pow(api_rad,2));
                        basCOM = (myTargetHeight/4)*(pow(bas_rad,2)+2*bas_rad*mid_rad+3*pow(mid_rad,2))/(pow(bas_rad,2)+bas_rad*mid_rad+pow(mid_rad,2));

                        if (apical_end) {
                            bas_rad = mid_rad;
                            myCOM = apiCOM;
                            float deltaCOM = myTargetHeight + apiCOM - basCOM;
                            myXCOM = refCell->xCOM + deltaCOM*deltaX;
                            myYCOM = refCell->yCOM + deltaCOM*deltaY;
                            myZCOM = refCell->zCOM + deltaCOM*deltaZ;
                        } else {
                            api_rad = mid_rad;
                            myCOM = basCOM;
                        }
                    }
                }
            }
            changeVecX = pt.x - myXCOM;
            changeVecY = pt.y - myYCOM;
            changeVecZ = pt.z - myZCOM;                     
	    }
    }
    if (cell_enabled) {
		float magnitude = sqrt(pow(changeVecX,2)+pow(changeVecY,2)+pow(changeVecZ,2));
    	if (magnitude > 0) {
        	changeVecX/=magnitude;
        	changeVecY/=magnitude;
        	changeVecZ/=magnitude;
            if (constrict_enabled) {
                dotProduct = (changeVecX * deltaX) + (changeVecY * deltaY) + (changeVecZ * deltaZ);
            } else {
                dotProduct = (changeVecX * elongNormalX) + (changeVecY * elongNormalY) + (changeVecZ * elongNormalZ);
            }
			float changeVecAngle = acos(dotProduct);
			float pointRadius = magnitude * sin(changeVecAngle);        	
            if (constrict_enabled) {  //compute cone myTargetRadius
                float pointHeight = magnitude * dotProduct + myCOM;
                myTargetRadius = (pointHeight*(api_rad - bas_rad)/myTargetHeight) + bas_rad;
                //std::cout <<  "\033[2;3api_rad/bas_rad/myTargetHeight = " << api_rad << "/" << bas_rad << "/" << myTargetHeight << "\033[0m\n";
                //std::cout <<  "\033[2;33mpointRadius/pointHeight/myTargetRadius = " << pointRadius << "/" << pointHeight << "/" << myTargetRadius << "\033[0m\n";
                if ((pointHeight > myTargetHeight) || (pointHeight < 0)) {
                    pointRadius = 2.0 * myTargetRadius;  //put out of bounds above and below compartment
                }
            } else {
                myTargetRadius = getConstraintWidth(ogCell)/2.0;
                //std::cout <<  "\033[2;33mpointRadius/myTargetRadius = " << pointRadius << "/" << myTargetRadius << "\033[0m\n";
            }
			if (pointRadius > myTargetRadius) {
                offset = pointRadius - myTargetRadius;
                //std::cout <<  "\033[2;35moffset = " << offset << "\033[0m\n";
                if (xml_energy_falloff > 0 ) {
                	energyFalloff = xml_energy_penalty * exp(-5.0 * offset/xml_energy_falloff); //Scaled for voxel widths
                	//https://en.wikipedia.org/wiki/Exponential_decay#/media/File:Plot-exponential-decay.svg
                }
                energyExp = xml_energy_penalty - energyFalloff;
                //std::cout <<  "Offset  = " << offset << "  |  Penalty = " << energyExp << "  |  PenaltyTemp = " << energyFalloff << "\n";
			}
        	if (newCell) {
        		energy += energyExp;
        	} else if (oldCell) {
				energy -= energyExp;
        	}
        }
    }
    return energy;
}


void OrientedGrowth2Plugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
    set<unsigned char> cellTypesSet;

    CC3DXMLElement * myElementOne = xmlData->getFirstElement("Penalty");
    if(myElementOne){
        xml_energy_penalty = myElementOne->getDouble();
    }else{
        xml_energy_penalty = 99999;
    }
    
    CC3DXMLElement * myElementTwo = xmlData->getFirstElement("Falloff");
    if(myElementTwo){
        xml_energy_falloff = myElementTwo->getDouble();
    }else{
        xml_energy_falloff = 1;
    }
    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();
}


std::string OrientedGrowth2Plugin::toString(){
    return "OrientedGrowth2";
}


std::string OrientedGrowth2Plugin::steerableName(){
    return toString();
}
