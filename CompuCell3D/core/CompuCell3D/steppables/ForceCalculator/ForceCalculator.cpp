



#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;



using namespace std;




#include "ForceCalculator.h"

ForceCalculator::ForceCalculator() : 
	cellFieldG(0),
	sim(0),
	potts(0),
	xmlData(0),
	boundaryStrategy(0),
	automaton(0),
	cellInventoryPtr(0),
	neighborOrder(1)
{}



ForceCalculator::~ForceCalculator() {

	deleteForceField();

}





void ForceCalculator::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData = _xmlData;

	potts = simulator->getPotts();

	cellInventoryPtr = &potts->getCellInventory();

	sim = simulator;

	cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

	fieldDim = cellFieldG->getDim();

	bool pluginAlreadyRegisteredFlag;

	// Get boundary pixel tracker plugin
	boundaryTrackerPlugin = (BoundaryPixelTrackerPlugin*)Simulator::pluginManager.get("BoundaryPixelTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *BoundaryPixelTrackerXML = simulator->getCC3DModuleData("Plugin", "BoundaryPixelTracker");
		boundaryTrackerPlugin->init(simulator, BoundaryPixelTrackerXML);
	}


	simulator->registerSteerableObject(this);

	if (fieldDim.z == 1) maxDimIdx = 2;
	else maxDimIdx = 3;

	InitializeForceField(fieldDim);

	update(_xmlData, true);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void ForceCalculator::extraInit(Simulator *simulator){

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ForceCalculator::start(){

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void ForceCalculator::step(const unsigned int currentStep){

    //right minus left is -F_x

	CellInventory::cellInventoryIterator cInvItr;

	CellG * cell=0;
	Point3D currentPoint;
	Point3D shiftedPoint_positive = currentPoint;
	Point3D shiftedPoint_negative = currentPoint;
	CellG* shiftedPoint_positive_cell = 0;
	CellG* shiftedPoint_negative_cell = 0;
    
    std::vector<Point3D > offsetPoint_positive = std::vector<Point3D >(3);
    std::vector<Point3D > offsetPoint_negative = std::vector<Point3D >(3);
    
    // dim 0: x
    // dim 1: y
    // dim 2: z
    offsetPoint_positive[0] = Point3D(1, 0, 0);
    offsetPoint_positive[1] = Point3D(0, 1, 0);
    offsetPoint_positive[2] = Point3D(0, 0, 1);
    
    offsetPoint_negative[0] = Point3D(-1, 0, 0);
    offsetPoint_negative[1] = Point3D(0, -1, 0);
    offsetPoint_negative[2] = Point3D(0, 0, -1);

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd();++cInvItr)

	{

		cell = cellInventoryPtr->getCell(cInvItr);

		std::set<BoundaryPixelTrackerData > *boundarySet = boundaryTrackerPlugin->getPixelSetForNeighborOrderPtr(cell, neighborOrder);
		for (std::set<BoundaryPixelTrackerData >::iterator bInvItr = boundarySet->begin(); bInvItr != boundarySet->end(); ++bInvItr)
		{
			currentPoint = bInvItr->pixel;
            
            std::vector<float> force = std::vector<float>(3, 0.0);
            
            for (int dimIdx = 0; dimIdx < maxDimIdx; ++dimIdx){
                
                shiftedPoint_positive = currentPoint + offsetPoint_positive[dimIdx];
                shiftedPoint_negative = currentPoint + offsetPoint_negative[dimIdx];
                
                // Do lattice checks here!

                shiftedPoint_positive_cell = cellFieldG->getQuick(const_cast<Point3D&>(shiftedPoint_positive));
                shiftedPoint_negative_cell = cellFieldG->getQuick(const_cast<Point3D&>(shiftedPoint_negative));

                double delta_H = 0.0;
			
                if (shiftedPoint_positive_cell != cell && shiftedPoint_negative_cell == cell)//forward boundary
                {
                    delta_H = potts->changeEnergy(shiftedPoint_positive, cell, shiftedPoint_positive_cell) - potts->changeEnergy(currentPoint, shiftedPoint_positive_cell, cell);// NEGATIVE OF THIS IS FORCE
                }

                else if (shiftedPoint_positive_cell == cell && shiftedPoint_negative_cell != cell)//backward boundary
                {
                    delta_H =  potts->changeEnergy(currentPoint, shiftedPoint_negative_cell, cell) - potts->changeEnergy(shiftedPoint_negative, cell, shiftedPoint_negative_cell);
                }
                
                else if (shiftedPoint_positive_cell != cell && shiftedPoint_negative_cell != cell)//both boundaries
                {
                    delta_H = potts->changeEnergy(shiftedPoint_positive, cell, shiftedPoint_positive_cell) - potts->changeEnergy(shiftedPoint_negative, cell, shiftedPoint_negative_cell);
                }

                force[dimIdx] = -float(delta_H / double(2.0));
            }
            
            // Store vector value in field
			ForceField->get(currentPoint)->setForce(force);

		}

	}



}





void ForceCalculator::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

    boundaryStrategy=BoundaryStrategy::getInstance();

}



std::string ForceCalculator::toString(){

   return "ForceCalculator";

}



std::string ForceCalculator::steerableName(){

   return toString();

}

float ForceCalculator::getForceComponent(Point3D &pt, unsigned int compIdx) {
	std::vector<float> force = ForceField->get(pt)->getForce();
	return force[compIdx];
}

// in case of lattice resize, deallocate this old field, and then create a new field
void ForceCalculator::InitializeForceField(Dim3D fieldDim) {
	ForceField = new Field3DImpl<ForceFieldData* >(fieldDim, 0);
	Point3D pt;
	ForceFieldData* ForceFieldDataLocal;
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);
				ForceFieldDataLocal = new ForceFieldData();
				ForceField->set(pt, ForceFieldDataLocal);
			}
}




        

void ForceCalculator::deleteForceField() {
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				Point3D pt = Point3D(x, y, z);
				ForceFieldData *d = ForceField->get(pt);
				delete d;
			}

	delete ForceField;
	ForceField = 0;//make other refs null ppinters
}