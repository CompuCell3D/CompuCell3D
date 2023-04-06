#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

using namespace std;


#include "FoamDataOutput.h"

FoamDataOutput::FoamDataOutput() :
        potts(0),
        neighborTrackerAccessorPtr(0),
        surFlag(false),
        volFlag(false),
        numNeighborsFlag(false),
        cellIDFlag(false) {}


void FoamDataOutput::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
    potts = _simulator->getPotts();
    cellInventoryPtr = &potts->getCellInventory();
    CC3DXMLElement *outputXMLElement = _xmlData->getFirstElement("Output");

    if (!outputXMLElement)
        throw CC3DException("You need to provide Output element to FoamDataOutput Steppable with at least file name");

    if (outputXMLElement) {
        if (outputXMLElement->findAttribute("FileName"))
            fileName = outputXMLElement->getAttribute("FileName");

        if (outputXMLElement->findAttribute("Volume"))
            volFlag = true;

        if (outputXMLElement->findAttribute("Surface"))
            surFlag = true;

        if (outputXMLElement->findAttribute("NumberOfNeighbors"))
            numNeighborsFlag = true;

        if (outputXMLElement->findAttribute("CellID"))
            numNeighborsFlag = cellIDFlag;
    }
}

void FoamDataOutput::extraInit(Simulator *simulator) {
    if (numNeighborsFlag) {
        bool pluginAlreadyRegisteredFlag;
        NeighborTrackerPlugin *neighborTrackerPluginPtr = (NeighborTrackerPlugin * )(
                Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag));
        if (!pluginAlreadyRegisteredFlag)
            neighborTrackerPluginPtr->init(simulator);
        if (!neighborTrackerPluginPtr) throw CC3DException("NeighborTracker plugin not initialized!");
        neighborTrackerAccessorPtr = neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();
        if (!neighborTrackerAccessorPtr) throw CC3DException("neighborAccessorPtr  not initialized!");
    }
}


void FoamDataOutput::start() {}

void FoamDataOutput::step(const unsigned int currentStep) {

	ostringstream str;
	str<<fileName<<"."<<currentStep;
	ofstream out(str.str().c_str());
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	std::set<NeighborSurfaceData > * neighborData;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        if (cellIDFlag)
            out << cell->id << "\t";

        if (volFlag)
            out << cell->volume << "\t";

        if (surFlag)
            out << cell->surface << "\t";

        if (numNeighborsFlag) {
            neighborData = &(neighborTrackerAccessorPtr->get(cell->extraAttribPtr)->cellNeighbors);
            out << neighborData->size() << "\t";
        }
        out << endl;
    }
}


std::string FoamDataOutput::toString() {
    return "FoamDataOutput";
}





