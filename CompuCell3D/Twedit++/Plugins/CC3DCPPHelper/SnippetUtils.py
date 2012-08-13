class SnippetUtils:
    def __init__(self):
        self.snippetDict={}
        
        self.initCodeSnippets()
        
    def getCodeSnippetsDict(self):
        return self.snippetDict
        
    def initCodeSnippets(self):

        self.snippetDict["Visit All Cells"]="""
CellInventory::cellInventoryIterator cInvItr;
CellG * cell;
std::set<NeighborSurfaceData > * neighborData;

for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
{
    //Put your code here
    cerr<<"cell id="<<cell->id<<endl;
}
"""
        self.snippetDict["Visit Pixel Neighbors"]="""
int maxNeighborIndexLocal=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder); // this line usually sits in the init or extra init function
Neighbor neighbor;
CellG * nCell;
WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG(); // you may store WatchableField3D<CellG *> *fieldG as a class member

Point3D px;

for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexLocal ; ++nIdx ){
    neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(px),nIdx);
    if(!neighbor.distance){
        //if distance is 0 then the neighbor returned is invalid
        continue;
    }
    nCell=fieldG->get(neighbor.pt);

    if (!nCell) {
        cerr<<"neighbor pixel cell id="<<nCell->id<<endl;
    }
        
}
"""

        self.snippetDict["Visit Compartments of a Cluster"]="""
CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(CELL->clusterId);
for (int i =0 ; i< compartments.size() ; ++i){
    cerr<<"compartment id="<<compartments[i]->id<<endl; 
}
"""

        self.snippetDict["Module Setup Preload Plugin"]="""
//This code is usually called from   init finction      
bool pluginAlreadyRegisteredFlag;
Plugin *plugin=Simulator::pluginManager.get("PLUGIN_NAME",&pluginAlreadyRegisteredFlag); //this will load PLUGIN_NAME plugin if it is not already loaded
if(!pluginAlreadyRegisteredFlag)
    plugin->init(simulator);
"""

#----------------------Includes
        self.snippetDict["Include Cell/Cluster Inventory"]="""
#include <CompuCell3D/Potts3D/CellInventory.h>
"""
        self.snippetDict["Include Plugin Files"]="""
#include <CompuCell3D/plugins/PLUGIN_NAME/PLUGIN_FILE.h>
"""

        self.snippetDict["Include Point3D/Dim3D"]="""
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
"""
        self.snippetDict["Include Field3D"]="""
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
"""

        self.snippetDict["Include Boundary Type Definitions"]="""
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>
"""

        self.snippetDict["Include Boundary Strategy"]="""
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
"""

        self.snippetDict["Include Automaton"]="""
#include <CompuCell3D/Automaton/Automaton.h>
"""

        self.snippetDict["Include Potts3D"]="""
#include <CompuCell3D/Potts3D/Potts3D.h>
"""

        self.snippetDict["Include Simulator"]="""
#include <CompuCell3D/Simulator.h>
"""

        self.snippetDict["Include Vector3"]="""
#include <PublicUtilities/Vector3.h>
"""

        self.snippetDict["Include StringUtilis"]="""
#include <PublicUtilities/StringUtils.h>
"""

        self.snippetDict["Include NumericalUtilis"]="""
#include <PublicUtilities/NumericalUtils.h>
"""

#----------------------Getters
        self.snippetDict["Get ExtraAttribute"]="""
ACCESSOR_NAME.get(CELL->extraAttribPtr)->ATTRIBUTE_COMPONENT
"""


# ---------------------XML Utils
        self.snippetDict["XML Utils Find Element"]="""
bool flag=_xmlData->findElement("ELEMENT_NAME");
"""
        self.snippetDict["XML Utils Get Element"]="""
CC3DXMLElement *elem=_xmlData->getFirstElement("ELEMENT_NAME");
"""
        self.snippetDict["XML Utils Get Element As Double"]="""
double val=_xmlData->getFirstElement("ELEMENT_NAME")->getDouble();
"""
        self.snippetDict["XML Utils Get Element As Int"]="""
int val=_xmlData->getFirstElement("ELEMENT_NAME")->getInt();
"""
        self.snippetDict["XML Utils Get Element As UInt"]="""
unsigned int val=_xmlData->getFirstElement("ELEMENT_NAME")->getUInt();
"""
        self.snippetDict["XML Utils Get Element As Short"]="""
short val=_xmlData->getFirstElement("ELEMENT_NAME")->getShort();
"""
        self.snippetDict["XML Utils Get Element As UShort"]="""
unsigned short val=_xmlData->getFirstElement("ELEMENT_NAME")->getUShort();
"""
        self.snippetDict["XML Utils Get Element As Bool"]="""
bool val=_xmlData->getFirstElement("ELEMENT_NAME")->getBool();
"""
        self.snippetDict["XML Utils Get Element As Text"]="""
std::string val=_xmlData->getFirstElement("ELEMENT_NAME")->getText();
"""
        self.snippetDict["XML Utils Find Attribute"]="""
bool flag=_xmlData->findAttribute("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As Text"]="""
std::string val=_xmlData->getAttribute("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As Double"]="""
double val=_xmlData->getAttributeAsDouble("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As Int"]="""
int val=_xmlData->getAttributeAsInt("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As UInt"]="""
unsigned int val=_xmlData->getAttributeAsUInt("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As Short"]="""
short val=_xmlData->getAttributeAsShort("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As UShort"]="""
unsigned short val=_xmlData->getAttributeAsUShort("ATTR_NAME");
"""
        self.snippetDict["XML Utils Get Attribute As Bool"]="""
bool val=_xmlData->getAttributeAsBool("ATTR_NAME");
"""
        self.snippetDict["XML Utils Process List of Elements"]="""
CC3DXMLElementList elemVec=_xmlData->getElements("ELEMENT_NAME");
for (int i = 0 ; i<elemVec.size(); ++i){
    //Put your code here
}
"""


