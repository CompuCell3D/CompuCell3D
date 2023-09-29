

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "PolygonFieldInitializer.h"
#include <Logger/CC3DLogger.h>
PolygonFieldInitializer::PolygonFieldInitializer() :
        potts(0), sim(0) {}


void PolygonFieldInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    sim = simulator;
    potts = simulator->getPotts();
    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellFieldG) throw CC3DException("initField() Cell field G cannot be null!");
    Dim3D dim = cellFieldG->getDim();


    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    //clearing vector storing PolygonFieldInitializerData (region definitions)
    initDataVec.clear();

    CC3DXMLElementList regionVec = _xmlData->getElements("Region");
    if (regionVec.size() > 0) {
        for (int i = 0; i < regionVec.size(); ++i) {
            PolygonFieldInitializerData initData;

            if (regionVec[i]->findElement("Gap"))
                initData.gap = regionVec[i]->getFirstElement("Gap")->getUInt();
            if (regionVec[i]->findElement("Width"))
                initData.width = regionVec[i]->getFirstElement("Width")->getUInt();

            if (!regionVec[i]->getFirstElement("Types"))
                throw CC3DException(
                        "PolygonInitializer requires Types element inside Region section.See manual for details.");
            initData.typeNamesString = regionVec[i]->getFirstElement("Types")->getText();
            parseStringIntoList(initData.typeNamesString, initData.typeNames, ",");


            //Parsing Edges groups
            CC3DXMLElementList edgesGroupsXMlList = regionVec[i]->getElements("EdgeList");
            for (int i = 0; i < edgesGroupsXMlList.size(); i++) { 
                cerr << "Found Edges Group" << endl;
                //Parsing Edge elements
                CC3DXMLElementList edgesXMlList = edgesGroupsXMlList[i]->getElements("Edge");
                for (int j = 0; j < edgesXMlList.size(); j++) { 
                    CC3DXMLElement * edge = edgesXMlList[j];
                    
                    if (edge->findElement("From") && edge->findElement("To")) {
                        cerr << "Found FromTo Edge" << endl;

                        CC3DXMLElement * srcPointXML = edge->getFirstElement("From");
                        Point3D src = Point3D();
                        src.x = srcPointXML->getAttributeAsUInt("x");
                        src.y = srcPointXML->getAttributeAsUInt("y");
                        initData.srcPoints.push_back(src);

                        CC3DXMLElement * dstPointXML = edge->getFirstElement("To");
                        Point3D dst = Point3D();
                        //TODO skip z if not present
                        dst.x = dstPointXML->getAttributeAsUInt("x");
                        dst.y = dstPointXML->getAttributeAsUInt("y");
                        initData.dstPoints.push_back(dst);
                    }
                    else {
                        CC3D_Log(LOG_TRACE) << "Malformed XML element 'Edge'" << endl; 
                    }
                }
            }

            initDataVec.push_back(initData);
        }
    }
    else {
        cerr << "failed if (regionVec.size() > 0)" << endl;
    }
}

void PolygonFieldInitializer::layOutCells(const PolygonFieldInitializerData &_initData) { 
    //TEMP: print edges
    std::vector <Point3D> srcPoints  = _initData.srcPoints;
    std::vector <Point3D> dstPoints  = _initData.dstPoints;
    for (int i = 0; i < srcPoints.size(); i++) { 
        cerr << "Edge i=" << to_string(i) << endl;
        cerr << "From (" << srcPoints[i].x << ", " << srcPoints[i].y << ", "  << srcPoints[i].z << ") ";
        cerr << "To (" << dstPoints[i].x  << ", " << dstPoints[i].y << ", "  << dstPoints[i].z << ") " << endl;
    }

    
}

unsigned char PolygonFieldInitializer::initCellType(const PolygonFieldInitializerData &_initData) {
    Automaton *automaton = potts->getAutomaton();
    if (_initData.typeNames.size() == 0) {//by default each newly created type will be 1
        return 1;
    } else { //user has specified more than one cell type - will pick randomly the type
        RandomNumberGenerator *randGen = sim->getRandomNumberGeneratorInstance();
        int index = randGen->getInteger(0, _initData.typeNames.size() - 1);


        return automaton->getTypeId(_initData.typeNames[index]);
    }

}

void PolygonFieldInitializer::start() {
    if (sim->getRestartEnabled()) {
        return;  // we will not initialize cells if restart flag is on
    }
    CC3D_Log(LOG_DEBUG) << "INSIDE START";

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");
    Dim3D dim = cellField->getDim();

    cerr << "initDataVec.size() = " << initDataVec.size() << endl;
    if (initDataVec.size() != 0) {
        for (int i = 0; i < initDataVec.size(); ++i) {
            layOutCells(initDataVec[i]);
        }
    }

}

std::string PolygonFieldInitializer::steerableName() {
    return toString();
}

std::string PolygonFieldInitializer::toString() {
    return "PolygonInitializer";
}