//https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "TubeFieldInitializer.h"
#include <Logger/CC3DLogger.h>
#include <limits>
TubeFieldInitializer::TubeFieldInitializer() :
        potts(0), sim(0) {}


void TubeFieldInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
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

    //clearing vector storing TubeFieldInitializerData (region definitions)
    initDataVec.clear();

    CC3DXMLElementList regionVec = _xmlData->getElements("Region");
    if (regionVec.size() > 0) {
        for (int i = 0; i < regionVec.size(); ++i) {
            TubeFieldInitializerData initData;

            if (regionVec[i]->findElement("Gap"))
                initData.gap = regionVec[i]->getFirstElement("Gap")->getUInt();
            if (regionVec[i]->findElement("Width"))
                initData.width = regionVec[i]->getFirstElement("Width")->getUInt();

            if (!regionVec[i]->getFirstElement("Types"))
                throw CC3DException(
                        "TubeInitializer requires Types element inside Region section.See manual for details.");
            initData.typeNamesString = regionVec[i]->getFirstElement("Types")->getText();
            parseStringIntoList(initData.typeNamesString, initData.typeNames, ",");

            if (regionVec[i]->findElement("InnerRadius"))
                initData.innerRadius = regionVec[i]->getFirstElement("InnerRadius")->getUInt();
            
            if (regionVec[i]->findElement("OuterRadius"))
                initData.outerRadius = regionVec[i]->getFirstElement("OuterRadius")->getUInt();
            
            if (regionVec[i]->findElement("Extrude")) {
                CC3DXMLElement * extrudeXML = regionVec[i]->getFirstElement("Extrude");
                if (extrudeXML->findElement("From") && extrudeXML->findElement("To")) {
                    CC3DXMLElement * fromPointXML = extrudeXML->getFirstElement("From");
                    initData.fromPoint.x = fromPointXML->getAttributeAsUInt("x");
                    initData.fromPoint.y = fromPointXML->getAttributeAsUInt("y");
                    initData.fromPoint.z = fromPointXML->getAttributeAsUInt("z");

                    CC3DXMLElement * toPointXML = extrudeXML->getFirstElement("To");
                    initData.toPoint.x = toPointXML->getAttributeAsUInt("x");
                    initData.toPoint.y = toPointXML->getAttributeAsUInt("y");
                    initData.toPoint.z = toPointXML->getAttributeAsUInt("z");
                }
                else {
                    throw CC3DException("The 'Extrude' XML tag requires two tags within it: 'From' and 'To'");
                }
            }
            else {
                throw CC3DException("TubeInitializer requires an XML element named 'Extrude'");
            }

            initDataVec.push_back(initData);
        }
    }
    else {
        cerr << "failed if (regionVec.size() > 0)" << endl;
    }
}

Dim3D TubeFieldInitializer::getTubeDimensions(const Dim3D &dim, int size) {
    Dim3D itDim;

    itDim.x = dim.x / size;
    if (dim.x % size) 
        itDim.x += 1;

    itDim.y = dim.y / size;
    if (dim.y % size) 
        itDim.y += 1;

    itDim.z = dim.z / size;
    if (dim.z % size) 
        itDim.z += 1;

    return itDim;

}

double TubeFieldInitializer::distance(double ax, double ay, double az, double bx, double by, double bz) {
    return sqrt((double) (ax - bx) * (ax - bx) +
                (double) (ay - by) * (ay - by) +
                (double) (az - bz) * (az - bz));
}




//Subtract 2 points
Point3D TubeFieldInitializer::subtract(Point3D p1, Point3D p2) {
    int x1 = p1.x - p2.x;
    int y1 = p1.y - p2.y;
    int z1 = p1.z - p2.z;

    return Point3D(x1, y1, z1);
}

//Dot product of 2 points
int TubeFieldInitializer::dotProduct(Point3D p1, Point3D p2) {
    int x1 = p1.x * p2.x;
    int y1 = p1.y * p2.y;
    int z1 = p1.z * p2.z;

    return x1 + y1 + z1;
}

//Cross product of 2 points
Point3D TubeFieldInitializer::crossProduct(Point3D p1, Point3D p2) {
    int x1 = p1.y * p2.z - p1.z * p2.y;
    int y1 = p1.z * p2.x - p1.x * p2.z;
    int z1 = p1.x * p2.y - p1.y * p2.x;
    
    return Point3D(x1, y1, z1);
}

float TubeFieldInitializer::magnitude(Point3D p) {
    return (float) sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
}

/**
Calculates the shortest distance from a point to a line in 3D.
*/
float TubeFieldInitializer::shortDistance(Point3D line_point1,
                                    Point3D line_point2,
                                    Point3D point)
{
    Point3D AB = subtract(line_point2, line_point1);
    Point3D AC = subtract(point, line_point1);
    float area = magnitude( (crossProduct(AB, AC)) );
    float CD = area / magnitude(AB);
    return CD;
}




void TubeFieldInitializer::layOutCells(const TubeFieldInitializerData &_initData) { 

    //TEMP
    cerr << "FromPt: " << _initData.fromPoint.x << ", " << _initData.fromPoint.y << ", " << _initData.fromPoint.z << endl;
    cerr << "ToPt: " << _initData.toPoint.x << ", " << _initData.toPoint.y << ", " << _initData.toPoint.z << endl;


    int size = _initData.gap + _initData.width;
    int cellWidth = _initData.width;

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellField->getDim();

    cerr << "dim:" << dim.x << ", " << dim.y << ", "  << dim.z << endl; 
	Dim3D itDim = getTubeDimensions(dim, size);
    cerr << "itDim:" << itDim.x << ", " << itDim.y << ", "  << itDim.z << endl;
	CC3D_Log(LOG_DEBUG) << "itDim="<<itDim;

    Point3D pt;
    Point3D cellPt;
    CellG *cell;

    double tubeLength = distance(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
            _initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z);

    for (int z = 0; z < itDim.z; z++)
        for (int y = 0; y < itDim.y; y++)
            for (int x = 0; x < itDim.x; x++) {
                pt.x = x * size;
                pt.y = y * size;
                pt.z = z * size;

                double dist = shortDistance(_initData.fromPoint, _initData.toPoint, pt);
                if (dist > _initData.outerRadius) {
                    continue;
                }

                double p1Dist = distance(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
                        pt.x, pt.y, pt.z);
                double p2Dist = distance(_initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z,
                        pt.x, pt.y, pt.z);

                double hypotenuse = sqrt( pow(_initData.outerRadius, 2) + pow(tubeLength, 2) );
                double distanceFromFace = max(p1Dist, p2Dist);
                if (distanceFromFace > hypotenuse) {
                    continue;
                }


                if (BoundaryStrategy::getInstance()->isValid(pt)) {
                    cell = potts->createCellG(pt);
                    cell->type = initCellType(_initData);
                    potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                    //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                    // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                    //inventory unless you call steppers(VolumeTrackerPlugin) explicitly


                } else {
                    continue;
                }

                for (cellPt.z = pt.z; cellPt.z < pt.z + cellWidth &&
                                      cellPt.z < dim.z; cellPt.z++)
                    for (cellPt.y = pt.y; cellPt.y < pt.y + cellWidth &&
                                          cellPt.y < dim.y; cellPt.y++)
                        for (cellPt.x = pt.x; cellPt.x < pt.x + cellWidth &&
                                              cellPt.x < dim.x; cellPt.x++) {

                            if (BoundaryStrategy::getInstance()->isValid(pt))
                                cellField->set(cellPt, cell);

                        }
                potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

            }


}

unsigned char TubeFieldInitializer::initCellType(const TubeFieldInitializerData &_initData) {
    Automaton *automaton = potts->getAutomaton();
    if (_initData.typeNames.size() == 0) {//by default each newly created type will be 1
        return 1;
    } else { //user has specified more than one cell type - will pick randomly the type
        RandomNumberGenerator *randGen = sim->getRandomNumberGeneratorInstance();
        int index = randGen->getInteger(0, _initData.typeNames.size() - 1);


        return automaton->getTypeId(_initData.typeNames[index]);
    }

}

void TubeFieldInitializer::start() {
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

std::string TubeFieldInitializer::steerableName() {
    return toString();
}

std::string TubeFieldInitializer::toString() {
    return "TubeInitializer";
}