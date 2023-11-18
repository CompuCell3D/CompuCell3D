/**
The checkInside algorithm is from: https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
Related functions adapted from GeeksForGeeks:
    - checkInside
    - onLine
    - direction
    - isIntersect
*/

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "PolygonFieldInitializer.h"
#include <Logger/CC3DLogger.h>
#include <limits>
#include <Utils/Coordinates3D.h>

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
                //Parsing Edge elements
                CC3DXMLElementList edgesXMlList = edgesGroupsXMlList[i]->getElements("Edge");
                for (int j = 0; j < edgesXMlList.size(); j++) { 
                    CC3DXMLElement * edge = edgesXMlList[j];
                    
                    if (edge->findElement("From") && edge->findElement("To")) {
                        CC3DXMLElement * srcPointXML = edge->getFirstElement("From");
                        Coordinates3D<double> src = Coordinates3D<double>();
                        src.x = srcPointXML->getAttributeAsDouble("x");
                        src.y = srcPointXML->getAttributeAsDouble("y");
                        initData.srcPoints.push_back(src);

                        CC3DXMLElement * dstPointXML = edge->getFirstElement("To");
                        Coordinates3D<double> dst = Coordinates3D<double>();
                        dst.x = dstPointXML->getAttributeAsDouble("x");
                        dst.y = dstPointXML->getAttributeAsDouble("y");
                        initData.dstPoints.push_back(dst);
                    }
                    else {
                        throw CC3DException("Malformed XML element 'Edge'");
                    }
                }
            }

            if (initData.srcPoints.size() < 3 || initData.dstPoints.size() < 3) {
                throw CC3DException("PolygonInitializer requires at least 3 edges");
            }
            if (initData.srcPoints.size() != initData.dstPoints.size()) {
                throw CC3DException("PolygonInitializer requires there to be an equal number of \
                    'From' and 'To' points.");
            }

            if (regionVec[i]->findElement("Extrude")) {
                CC3DXMLElement * extrudeXML = regionVec[i]->getFirstElement("Extrude");
                initData.zMin = extrudeXML->getAttributeAsUInt("zMin");
                initData.zMax = extrudeXML->getAttributeAsUInt("zMax");
            }
            else {
                //Set default value of extrusion so that the entire field is covered
                WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
                potts->getCellFieldG();
                if (!cellField) throw CC3DException("initField() Cell field cannot be null!");
                Dim3D dim = cellField->getDim();

                initData.zMin = 0;
                initData.zMax = dim.z;
            }
            
            initDataVec.push_back(initData);
        }
    }
    else {
        cerr << "failed if (regionVec.size() > 0)" << endl;
    }
}

Dim3D PolygonFieldInitializer::getPolygonDimensions(const Dim3D &dim, int size) {
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

bool PolygonFieldInitializer::onLine(Coordinates3D<double> lineStart, Coordinates3D<double> lineEnd, Coordinates3D<double> pt) {
    //Check whether p is on the line or not
    if (pt.x <= max(lineStart.x, lineEnd.x)
        && pt.x >= min(lineStart.x, lineEnd.x)
        && pt.y <= max(lineStart.y, lineEnd.y)
        && pt.y >= min(lineStart.y, lineEnd.y))
        return true;
    return false;
}

int PolygonFieldInitializer::direction(Coordinates3D<double> a, Coordinates3D<double> b, Coordinates3D<double> c) {
    double val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (val == 0)
        //Collinear
        return 0;
    else if (val < 0)
        //Counter-clockwise direction
        return 2;
    //Clockwise direction
    return 1;
}


bool PolygonFieldInitializer::isIntersect(Coordinates3D<double> src1, Coordinates3D<double> dst1, Coordinates3D<double> src2, Coordinates3D<double> dst2) {
    //Four direction for two lines and points of other line
    int dir1 = direction(src1, dst1, src2);
    int dir2 = direction(src1, dst1, dst2);
    int dir3 = direction(src2, dst2, src1);
    int dir4 = direction(src2, dst2, dst1);
 
    //When intersecting
    if (dir1 != dir2 && dir3 != dir4)
        return true;
 
    //When p2 of line2 is on line1
    if (dir1 == 0 && onLine(src1, dst1, src2))
        return true;
 
    //When p1 of line2 is on line1
    if (dir2 == 0 && onLine(src1, dst1, dst2))
        return true;
 
    //When p2 of line1 is on line2
    if (dir3 == 0 && onLine(src2, dst2, src1))
        return true;
 
    //When p1 of line1 is on line2
    if (dir4 == 0 && onLine(src2, dst2, dst1))
        return true;
 
    return false;
}

bool PolygonFieldInitializer::checkInside(Coordinates3D<double> pt, std::vector <Coordinates3D<double>> srcPoints, std::vector <Coordinates3D<double>> dstPoints) {
    int n = srcPoints.size();

    //The below loop fixes a bug where 
    //isIntersect() would consider the imaginary line to cross through two edges
    //just because it is hitting a vertex.
    int numPointsAtSameLevelOf = 0; 
    for (int i = 0; i < srcPoints.size(); i++) { 
        if (srcPoints[i].y == pt.y || dstPoints[i].y == pt.y) {
            pt.y += 0.00001;
        }
    }
    
    
    //Algorithm:
    //pt will connect to an imaginary point at (x="infinity", y=pt.y), forming a line.
    //If the line intersects the polygon once, then the point pt must be inside the polygon.
    //Special case: If the polygon is a U-shape and pt is on the left, then the line would cross through multiple edges.
    //Thus, we are looking for an ODD number of intersections with the polygon.
    
    double MAX_DOUBLE = (std::numeric_limits<float>::max)();
    Coordinates3D<double> ptInfinity = Coordinates3D<double>();
    ptInfinity.x = MAX_DOUBLE;
    ptInfinity.y = pt.y;
    ptInfinity.z = pt.z;

    int count = 0;
    int i = 0;
    while (i < n) {
        //Check "line i", which connects a point from srcPoints to dstPoints
        if (isIntersect(srcPoints[i], dstPoints[i], pt, ptInfinity)) {
            //If this line is intersects line i
            if (direction(srcPoints[i], pt, dstPoints[i]) == 0) {
                return onLine(srcPoints[i], dstPoints[i], pt);
            }
            count += 1;
        }

        i++;
    }
    
    //When count is odd
    return count & 1;
}


void PolygonFieldInitializer::layOutCells(const PolygonFieldInitializerData &_initData) { 
    std::vector <Coordinates3D<double>> srcPoints  = _initData.srcPoints;
    std::vector <Coordinates3D<double>> dstPoints  = _initData.dstPoints;

    int size = _initData.gap + _initData.width;
    int cellWidth = _initData.width;

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellField->getDim();
 
	Dim3D itDim = getPolygonDimensions(dim, size);
	CC3D_Log(LOG_DEBUG) << "itDim="<<itDim;

    Point3D pt;
    Point3D cellPt;
    CellG *cell;
    Coordinates3D<double> doublePoint = Coordinates3D<double>();

    //Only Z is dependent on the Extrude parameter
    for (int z = _initData.zMin / size; z < _initData.zMax / size; z++)
        for (int y = 0; y < itDim.y; y++)
            for (int x = 0; x < itDim.x; x++) {
                pt.x = x * size;
                pt.y = y * size;
                pt.z = z * size;
                
                doublePoint.x = (double) pt.x;
                doublePoint.y = (double) pt.y;
                doublePoint.z = (double) pt.z;
                if (!checkInside(doublePoint, srcPoints, dstPoints)) {
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