/**
Full credit for the following functions to https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/
- distanceToLine
- crossProduct
- magnitude
*/


#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "TubeFieldInitializer.h"
#include <Logger/CC3DLogger.h>
#include <limits>
#include <PublicUtilities/NumericalUtils.h>

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

            if (regionVec[i]->findElement("NumSlices")) {
                initData.numSlices = regionVec[i]->getFirstElement("NumSlices")->getUInt();
                if (initData.numSlices < 1)
                    initData.numSlices = DEFAULT_NUM_SLICES;
            }

            if (regionVec[i]->findElement("Width"))
                initData.width = regionVec[i]->getFirstElement("Width")->getUInt();

            if (regionVec[i]->findElement("CellShape")) {
                std::string cellShapeStr = regionVec[i]->getFirstElement("CellShape")->getText();
                changeToLower(cellShapeStr);
                if (cellShapeStr == "cube")
                    initData.cellShape = CUBE;
                else
                    initData.cellShape = WEDGE;
            }

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

//TODO; it's same for blob, tube, poly
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


Point3D TubeFieldInitializer::crossProduct(Point3D p1, Point3D p2) {
    int x1 = p1.y * p2.z - p1.z * p2.y;
    int y1 = p1.z * p2.x - p1.x * p2.z;
    int z1 = p1.x * p2.y - p1.y * p2.x;
    
    return Point3D(x1, y1, z1);
}

double TubeFieldInitializer::magnitude(Point3D p) {
    return (double) sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
}

/**
Calculates the perpendicular distance from a point to a line.
*/
double TubeFieldInitializer::distanceToLine(Point3D line_point1,
                                    Point3D line_point2,
                                    Point3D point)
{
    //AB: the length of the given line.
    //CD: the imaginary perpendicular line segment between `point` and the line.
    //AC: the imaginary segment connecting `line_point1` to `point`

    //The goal is to treat the points as a parallelogram which has an
    //area of Base * Height = AB * CD.
    //This gives CD = |ABxAC| / |AB|
    //where |ABxAC| serves as Base*Height and the `area` below.

    Point3D AB = line_point2 - line_point1;
    Point3D AC = point - line_point1;
    double area = magnitude( (crossProduct(AB, AC)) );
    double CD = area / magnitude(AB);
    return CD;
}




void TubeFieldInitializer::layOutCellsCube(const TubeFieldInitializerData &_initData) { 

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

    double tubeLength = dist(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
            _initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z);
    double hypotenuse = sqrt( pow(_initData.outerRadius, 2) + pow(tubeLength, 2) );

    for (int z = 0; z < itDim.z; z++)
        for (int y = 0; y < itDim.y; y++)
            for (int x = 0; x < itDim.x; x++) {
                pt.x = x * size;
                pt.y = y * size;
                pt.z = z * size;

                //Step 1: Is the point close/far enough to the center axis of the tube?
                double lineDist = distanceToLine(_initData.fromPoint, _initData.toPoint, pt);
                if (lineDist > _initData.outerRadius || lineDist < _initData.innerRadius) {
                    continue;
                }

                //Step 2: Is the point too far from the face of the tube/cylinder? (Pythagorean Thm.)
                //This trims the tube to its desired length.
                //1. Choose the face that's farther away from the point pt.
                double fromDist = dist(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
                        pt.x, pt.y, pt.z);
                double toDist = dist(_initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z,
                        pt.x, pt.y, pt.z);
                double distanceFromFace = max(fromDist, toDist);
                //2. Check the distance from `pt` to the farther face against hypotenuse.
                //Ex: If they are equal, the point would be at the bottom edge of the tube.
                //Ex: If the lineDist is too great, the point is beyond the tube's length. 
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


// Define a function to calculate the cross product of two vectors
std::vector<double> TubeFieldInitializer::crossProductVec(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}





void TubeFieldInitializer::layOutCellsWedge(const TubeFieldInitializerData &_initData) { 
    /**
    This algorithm creates the tube one ring at a time using superAxisIter.
    The rings are placed along a direction vector that places the points along any user-defined line.
    Each time a ring is generated, the algorithm tries to place all the pixels of a cell
    before moving on to create the next cell. 
    */

    int cellWidth = _initData.width;

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellField->getDim();
    cerr << "dim:" << dim.x << ", " << dim.y << ", "  << dim.z << endl;

    Point3D pt = Point3D();
    CellG *cell;

    double tubeLength = dist(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
            _initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z);

    std::vector<double> directionVec(3);
    directionVec[0] = double(_initData.fromPoint.x) - double(_initData.toPoint.x);
    directionVec[1] = double(_initData.fromPoint.y) - double(_initData.toPoint.y);
    directionVec[2] = double(_initData.fromPoint.z) - double(_initData.toPoint.z);
    //Normalize the direction vector
    for (int i = 0; i < 3; i++) {
        directionVec[i] /= tubeLength;
    }

    const int NUM_RING_POINTS = 60; //arbitrary

    //Do a linear interpolation between fromPoint and toPoint
    short numAxisPoints = tubeLength;
    double dx = double(_initData.toPoint.x - _initData.fromPoint.x) / max(numAxisPoints - 1, 1);
    double dy = double(_initData.toPoint.y - _initData.fromPoint.y) / max(numAxisPoints - 1, 1);
    double dz = double(_initData.toPoint.z - _initData.fromPoint.z) / max(numAxisPoints - 1, 1);
    //Add gap along the axis
    if (dx != 0.0)
        dx += (dx/dx) * _initData.gap;
    if (dy != 0.0)
        dy += (dy/dy) * _initData.gap;
    if (dz != 0.0)
        dz += (dz/dz) * _initData.gap;
    short centerX, centerY, centerZ;

    double axisMagnitude = sqrt(directionVec[0] * directionVec[0] + directionVec[1] * directionVec[1] + directionVec[2] * directionVec[2]);

    //Calculate the normalized direction vector
    std::vector<double> normalizedDir(3);
    normalizedDir[0] = directionVec[0] / axisMagnitude;
    normalizedDir[1] = directionVec[1] / axisMagnitude;
    normalizedDir[2] = directionVec[2] / axisMagnitude;

    //Create an orthonormal basis around the x-axis
    std::vector<double> a = crossProductVec(normalizedDir, {1.0, 0.0, 0.0});

    //Check if a is a zero vector
    if (a[0] == 0 && a[1] == 0 && a[2] == 0) {
        a = crossProductVec(normalizedDir, {0.0, 1.0, 0.0});
    }

    //Normalize vector a
    double aMagnitude = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    a[0] /= aMagnitude;
    a[1] /= aMagnitude;
    a[2] /= aMagnitude;

    std::vector<double> b = crossProductVec(normalizedDir, a);
        
    bool justFormedCell = false; //This is false only when all of the pixels of the current cell are placed.
    
    
    for (short superAxisIter = 0; superAxisIter < short(tubeLength); superAxisIter += short(cellWidth)) {

        //Convert the 'gap' into arc length measured in degrees
        double approxRadius = (_initData.outerRadius - _initData.innerRadius) / 2 + _initData.innerRadius;
        double cellWidthDegrees =  2*M_PI / _initData.numSlices;
        double gapDegrees = (_initData.gap * approxRadius) * (M_PI / 180.0);

        for (double superAngle = 0.0; superAngle < 2*M_PI - gapDegrees; superAngle += cellWidthDegrees + gapDegrees) {
            for (double angle = superAngle; angle < superAngle + cellWidthDegrees; angle += M_PI/180.0) {
                //Increment by at least 0.5 just to avoid having empty pixels in the rings.
                for (double radius = _initData.innerRadius; radius < _initData.outerRadius; radius += 0.5) {
                    //Give the cell some thickness along the direction vector
                    for (short axisIter = superAxisIter; axisIter < superAxisIter + cellWidth; axisIter++) {

                        centerX = short(_initData.fromPoint.x + axisIter * dx);
                        centerY = short(_initData.fromPoint.y + axisIter * dy);
                        centerZ = short(_initData.fromPoint.z + axisIter * dz);

                        pt.x = short(round(centerX + radius * cos(angle) * a[0] + radius * sin(angle) * b[0]));
                        pt.y = short(round(centerY + radius * cos(angle) * a[1] + radius * sin(angle) * b[1]));
                        pt.z = short(round(centerZ + radius * cos(angle) * a[2] + radius * sin(angle) * b[2]));
                        
                        if (!BoundaryStrategy::getInstance()->isValid(pt))
                            continue;

                        if (!justFormedCell) {
                            cell = potts->createCellG(pt);
                            cell->type = initCellType(_initData);
                            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly
                            justFormedCell = true;
                        }

                        // if (cellField->get(pt) != NULL && cellField->get(pt) != nullptr)
                        cellField->set(pt, cell);
                    }
                }
            }
            justFormedCell = false;

            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                    //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                    // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                    //inventory unless you call steppers(VolumeTrackerPlugin) explicitly
        }
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
            if (initDataVec[i].cellShape == CUBE) {
                layOutCellsCube(initDataVec[i]);
            }
            else {
                layOutCellsWedge(initDataVec[i]);
            }            
        }
    }

}

std::string TubeFieldInitializer::steerableName() {
    return toString();
}

std::string TubeFieldInitializer::toString() {
    return "TubeInitializer";
}