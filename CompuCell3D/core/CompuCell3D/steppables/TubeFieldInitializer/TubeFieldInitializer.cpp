/**
Full credit for the following functions to https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/
- distanceToLine
- subtractPoints
- dotProduct
- crossProduct
- magnitude
*/


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





Point3D TubeFieldInitializer::subtractPoints(Point3D p1, Point3D p2) {
    int x1 = p1.x - p2.x;
    int y1 = p1.y - p2.y;
    int z1 = p1.z - p2.z;

    return Point3D(x1, y1, z1);
}

int TubeFieldInitializer::dotProduct(Point3D p1, Point3D p2) {
    int x1 = p1.x * p2.x;
    int y1 = p1.y * p2.y;
    int z1 = p1.z * p2.z;

    return x1 + y1 + z1;
}

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
Calculates the perpendicular distance from a point to a line.
*/
float TubeFieldInitializer::distanceToLine(Point3D line_point1,
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

    Point3D AB = subtractPoints(line_point2, line_point1);
    Point3D AC = subtractPoints(point, line_point1);
    float area = magnitude( (crossProduct(AB, AC)) );
    float CD = area / magnitude(AB);
    return CD;
}




// void TubeFieldInitializer::layOutCells(const TubeFieldInitializerData &_initData) { 

//     int size = _initData.gap + _initData.width;
//     int cellWidth = _initData.width;

//     WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
//     potts->getCellFieldG();
//     if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

//     Dim3D dim = cellField->getDim();

//     cerr << "dim:" << dim.x << ", " << dim.y << ", "  << dim.z << endl; 
// 	Dim3D itDim = getTubeDimensions(dim, size);
//     cerr << "itDim:" << itDim.x << ", " << itDim.y << ", "  << itDim.z << endl;
// 	CC3D_Log(LOG_DEBUG) << "itDim="<<itDim;

//     Point3D pt;
//     Point3D cellPt;
//     CellG *cell;

//     double tubeLength = distance(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
//             _initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z);
//     double hypotenuse = sqrt( pow(_initData.outerRadius, 2) + pow(tubeLength, 2) );

//     for (int z = 0; z < itDim.z; z++)
//         for (int y = 0; y < itDim.y; y++)
//             for (int x = 0; x < itDim.x; x++) {
//                 pt.x = x * size;
//                 pt.y = y * size;
//                 pt.z = z * size;

//                 //Step 1: Is the point close/far enough to the center axis of the tube?
//                 double dist = distanceToLine(_initData.fromPoint, _initData.toPoint, pt);
//                 if (dist > _initData.outerRadius || dist < _initData.innerRadius) {
//                     continue;
//                 }

//                 //Step 2: Is the point too far from the face of the tube/cylinder? (Pythagorean Thm.)
//                 //This trims the tube to its desired length.
//                 //1. Choose the face that's farther away from the point pt.
//                 double fromDist = distance(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
//                         pt.x, pt.y, pt.z);
//                 double toDist = distance(_initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z,
//                         pt.x, pt.y, pt.z);
//                 double distanceFromFace = max(fromDist, toDist);
//                 //2. Check the distance from `pt` to the farther face against hypotenuse.
//                 //Ex: If they are equal, the point would be at the bottom edge of the tube.
//                 //Ex: If the dist is too great, the point is beyond the tube's length. 
//                 if (distanceFromFace > hypotenuse) {
//                     continue;
//                 }


//                 if (BoundaryStrategy::getInstance()->isValid(pt)) {
//                     cell = potts->createCellG(pt);
//                     cell->type = initCellType(_initData);
//                     potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
//                     //It is necessary to do it this way because steppers are called only when we are performing pixel copies
//                     // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
//                     //inventory unless you call steppers(VolumeTrackerPlugin) explicitly


//                 } else {
//                     continue;
//                 }

//                 for (cellPt.z = pt.z; cellPt.z < pt.z + cellWidth &&
//                                       cellPt.z < dim.z; cellPt.z++)
//                     for (cellPt.y = pt.y; cellPt.y < pt.y + cellWidth &&
//                                           cellPt.y < dim.y; cellPt.y++)
//                         for (cellPt.x = pt.x; cellPt.x < pt.x + cellWidth &&
//                                               cellPt.x < dim.x; cellPt.x++) {

//                             if (BoundaryStrategy::getInstance()->isValid(pt))
//                                 cellField->set(cellPt, cell);

//                         }
//                 potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
//                 //It is necessary to do it this way because steppers are called only when we are performing pixel copies
//                 // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
//                 //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

//             }


// }

// Define a function to calculate the cross product of two vectors
std::vector<double> TubeFieldInitializer::crossProductVec(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}




void TubeFieldInitializer::layOutCells(const TubeFieldInitializerData &_initData) { 

    for (int offset = 0; offset < 2; offset++) {
    int size = _initData.gap + _initData.width;
    int cellWidth = _initData.width;

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellField->getDim();
    cerr << "dim:" << dim.x << ", " << dim.y << ", "  << dim.z << endl;

    Point3D pt;
    CellG *cell;

    double tubeLength = distance(_initData.fromPoint.x, _initData.fromPoint.y, _initData.fromPoint.z,
            _initData.toPoint.x, _initData.toPoint.y, _initData.toPoint.z);

    Point3D directionVec = _initData.fromPoint - _initData.toPoint;
    //Normalize the direction vector
    directionVec.x /= tubeLength;
    directionVec.y /= tubeLength;
    directionVec.z /= tubeLength;

    Point3D center = Point3D();
    center.x = _initData.fromPoint.x;
    center.y = _initData.fromPoint.y;
    center.z = offset;

    for (double radius = 8.0; radius < 11.0; radius += 0.5) {
        int num_points = 60;
        double axisMagnitude = sqrt(directionVec.x * directionVec.x + directionVec.y * directionVec.y + directionVec.z * directionVec.z);

        // Calculate unit vector c
        std::vector<float> c(3);
        // for (int i = 0; i < 3; i++) {
        //     c[i] = directionVec[i] / axisMagnitude;
        // }

        c[0] = directionVec.x / axisMagnitude;
        c[1] = directionVec.y / axisMagnitude;
        c[2] = directionVec.z / axisMagnitude;

        // Point3D c;
        // c.x = directionVec.x / axisMagnitude;
        // c.y = directionVec.y / axisMagnitude;
        // c.z = directionVec.z / axisMagnitude;


        // Define a vector a
        std::vector<float> a(3);
        // Point3D a;

        //Create an orthonormal basis around the axis
        a = crossProductVec(c, {1.0, 0.0, 0.0});
        // a = crossProduct(c, Point3D(1.0, 0.0, 0.0));

        // Check if a is a zero vector
        if (a[0] == 0 && a[1] == 0 && a[2] == 0) {
            a = crossProductVec(c, {0.0, 1.0, 0.0});
        }

        // Calculate the magnitude of vector a
        float aMagnitude = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

        // Normalize vector a
        // for (int i = 0; i < 3; i++) {
        //     a[i] /= aMagnitude;
        // }
        a[0] /= aMagnitude;
        a[1] /= aMagnitude;
        a[2] /= aMagnitude;

        // Calculate vector b
        std::vector<double> b = crossProductVec(c, a);
        // Point3D b = crossProduct(c, a);



        // angle_alt = 0
        // color2 = np.array([0, 0.4, 0.9])
        // color1 = np.array([0.9, 0, 0])
        // curr_color = color1
        // if alt_color:
        //     curr_color = color2

        //Calculate the ring coordinates
        for (int i = 0; i < num_points; i++) {
            double angle = 2 * M_PI * i / num_points;
            int x_coord = round(center.x + radius * cos(angle) * a[0] + radius * sin(angle) * b[0]);
            int y_coord = round(center.y + radius * cos(angle) * a[1] + radius * sin(angle) * b[1]);
            int z_coord = round(center.z + radius * cos(angle) * a[2] + radius * sin(angle) * b[2]);

            if (x_coord < dim.x && y_coord < dim.y && z_coord < dim.z && x_coord >= 0 && y_coord >= 0 && z_coord >= 0) {
                
                pt.x = x_coord;
                pt.y = y_coord;
                pt.z = z_coord;
                if (BoundaryStrategy::getInstance()->isValid(pt))
                    cellField->set(pt, cell);

                // grid[x_coord][y_coord][z_coord] = 1
                // if angle_alt >= 5:
                //     if id(curr_color) == id(color1): #same memory address?
                //         curr_color = color2
                //     else:
                //         curr_color = color1
                //     angle_alt = 0
                // colors[x_coord][y_coord][z_coord] = curr_color
            }
            // angle_alt += 1
        }
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