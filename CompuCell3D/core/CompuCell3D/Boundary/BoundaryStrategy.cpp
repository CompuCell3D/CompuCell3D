
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Field3D/Neighbor.h>
#include <CompuCell3D/Field3D/NeighborFinder.h>
#include <map>
#include <sstream>
#include <algorithm>
#include <Utils/Coordinates3D.h>


#include "BoundaryStrategy.h"
//Field3DImpl.h includes boundary Strategy and for this reason has to be listed after #define EXP_STL
#include <CompuCell3D/Field3D/Field3DImpl.h>

#include "Boundary.h"
#include "BoundaryFactory.h"

#include "AlgorithmFactory.h"
#include "Algorithm.h"
#include <Logger/CC3DLogger.h>

#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


using namespace std;
using namespace CompuCell3D;

BoundaryStrategy *BoundaryStrategy::singleton;


BoundaryStrategy::BoundaryStrategy() {
    boundaryConditionIndicator.assign(3, 0);
    strategy_x = BoundaryFactory::createBoundary(BoundaryFactory::no_flux);
    strategy_y = BoundaryFactory::createBoundary(BoundaryFactory::no_flux);
    strategy_z = BoundaryFactory::createBoundary(BoundaryFactory::no_flux);
    algorithm = AlgorithmFactory::createAlgorithm(AlgorithmFactory::Default, 0, 0, "None");

    regular = true;
    neighborListsInitializedFlag = false;
    latticeType = SQUARE_LATTICE;
    dimensionType = DIM_DEFAULT;
    maxNeighborOrder = 0;


}


BoundaryStrategy::BoundaryStrategy(const string &boundary_x, const string &boundary_y,
                                   const string &boundary_z, string alg, int index, int size, string inputfile,
                                   LatticeType latticeType, DimensionType dimensionType) {

    boundaryConditionIndicator.assign(3, 0);

    strategy_x = BoundaryFactory::createBoundary(boundary_x);
    strategy_y = BoundaryFactory::createBoundary(boundary_y);
    strategy_z = BoundaryFactory::createBoundary(boundary_z);

    boundaryConditionIndicator[0] = boundary_x == "Periodic" ? 1 : 0;
    boundaryConditionIndicator[1] = boundary_y == "Periodic" ? 1 : 0;
    boundaryConditionIndicator[2] = boundary_z == "Periodic" ? 1 : 0;


    algorithm = AlgorithmFactory::createAlgorithm(alg, index, size, inputfile);
    regular = true;
    neighborListsInitializedFlag = false;
    this->latticeType = latticeType;
    this->dimensionType = dimensionType;
    maxNeighborOrder = 0;

}


BoundaryStrategy::~BoundaryStrategy() {
    CC3D_Log(LOG_DEBUG) << "strategy_x=" << strategy_x;
    CC3D_Log(LOG_DEBUG) << "strategy_y=" << strategy_y;
    CC3D_Log(LOG_DEBUG) << "strategy_z=" << strategy_z;
    if (strategy_x) {
        delete strategy_x;
        strategy_x = nullptr;
    }

    if (strategy_y) {
        delete strategy_y;
        strategy_y = nullptr;
    }

    if (strategy_z) {
        delete strategy_z;
        strategy_z = nullptr;
    }

}

void BoundaryStrategy::setDim(const Dim3D theDim) {

    Dim3D oldDim(dim);

    dim = theDim;
    algorithm->setDim(theDim);
    if (!neighborListsInitializedFlag) {
        prepareNeighborLists();
        neighborListsInitializedFlag = true;
    }

    if (latticeType == HEXAGONAL_LATTICE) {
        latticeSizeVector.x = dim.x;
        latticeSizeVector.y = dim.y * sqrt(3.0) / 2.0;
        latticeSizeVector.z = dim.z * sqrt(6.0) / 3.0;

        latticeSpanVector.x = dim.x - 1;
        latticeSpanVector.y = (dim.y - 1) * sqrt(3.0) / 2.0;
        latticeSpanVector.z = (dim.z - 1) * sqrt(6.0) / 3.0;

    } else {
        latticeSizeVector.x = dim.x;
        latticeSizeVector.y = dim.y;
        latticeSizeVector.z = dim.z;

        latticeSpanVector.x = dim.x - 1;
        latticeSpanVector.y = dim.y - 1;
        latticeSpanVector.z = dim.z - 1;
    }

}

const std::vector<Point3D> &BoundaryStrategy::getOffsetVec() const { return offsetVec; }

const std::vector<Point3D> &BoundaryStrategy::getOffsetVec(Point3D &pt) const {
    if (latticeType == HEXAGONAL_LATTICE) {
        return hexOffsetArray[(pt.z % 3) * 2 + pt.y % 2];
    } else {
        return offsetVec;
    }
}

bool BoundaryStrategy::isValid(const int coordinate, const int max_value) const {

    return (0 <= coordinate && coordinate < max_value);
}

bool BoundaryStrategy::isValidCustomDim(const Point3D &pt, const Dim3D &customDim) const {
    // check to see if the point lies in the dimensions before applying the
    // shape algorithm

    if (0 <= pt.x && pt.x < customDim.x &&
        0 <= pt.y && pt.y < customDim.y &&
        0 <= pt.z && pt.z < customDim.z) {
        return algorithm->inGrid(pt);
    }

    return false;

}

Point3D
BoundaryStrategy::getNeighbor(const Point3D &pt, unsigned int &token, double &distance, bool checkBounds) const {
    Neighbor n;
    Point3D p;
    int x;
    int y;
    int z;
    bool x_bool;
    bool y_bool;
    bool z_bool;

    NeighborFinder::destroy();

    while (true) {
        // Get a neighbor from the NeighborFinder
        n = NeighborFinder::getInstance()->getNeighbor(token);
        x = (pt + n.pt).x;
        y = (pt + n.pt).y;
        z = (pt + n.pt).z;

        token++;

        if (!checkBounds || isValid(pt + n.pt)) {
            // Valid Neighbor
            break;
        } else {
            if (regular) {
                // For each coordinate, if it is not valid, apply condition
                x_bool = (isValid(x, dim.x) ? true : strategy_x->applyCondition(x, dim.x));
                y_bool = (isValid(y, dim.y) ? true : strategy_y->applyCondition(y, dim.y));
                z_bool = (isValid(z, dim.z) ? true : strategy_z->applyCondition(z, dim.z));

                // If all the coordinates of the neighbor are valid then return the
                // neighbor
                if (x_bool && y_bool && z_bool) {
                    break;
                }
            }
        }
    }

    distance = n.distance;
    p.x = x;
    p.y = y;
    p.z = z;

    return p;

}


bool BoundaryStrategy::isValid(const Point3D &pt) const {

    // check to see if the point lies in the dimensions before applying the
    // shape algorithm

    if (0 <= pt.x && pt.x < dim.x &&
        0 <= pt.y && pt.y < dim.y &&
        0 <= pt.z && pt.z < dim.z) {
        return algorithm->inGrid(pt);
    }

    return false;
}


bool BoundaryStrategy::checkIfOffsetAlreadyStacked(Point3D &_ptToCheck, std::vector<Point3D> &_offsetVec) const {

    for (auto &i: _offsetVec) {
        if (i.x == _ptToCheck.x && i.y == _ptToCheck.y && i.z == _ptToCheck.z)
            return true;
    }
    return false;
}

double BoundaryStrategy::calculateDistance(Coordinates3D<double> &_pt1, Coordinates3D<double> &_pt2) const {
    return sqrt((double) (_pt1.x - _pt2.x) * (_pt1.x - _pt2.x) + (_pt1.y - _pt2.y) * (_pt1.y - _pt2.y) +
                (_pt1.z - _pt2.z) * (_pt1.z - _pt2.z));
}

bool BoundaryStrategy::checkEuclidianDistance(Coordinates3D<double> &_pt1, Coordinates3D<double> &_pt2,
                                              float _distance) const {
    //checks if distance between two points is smaller than _distance
    //used to eliminate in offsetVec offsets that come from periodic conditions (opposite side of the lattice)
    return calculateDistance(_pt1, _pt2) < _distance + 0.1;

}

Coordinates3D<double> BoundaryStrategy::HexCoord(const Point3D &_pt) const {
    //the transformations formulas for hex latice are written in such a way that distance between pixels is set to 1

    if ((_pt.z % 3) == 1) {//odd z e.g. z=1
        //(-0.5,+sqrt(3)/6)
        if (_pt.y % 2)
            return Coordinates3D<double>(_pt.x + 0.5, sqrt(3.0) / 2.0 * (_pt.y + 2.0 / 6.0), _pt.z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(_pt.x, sqrt(3.0) / 2.0 * (_pt.y + 2.0 / 6.0), _pt.z * sqrt(6.0) / 3.0);

    } else if ((_pt.z % 3) == 2) { //e.g. z=2

        if (_pt.y % 2)
            return Coordinates3D<double>(_pt.x + 0.5, sqrt(3.0) / 2.0 * (_pt.y - 2.0 / 6.0), _pt.z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(_pt.x, sqrt(3.0) / 2.0 * (_pt.y - 2.0 / 6.0), _pt.z * sqrt(6.0) / 3.0);

    } else {//z divible by 3 - includes z=0
        if (_pt.y % 2)
            return Coordinates3D<double>(_pt.x, sqrt(3.0) / 2.0 * _pt.y, _pt.z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(_pt.x + 0.5, sqrt(3.0) / 2.0 * _pt.y, _pt.z * sqrt(6.0) / 3.0);
    }

}


void BoundaryStrategy::getOffsetsAndDistances(
        Point3D ctPt,
        float maxDistance,
        Field3DImpl<char> const &tempField,
        vector<Point3D> &offsetVecTmp,
        vector<float> &distanceVecTmp,
        vector<unsigned int> &neighborOrderIndexVecTmp
) const {

    Point3D n;

    unsigned int token = 0;
    double distance = 0;
    Coordinates3D<double> ctPtTrans, nTrans;
    Point3D offset;
    double distanceTrans = 0.0;

    offsetVecTmp.clear();
    distanceVecTmp.clear();
    neighborOrderIndexVecTmp.clear();

    if (latticeType == HEXAGONAL_LATTICE) {
        ctPtTrans = HexCoord(ctPt);
    } else {
        ctPtTrans = Coordinates3D<double>(ctPt.x, ctPt.y, ctPt.z);
    }

    Dim3D tmpFieldDim = tempField.getDim();

    while (true) {
        // calling  getNeighbor via field interface changes checkBounds from false to true...
        // calling getNeighbor directly requires does not set checkBounds to true
        // notice that we cannot in general use regular getNeighbor because this fcn assumes that dimension
        // of the thmField are same as the dimensions of simulation field
        n = getNeighborCustomDim(ctPt, token, distance, tmpFieldDim,
                                 true);
        if (distance > maxDistance * 2.0)
            break; //2.0 factor is to ensure you visit enough neighbors for different kind of lattices
        //This factor is purely heuristic and may need to be increased in certain cases

        offset = n - ctPt;

        if (latticeType == HEXAGONAL_LATTICE) {
            //the transformations formulas for hex lattice
            // are written in such a way that distance between pixels is set to 1
            ctPtTrans = HexCoord(ctPt);
            nTrans = HexCoord(n);
            distanceTrans = calculateDistance(ctPtTrans, nTrans);

        } else {
            ctPtTrans = Coordinates3D<double>(ctPt.x, ctPt.y, ctPt.z);
            nTrans = Coordinates3D<double>(n.x, n.y, n.z);
            distanceTrans = distance;
        }

        if (!checkIfOffsetAlreadyStacked(offset, offsetVecTmp) && distanceTrans < maxDistance + 0.1) {
            CC3D_Log(LOG_TRACE) << "distanceTrans=" << distanceTrans << " offset=" << offset;
            offsetVecTmp.push_back(offset);
            distanceVecTmp.push_back(distanceTrans);
        }
    }

    //at this point we have all the offsets for the given simulation but they are unsorted.
    //Sorting  neighbors
    multimap<float, Point3D> sortingMap;
    for (int i = 0; i < offsetVecTmp.size(); ++i) {
        sortingMap.insert(make_pair(distanceVecTmp[i], offsetVecTmp[i]));
    }

    //clearing offsetVecTmp and distanceVecTmp
    offsetVecTmp.clear();
    distanceVecTmp.clear();
    //Writing sorted  by distance content of offsetVecTmp and distanceVecTmp
    for (multimap<float, Point3D>::iterator mitr = sortingMap.begin(); mitr != sortingMap.end(); ++mitr) {
        //       distanceVecTmp.push_back(mitr->first*lmf.lengthMF);
        distanceVecTmp.push_back(mitr->first);
        offsetVecTmp.push_back(mitr->second);
    }
#ifdef _DEBUG
    CC3D_Log(LOG_DEBUG) << "distanceVecTmp.size()=" << distanceVecTmp.size();
#endif
    //creating a vector indexed by neighbor order  - entries of this vector are the highest indices of offsets for a
    //given neighbor order
    float currentDistance = 1.0;


    for (int i = 0; i < distanceVecTmp.size(); ++i) {
        if (currentDistance < distanceVecTmp[i]) {
            neighborOrderIndexVecTmp.push_back(i - i);
            currentDistance = distanceVecTmp[i];
        }
    }


}


Point3D
BoundaryStrategy::getNeighborCustomDim(const Point3D &pt, unsigned int &token, double &distance, const Dim3D &customDim,
                                       bool checkBounds) const {

    Neighbor n;
    Point3D p;
    int x;
    int y;
    int z;
    bool x_bool;
    bool y_bool;
    bool z_bool;

    NeighborFinder::destroy();

    while (true) {
        // Get a neighbor from the NeighborFinder
        n = NeighborFinder::getInstance()->getNeighbor(token);
        x = (pt + n.pt).x;
        y = (pt + n.pt).y;
        z = (pt + n.pt).z;

        token++;

        if (!checkBounds || isValidCustomDim(pt + n.pt, customDim)) {
            // Valid Neighbor
            break;
        } else {
            if (regular) {
                // For each coordinate, if it is not valid, apply condition
                x_bool = (isValid(x, customDim.x) ? true : strategy_x->applyCondition(x, customDim.x));
                y_bool = (isValid(y, customDim.y) ? true : strategy_y->applyCondition(y, customDim.y));
                z_bool = (isValid(z, customDim.z) ? true : strategy_z->applyCondition(z, customDim.z));

                // If all the coordinates of the neighbor are valid then return the
                // neighbor
                if (x_bool && y_bool && z_bool) {
                    break;
                }
            }
        }
    }

    distance = n.distance;
    p.x = x;
    p.y = y;
    p.z = z;

    return p;

}


void BoundaryStrategy::prepareNeighborListsSquare(float _maxDistance) {

    char a = '0';
    Dim3D dim_test_field;

    if (dim.x == 1 || dim.y == 1 || dim.z == 1) {
        // we are dealing with 2D case 
        dim_test_field = dim;
    } else {
        // we are dealing with 3D case but we want to make sure that if we set one dimension to 2 we
        // get center point that is truly in the middle of the lattice therefore minimum
        // dimension in 3D for a test field is set to 3
        dim_test_field.x = std::max((short) 3, dim.x);
        dim_test_field.y = std::max((short) 3, dim.y);
        dim_test_field.z = std::max((short) 3, dim.z);
    }

    Field3DImpl<char> tempField(dim_test_field, a);
    int margin = 2 * (int) fabs(_maxDistance) + 1;
    Point3D ctPt(dim_test_field.x / 2, dim_test_field.y / 2, dim_test_field.z / 2);

    if (3 * _maxDistance > dim_test_field.x && 3 * _maxDistance > dim_test_field.y &&
        3 * _maxDistance > dim_test_field.z) {

        ostringstream outStr;
        outStr << "NeighborOrder too large for this lattice. Increase lattice size so that at least two dimensions ";
        outStr << "are greater than 3*_maxDistance. " << endl;
        outStr << " Trying to fetch neighbors with _maxDistance =" << _maxDistance << endl;
        throw CC3DException(outStr.str().c_str());
    }

    getOffsetsAndDistances(ctPt, _maxDistance, tempField, offsetVec, distanceVec,
                           neighborOrderIndexVec);

    // initializing neighbor vectors that will be used in pixel copy
    this->offsetVecVoxelCopy = offsetVec;
    this->distanceVecVoxelCopy = distanceVec;
    this->neighborOrderIndexVecVoxelCopy = neighborOrderIndexVec;

#ifdef _DEBUG
    for (size_t i = 0; i < offsetVec.size(); ++i) {
        cerr<<"i="<<i<<" offsetVec="<<offsetVec[i]<<" distanceVec="<<distanceVec[i]<<" neighborOrderIndexVec="<<neighborOrderIndexVec[i]<<endl;
    }
#endif

    // removing offsets where z != 0
    if (this->dimensionType == DIM_2_5) {
        this->prepare_2_5_d_voxel_copy_neighbors(this->offsetVecVoxelCopy, this->distanceVecVoxelCopy,
                                                 this->neighborOrderIndexVecVoxelCopy);


    }

#ifdef _DEBUG
    for (int i = 0; i < offsetVec.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << " This is offset[" << i << "]=" << offsetVec[i] << " distance=" << distanceVec[i];
    }
#endif
}

void BoundaryStrategy::prepare_2_5_d_voxel_copy_neighbors(std::vector<Point3D> &offsetVecTemplate,
                                                          std::vector<float> &distanceVecTemplate,
                                                          std::vector<unsigned int> &neighborOrderIndexVecTemplate) {
    // Check dimensions and remove elements accordingly
#ifdef _DEBUG
    cerr<<"INSIDE prepare_2_5_d_voxel_copy_neighbors"<<endl;
#endif

    std::vector<Point3D> offsetVecNew;
    std::vector<float> distanceVecNew;
    std::vector<unsigned int> neighborOrderIndexVecNew;

#ifdef _DEBUG
    for (size_t i = 0; i < offsetVecTemplate.size(); ++i) {
        cerr<<"i="<<i<<" offsetVecTemplate="<<offsetVecTemplate[i]<<" distanceVecTemplate="<<distanceVecTemplate[i]<<" neighborOrderIndexVecTemplate="<<neighborOrderIndexVecTemplate[i]<<endl;
    }
#endif

    for (size_t i = 0; i < offsetVecTemplate.size(); ++i) {
        if (offsetVecTemplate[i].z == 0) {
            offsetVecNew.push_back(offsetVecTemplate[i]);
            distanceVecNew.push_back(distanceVecTemplate[i]);
            neighborOrderIndexVecNew.push_back(neighborOrderIndexVecTemplate[i]);
        }


    }
    offsetVecTemplate.assign(offsetVecNew.begin(), offsetVecNew.end());
    distanceVecTemplate.assign(distanceVecNew.begin(), distanceVecNew.end());
    neighborOrderIndexVecTemplate.assign(neighborOrderIndexVecNew.begin(), neighborOrderIndexVecNew.end());
#ifdef _DEBUG
    cerr<<"prepare_2_5_d_voxel_copy_neighbors DONE"<<endl;


    for (size_t i = 0; i < offsetVecTemplate.size(); ++i) {
        cerr<<"i="<<i<<" offsetVecTemplate="<<offsetVecTemplate[i]<<" distanceVecTemplate="<<distanceVecTemplate[i]<<" neighborOrderIndexVecTemplate="<<neighborOrderIndexVecTemplate[i]<<endl;
    }

#endif


}


LatticeMultiplicativeFactors BoundaryStrategy::getLatticeMultiplicativeFactors() const {
    return lmf;
}

LatticeMultiplicativeFactors
BoundaryStrategy::generateLatticeMultiplicativeFactors(LatticeType _latticeType, Dim3D dim) {
    LatticeMultiplicativeFactors lFactors;
    if (_latticeType == HEXAGONAL_LATTICE) {
        if (dim.x == 1 || dim.y == 1 ||
            dim.z == 1) {//2D case for hex lattice // might need to tune it further to account for 1D case
            //area of hexagon with edge l = 6*sqrt(3)/4 * l^2

            lFactors.volumeMF = 1.0;
            lFactors.surfaceMF = sqrt(2.0 / (3.0 * sqrt(3.0)));
            lFactors.lengthMF = lFactors.surfaceMF * sqrt(3.0);
            return lFactors;
        } else {//3D case for hex lattice
            //Volume of rhombic dodecahedron = 16/9 *sqrt(3)*b^3
            //Surface of rhombic dodecahedron = 9*sqrt(2)*b^2
            //b - rhombus edge length
            lFactors.volumeMF = 1.0;
            lFactors.surfaceMF = 8.0 / 12.0 * sqrt(2.0) * pow(9.0 / (16.0 * sqrt(3.0)), 1.0 / 3.0) *
                                 pow(9.0 / (16.0 * sqrt(3.0)), 1.0 / 3.0);
            lFactors.lengthMF = 2.0 * sqrt(2.0 / 3.0) * pow(9.0 / (16.0 * sqrt(3.0)), 1.0 / 3.0);
            return lFactors;
        }
    } else {
        lFactors.volumeMF = 1.0;
        lFactors.surfaceMF = 1.0;
        lFactors.lengthMF = 1.0;
        return lFactors;
    }
}

void BoundaryStrategy::prepareNeighborListsHex(float _maxDistance) {
#ifdef _DEBUG
    CC3D_Log(LOG_DEBUG) << "INSIDE prepareNeighborListsHex";
#endif


    unsigned int maxHexArraySize = 6;

    hexOffsetArray.assign(maxHexArraySize, vector<Point3D>());
    hexDistanceArray.assign(maxHexArraySize, vector<float>());
    hexNeighborOrderIndexArray.assign(maxHexArraySize, vector<unsigned int>());

    char a = '0';

    vector<Point3D> offsetVecTmp;
    vector<float> distanceVecTmp;
    Dim3D tmpFieldDim;

    tmpFieldDim = dim;
    if (dim.z != 1 && tmpFieldDim.z < 15) {
        // to generate correct offsets we need to have tmpField which is large enough
        // for our algorithm in non-flat z dimension
        tmpFieldDim.z = 15;
    }

    if (dim.y != 1 && tmpFieldDim.y < 10) {
        // to generate correct offsets we need to have tmpField which is large enough
        // for our algorithm in non-flat y dimension
        tmpFieldDim.z = 10;
    }

    if (dim.x != 1 && tmpFieldDim.x < 10) {
        // to generate coreect offsets we need to have tmpField which is large enoug
        // for our algorithm in non-flat z dimension
        tmpFieldDim.x = 10;
    }

    Field3DImpl<char> tempField(tmpFieldDim, a);
    Point3D ctPt(tmpFieldDim.x / 2, tmpFieldDim.y / 2, tmpFieldDim.z / 2);

    Point3D ctPtTmp;
    unsigned int indexHex;
    //For hex lattice we have four different offset lists
    //y-even z_even

    ctPtTmp = ctPt;

    //there are 3 layers of z planes which are interlaced therefore we need to consider pt.z%3
    //indexHexFormula=(pt.z%3)*2+(pt.y%2);

    //indexHex=Y_EVEN|Z_EVEN;
    indexHex = 0; // e.g. z=21,y=20

    if (dim.z > 1) {//make sure not 2D with z direction flat
        ctPtTmp.y += ctPtTmp.y % 2; //make it even
        ctPtTmp.z += 3 - ctPtTmp.z % 3;// make it divisible by 3 in case it is not
#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "ctPtTmp.y % 2 =" << ctPtTmp.y % 2 << " ctPtTmp.y % 2=" << ctPtTmp.y % 2;
        CC3D_Log(LOG_TRACE) << "  WILL USE CENTER POINT="<<ctPtTmp<<"Y_EVEN|Z_EVEN "<<(Y_EVEN|Z_EVEN);
#endif
        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "ctPtTmp.y % 2 =" << ctPtTmp.y % 2;
#endif
        ctPtTmp.y += ctPtTmp.y % 2; //make it even
        ctPtTmp.z += 0;// make it divisible by 3 in case it is not



#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "even even ctPtTmp=" << ctPtTmp;
#endif
        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    }

    //y-odd z_even
    ctPtTmp = ctPt;
    indexHex = 1; //e.g. z=21 y=21
    //indexHex=Y_ODD|Z_EVEN;

    if (dim.z > 1) {//make sure not 2D with z direction flat

        ctPtTmp.y += (ctPtTmp.y % 2 - 1); //make it odd
        ctPtTmp.z += 3 - ctPtTmp.z % 3;// make it divisible by 3 in case it is not

#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "ctPtTmp.y % 2 =" << ctPtTmp.y % 2 << " !ctPtTmp.y % 2=" << !(ctPtTmp.y % 2);
CC3D_Log(LOG_TRACE) << "  WILL USE CENTER POINT="<<ctPtTmp<<"Y_ODD|Z_EVEN "<<(Y_ODD|Z_EVEN);
#endif
        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "ctPtTmp.y % 2 =" << ctPtTmp.y % 2 << " !ctPtTmp.y % 2=" << !(ctPtTmp.y % 2);
#endif

        ctPtTmp.y += (ctPtTmp.y % 2 - 1); //make it odd
        ctPtTmp.z += 0;   // make it divisible by 3 in case it is not

#ifdef _DEBUG
        CC3D_Log(LOG_DEBUG) << "odd even ctPtTmp=" << ctPtTmp;
#endif
        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    }

    ctPtTmp = ctPt;


    indexHex = 2;// e.g. z=22 y=20

    if (dim.z > 1) {//make sure not 2D with z direction flat

        ctPtTmp.y += ctPtTmp.y % 2; //make it even

        ctPtTmp.z += 3 - ctPtTmp.z % 3 - 2;// make it divisible by 3 with z%3=1 in case it is not

        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
        //ignore this case
    }

    //y-even z_odd
    ctPtTmp = ctPt;

    indexHex = 3;


    if (dim.z > 1) {//make sure not 2D with z direction flat

        ctPtTmp.y += (ctPtTmp.y % 2 - 1); //make it odd
        ctPtTmp.z += 3 - ctPtTmp.z % 3 - 2;// make it divisible by 3 with z%3=1 in case it is not


        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
        //ignore this case
    }


    ctPtTmp = ctPt;

    indexHex = 4;


    if (dim.z > 1) {//make sure not 2D with z direction flat

        ctPtTmp.y += ctPtTmp.y % 2; //make it even
        ctPtTmp.z += 3 - ctPtTmp.z % 3 - 1;// make it divisible by 3 with z%3=2 in case it is not


        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
        //ignore this case
    }

    ctPtTmp = ctPt;

    indexHex = 5;

    if (dim.z > 1) {//make sure not 2D with z direction flat

        ctPtTmp.y += (ctPtTmp.y % 2 - 1); //make it odd
        ctPtTmp.z += 3 - ctPtTmp.z % 3 - 1;// make it divisible by 3 with z%3=2 in case it is not

        getOffsetsAndDistances(ctPtTmp, _maxDistance, tempField, hexOffsetArray[indexHex],
                               hexDistanceArray[indexHex],
                               hexNeighborOrderIndexArray[indexHex]);

    } else {//2D case
        //ignore this case
    }

    //we will copy arrays 0 and 1 to (2,4) (3,5) respectively for 2D case
    {
        maxOffset = 6;
        if (dim.z == 1) {
            maxOffset = 6;

            hexOffsetArray[2] = hexOffsetArray[0];
            hexOffsetArray[4] = hexOffsetArray[0];
            hexOffsetArray[3] = hexOffsetArray[1];
            hexOffsetArray[5] = hexOffsetArray[1];

            hexDistanceArray[2] = hexDistanceArray[0];
            hexDistanceArray[4] = hexDistanceArray[0];
            hexDistanceArray[3] = hexDistanceArray[1];
            hexDistanceArray[5] = hexDistanceArray[1];

            hexNeighborOrderIndexArray[2] = hexNeighborOrderIndexArray[0];
            hexNeighborOrderIndexArray[4] = hexNeighborOrderIndexArray[0];
            hexNeighborOrderIndexArray[3] = hexNeighborOrderIndexArray[1];
            hexNeighborOrderIndexArray[5] = hexNeighborOrderIndexArray[1];

        } else {
            maxOffset = 12;
        }

    }

    // creating neighbor arrays used during voxel copy
    hexOffsetArrayVoxelCopy = hexOffsetArray;
    hexDistanceArrayVoxelCopy = hexDistanceArray;
    hexNeighborOrderIndexArrayVoxelCopy = hexNeighborOrderIndexArray;
    if (this->dimensionType == DIM_2_5) {
        for (size_t i = 0; i < hexOffsetArrayVoxelCopy.size(); ++i) {
            prepare_2_5_d_voxel_copy_neighbors(hexOffsetArrayVoxelCopy[i], hexDistanceArrayVoxelCopy[i],
                                               hexNeighborOrderIndexArrayVoxelCopy[i]);
        }
    }


#ifdef _DEBUG

    indexHex = 0;
    for (indexHex = 0; indexHex<maxHexArraySize; ++indexHex) {
        CC3D_Log(LOG_DEBUG) << "INDEX HEX=" << indexHex << " hexOffsetArray[indexHex].size()=" << hexOffsetArray[indexHex].size();

        for (int i = 0; i < hexOffsetArray[indexHex].size(); ++i) {
            CC3D_Log(LOG_DEBUG) << " This is offset[" << i << "]=" << hexOffsetArray[indexHex][i] << " distance=" << hexDistanceArray[indexHex][i];
        }
    }



    Neighbor n;
    Point3D testPt(10, 10, 0);
    unsigned int idx = 3;
    n = getNeighborDirect(testPt, idx);
    CC3D_Log(LOG_DEBUG) << "Neighbor=" << n;
    testPt = Point3D(10, 11, 0);
    n = getNeighborDirect(testPt, idx);
    CC3D_Log(LOG_DEBUG) << "Neighbor=" << n;
    testPt = Point3D(11, 11, 0);
    n = getNeighborDirect(testPt, idx);
    CC3D_Log(LOG_DEBUG) << "Neighbor=" << n;
    CC3D_Log(LOG_DEBUG) << " ****************************Checking Bondary ";

    testPt = Point3D(0, 0, 0);
    CC3D_Log(LOG_DEBUG) << "HexCoord(testPt)=" << HexCoord(testPt);
    for (int i = 0; i<6; ++i) {
        n = getNeighborDirect(testPt, i);
        if (n.distance>0) {
            CC3D_Log(LOG_DEBUG) << "Neighbor=" << n;
        }
        else {
            CC3D_Log(LOG_DEBUG) << "************************Not a neighbor= " << n;
        }
    }
    CC3D_Log(LOG_DEBUG) << " *****************Checkup Boundary";

    testPt = Point3D(0, dim.y - 1, 0);
    CC3D_Log(LOG_DEBUG) << "HexCoord(testPt)=" << HexCoord(testPt);
    for (int i = 0; i<6; ++i) {
        n = getNeighborDirect(testPt, i);
        if (n.distance>0) {
            CC3D_Log(LOG_DEBUG) << "Neighbor=" << n;
        }
        else {
            CC3D_Log(LOG_DEBUG) << "*****************Not a neighbor= " << n;
        }
    }


    for (int i = 1; i <= 11; ++i) {
        unsigned int maxIdx = getMaxNeighborIndexFromNeighborOrder(i);
        CC3D_Log(LOG_DEBUG) << "NEIGHBOR ORDER =" << i << " maxIdx=" << maxIdx;

    }

#endif

}


void BoundaryStrategy::prepareNeighborLists(float _maxDistance) {
    maxDistance = _maxDistance;

    lmf = generateLatticeMultiplicativeFactors(latticeType, dim);

    if (latticeType == HEXAGONAL_LATTICE) {

        prepareNeighborListsHex(_maxDistance);

    } else {
        prepareNeighborListsSquare(_maxDistance);
    }


}

unsigned int BoundaryStrategy::getMaxNeighborIndexFromNeighborOrderNoGen(unsigned int _neighborOrder) const {
    //this function returns whatever maxNeighborIndex exist for a given neighbororder. If neighborOrder is higher that maxNeighborOrder
    //this function DOES NOT generate extra offsets so the maxNeighborOrder may correspond to a smaller neighbor order than in the requested _neighborOrder

    return getMaxNeighborIndexFromNeighborOrderNoGenImpl(_neighborOrder,
                                                         distanceVec,
                                                         hexDistanceArray);

}


unsigned int BoundaryStrategy::getMaxNeighborIndexFromNeighborOrderNoGenVoxelCopy(unsigned int _neighborOrder) const {

    return getMaxNeighborIndexFromNeighborOrderNoGenImpl(_neighborOrder,
                                                         distanceVecVoxelCopy,
                                                         hexDistanceArrayVoxelCopy);
}


unsigned int BoundaryStrategy::getMaxNeighborIndexFromNeighborOrderNoGenImpl(
        unsigned int _neighborOrder,
        const std::vector<float> &distanceVecRef,
        const std::vector<std::vector<float>> &hexDistanceArrayRef) const {

    //this function returns whatever maxNeighborIndex exist for a given neighbororder.
    // If neighborOrder is higher that maxNeighborOrder
    //this function DOES NOT generate extra offsets so the maxNeighborOrder may correspond to a smaller neighbor order
    // than in the requested _neighborOrder

    //Now determine max neighbor index from a list of neighbor offsets
    unsigned int maxNeighborIndex = 0;
    unsigned int orderCounter = 1;


    if (latticeType == HEXAGONAL_LATTICE) {
        //unsigned int indexHex=Y_EVEN|Z_EVEN;
        unsigned int indexHex = 0;
        double currentDepth = hexDistanceArrayRef[indexHex][0];


        for (int i = 0; i < hexDistanceArrayRef[indexHex].size(); ++i) {

            ++maxNeighborIndex;
            if (hexDistanceArrayRef[indexHex][i] > (currentDepth +
                                                    0.005)) {//0.005 is to account for possible numerical approximations in double or float numbers
                currentDepth = hexDistanceArrayRef[indexHex][i];
                ++orderCounter;
                if (orderCounter > _neighborOrder) {
                    maxNeighborIndex = i - 1;
                    return maxNeighborIndex;
                }
            }
        }

        return --maxNeighborIndex;

    } else {

        double currentDepth = distanceVecRef[0];

        for (int i = 0; i < distanceVecRef.size(); ++i) {
            ++maxNeighborIndex;
            if (distanceVecRef[i] > (currentDepth + 0.005)) {
                //0.005 is to account for possible numerical approximations in double or float numbers
                currentDepth = distanceVecRef[i];
                ++orderCounter;

                if (orderCounter > _neighborOrder) {
                    maxNeighborIndex = i - 1;

                    return maxNeighborIndex;
                }
            }
        }

        return --maxNeighborIndex;
    }

}


unsigned int BoundaryStrategy::getMaxNeighborOrder() {

    //determining max neighborOrder
    unsigned int maxNeighborOrder = 1;
    unsigned int previousMaxIdx = 0;
    unsigned int currentMaxIdx = 0;


    while (true) {
        currentMaxIdx = getMaxNeighborIndexFromNeighborOrderNoGen(maxNeighborOrder);

        if (previousMaxIdx == currentMaxIdx)
            break;

        previousMaxIdx = currentMaxIdx;
        ++maxNeighborOrder;
    }

    return --maxNeighborOrder;

}

//
void BoundaryStrategy::prepareNeighborListsBasedOnNeighborOrder(unsigned int _neighborOrder) {

    maxNeighborOrder = getMaxNeighborOrder();

    while ((maxNeighborOrder - 4) < _neighborOrder) {
        //making sure there is enough higher order neighbors in the list
        // this results in faster generation of neighbors for reasonable neighbor order
        prepareNeighborLists(maxDistance + 2.0);
        maxNeighborOrder = getMaxNeighborOrder();
    }

}

unsigned int BoundaryStrategy::getMaxNeighborIndexFromNeighborOrder(unsigned int _neighborOrder) {
    //this function first checks if there is  enough offsets generated
    // and if not it generates extra offsets and then returns correct neighbor order


    //Now determine max neighbor index from a list of neighbor offsets
    unsigned int maxNeighborIndex = 0;
    unsigned int orderCounter = 1;

    //we check if existing max neighbor order is less that requested neighbor order
    // and if so we generate more neighbor offsets
    if (maxNeighborOrder < _neighborOrder) {
        prepareNeighborListsBasedOnNeighborOrder(_neighborOrder);

    }

    return getMaxNeighborIndexFromNeighborOrderNoGen(_neighborOrder);

}

unsigned int BoundaryStrategy::getMaxNeighborIndexFromDepth(float depth) const {
    return getMaxNeighborIndexFromDepthImpl(
            depth,
            distanceVec,
            hexDistanceArray
    );
}

unsigned int BoundaryStrategy::getMaxNeighborIndexFromDepthVoxelCopy(float depth) const {
    return getMaxNeighborIndexFromDepthImpl(
            depth,
            distanceVecVoxelCopy,
            hexDistanceArrayVoxelCopy
    );
}

unsigned int BoundaryStrategy::getMaxNeighborIndexFromDepthImpl(
        float depth,
        const std::vector<float> &distanceVecRef,
        const std::vector<std::vector<float>> &hexDistanceArrayRef
) const {
    //Now determine max neighbor index from a list of neighbor offsets

    unsigned int maxNeighborIndex = 0;

    if (latticeType == HEXAGONAL_LATTICE) {
        //unsigned int indexHex=Y_EVEN|Z_EVEN;
        unsigned int indexHex = 0;

        for (int i = 0; i < hexDistanceArrayRef[indexHex].size(); ++i) {
            maxNeighborIndex = i;
            if (hexDistanceArrayRef[indexHex][i] > depth) {
                maxNeighborIndex = i - 1;
                break;
            }
        }
        return maxNeighborIndex;

    } else {

        for (int i = 0; i < distanceVecRef.size(); ++i) {
            maxNeighborIndex = i;
            if (distanceVecRef[i] > depth) {
                maxNeighborIndex = i - 1;
                break;
            }
        }
        return maxNeighborIndex;
    }
}


Coordinates3D<double> BoundaryStrategy::calculatePointCoordinates(const Point3D &_pt) const {
    if (latticeType == HEXAGONAL_LATTICE) {
        Coordinates3D<double> hexCoord = HexCoord(_pt);
        return hexCoord;
    } else {
        return Coordinates3D<double>(_pt.x, _pt.y, _pt.z);
    }

}


Neighbor
BoundaryStrategy::getNeighborDirect(Point3D &pt, unsigned int idx, bool checkBounds, bool calculatePtTrans) const {
    return getNeighborDirectImpl(pt, idx, checkBounds, calculatePtTrans,
                                 offsetVec, distanceVec,
                                 hexOffsetArray, hexDistanceArray);
}

Neighbor
BoundaryStrategy::getNeighborDirectVoxelCopy(Point3D &pt, unsigned int idx, bool checkBounds,
                                             bool calculatePtTrans) const {

    Neighbor n = getNeighborDirectImpl(pt, idx, checkBounds, calculatePtTrans,
                                       offsetVecVoxelCopy, distanceVecVoxelCopy,
                                       hexOffsetArrayVoxelCopy, hexDistanceArrayVoxelCopy);
#ifdef _DEBUG
    if (pt.z-n.pt.z) {
        cerr << "pt=" << pt << "n=" << n.pt << " delta z " << pt.z - n.pt.z << endl;
    }
#endif
    return n;
}


Neighbor
BoundaryStrategy::getNeighborDirectImpl(
        Point3D &pt, unsigned int idx, bool checkBounds, bool calculatePtTrans,
        const std::vector<Point3D> &offsetVec,
        const std::vector<float> &distanceVec,
        const std::vector<std::vector<Point3D>> &hexOffsetArray,
        const std::vector<std::vector<float>> &hexDistanceArray) const {
    Neighbor n;
    unsigned int indexHex;

    if (latticeType == HEXAGONAL_LATTICE) {
        indexHex = (pt.z % 3) * 2 + (pt.y % 2);
        // todo - add handling of dimension type
        n.pt = pt + hexOffsetArray[indexHex][idx];

    } else {

        n.pt = pt + offsetVec[idx];
    }

    //Here I will add condition  if (flagField[pt] ) ...

    if (!checkBounds || isValid(n.pt)) {

        // Valid Neighbor
        n.ptTrans = calculatePointCoordinates(n.pt);
        if (latticeType == HEXAGONAL_LATTICE) {
            n.distance = hexDistanceArray[indexHex][idx];
            if (calculatePtTrans)
                n.ptTrans = HexCoord(n.pt);

        } else {
            n.distance = distanceVec[idx] * lmf.lengthMF;
            if (calculatePtTrans)
                n.ptTrans = Coordinates3D<double>(pt.x, pt.y, pt.z);
        }

        return n;

    } else {

        if (regular) {
            bool x_bool;
            bool y_bool;
            bool z_bool;
            int x = n.pt.x;
            int y = n.pt.y;
            int z = n.pt.z;

            // For each coordinate, if it is not valid, apply condition
            x_bool = (isValid(x, dim.x) ? true : strategy_x->applyCondition(x, dim.x));
            y_bool = (isValid(y, dim.y) ? true : strategy_y->applyCondition(y, dim.y));
            z_bool = (isValid(z, dim.z) ? true : strategy_z->applyCondition(z, dim.z));

            // If all the coordinates of the neighbor are valid then return the
            // neighbor
            if (x_bool && y_bool && z_bool) {
                n.pt.x = x;
                n.pt.y = y;
                n.pt.z = z;
                n.ptTrans = calculatePointCoordinates(n.pt);
                if (latticeType == HEXAGONAL_LATTICE) {
                    n.distance = hexDistanceArray[indexHex][idx];
                    //                   n.ptTrans=HexCoord(n.pt);
                } else {
                    n.distance = distanceVec[idx] * lmf.lengthMF;
                    //                   n.ptTrans=Coordinates3D<double>(pt.x,pt.y,pt.z);
                }
                return n;

            } else {
                //requesed neighbor does not belong to the lattice
                n.distance = 0.0;
                return n;
            }

        }

    }

}


Point3D BoundaryStrategy::Hex2Cartesian(const Coordinates3D<double> &_coord) const {
    //this transformation takes coordinates of a point on the hex lattice
    // and returns integer coordinates of cartesian pixel that is the nearest given point on hex lattice
    //It is the inverse transformation of the one coded in HexCoord 
    int z_segments = (int) roundf(_coord.z / (sqrt(6.0) / 3.0));

    if ((z_segments % 3) == 1) {
        int y_segments = (int) roundf(_coord.y / (sqrt(3.0) / 2.0) - 2.0 / 6.0);

        if (y_segments % 2) {

            return Point3D((int) roundf(_coord.x - 0.5), y_segments, z_segments);
        } else {

            return Point3D((int) roundf(_coord.x), y_segments, z_segments);
        }

    } else if ((z_segments % 3) == 2) {

        int y_segments = (int) roundf(_coord.y / (sqrt(3.0) / 2.0) + 2.0 / 6.0);


        if (y_segments % 2) {

            return Point3D((int) roundf(_coord.x - 0.5), y_segments, z_segments);
        } else {

            return Point3D((int) roundf(_coord.x), y_segments, z_segments);
        }

    } else {

        int y_segments = (int) roundf(_coord.y / (sqrt(3.0) / 2.0));
        if (y_segments % 2) {

            return Point3D((int) roundf(_coord.x), y_segments, z_segments);
        } else {

            return Point3D((int) roundf(_coord.x - 0.5), y_segments, z_segments);
        }


    }
}

