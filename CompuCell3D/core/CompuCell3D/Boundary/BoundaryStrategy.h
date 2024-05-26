#ifndef BOUNDARY_STRATEGY_H
#define BOUNDARY_STRATEGY_H

#include "BoundaryDLLSpecifier.h"
#include "Boundary.h"
#include "Algorithm.h"
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Neighbor.h>
#include <CompuCell3D/CC3DExceptions.h>

#include <vector>
#include <iostream>
#include <vector>
#include "BoundaryTypeDefinitions.h"
#include <Logger/CC3DLogger.h>

using namespace std;

template<typename T>
class Coordinates3D;


namespace CompuCell3D {

    /*
     * Implements the singleton for Boundary strategies
     * Each axis is assigned its own boundary strategy
     */
    //need to include it to avoid problems with "inclomplete types"
    /*   template <typename Y>
       class Field3DImpl;*/


    template<typename T>
    class Field3DImpl;

    class BOUNDARYSHARED_EXPORT BoundaryStrategy {

        static BoundaryStrategy *singleton;
        LatticeMultiplicativeFactors lmf;
        Dim3D dim;
//        int currentStep;
        bool regular;
        Boundary *strategy_x;
        Boundary *strategy_y;
        Boundary *strategy_z;

        bool isValid(int coordinate, const int max_value) const;

        std::vector <Point3D> offsetVec;
        std::vector<float> distanceVec;
        std::vector<unsigned int> neighborOrderIndexVec;

        std::vector <std::vector<Point3D>> hexOffsetArray;
        std::vector <std::vector<float>> hexDistanceArray;
        std::vector <std::vector<unsigned int>> hexNeighborOrderIndexArray;


        // those are vectors that are used only in Potts3D during voxel copy We need them to implement 2.5D simulations
        // where pixel copy offsets vector is different from the one that is used in energy computations
        std::vector <Point3D> offsetVecVoxelCopy;
        std::vector<float> distanceVecVoxelCopy;
        std::vector<unsigned int> neighborOrderIndexVecVoxelCopy;

        std::vector <std::vector<Point3D>> hexOffsetArrayVoxelCopy;
        std::vector <std::vector<float>> hexDistanceArrayVoxelCopy;
        std::vector <std::vector<unsigned int>> hexNeighborOrderIndexArrayVoxelCopy;

//        std::vector <Point3D> *offsetVecPtr= nullptr;
//        std::vector<float> * distanceVecPtr= nullptr;
//        std::vector <std::vector<Point3D> > *hexOffsetArrayPtr = nullptr;
//        std::vector <std::vector<float> > *hexDistanceArrayPtr = nullptr;

        bool checkIfOffsetAlreadyStacked(Point3D &, std::vector <Point3D> &) const;

        bool checkEuclidianDistance(Coordinates3D<double> &, Coordinates3D<double> &, float) const;

        // method that prepares 2.5D neighbors - used in voxel copying only
        void prepare_2_5_d_voxel_copy_neighbors(std::vector <Point3D> & offsetVecTemplate,
                                                std::vector<float> & distanceVecTemplate,
                                                std::vector<unsigned int> & neighborOrderIndexVecTemplate);

        unsigned int getMaxNeighborIndexFromNeighborOrderNoGenImpl(
                unsigned int _neighborOrder,
                const std::vector<float>& distanceVecRef,
                const std::vector <std::vector<float>> & hexDistanceArrayRef) const;

        unsigned int getMaxNeighborIndexFromDepthImpl(
                float depth,
                const std::vector<float> &distanceVecRef,
                const std::vector<std::vector<float>> &hexDistanceArrayRef
        ) const;


        //void initializeQuickCheckField(Dim3D);
        float maxDistance = 0.;
        bool neighborListsInitializedFlag;

        void getOffsetsAndDistances(
                Point3D ctPt,
                float maxDistance,
                Field3DImpl<char> const &tempField,
                std::vector <Point3D> &offsetVecTmp,
                std::vector<float> &distanceVecTmp,
                std::vector<unsigned int> &neighborOrderIndexVecTmp
        ) const;



        // determines actual size of the lattice in x,y,z directions
        // the dimensions are different for hex and square lattice
        Coordinates3D<double> latticeSizeVector;
        //determines maximum allowed point coordinate which is considered to be still in the lattice
        // used in distanceInvariant calculations in NumericalUtils.cpp
        Coordinates3D<double> latticeSpanVector;

        LatticeType latticeType;
        DimensionType dimensionType;
        int maxOffset = 0;

        Algorithm *algorithm;
        unsigned int maxNeighborOrder;

        BoundaryStrategy(const string& boundary_x, const string& boundary_y,
                         const string& boundary_z, string alg, int index, int size, string inputfile,
                         LatticeType latticeType = SQUARE_LATTICE, DimensionType dimensionType=DIM_DEFAULT);

        BoundaryStrategy();

        std::vector<unsigned int> boundaryConditionIndicator;


    public:
        Coordinates3D<double>
        //maximum allowed point coordinate which is considered to be still in the lattice
        getLatticeSpanVector() const { return latticeSpanVector; }
        Coordinates3D<double>
        //actual size of the lattice in x,y,z directions
        getLatticeSizeVector() const { return latticeSizeVector; }
        LatticeType getLatticeType() const { return latticeType; }

        float getMaxDistance() const { return maxDistance; }

        std::vector<unsigned int> getBoundaryConditionIndicator() {
            return boundaryConditionIndicator;
        };

        ~BoundaryStrategy();

        static void instantiate(string boundary_x, string boundary_y,
                                string boundary_z, string alg,
                                int index, int size, string inputfile,
                                LatticeType latticeType = SQUARE_LATTICE,DimensionType dimensionType=DIM_DEFAULT) {


            if (!singleton) {

                singleton = new BoundaryStrategy(boundary_x, boundary_y,
                                                 boundary_z, alg, index, size, inputfile, latticeType, dimensionType);

            }
        }

        static BoundaryStrategy *getInstance() {
            using namespace std;
            if (!singleton) {
				CC3D_Log(LOG_DEBUG) << "CONSTRUCTING an instance";
                singleton = new BoundaryStrategy();
            }

            return singleton;
        }

        static void destroy() {
			CC3D_Log(LOG_DEBUG) << "destroy fcn: destroying boundary strategy";
            if (singleton)
			{
				CC3D_Log(LOG_DEBUG) << "will destroy boundary strategy singleton = " << singleton;

				delete singleton;
                singleton = 0;
				CC3D_Log(LOG_DEBUG) << "BoundaryStrategy singleton is DEAD!";
            }
            else
            {
                CC3D_Log(LOG_DEBUG) << "BoundaryStrategy singleton WAS NOT DESTROYED BECAUSE IT IS DEAD!";
            }

        }

        double calculateDistance(Coordinates3D<double> &, Coordinates3D<double> &) const;

        Point3D getNeighbor(const Point3D &pt, unsigned int &token, double &distance, bool checkBounds = true) const;

        Coordinates3D<double> HexCoord(const Point3D &_pt) const;

        Point3D Hex2Cartesian(const Coordinates3D<double> &_coord) const;

        Point3D getNeighborCustomDim(const Point3D &pt, unsigned int &token,
                                     double &distance, const Dim3D &customDim,
                                     bool checkBounds = true) const; // this function returns neighbor but takes extra dim as an argument  menaning we can use it for lattices of size different than simulation dim. used in prepareOffsets functions

        bool isValid(const Point3D &pt) const;

        bool isValidCustomDim(const Point3D &pt, const Dim3D &customDim) const;

        void prepareNeighborListsSquare(float _maxDistance = 4.0);

        LatticeMultiplicativeFactors generateLatticeMultiplicativeFactors(LatticeType _latticeType, Dim3D dim);

        LatticeMultiplicativeFactors getLatticeMultiplicativeFactors() const;


        void prepareNeighborListsHex(float _maxDistance = 4.0);

        void prepareNeighborLists(float _maxDistance = 4.0);

        unsigned int getMaxNeighborIndexFromNeighborOrderNoGen(unsigned int _neighborOrder) const;
        unsigned int getMaxNeighborIndexFromNeighborOrderNoGenVoxelCopy(unsigned int _neighborOrder) const;

        unsigned int getMaxNeighborOrder();

        void prepareNeighborListsBasedOnNeighborOrder(unsigned int _neighborOrder);

        unsigned int getMaxNeighborIndexFromNeighborOrder(unsigned int _neighborOrder);

        unsigned int getMaxNeighborIndexFromDepth(float depth) const;
        unsigned int getMaxNeighborIndexFromDepthVoxelCopy(float depth) const;

        Neighbor
        getNeighborDirect(Point3D &pt, unsigned int idx, bool checkBounds = true, bool calculatePtTrans = false) const;

        Neighbor
        getNeighborDirectVoxelCopy(Point3D &pt, unsigned int idx, bool checkBounds = true, bool calculatePtTrans = false) const;


        Neighbor
        getNeighborDirectImpl(
                Point3D &pt, unsigned int idx, bool checkBounds, bool calculatePtTrans,
                const std::vector <Point3D> & offsetVec,
                const std::vector<float> &distanceVec,
                const std::vector <std::vector<Point3D>> & hexOffsetArray,
                const std::vector <std::vector<float>> & hexDistanceArray) const;

        Coordinates3D<double> calculatePointCoordinates(const Point3D &_pt) const;

        void setDim(const Dim3D theDim);

        const std::vector <Point3D> &getOffsetVec(Point3D &pt) const;

        const std::vector <Point3D> &getOffsetVec() const;

        void getHexOffsetArray(std::vector <std::vector<Point3D>> &hoa) const {
            hoa = hexOffsetArray;
        }

        int getMaxOffset() const { return maxOffset; }

    };
};

#endif
