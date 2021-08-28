/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef BOUNDARY_STRATEGY_H
#define BOUNDARY_STRATEGY_H

 //#include <CompuCell3D/dllDeclarationSpecifier.h>
#include "BoundaryDLLSpecifier.h"
#include "Boundary.h"
#include "Algorithm.h"
// #include <CompuCell3D/Field3D/Field3DImpl.h>

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Neighbor.h>
#include <CompuCell3D/CC3DExceptions.h>

#include <vector>
#include <iostream>
#include <vector>
#include "BoundaryTypeDefinitions.h"

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

	 //enum HexOddEvenFlags{Y_ODD=1,Z_ODD=2,X_ODD=0,Y_EVEN=0,Z_EVEN=0,X_EVEN=0};




	template <typename T>
	class Field3DImpl;

	class BOUNDARYSHARED_EXPORT BoundaryStrategy {

		static BoundaryStrategy *singleton;
		LatticeMultiplicativeFactors lmf;
		Dim3D dim;
		int currentStep;
		bool regular;
		Boundary *strategy_x;
		Boundary *strategy_y;
		Boundary *strategy_z;

		bool isValid(const int coordinate, const int max_value)const;

		std::vector<Point3D> offsetVec;
		std::vector<float> distanceVec;
		std::vector<unsigned int> neighborOrderIndexVec;
		bool checkIfOffsetAlreadyStacked(Point3D &, std::vector<Point3D> &)const;
		bool checkEuclidianDistance(Coordinates3D<double> &, Coordinates3D<double> &, float)const;

		//void initializeQuickCheckField(Dim3D);
		float maxDistance;
		bool neighborListsInitializedFlag;

		void getOffsetsAndDistances(
			Point3D ctPt,
			float maxDistance,
			Field3DImpl<char> const & tempField,
			std::vector<Point3D> & offsetVecTmp,
			std::vector<float> &distanceVecTmp,
			std::vector<unsigned int> &neighborOrderIndexVecTmp
			)const;


		std::vector<std::vector<Point3D> > hexOffsetArray;
		std::vector<std::vector<float> > hexDistanceArray;
		//                std::vector<std::vector<float> > hexDistanceArrayScaled;
		std::vector<std::vector<unsigned int> > hexNeighborOrderIndexArray;

		Coordinates3D<double> latticeSizeVector; // determines actual size of the lattice in x,y,z directions - the dimensions are different for hex and square lattice
		Coordinates3D<double> latticeSpanVector; //determines maximum allowed point coordinate which is considered to be still in the lattice - used in distanceInvariant calculations in NumericalUtils.cpp

		LatticeType latticeType;
		int maxOffset;


		Algorithm* algorithm;
		unsigned int maxNeighborOrder;

		BoundaryStrategy(string boundary_x, string boundary_y,
			string boundary_z, string alg, int index, int size, string inputfile, LatticeType latticeType = SQUARE_LATTICE);


		BoundaryStrategy();
        std::vector<unsigned int> boundaryConditionIndicator;


	public:
		Coordinates3D<double> getLatticeSpanVector()const { return latticeSpanVector; } //maximum allowed point coordinate which is considered to be still in the lattice
		Coordinates3D<double> getLatticeSizeVector()const { return latticeSizeVector; } //actual size of the lattice in x,y,z directions
		LatticeType getLatticeType()const { return latticeType; }
		float getMaxDistance()const { return maxDistance; }
        std::vector<unsigned int> getBoundaryConditionIndicator() {
            return boundaryConditionIndicator;
        };

		~BoundaryStrategy();

		static void instantiate(string boundary_x, string boundary_y,
			string boundary_z, string alg,
			int index, int size, string inputfile, LatticeType latticeType = SQUARE_LATTICE) {


			if (!singleton) {

				singleton = new BoundaryStrategy(boundary_x, boundary_y,
					boundary_z, alg, index, size, inputfile, latticeType);


			}

		}
		static BoundaryStrategy *getInstance() {
			using namespace std;
			if (!singleton) {
				cerr << "CONSTRUCTING an instance" << endl;
				singleton = new BoundaryStrategy();
			}

			return singleton;
		}

		static void destroy() {
			cerr << "destroy fcn: destroying bondary strategy" << endl;
			if (singleton)
			{
				cerr << "will destroy boundary strategy singleton = " << singleton << endl;

				delete singleton;
				singleton = 0;

				cerr << "BoundaryStrategy singleton is DEAD!\n";
			}
			else
			{
				cerr << "BoundaryStrategy singleton WAS NOT DeSTROYED BECAUSE IT IS DEAD!\n";
			}

		}

		double calculateDistance(Coordinates3D<double> &, Coordinates3D<double> &)const;

		Point3D getNeighbor(const Point3D& pt, unsigned int& token, double& distance, bool checkBounds = true)const;
		Coordinates3D<double> HexCoord(const Point3D & _pt)const;
        Point3D Hex2Cartesian(const Coordinates3D<double> & _coord)const;
		Point3D getNeighborCustomDim(const Point3D& pt, unsigned int& token,
			double& distance, const Dim3D & customDim, bool checkBounds = true)const; // this function returns neighbor but takes extra dim as an argument  menaning we can use it for lattices of size different than simulation dim. used in prepareOffsets functions

		bool isValid(const Point3D &pt) const;
		bool isValidCustomDim(const Point3D &pt, const Dim3D & customDim) const;
		void prepareNeighborListsSquare(float _maxDistance = 4.0);
		LatticeMultiplicativeFactors generateLatticeMultiplicativeFactors(LatticeType _latticeType, Dim3D dim);
		LatticeMultiplicativeFactors getLatticeMultiplicativeFactors()const;


		void prepareNeighborListsHex(float _maxDistance = 4.0);
		void prepareNeighborLists(float _maxDistance = 4.0);
		unsigned int getMaxNeighborIndexFromNeighborOrderNoGen(unsigned int _neighborOrder) const;
		unsigned int getMaxNeighborOrder();
		void prepareNeighborListsBasedOnNeighborOrder(unsigned int _neighborOrder);
		unsigned int getMaxNeighborIndexFromNeighborOrder(unsigned int _neighborOrder);
		unsigned int getMaxNeighborIndexFromDepth(float depth);
		Neighbor getNeighborDirect(Point3D & pt, unsigned int idx, bool checkBounds = true, bool calculatePtTrans = false)const;
		Coordinates3D<double> calculatePointCoordinates(const Point3D & _pt)const;
		void setDim(const Dim3D theDim);

		const std::vector<Point3D> & getOffsetVec(Point3D & pt) const;
		const std::vector<Point3D> & getOffsetVec() const;

        void getHexOffsetArray(std::vector<std::vector<Point3D> > &hoa)const{
        	hoa=hexOffsetArray;
        }
        int getMaxOffset() const { return maxOffset; }

	};
};

#endif
