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

#ifndef GLOBALBOUNDARYPIXELTRACKERPLUGIN_H
#define GLOBALBOUNDARYPIXELTRACKERPLUGIN_H
#include <CompuCell3D/CC3D.h>
#include "GlobalBoundaryPixelTrackerDLLSpecifier.h"
#include <unordered_set>

class CC3DXMLElement;
namespace CompuCell3D {

	class Cell;
	class Field3DIndex;
	class Potts3D;
	template <class T> class Field3D;
	template <class T> class WatchableField3D;
	class BoundaryStrategy;

	//struct Point3DHasher {
	//public:
	//	size_t operator()(const Point3D & pt) const {

	//		long long int hash_val = 1e12*pt.x+1e6*pt.y+pt.z;
	//		return std::hash<long long int>()(hash_val);
	//	}
	//};

	//// Custom comparator that compares the string objects by length
	//struct Point3DComparator {
	//public:
	//	bool operator()(const Point3D & pt1 , const Point3D & pt2) const {
	//		long long int hash_val_1 = 1e12*pt1.x + 1e6*pt1.y + pt1.z;
	//		long long int hash_val_2 = 1e12*pt2.x + 1e6*pt2.y + pt2.z;
	//		if (hash_val_1 == hash_val_2)
	//			return true;
	//		else
	//			return false;
	//	}
	//};


	class GLOBALBOUNDARYPIXELTRACKER_EXPORT GlobalBoundaryPixelTrackerPlugin : public Plugin, public CellGChangeWatcher {

		ParallelUtilsOpenMP *pUtils;
		ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

		//WatchableField3D<CellG *> *cellFieldG;
		Dim3D fieldDim;
		Simulator *simulator;
		Potts3D* potts;
		unsigned int maxNeighborIndex;
		float container_refresh_fraction;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;
		//std::set<Point3D> * boundaryPixelSetPtr;
		//std::set<Point3D>  * justInsertedBoundaryPixelSetPtr;
		//std::set<Point3D>  * justDeletedBoundaryPixelSetPtr;
		std::unordered_set<Point3D, Point3DHasher, Point3DComparator> *  boundaryPixelSetPtr;
		std::unordered_set<Point3D, Point3DHasher, Point3DComparator> *justInsertedBoundaryPixelSetPtr;
		std::unordered_set<Point3D, Point3DHasher, Point3DComparator> *justDeletedBoundaryPixelSetPtr;
		std::vector<Point3D> * boundaryPixelVectorPtr;

		//std::unordered_set<Point3D, Point3DHasher, Point3DComparator> justInsertedBoundaryPixelSet;

		void insertPixel(Point3D & pt);
		void removePixel(Point3D & pt);

		void refreshContainers();


	public:
		GlobalBoundaryPixelTrackerPlugin();
		virtual ~GlobalBoundaryPixelTrackerPlugin();


		// Field3DChangeWatcher interface
		virtual void field3DChange(const Point3D &pt, CellG *newCell,
			CellG *oldCell);

		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);
		virtual void extraInit(Simulator *_simulators);
		virtual void handleEvent(CC3DEvent & _event);

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);
		virtual std::string steerableName();
		virtual std::string toString();



	};
};
#endif
