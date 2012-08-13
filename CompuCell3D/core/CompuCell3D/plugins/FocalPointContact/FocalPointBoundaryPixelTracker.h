#ifndef FOCALPOINTFOCALPOINTBOUNDARYPIXELTRACKER_H
#define FOCALPOINTFOCALPOINTBOUNDARYPIXELTRACKER_H


/**
@author m
*/
#include <map>
#include <CompuCell3D/Field3D/Point3D.h>
#include "FocalPointContactDLLSpecifier.h"
#include <iostream>
#include <list>
#include <set>
#include <map>

namespace CompuCell3D {

   
   class CellG;
   //common surface area is expressed in unitsa of elementary surfaces not actual physical units. If necessary it may 
   //need to be transformed to physical units by multiplying it by surface latticemultiplicative factor 
   class FOCALPOINTCONTACT_EXPORT FocalPointBoundaryPixelTrackerData{
      public:
      FocalPointBoundaryPixelTrackerData():cadLevel(0.0),inContact(false),cell1(0),cell2(0){
				// pixel=Point3D();
			}
         // FocalPointBoundaryPixelTrackerData(Point3D _pixel)
         // :pixel(_pixel)
          
          // {}
         
         ///have to define < operator if using a class in the set and no < operator is defined for this class
         // bool operator<(const FocalPointBoundaryPixelTrackerData & _rhs) const{
            // return pixel.x < _rhs.pixel.x || (!(_rhs.pixel.x < pixel.x)&& pixel.y < _rhs.pixel.y)
					// ||(!(_rhs.pixel.x < pixel.x)&& !(_rhs.pixel.y <pixel.y )&& pixel.z < _rhs.pixel.z);
         // }
         
         // bool operator==(const FocalPointBoundaryPixelTrackerData & _rhs)const{
            // return pixel==_rhs.pixel;
         // }
         
         ///members
         // Point3D pixel;
			//double targetDistance;
			double lambda;
			Point3D pt1;
			Point3D pt2;
			CellG * cell1;
			CellG * cell2;

			

         double cadLevel;
			bool inContact;

			bool operator==(const FocalPointBoundaryPixelTrackerData & _rhs)const{
            return (pt1==_rhs.pt1)&&(pt2==_rhs.pt2)&&(cell1==_rhs.cell1)&&(cell2==_rhs.cell2);
         }

                  
   };


	inline std::ostream & operator<<(std::ostream & _out,const FocalPointBoundaryPixelTrackerData & _fpbPixData){
		using namespace std;
		_out<<"********FocalPointBoundaryPixelTrackerData**********"<<std::endl;
		_out<<"pt1="<<_fpbPixData.pt1<<" pt2="<<_fpbPixData.pt2<<std::endl;
		_out<<"cell1="<<_fpbPixData.cell1<<" cell2="<<_fpbPixData.cell2;
		_out<<"cell1.id="<<_fpbPixData.cell1->id<<" cell2.id="<<_fpbPixData.cell2->id;
		return _out;

	}
   
   
   class FOCALPOINTCONTACT_EXPORT  FocalPointBoundaryPixelTracker{
      public:
         FocalPointBoundaryPixelTracker(){};
         
         ~FocalPointBoundaryPixelTracker(){};
         std::map<Point3D, FocalPointBoundaryPixelTrackerData> pixelCadMap; //stores cadherin levels for boundary pixels
			std::list<FocalPointBoundaryPixelTrackerData> focalJunctionList;
         std::set<Point3D> junctionPointsSet;
			double totalCadLevel;
			double reservoir;
			int junctionPool;
			
         
   };
};


#endif
