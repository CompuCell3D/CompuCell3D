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

#ifndef BLOBFIELDINITIALIZER_H
#define BLOBFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Steppable.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <vector>
// // // #include <string>
//#include "BlobInitializerParseData.h"

// #include <CompuCell3D/dllDeclarationSpecifier.h>
#include "BlobFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class Simulator;
  class Dim3D;
  class Point3D;

  class BLOBFIELDINITIALIZER_EXPORT BlobFieldInitializerData{
      public:
      BlobFieldInitializerData():
      radius(1),width(1),gap(0),randomize(false)
      {}
      Point3D center;
      int radius;
      std::vector<std::string> typeNames;
      std::string typeNamesString;
      int width;
      int gap;
      bool randomize;

      void Gap(int _gap){gap=_gap;}
      void Width(int _width){width=_width;}
      void Center(Point3D _center){center=_center;}
      void Radius(int _radius){radius=_radius;}
      void Types(std::string  _type){
         typeNames.push_back(_type);
      }
   };

    class BLOBFIELDINITIALIZER_EXPORT EngulfmentData{
      public:
      EngulfmentData():engulfment(false){}
      std::string bottomType;
      std::string topType;
      std::string engulfmentCoordinate;
      unsigned int engulfmentCutoff;
      bool engulfment;
      void BottomType(std::string _bottomType){bottomType=_bottomType;}
      void TopType(std::string _topType){topType=_topType;}
      void EngulfmentCoordinate(std::string _engulfmentCoordinate){engulfmentCoordinate=_engulfmentCoordinate;}
      void EngulfmentCutoff(unsigned int _engulfmentCutoff){engulfmentCutoff=_engulfmentCutoff;}
  };
   
  class BLOBFIELDINITIALIZER_EXPORT BlobFieldInitializer : public Steppable {
  protected:
    Potts3D *potts;
	Simulator *sim;

//     int gap;
//     int width;
//     int radius;
    Dim3D blobDim;
    bool cellSortInit;
	 std::vector<BlobFieldInitializerData> blobInitializerData;
	 BlobFieldInitializerData oldStyleInitData;
	 EngulfmentData engulfmentData;
//     bool engulfment;
//     std::string bottomType;
//     std::string topType;
//     std::string engulfmentCoordinate;
//     unsigned int engulfmentCutoff;
//     std::vector<BlobFieldInitializerData> initDataVec;
    void layOutCells(const BlobFieldInitializerData & _initData);
    unsigned char initCellType(const BlobFieldInitializerData & _initData);

    
  public:
    //BlobInitializerParseData bipd;
    //BlobInitializerParseData * bipdPtr;
	 CC3DXMLElement *moduleXMLDataPtr;
    BlobFieldInitializer();
    
    virtual ~BlobFieldInitializer(){}
    void setPotts(Potts3D *potts) {this->potts = potts;}
    double distance(double, double, double, double, double, double);
    Dim3D getBlobDim(){return blobDim;}
    
    // SimObject interface
	 virtual void init(Simulator *simulator,  CC3DXMLElement * _xmlData=0);

    // Begin Steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep) {}
    virtual void finish() {}
    // End Steppable interface
    Dim3D getBlobDimensions(const Dim3D & dim,int size);
    void initializeCellTypesCellSort();
    void initializeEngulfment();
    virtual std::string steerableName();
	 virtual std::string toString();
  };
};
#endif
