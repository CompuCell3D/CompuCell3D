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

#ifndef CELL_H
#define CELL_H


#ifndef PyObject_HEAD
struct _object; //forward declare
typedef _object PyObject; //type redefinition
#endif

class BasicClassGroup;

namespace CompuCell3D {

  /**
   * A Potts3D cell.
   */

   class CellG{
   public:
      typedef unsigned char CellType_t;
      CellG():
        volume(0),
        targetVolume(0.0),
        lambdaVolume(0.0),
        surface(0),
        targetSurface(0.0),
        lambdaSurface(0.0),
		clusterSurface(0.0),
		targetClusterSurface(0.0),
		lambdaClusterSurface(0.0),
        type(0),
        xCM(0),yCM(0),zCM(0),
		xCOM(0),yCOM(0),zCOM(0),
		xCOMPrev(0),yCOMPrev(0),zCOMPrev(0),
        iXX(0), iXY(0), iXZ(0), iYY(0), iYZ(0), iZZ(0),
        lX(0.0),
        lY(0.0),
        lZ(0.0),
        lambdaVecX(0.0),
        lambdaVecY(0.0),
        lambdaVecZ(0.0),
        flag(0),
        id(0),
        clusterId(0),
		fluctAmpl(-1.0),
		connectivityOn(false),
        extraAttribPtr(0),
        pyAttrib(0)
      {}
      long volume;
      float targetVolume;
      float lambdaVolume;
      double surface;
      float targetSurface;
      float angle;
      float lambdaSurface;
	  double clusterSurface;
	  float targetClusterSurface;
	  float lambdaClusterSurface;
      unsigned char type;
      unsigned char subtype;
      double xCM,yCM,zCM; // numerator of center of mass expression (components)
	  double xCOM,yCOM,zCOM; // numerator of center of mass expression (components)
	  double xCOMPrev,yCOMPrev,zCOMPrev; // previous center of mass 
      double iXX, iXY, iXZ, iYY, iYZ, iZZ; // tensor of inertia components
      float lX,lY,lZ; //orientation vector components - set by MomentsOfInertia Plugin - read only
      float ecc; // cell eccentricity
      float lambdaVecX,lambdaVecY,lambdaVecZ; // external potential lambda vector components
      unsigned char flag;
      float averageConcentration;
      long id;
      long clusterId;
	  double fluctAmpl;
	  bool connectivityOn;
      BasicClassGroup *extraAttribPtr;

      PyObject *pyAttrib;
   };


  class Cell {
  };

  class CellPtr{
   public:
   Cell * cellPtr;
  };
};
#endif
