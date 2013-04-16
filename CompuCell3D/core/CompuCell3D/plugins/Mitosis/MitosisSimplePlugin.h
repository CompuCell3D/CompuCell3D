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

#ifndef MITOSISSIMPLEPLUGIN_H
#define MITOSISSIMPLEPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "MitosisPlugin.h"
#include "MitosisDLLSpecifier.h"
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <Utils/Coordinates3D.h>

namespace CompuCell3D {
  class Potts3D;
  class PixelTracker;
  class PixelTrackerPlugin;
  
  class MITOSIS_EXPORT OrientationVectorsMitosis{
  public:
	  OrientationVectorsMitosis(){}


		Coordinates3D<double> semiminorVec;
		Coordinates3D<double> semimajorVec;
		

  };

  class MITOSIS_EXPORT MitosisSimplePlugin : public MitosisPlugin {
  public:

	 typedef bool (MitosisSimplePlugin::*doDirectionalMitosis2DPtr_t)();
	 doDirectionalMitosis2DPtr_t doDirectionalMitosis2DPtr;

	 //typedef OrientationVectorMitosis (MitosisSimplePlugin::*getOrientationVectorMitosis2DPtr_t)(CellG *);
	 //getOrientationVectorMitosis2DPtr_t getOrientationVectorMitosis2DPtr;

	 typedef OrientationVectorsMitosis (MitosisSimplePlugin::*getOrientationVectorsMitosis2DPtr_t)(CellG *);
	 getOrientationVectorsMitosis2DPtr_t getOrientationVectorsMitosis2DPtr;

    MitosisSimplePlugin();
    virtual ~MitosisSimplePlugin();
	 bool divideAlongMinorAxisFlag;
	 bool divideAlongMajorAxisFlag;
	 bool flag3D;
	 
     
     
	 BasicClassAccessor<PixelTracker> *pixelTrackerAccessorPtr;
	 PixelTrackerPlugin * pixelTrackerPlugin;

     virtual void handleEvent(CC3DEvent & _event);
     
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
	 void setDivideAlongMinorAxis();
	 void setDivideAlongMajorAxis();
	 

	 OrientationVectorsMitosis getOrientationVectorsMitosis(CellG * );
	 OrientationVectorsMitosis getOrientationVectorsMitosis2D_xy(CellG *);
	 OrientationVectorsMitosis getOrientationVectorsMitosis2D_xz(CellG *);
	 OrientationVectorsMitosis getOrientationVectorsMitosis2D_yz(CellG *);
	 OrientationVectorsMitosis getOrientationVectorsMitosis3D(CellG *);

	 bool doDirectionalMitosis();
	 bool doDirectionalMitosis2D_xy();
	 bool doDirectionalMitosis2D_xz();
	 bool doDirectionalMitosis2D_yz();
	 bool doDirectionalMitosis3D();
	 bool doDirectionalMitosisOrientationVectorBased(double _nx, double _ny, double _nz);

	 void setMitosisFlag(bool _flag);
	 bool getMitosisFlag();
	 
  };
};
#endif
