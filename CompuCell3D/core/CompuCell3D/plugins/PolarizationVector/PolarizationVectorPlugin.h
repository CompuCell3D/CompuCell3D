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

#ifndef POLARIZATIONVECTORPLUGIN_H
#define POLARIZATIONVECTORPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>


// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
// // // #include <vector>
#include "PolarizationVector.h"


#include "PolarizationVectorDLLSpecifier.h"

namespace CompuCell3D {

  class CellG;
  class POLARIZATIONVECTOR_EXPORT PolarizationVectorPlugin : public Plugin{

    BasicClassAccessor<PolarizationVector> polarizationVectorAccessor;

  public:

    PolarizationVectorPlugin();
    virtual ~PolarizationVectorPlugin();

    BasicClassAccessor<PolarizationVector> * getPolarizationVectorAccessorPtr(){return &polarizationVectorAccessor;}

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);

    void setPolarizationVector(CellG * _cell, float _x, float _y, float _z);
	std::vector<float> getPolarizationVector(CellG * _cell);
  };
};
#endif
