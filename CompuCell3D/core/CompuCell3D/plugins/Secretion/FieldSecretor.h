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


#ifndef FIELDSECRETOR_H
#define FIELDSECRETOR_H

#include <CompuCell3D/CC3D.h>

#include "SecretionDLLSpecifier.h"


namespace CompuCell3D {

	class CellG;
	class Simulator;
	class BoundaryStrategy;
	class BoundaryPixelTrackerPlugin;
	class PixelTrackerPlugin;


	template <typename Y> class WatchableField3D;
	template <typename Y> class Field3DImpl;

	class SECRETION_EXPORT FieldSecretorPixelData{
	public:
		FieldSecretorPixelData(){
			pixel=Point3D();
		}
		FieldSecretorPixelData(Point3D _pixel)
			:pixel(_pixel)

		{}

		///have to define < operator if using a class in the set and no < operator is defined for this class
		bool operator<(const FieldSecretorPixelData & _rhs) const{
			return pixel.x < _rhs.pixel.x || (!(_rhs.pixel.x < pixel.x)&& pixel.y < _rhs.pixel.y)
				||(!(_rhs.pixel.x < pixel.x)&& !(_rhs.pixel.y <pixel.y )&& pixel.z < _rhs.pixel.z);
		}

		bool operator==(const FieldSecretorPixelData & _rhs)const{
			return pixel==_rhs.pixel;
		}

		///members
		Point3D pixel;


	};

	class SECRETION_EXPORT  FieldSecretor{
	private:
		double round(double d)
		{
			return floor(d + 0.5);
		}

	public:

		FieldSecretor();
		~FieldSecretor();
		Field3D<float> * concentrationFieldPtr;
		BoundaryPixelTrackerPlugin *boundaryPixelTrackerPlugin;
		PixelTrackerPlugin *pixelTrackerPlugin;
		BoundaryStrategy *boundaryStrategy;
		unsigned int maxNeighborIndex;
		WatchableField3D<CellG *> *cellFieldG;


		bool secreteInsideCell(CellG * _cell, float _amount);
		bool secreteInsideCellAtBoundary(CellG * _cell, float _amount);
		bool secreteOutsideCellAtBoundary(CellG * _cell, float _amount);
		bool secreteInsideCellAtCOM(CellG * _cell, float _amount);

		bool uptakeInsideCell(CellG * _cell, float _maxUptake, float _relativeUptake);
		bool uptakeInsideCellAtBoundary(CellG * _cell, float _maxUptake, float _relativeUptake);
		bool uptakeOutsideCellAtBoundary(CellG * _cell, float _maxUptake, float _relativeUptake);
		bool uptakeInsideCellAtCOM(CellG * _cell, float _maxUptake, float _relativeUptake);

	};

};
#endif

