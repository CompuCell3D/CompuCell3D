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

#ifndef VOLUMEPLUGIN_H
#define VOLUMEPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "VolumeDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
	class Potts3D;
	class CellG;


	class VOLUME_EXPORT VolumeEnergyParam{
	public:
		VolumeEnergyParam():targetVolume(0.0),lambdaVolume(0.0){}
		double targetVolume;
		double lambdaVolume;
		std::string typeName;
	};

	class VOLUME_EXPORT VolumePlugin : public Plugin , public EnergyFunction {
		Potts3D *potts;
		CC3DXMLElement *xmlData;
		ParallelUtilsOpenMP *pUtils;
		ExpressionEvaluatorDepot eed;
		bool energyExpressionDefined;


		std::string pluginName;

		double targetVolume;
		double lambdaVolume;
		enum FunctionType {GLOBAL=0,BYCELLTYPE=1,BYCELLID=2};
		FunctionType functionType;
		std::vector<VolumeEnergyParam> volumeEnergyParamVector;


		typedef double (VolumePlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

		VolumePlugin::changeEnergy_t changeEnergyFcnPtr;
		double changeEnergyGlobal(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double changeEnergyByCellType(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double changeEnergyByCellId(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double customExpressionFunction(double _lambdaVolume,double _targetVolume, double _volumeBefore,double _volumeAfter);

	public:
		VolumePlugin():potts(0),energyExpressionDefined(false),pUtils(0),pluginName("Volume"){};
		virtual ~VolumePlugin();

		// SimObject interface
		virtual void extraInit(Simulator *simulator);
		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual void handleEvent(CC3DEvent & _event);

		//EnergyFunction interface
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		virtual std::string steerableName();
		virtual std::string toString();
	};
};
#endif
