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

#ifndef TEMPLATEPLUGIN_H
#define TEMPLATEPLUGIN_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Plugin.h>
// // // #include <CompuCell3D/Potts3D/Stepper.h>
// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// #include <CompuCell3D/dllDeclarationSpecifier.h>
#include "TemplatePluginDLLSpecifier.h"
// // // #include <vector>
// // // #include <string>

class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class CellG;

  class TEMPLATE_EXPORT TemplateEnergyParam{
  public:
	  TemplateEnergyParam():targetTemplate(0.0),lambdaTemplate(0.0){}
	  double targetTemplate;
	  double lambdaTemplate;
	  std::string typeName;

  };

  class TEMPLATE_EXPORT TemplatePlugin : public Plugin , public EnergyFunction {
	Potts3D *potts;
	CC3DXMLElement *xmlData;
    Simulator* sim;

	std::string pluginName;

    double targetTemplate;
    double lambdaTemplate;
	enum FunctionType {GLOBAL=0,BYCELLTYPE=1,BYCELLID=2};
	FunctionType functionType;
	std::vector<TemplateEnergyParam> volumeEnergyParamVector;



	typedef double (TemplatePlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

	TemplatePlugin::changeEnergy_t changeEnergyFcnPtr;
	double changeEnergyGlobal(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
	double changeEnergyByCellType(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
	double changeEnergyByCellId(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

  public:

	  std::string TemplateChemicalFieldName;
	  std::string TemplateChemicalFieldSource;
      std::vector<Field3D<float> *> fieldVec;
    TemplatePlugin():potts(0),pluginName("Template"){};
    virtual ~TemplatePlugin();

    // SimObject interface
	virtual void extraInit(Simulator *simulator);
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

	 //EnergyFunction interface
	virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
    virtual std::string steerableName();
	virtual std::string toString();
  };
};
#endif
