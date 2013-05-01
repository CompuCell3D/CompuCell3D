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

#ifndef CHEMOTAXISDICTYPLUGIN_H
#define CHEMOTAXISDICTYPLUGIN_H

 #include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>

// // // #include <CompuCell3D/Plugin.h>
// // // //#include <CompuCell3D/Potts3D/Stepper.h>
// // // #include <BasicUtils/BasicClassGroup.h>
// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include "ChemotaxisDictyDLLSpecifier.h"

// // // template <typename T>
// // // class BasicClassAccessor;

class CC3DXMLElement;

namespace CompuCell3D {

  

  template <class T>
  class Field3D;
  template <class T>
  class Field3DImpl;


  class Potts3D;
  class Simulator;
  class SimpleClock;



  class CHEMOTAXISDICTY_EXPORT ChemotaxisDictyPlugin : public Plugin, public CellGChangeWatcher,public EnergyFunction{

    Simulator* sim;
    Field3D<float>* concentrationField;
	//EnergyFunction Data    
    Field3D<float> *field;

   
    Potts3D *potts;
    BasicClassAccessor<SimpleClock> *simpleClockAccessorPtr;
    
    double lambda;

    std::string chemicalFieldSource;
    std::string chemicalFieldName;
    // bool chemotax;
    bool gotChemicalField;

    std::vector<unsigned char> nonChemotacticTypeVector;
	 CC3DXMLElement * xmlData;

    
  public:
    ChemotaxisDictyPlugin();
    virtual ~ChemotaxisDictyPlugin();

	//plugin interface
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
	virtual void extraInit(Simulator *_simulator);


    ///CellChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);

    
    virtual void step(){}
    


	 //energyFunction interface
	  virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);

		//steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();




	//EnergyFunction methods
    double getConcentration(const Point3D &pt);

    double getLambda() {return lambda;}
    

    Field3D<float>* getField() {return (Field3D<float>* )field;}

    void initializeField();


  };
};
#endif
