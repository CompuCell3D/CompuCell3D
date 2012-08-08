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

#ifndef GROWTHPLUGIN_H
#define GROWTHPLUGIN_H

#include <CompuCell3D/dllDeclarationSpecifier.h>
#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <string>

namespace CompuCell3D {
  class Simulator;
  class Potts3D;
  class Cell;
  class MitosisPlugin;
  
  class DECLSPECIFIER GrowthPlugin : public Plugin, 
                       public CellGChangeWatcher,
                       public Stepper {
    
    Simulator* sim;
    Potts3D* potts;

    unsigned int delay;
    unsigned int delta;
    int evolution;
    
    int numcellpixels;
    double density;
    unsigned int currstep;

    double dThreshold;
    double fgfThreshold;
    bool dTrigger;
    
    MitosisPlugin *mitosisPlugin;

    Dim3D dim;
    std::string autoName;
  
    int stopaer;
    int zAER;

  public:
    GrowthPlugin();
    virtual ~GrowthPlugin();

    // SimObject interface
    virtual void init(Simulator *simulator);
    virtual void extraInit(Simulator *simulator);
    // CellChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);

    double FGF(int);
    double getFGFThreshold() ;

    void grow();

    // Stepper interface
    virtual void step();
    
    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
  };
};
#endif
