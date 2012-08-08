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

#ifndef CONCENTRATIONPLUGIN_H
#define CONCENTRATIONPLUGIN_H

#include <CompuCell3D/Plugin.h>


#include <CompuCell3D/Potts3D/Stepper.h>

#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include <CompuCell3D/plugins/Concentration/Concentration.h>

namespace CompuCell3D {
  class Potts3D;

  class Cell;
  template <typename Y> class Field3DImpl;
  
  class ConcentrationPlugin : public Plugin, public CellGChangeWatcher,
		       public Stepper {
    
    BasicClassAccessor<Concentration> concentrationAccessor;
        
    
    Field3DImpl<float> *concentrationFieldPtr;
    Potts3D *potts;
        

  public:
    ConcentrationPlugin();
    virtual ~ConcentrationPlugin();

    BasicClassAccessor<Concentration> * getConcentrationAccessorPtr(){return &concentrationAccessor;}
    // SimObject interface
    virtual void init(Simulator *simulator);

    void setConcentrationFieldPtr( Field3DImpl<float> *_concentrationFieldPtr){concentrationFieldPtr=_concentrationFieldPtr;}
    
    
    ///CellChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);


   
    // Stepper interface
    virtual void step();

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
  };
};
#endif
