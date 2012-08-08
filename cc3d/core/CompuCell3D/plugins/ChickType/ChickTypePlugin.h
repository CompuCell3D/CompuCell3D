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

#ifndef CHICKTYPEPLUGIN_H
#define CHICKTYPEPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/CellChangeWatcher.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <string>

namespace CompuCell3D {
  class Potts3D;
  class Cell;

  class ChickTypePlugin : public Plugin, public Automaton {
    Simulator* sim;
    Potts3D* potts;
    std::string fieldSource;
    std::string fieldName;
    float threshold;
  public:
    ChickTypePlugin();
    virtual ~ChickTypePlugin();

    // SimObject interface
    virtual void init(Simulator *simulator);

    unsigned char getCellType(const CellG *cell) const;
    std::string getTypeName(const char type) const;
    unsigned char getTypeId(const std::string typeName) const;

    float getThreshold() {return threshold;}
    float getConcentration(Point3D pt);

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
  };
};
#endif
