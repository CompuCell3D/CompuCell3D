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

#ifndef STEPPABLE_H
#define STEPPABLE_H

#include "SimObject.h"

namespace CompuCell3D {
  class Simulator;

  class Steppable : public SimObject {
  public:
    int frequency;

    Steppable() : frequency(1) {}
    virtual ~Steppable() {}

//     virtual void start() = 0;
//     virtual void step(const unsigned int currentStep) = 0;
//     virtual void finish() = 0;
//     virtual std::string toString(){return "Steppable";}

    virtual void start() {};
    virtual void step(const unsigned int currentStep) {};
    virtual void finish() {};
    virtual std::string toString(){return "Steppable";}

  };
};
#endif
