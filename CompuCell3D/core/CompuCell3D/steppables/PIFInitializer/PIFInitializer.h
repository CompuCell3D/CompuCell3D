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

#ifndef PIFINITIALIZER_H
#define PIFINITIALIZER_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Steppable.h>

#include <string>


#include "PIFInitializerDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class Simulator;

  class PIFINITIALIZER_EXPORT PIFInitializer : public Steppable {
    Potts3D *potts;
	Simulator *sim;

    std::string pifname;

  public:


    PIFInitializer();
    PIFInitializer(std::string);

    void setPotts(Potts3D *potts) {this->potts = potts;}

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    // Begin Steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep) {}
    virtual void finish() {}
    // End Steppable interface


  };
};
#endif
