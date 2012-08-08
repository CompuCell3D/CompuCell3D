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

#ifndef DEFAULTACCEPTANCEFUNCTION_H
#define DEFAULTACCEPTANCEFUNCTION_H

#include "AcceptanceFunction.h"

#include <math.h>

namespace CompuCell3D {

  /** 
   * The default Boltzman acceptance function.
   */
  class DefaultAcceptanceFunction: public AcceptanceFunction {
    double k;
    double offset;

  public:
    DefaultAcceptanceFunction(const double _k = 1.0,const double _offset=0.0) : k(_k),offset(_offset) {}
    virtual void setOffset(double _offset){offset=_offset;}
    virtual void setK(double _k){k=_k;}
    double accept(const double temp, const double change) {
      if (temp <= 0) {
	if (change > 0) return  0.0;
	if (change == 0) return 0.5;
	return  1.0;

      } else {
	if (change <= offset) return 1.0;
	return exp(-(change -offset)/ (k * temp));
      }
    }
  };
};
#endif
