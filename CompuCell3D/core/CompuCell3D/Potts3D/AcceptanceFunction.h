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

#ifndef ACCEPTANCEFUNCTION_H
#define ACCEPTANCEFUNCTION_H

namespace CompuCell3D {

  /** 
   * The Potts3D acceptance function interface.
   *
   * See DefaultAcceptanceFunction.
   */
  class AcceptanceFunction {
  public:

    /** 
     * Calculates the probability that a change should be accepted
     * based on the current temperature and the energy cost.
     * 
     * @param temp The current temperature.
     * @param change The change energy.
     * 
     * @return The probability that the change should be accepted as
     *         a number in the range [0,1).
     */
    virtual double accept(const double temp, const double change) = 0;
    virtual void setOffset(double _offset)=0;
    virtual void setK(double _k)=0;

  };
};
#endif
