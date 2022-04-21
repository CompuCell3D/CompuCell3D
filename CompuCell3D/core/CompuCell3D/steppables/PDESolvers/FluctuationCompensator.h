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

#ifndef FLUCTUATIONCOMPENSATOR_H
#define FLUCTUATIONCOMPENSATOR_H

#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>

#include "PDESolversDLLSpecifier.h"


namespace CompuCell3D {

    /**
    @author T.J. Sego, Ph.D.

    This PDE solver add-on performs an algorithm partially described in

    "How Cells Integrate Complex Stimuli: The Effect of
    Feedback from Phosphoinositides and Cell Shape on Cell
    Polarization and Motility", Maree et. al., 2012

    The algorithm is performend on each cell subdomain and the medium,
    according to the following two steps:

    1.	For every spin flip of a Monte Carlo step, the solver field values
        in the copying site are exactly copied to the site of the spin flip.
    2.	At the conclusion of a Monte Carlo step, a correction factor is
        uniformly applied such that, for each subdomain, the total mass
        of each species is unchanged.
    */

    class Potts3D;

    class Simulator;

    class CellG;

    class CellInventory;

    class PixelTrackerData;

    class PixelTrackerPlugin;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class FluctuationCompensatorCellData;

    class PDESOLVERS_EXPORT FluctuationCompensator : public CellGChangeWatcher {

        ParallelUtilsOpenMP *pUtils;

        PixelTrackerPlugin *pixelTrackerPlugin;

        Dim3D fieldDim;

        std::vector<float> concentrationVecIncSMC;
        std::vector<float> concentrationVecCopiesMedium;
        std::vector <std::vector<float>> concentrationVecCopiesMediumTheads;
        std::vector<float> concentrationVecTotalMedium;

        std::vector<Field3DImpl<float> *> diffusibleFields;
        std::vector <std::string> diffusibleFieldNames;
        std::vector<Field3DImpl<float> *>::iterator diffusibleFieldsItr;
        int numFields;

        std::map<CellG *, FluctuationCompensatorCellData *> cellCompensatorData;
        std::map<CellG *, FluctuationCompensatorCellData *>::iterator cellCompensatorDataItr;

        void updateTotalCellConcentrations();

        void updateTotalMediumConcentration();

        void resetCellConcentrations();

        void resetMediumConcentration();

        bool needsInitialized;

        std::vector<float> totalCellConcentration(const CellG *_cell);

        std::vector<float> totalMediumConcentration();

        std::vector<float> totalPixelSetConcentration(std::vector <Point3D> _pixelVec);

        FluctuationCompensatorCellData *getFluctuationCompensatorCellData(CellG *_cell, bool _fullInit = true);

        std::vector<float> getConcentrationVec(const Point3D &_pt);

        void setConcentrationVec(const Point3D &_pt, std::vector<float> _vec);

    protected:

        Simulator *sim;
        Potts3D *potts;
        Automaton *automaton;
        CellInventory *cellInventory;
        CellInventory::cellInventoryIterator cell_itr;
        Field3DImpl<CellG *> *cellFieldG;

    public:

        FluctuationCompensator(Simulator *_sim);

        virtual ~FluctuationCompensator();

        std::vector <Point3D> getCellPixelVec(const CellG *_cell);

        std::vector <Point3D> getMediumPixelVec();

        // Cell field watcher interface

        void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {};

        virtual void field3DChange(const Point3D &pt, const Point3D &addPt, CellG *newCell, CellG *oldCell);


        // Solver interface

        void loadFieldName(std::string _fieldName);

        virtual void loadFields();

        // This should occur before solver integration
        virtual void applyCorrections();

        // This should occur after solver integration
        virtual void resetCorrections();

        // Updates total concentrations according to current state of all solver fields
        virtual void updateTotalConcentrations();

    };

    class PDESOLVERS_EXPORT FluctuationCompensatorCellData {

    public:

        FluctuationCompensatorCellData() {};

        FluctuationCompensatorCellData(int _numFields) {

            concentrationVecCopies = std::vector<float>(_numFields, 0.0);
            concentrationVecTotals = std::vector<float>(_numFields, 0.0);

        }

        virtual ~FluctuationCompensatorCellData() {}

        std::vector<float> concentrationVecCopies;
        std::vector<float> concentrationVecTotals;

    };

}
#endif