#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h>


using namespace CompuCell3D;
using namespace std;


#include "FieldSecretor.h"

FieldSecretor::FieldSecretor() :
        concentrationFieldPtr(0),
        boundaryPixelTrackerPlugin(0),
        pixelTrackerPlugin(0),
        boundaryStrategy(0),
        maxNeighborIndex(0),
        cellFieldG(0) {}

FieldSecretor::~FieldSecretor() {}

// NOTICE, exceptions are thrown from the python wrapper functions defined in CompuCellExtraDeclarations.i

FieldSecretorResult FieldSecretor::_secreteInsideCellTotalCount(CellG *_cell, float _amount) {

    FieldSecretorResult res;
    if (!pixelTrackerPlugin) {
        res.success_flag = false;
        return res;

    }


    ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();
    set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

    for (set<PixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        concentrationFieldPtr->set(sitr->pixel, concentrationFieldPtr->get(sitr->pixel) + _amount);

    }


    res.tot_amount = pixelSetRef.size() * _amount;


    return res;

}


bool FieldSecretor::_secreteInsideCell(CellG *_cell, float _amount) {

    FieldSecretorResult res = _secreteInsideCellTotalCount(_cell, _amount);
    return res.success_flag;

}


FieldSecretorResult FieldSecretor::_secreteInsideCellConstantConcentrationTotalCount(CellG *_cell, float _amount) {

    FieldSecretorResult res;

    if (!pixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    float total_amount = 0.0;
    ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();
    set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;
    for (set<PixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        //this may cause total "secreted" amount to be negative - this is property of constant concentration secretion mode
        total_amount += _amount - concentrationFieldPtr->get(sitr->pixel);

        concentrationFieldPtr->set(sitr->pixel, _amount);

    }

    res.tot_amount = total_amount;

    return res;


}

bool FieldSecretor::_secreteInsideCellConstantConcentration(CellG *_cell, float _amount) {
    FieldSecretorResult res = _secreteInsideCellConstantConcentrationTotalCount(_cell, _amount);
    return res.success_flag;
}


FieldSecretorResult FieldSecretor::_secreteInsideCellAtBoundaryTotalCount(CellG *_cell, float _amount) {
    FieldSecretorResult res;

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;


    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        concentrationFieldPtr->set(sitr->pixel, concentrationFieldPtr->get(sitr->pixel) + _amount);

    }


    res.tot_amount = pixelSetRef.size() * _amount;

    return res;


}

bool FieldSecretor::_secreteInsideCellAtBoundary(CellG *_cell, float _amount) {

    FieldSecretorResult res = _secreteInsideCellAtBoundaryTotalCount(_cell, _amount);
    return res.success_flag;

}


FieldSecretorResult FieldSecretor::_secreteInsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _amount,
                                                                                       const std::vector<unsigned char> &_onContactVec) {

    FieldSecretorResult res;

    set<unsigned char> onContactSet(_onContactVec.begin(), _onContactVec.end());

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    float total_amount = 0.0;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);
            if (nCell != _cell && !nCell && onContactSet.find(0) != onContactSet.end()) {
                //user requested secrete on contact with medium and we found medium pixel
                concentrationFieldPtr->set(sitr->pixel, concentrationFieldPtr->get(sitr->pixel) + _amount);
                total_amount += _amount;
                break; //after secreting do not try to secrete more
            }

            if (nCell != _cell && nCell && onContactSet.find(nCell->type) != onContactSet.end()) {
                //user requested secretion on contact with cell type whose pixel we have just found
                concentrationFieldPtr->set(sitr->pixel, concentrationFieldPtr->get(sitr->pixel) + _amount);
                total_amount += _amount;
                break;//after secreting do not try to secrete more
            }

        }

    }

    res.tot_amount = total_amount;

    return res;

}

bool FieldSecretor::_secreteInsideCellAtBoundaryOnContactWith(CellG *_cell, float _amount,
                                                              const std::vector<unsigned char> &_onContactVec) {

    FieldSecretorResult res = _secreteInsideCellAtBoundaryOnContactWithTotalCount(_cell, _amount, _onContactVec);
    return res.success_flag;
}


FieldSecretorResult FieldSecretor::_secreteOutsideCellAtBoundaryTotalCount(CellG *_cell, float _amount) {

    FieldSecretorResult res;
    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    set <FieldSecretorPixelData> visitedPixels;

    float total_amount = 0.0;
    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {


        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);
            if (nCell != _cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt)) == visitedPixels.end()) {

                concentrationFieldPtr->set(nPt, concentrationFieldPtr->get(nPt) + _amount);
                total_amount += _amount;
                visitedPixels.insert(FieldSecretorPixelData(nPt));

            }

        }

    }

    res.tot_amount = total_amount;

    return res;

}

bool FieldSecretor::_secreteOutsideCellAtBoundary(CellG *_cell, float _amount) {

    FieldSecretorResult res = _secreteOutsideCellAtBoundaryTotalCount(_cell, _amount);
    return res.success_flag;

}


FieldSecretorResult FieldSecretor::_secreteOutsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _amount,
                                                                                        const std::vector<unsigned char> &_onContactVec) {
    FieldSecretorResult res;
    set<unsigned char> onContactSet(_onContactVec.begin(), _onContactVec.end());

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    set <FieldSecretorPixelData> visitedPixels;

    float total_amount = 0.0;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {


        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);
            if (nCell != _cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt)) == visitedPixels.end()) {
                if (!nCell && onContactSet.find(0) != onContactSet.end()) {
                    //checking if the unvisited pixel belongs to Medium and if Medium is a  cell type listed in the onContactSet
                    concentrationFieldPtr->set(nPt, concentrationFieldPtr->get(nPt) + _amount);
                    total_amount += _amount;
                    visitedPixels.insert(FieldSecretorPixelData(nPt));
                }

                if (nCell && onContactSet.find(nCell->type) != onContactSet.end()) {
                    //checking if the unvisited pixel belongs to a  cell type listed in the onContactSet
                    concentrationFieldPtr->set(nPt, concentrationFieldPtr->get(nPt) + _amount);
                    total_amount += _amount;
                    visitedPixels.insert(FieldSecretorPixelData(nPt));
                }

            }

        }

    }


    res.tot_amount = total_amount;

    return res;
}

bool FieldSecretor::_secreteOutsideCellAtBoundaryOnContactWith(CellG *_cell, float _amount,
                                                               const std::vector<unsigned char> &_onContactVec) {

    FieldSecretorResult res = _secreteOutsideCellAtBoundaryOnContactWithTotalCount(_cell, _amount, _onContactVec);
    return res.success_flag;
}


FieldSecretorResult FieldSecretor::secreteInsideCellAtCOMTotalCount(CellG *_cell, float _amount) {

    FieldSecretorResult res;

    Point3D pt((int) round(_cell->xCM / _cell->volume), (int) round(_cell->yCM / _cell->volume),
               (int) round(_cell->zCM / _cell->volume));

    concentrationFieldPtr->set(pt, concentrationFieldPtr->get(pt) + _amount);


    res.tot_amount = _amount;

    return res;
}

bool FieldSecretor::secreteInsideCellAtCOM(CellG *_cell, float _amount) {
    FieldSecretorResult res = secreteInsideCellAtCOMTotalCount(_cell, _amount);
    return res.success_flag;
}


FieldSecretorResult FieldSecretor::_uptakeInsideCellTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res;

    if (!pixelTrackerPlugin) {
        res.success_flag = false;
        return res;

    }


    ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();
    set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

    float currentConcentration;
    float total_amount = 0.0;
    float uptake_amount;
    for (set<PixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {
        currentConcentration = concentrationFieldPtr->get(sitr->pixel);
        if (currentConcentration * _relativeUptake > _maxUptake) {
            uptake_amount = -_maxUptake;

        } else {
            uptake_amount = -currentConcentration * _relativeUptake;

        }

        concentrationFieldPtr->set(sitr->pixel, currentConcentration + uptake_amount);
        total_amount += uptake_amount;
    }


    res.tot_amount = total_amount;


    return res;


}


bool FieldSecretor::_uptakeInsideCell(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res = _uptakeInsideCellTotalCount(_cell, _maxUptake, _relativeUptake);
    return res.success_flag;

}


FieldSecretorResult
FieldSecretor::_uptakeInsideCellAtBoundaryTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake) {
    FieldSecretorResult res;

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    float currentConcentration;
    float total_amount = 0.0;
    float uptake_amount;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        currentConcentration = concentrationFieldPtr->get(sitr->pixel);
        if (currentConcentration * _relativeUptake > _maxUptake) {
            uptake_amount = -_maxUptake;

        } else {
            uptake_amount = -currentConcentration * _relativeUptake;

        }

        concentrationFieldPtr->set(sitr->pixel, currentConcentration + uptake_amount);
        total_amount += uptake_amount;

    }


    res.tot_amount = total_amount;


    return res;


}

bool FieldSecretor::_uptakeInsideCellAtBoundary(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res = _uptakeInsideCellAtBoundaryTotalCount(_cell, _maxUptake, _relativeUptake);
    return res.success_flag;
}


FieldSecretorResult
FieldSecretor::_uptakeInsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                                  const std::vector<unsigned char> &_onContactVec) {
    FieldSecretorResult res;

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }


    set<unsigned char> onContactSet(_onContactVec.begin(), _onContactVec.end());

    if (!boundaryPixelTrackerPlugin) {
        return false;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    float currentConcentration;
    float total_amount = 0.0;
    float uptake_amount;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {


        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);
            if (nCell != _cell && !nCell && onContactSet.find(0) != onContactSet.end()) {
                //user requested secrete on contact with medium and we found medium pixel
                currentConcentration = concentrationFieldPtr->get(sitr->pixel);
                if (currentConcentration * _relativeUptake > _maxUptake) {
                    uptake_amount = -_maxUptake;
                } else {
                    uptake_amount = -currentConcentration * _relativeUptake;

                }

                concentrationFieldPtr->set(sitr->pixel, currentConcentration + uptake_amount);
                total_amount += uptake_amount;
                break; //after secreting do not try to secrete more
            }

            if (nCell != _cell && nCell && onContactSet.find(nCell->type) != onContactSet.end()) {
                //user requested secretion on contact with cell type whose pixel we have just found
                currentConcentration = concentrationFieldPtr->get(sitr->pixel);
                if (currentConcentration * _relativeUptake > _maxUptake) {
                    uptake_amount = -_maxUptake;
                } else {
                    uptake_amount = -currentConcentration * _relativeUptake;

                }

                concentrationFieldPtr->set(sitr->pixel, currentConcentration + uptake_amount);
                total_amount += uptake_amount;


                break;//after secreting do not try to secrete more
            }

        }

    }


    res.tot_amount = total_amount;


    return res;

}

bool FieldSecretor::_uptakeInsideCellAtBoundaryOnContactWith(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                             const std::vector<unsigned char> &_onContactVec) {

    FieldSecretorResult res = _uptakeInsideCellAtBoundaryOnContactWithTotalCount(_cell, _maxUptake, _relativeUptake,
                                                                                 _onContactVec);
    return res.success_flag;
}


FieldSecretorResult
FieldSecretor::_uptakeOutsideCellAtBoundaryTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake) {
    FieldSecretorResult res;

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    set <FieldSecretorPixelData> visitedPixels;

    float currentConcentration;
    float total_amount = 0.0;
    float uptake_amount;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {


        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);
            if (nCell != _cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt)) == visitedPixels.end()) {

                currentConcentration = concentrationFieldPtr->get(nPt);
                if (currentConcentration * _relativeUptake > _maxUptake) {
                    uptake_amount = -_maxUptake;
                } else {
                    uptake_amount = -currentConcentration * _relativeUptake;

                }

                concentrationFieldPtr->set(nPt, currentConcentration + uptake_amount);
                total_amount += uptake_amount;


                visitedPixels.insert(FieldSecretorPixelData(nPt));

            }

        }

    }


    res.tot_amount = total_amount;


    return res;

}

bool FieldSecretor::_uptakeOutsideCellAtBoundary(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res = _uptakeOutsideCellAtBoundaryTotalCount(_cell, _maxUptake, _relativeUptake);
    return res.success_flag;


}

FieldSecretorResult FieldSecretor::_uptakeOutsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _maxUptake,
                                                                                       float _relativeUptake,
                                                                                       const std::vector<unsigned char> &_onContactVec) {


    FieldSecretorResult res;

    if (!boundaryPixelTrackerPlugin) {
        res.success_flag = false;
        return res;
    }

    set<unsigned char> onContactSet(_onContactVec.begin(), _onContactVec.end());

    if (!boundaryPixelTrackerPlugin) {
        return false;
    }

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            _cell->extraAttribPtr)->pixelSet;

    Point3D nPt;

    CellG *nCell = 0;

    Neighbor neighbor;

    set <FieldSecretorPixelData> visitedPixels;

    float currentConcentration;

    float total_amount = 0.0;
    float uptake_amount;

    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {


        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nPt = neighbor.pt;
            nCell = cellFieldG->get(neighbor.pt);

            if (nCell != _cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt)) == visitedPixels.end()) {

                if (!nCell && onContactSet.find(0) != onContactSet.end()) {
                    //checking if the unvisited pixel belongs to Medium and if Medium is a  cell type listed in the onContactSet
                    currentConcentration = concentrationFieldPtr->get(nPt);
                    if (currentConcentration * _relativeUptake > _maxUptake) {
                        uptake_amount = -_maxUptake;
                    } else {
                        uptake_amount = -currentConcentration * _relativeUptake;

                    }

                    concentrationFieldPtr->set(nPt, currentConcentration + uptake_amount);
                    total_amount += uptake_amount;
                }

                if (nCell && onContactSet.find(nCell->type) != onContactSet.end()) {
                    //checking if the unvisited pixel belongs to a  cell type listed in the onContactSet
                    currentConcentration = concentrationFieldPtr->get(nPt);
                    if (currentConcentration * _relativeUptake > _maxUptake) {
                        uptake_amount = -_maxUptake;
                    } else {
                        uptake_amount = -currentConcentration * _relativeUptake;

                    }

                    concentrationFieldPtr->set(nPt, currentConcentration + uptake_amount);
                    total_amount += uptake_amount;
                }
                visitedPixels.insert(FieldSecretorPixelData(nPt));
            }

        }

    }


    res.tot_amount = total_amount;

    return res;

}

bool FieldSecretor::_uptakeOutsideCellAtBoundaryOnContactWith(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                              const std::vector<unsigned char> &_onContactVec) {

    FieldSecretorResult res = _uptakeOutsideCellAtBoundaryOnContactWithTotalCount(_cell, _maxUptake, _relativeUptake,
                                                                                  _onContactVec);

    return res.success_flag;

}


FieldSecretorResult
FieldSecretor::uptakeInsideCellAtCOMTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res;

    Point3D pt((int) round(_cell->xCM / _cell->volume), (int) round(_cell->yCM / _cell->volume),
               (int) round(_cell->zCM / _cell->volume));

    float currentConcentration = concentrationFieldPtr->get(pt);
    float total_amount = 0.0;
    float uptake_amount;


    if (currentConcentration * _relativeUptake > _maxUptake) {
        uptake_amount = -_maxUptake;
    } else {
        uptake_amount = -currentConcentration * _relativeUptake;

    }

    concentrationFieldPtr->set(pt, currentConcentration + uptake_amount);
    total_amount += uptake_amount;

    res.tot_amount = total_amount;

    return res;
}


bool FieldSecretor::uptakeInsideCellAtCOM(CellG *_cell, float _maxUptake, float _relativeUptake) {

    FieldSecretorResult res = uptakeInsideCellAtCOMTotalCount(_cell, _maxUptake, _relativeUptake);
    return res.success_flag;


}

float FieldSecretor::_amountSeenByCell(CellG *_cell) {

    if (!pixelTrackerPlugin) {
        return -1.0;
    }

    float amount_seen = 0.0;
    ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();
    set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

    for (set<PixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {

        amount_seen += concentrationFieldPtr->get(sitr->pixel);

    }

    return amount_seen;

}

float FieldSecretor::totalFieldIntegral() {
    Dim3D dim = concentrationFieldPtr->getDim();

    float tot_amount = 0.0;
    for (int x = 0; x < dim.x; ++x)
        for (int y = 0; y < dim.y; ++y)
            for (int z = 0; z < dim.z; ++z) {
                tot_amount += concentrationFieldPtr->get(Point3D(x, y, z));
            }
    return tot_amount;
}
