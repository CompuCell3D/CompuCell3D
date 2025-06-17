#include <string>
#include <CompuCell3D/CC3D.h>
#include <PublicUtilities/StringUtils.h>
#include "PersistencePlugin.h"


using namespace CompuCell3D;
using namespace std;

namespace PersistencePluginModel {
    static const std::string modelLabelSR = "self-reinforcing";
    static const std::string modelLabelAN = "angularnoise";

    static const std::string forceModeExtension = "extension";
    static const std::string forceModeRetraction = "retraction";
    static const std::string forceModeReciprocal = "reciprocal";

    static const std::string dispTypeRegular = "regular";
    static const std::string dispTypeNorm = "normalized";
    static const std::string dispTypeMass = "mass";

    double drawUniform(const double& lower, const double& upper, std::mt19937& prng) {
        std::uniform_real_distribution<> dist(lower, upper);
        return dist(prng);
    }

    double drawNormal(const double& mean, const double& stdev, std::mt19937& prng) {
        std::normal_distribution<> dist(mean, stdev);
        return dist(prng);
    }

    void applyVectorTransformationRotateByAxesRadians(Coordinates3D<double>& vec, const double& radX, const double& radY, const double& radZ) {
        double x, y, z;

        double cx = cos(radX), sx = sin(radX);
        double cy = cos(radY), sy = sin(radY);
        double cz = cos(radZ), sz = sin(radZ);

        x = vec.X(); y = vec.Y(); z = vec.Z();
        vec.YRef() = y * cx - z * sx;
        vec.ZRef() = y * sx + z * cx;

        x = vec.X(); y = vec.Y(); z = vec.Z();
        vec.XRef() = x * cy + z * sy;
        vec.ZRef() = -x * sy + z * cy;

        x = vec.X(); y = vec.Y(); z = vec.Z();
        vec.XRef() = x * cz - y * sz;
        vec.YRef() = x * sz + y * cz;
    }

    void applyVectorTransformationRotateByAxesDegrees(Coordinates3D<double>& vec, const double& degX, const double& degY, const double& degZ) {
        static const double cf = M_PI / 180.0;
        applyVectorTransformationRotateByAxesRadians(vec, degX * cf, degY * cf, degZ * cf);
    }

    void applyEulerAnglesToRadians(Coordinates3D<double>& vec, const double& radZ, const double& radX, const double& radZ2) {
        const double sZ = sin(radZ),   cZ = cos(radZ);
        const double sX = sin(radX),   cX = cos(radX);
        const double sZ2 = sin(radZ2), cZ2 = cos(radZ2);

        Coordinates3D<double> tmp{
            (cZ * cZ2 - cX * sZ * sZ2) * vec.X() - (cZ * sZ2 + cX * sZ2 * sZ) * vec.Y() + (sZ * sX) * vec.Z(),
            (cZ2 * sZ + cZ * cX * sZ2) * vec.X() + (cZ * cX * cZ2 - sZ * sZ2) * vec.Y() - (cZ * sX) * vec.Z(),
            (sX * sZ2)                 * vec.X() + (cZ2 * sX)                 * vec.Y() + (cX)      * vec.Z()
        };
        vec = tmp;
    }

    void perturbZXZEuler(Coordinates3D<double>& eangles, const double& radX, const double& radY, const double& radZ) {
        // Compute trigonometric terms
        const double sA = std::sin(eangles.X()), cA = std::cos(eangles.X());
        const double sB = std::sin(eangles.Y()), cB = std::cos(eangles.Y());

        // Construct Jacobian for Z-X-Z Euler sequence
        double J[3][3] = {
            {1, sB     , 0  },
            {0, cB * cA, -sA},
            {0, cB * sA, cA }
        };

        // Compute Euler angle perturbations
        eangles.XRef() += J[0][0] * radX + J[0][1] * radY + J[0][2] * radZ;
        eangles.YRef() += J[1][0] * radX + J[1][1] * radY + J[1][2] * radZ;
        eangles.ZRef() += J[2][0] * radX + J[2][1] * radY + J[2][2] * radZ;
    }

    double parseValueSpecification(CC3DXMLElement* xmlElement, std::mt19937& prng) {
        for (auto& el : xmlElement->children) {
            if (el->name == "Constant") return el->getDouble();
            else if (el->name == "Uniform") {
                const double lower = el->findAttribute("Min") ? el->getAttributeAsDouble("Min") : 0.0;
                const double upper = el->findAttribute("Max") ? el->getAttributeAsDouble("Max") : lower + 1.0;
                return drawUniform(lower, upper, prng);
            }
            else if (el->name == "Normal") {
                const double mean = el->findAttribute("Mean") ? el->getAttributeAsDouble("Mean") : 0.0;
                const double stdev = el->findAttribute("StDev") ? el->getAttributeAsDouble("StDev") : 1.0;
                return drawNormal(mean, stdev, prng);
            }
        }
        return 0.0;
    }

    std::string parseAxisSpecification(CC3DXMLElement* xmlElement) {
        if (!xmlElement->findAttribute("Axis")) throw CC3DException("No axis specified for persistence vector component set");
        auto axisLabel = xmlElement->getAttribute("Axis");
        changeToLower(axisLabel);
        return axisLabel;
    }

    std::string parseRotationSpecification(CC3DXMLElement* xmlElement) {
        if (!xmlElement->findAttribute("Units")) return "degrees";
        auto units = xmlElement->getAttribute("Units");
        changeToLower(units);
        if (units == "degrees") return units;
        else if (units == "radians") return units;
        throw CC3DException("Unknown rotation units. Valid units are Degrees and Radians");
    }

    void applyVectorTransformation(Coordinates3D<double>& vec, CC3DXMLElement* xmlElement, std::mt19937& prng, double& dval, const bool& useDval) {
        if (!xmlElement) return;

        double val;

        if (xmlElement->name == "RotateVector") {
            if (useDval) val = dval;
            else val = parseValueSpecification(xmlElement, prng);
            dval = val;
            
            auto axisLabel = parseAxisSpecification(xmlElement);
            auto units = parseRotationSpecification(xmlElement);
            auto fcn = units == "radians" ? &applyVectorTransformationRotateByAxesRadians : &applyVectorTransformationRotateByAxesDegrees;
            if (axisLabel == "x") fcn(vec, val, 0.0, 0.0);
            else if (axisLabel == "y") fcn(vec, 0.0, val, 0.0);
            else if (axisLabel == "z") fcn(vec, 0.0, 0.0, val);
            else throw CC3DException("Unknown axis specified for persistence vector rotation. Valid labels are X, Y, Z.");
        }
    }

    void applyCSysTransformation(Coordinates3D<double>& e1, Coordinates3D<double>& e2, Coordinates3D<double>& e3, CC3DXMLElement* xmlElement, std::mt19937& prng) {
        double dval;
        applyVectorTransformation(e1, xmlElement, prng, dval, false);
        applyVectorTransformation(e2, xmlElement, prng, dval, true);
        applyVectorTransformation(e3, xmlElement, prng, dval, true);
    }

}

PersistenceModel* _persistenceModelConstructor(CC3DXMLElement* xmlData, Simulator* _simulator, PersistencePlugin* _plugin, CellG* _cell) {
    if (!xmlData || !xmlData->findAttribute("Model")) 
        throw CC3DException("No specified persistence model");

    auto modelData = xmlData->getAttribute("Model");
    changeToLower(modelData);

    PersistenceModel* result;
    if (modelData == PersistencePluginModel::modelLabelSR) result = new SRPersistenceModel(_simulator, _plugin, _cell);
    else if (modelData == PersistencePluginModel::modelLabelAN) result = new ANPersistenceModel(_simulator, _plugin, _cell);
    else throw CC3DException("Unknown persistence model");

    return result;
}

Coordinates3D<double> _extractEulerAngles(const Coordinates3D<double>& e1, const Coordinates3D<double>& e2, const Coordinates3D<double>& e3) {
    Coordinates3D<double> result;
    
    result.YRef() = acos(e3.Z());

    if (fabs(e3.Z()) < 1.0) {
        result.XRef() = atan2(e3.Y(), e3.X());
        result.ZRef() = atan2(e2.Z(), -e1.Z());
    }
    else {
        result.XRef() = 0.0;
        result.ZRef() = atan2(e2.X(), e2.Y());
    }
    return result;
}

PersistencePlugin::PersistencePlugin() {}

PersistencePlugin::~PersistencePlugin() {
    for (auto& itr : modelInventory) 
        delete itr.second;
    modelInventory.clear();
}

void PersistencePlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
    simulator = _simulator;
    xmlData = _xmlData;

    auto potts = _simulator->getPotts();
    fieldDim = potts->getCellFieldG()->getDim();
    simulator->registerSteerableObject(this);
    simulator->getClassRegistry()->addStepper(this->toString(), this);
    potts->registerEnergyFunction(this);
    potts->getCellInventory().registerWatcher(this);
    potts->getCellFactoryGroupPtr()->registerClass(&persistenceDataAccessor);

    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass", &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(_simulator);

    if (_xmlData->findElement("RandomSeed")) randomSeed = _xmlData->getFirstElement("RandomSeed")->getUInt();
    else randomSeed = std::random_device()();
    prng = std::mt19937(randomSeed);

    cellTypeModelElements = _pullTypeModelElements(_xmlData);
}

void PersistencePlugin::extraInit(Simulator* simulator) {
    update(xmlData, true);
}

void PersistencePlugin::update(CC3DXMLElement* _xmlData, bool _fullInitFlag) {
    cellTypeModelElements = _pullTypeModelElements(_xmlData);
    _applyTypeModelElements();
}

double PersistencePlugin::changeEnergy(const Point3D& pt, const CellG* newCell, const CellG* oldCell) {
    auto ptNew = simulator->getPotts()->getFlipNeighbor();
    auto copyVec = distanceVectorCoordinatesInvariant(
        {static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z)}, 
        {static_cast<double>(ptNew.x), static_cast<double>(ptNew.y), static_cast<double>(ptNew.z)}, 
        fieldDim, BoundaryStrategy::getInstance()
    );

    double energy = 0.0;
    PersistenceModel* model;
    if (oldCell) {
        auto modelItr = modelInventory.find(oldCell->id);
        if (modelItr != modelInventory.end()) {
            model = modelItr->second;
            energy += model->changeEnergy(pt, copyVec, false);
        }
    }
    if (newCell) {
        auto modelItr = modelInventory.find(newCell->id);
        if (modelItr != modelInventory.end()) {
            model = modelItr->second;
            energy += model->changeEnergy(pt, copyVec, true);
        }
    }

    return energy;
}

void PersistencePlugin::_recordToInitialize(const long& _id) { _toInitialize.insert(_id); }

void PersistencePlugin::_recordInitialized(const long& _id) {
    auto idItr = _toInitialize.find(_id);
    if (idItr != _toInitialize.end()) 
        _toInitialize.erase(idItr);
}

PersistenceModel* PersistencePlugin::_maybePersistenceModelConstruct(CellG* cell) {
    PersistenceModel* model = NULL;
    if (!cell) return model;

    auto modelItr = modelInventory.find(cell->id);
    if (modelItr == modelInventory.end()) {
        auto modelDataItr = cellTypeModelElements.find(cell->type);
        if (modelDataItr != cellTypeModelElements.end() && modelDataItr->second) {
            model = _persistenceModelConstructor(modelDataItr->second, simulator, this, cell);
            model->applyModelData(modelDataItr->second, true);
            modelInventory[cell->id] = model;
        }
    } 
    else 
        model = modelItr->second;
    return model;
}

void PersistencePlugin::step(const unsigned int currentStep) {
    if (_toInitialize.size() > 0) {
        std::vector<long> toInitialize(_toInitialize.begin(), _toInitialize.end());
        auto cellInventory = simulator->getPotts()->getCellInventory();
        for (auto id : toInitialize) {
            CellG* cell = cellInventory.attemptFetchingCellById(id);
            if (cell && _maybePersistenceModelConstruct(cell)) 
                _recordInitialized(id);
        }
    }

    for (auto& itr : modelInventory) 
        itr.second->update();
}

void PersistencePlugin::onCellAdd(CellG* cell) {
    if (!cell) return;

    // Handle when created without set type
    if (cell->type == 0 || !cell->extraAttribPtr) {
        _recordToInitialize(cell->id);
        return;
    }

    _maybePersistenceModelConstruct(cell);
    _recordInitialized(cell->id);
}

void PersistencePlugin::onCellRemove(CellG* cell) {
    if (!cell) return;

    _recordInitialized(cell->id);

    auto modelItr = modelInventory.find(cell->id);
    if (modelItr != modelInventory.end()) {
        delete modelItr->second;
        modelInventory.erase(modelItr);
    }
}

PersistenceData* PersistencePlugin::getPersistenceData(CellG* cell) {
    if (!cell || !cell->extraAttribPtr) return NULL;
    return persistenceDataAccessor.get(cell->extraAttribPtr);
}

PersistenceModel* PersistencePlugin::getModel(CellG* cell) {
    PersistenceModel* model = NULL;
    if (!cell) 
        return model;
    
    auto modelItr = modelInventory.find(cell->id);
    if (modelItr != modelInventory.end()) 
        model = modelItr->second;
    return model;
}

// PersistenceModel

void PersistenceModel::applyModelData(CC3DXMLElement* xmlData, const bool& initialize) {
    if (!xmlData || !xmlData->findElement("ForceMode")) {
        forceMode = PersistencePlugin::ForceMode::FORCETYPE_RECIPROCAL;
    }
    else {
        auto forceModeEl = xmlData->getFirstElement("ForceMode");
        auto forceModeSpec = forceModeEl->getText();
        changeToLower(forceModeSpec);
        if (forceModeSpec == PersistencePluginModel::forceModeExtension) forceMode = PersistencePlugin::ForceMode::FORCETYPE_EXTENSION;
        else if (forceModeSpec == PersistencePluginModel::forceModeRetraction) forceMode = PersistencePlugin::ForceMode::FORCETYPE_RETRACTION;
        else if (forceModeSpec == PersistencePluginModel::forceModeReciprocal) forceMode = PersistencePlugin::ForceMode::FORCETYPE_RECIPROCAL;
        else throw CC3DException("Unknown force mode");
    }

    if (!xmlData || !xmlData->findElement("WorkTerm")) {
        dispType = PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_REGULAR;
    } 
    else {
        auto dispTypeEl = xmlData->getFirstElement("WorkTerm");
        auto dispTypeSpec = dispTypeEl->getText();
        changeToLower(dispTypeSpec);
        if (dispTypeSpec == PersistencePluginModel::dispTypeRegular) dispType = PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_REGULAR;
        else if (dispTypeSpec == PersistencePluginModel::dispTypeNorm) dispType = PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_NORMALIZED;
        else if (dispTypeSpec == PersistencePluginModel::dispTypeMass) dispType = PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_MASS;
        else throw CC3DException("Unknown work term");
    }

    PersistenceData* pdata = plugin->getPersistenceData(cell);
    if (!pdata) 
        return;
    
    if (xmlData && xmlData->findElement("Magnitude")) {
        
        pdata->magnitude = PersistencePluginModel::parseValueSpecification(xmlData->getFirstElement("Magnitude"), plugin->prng);
    }

    Coordinates3D<double> e1{1.0, 0.0, 0.0}, e2{0.0, 1.0, 0.0}, e3{0.0, 0.0, 1.0};
    if (xmlData && xmlData->findElement("VectorInit")) 
        for (auto& el : xmlData->getFirstElement("VectorInit")->children) 
            PersistencePluginModel::applyCSysTransformation(e1, e2, e3, el, plugin->prng);
    pdata->persistenceAngles = _extractEulerAngles(e1, e2, e3);
}

Coordinates3D<double> PersistenceModel::_calculateDisplacement(const Point3D& pt, const Coordinates3D<double>& copyVector, const bool& isNewCell) {
    if (dispType == PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_REGULAR) 
        return copyVector;
    else if (dispType == PersistencePlugin::DisplacementType::DISPLACEMENTTYPE_NORMALIZED) {
        const double dispLen = sqrt(copyVector * copyVector);
        return {copyVector.X() / dispLen, copyVector.Y() / dispLen, copyVector.Z() / dispLen};
    }

    Coordinates3D<double> result{static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z)};
    const Coordinates3D<double> dim{static_cast<double>(plugin->fieldDim.x), static_cast<double>(plugin->fieldDim.y), static_cast<double>(plugin->fieldDim.z)};
    result.XRef() -= cell->xCOM;
    result.YRef() -= cell->yCOM;
    result.ZRef() -= cell->zCOM;
    result.XRef() = result.XRef() >= dim.X() / 2.0 ? result.XRef() - dim.X() : result.XRef() < - dim.X() / 2 ? result.XRef() + dim.X() : result.XRef();
    result.YRef() = result.YRef() >= dim.Y() / 2.0 ? result.YRef() - dim.Y() : result.YRef() < - dim.Y() / 2 ? result.YRef() + dim.Y() : result.YRef();
    result.ZRef() = result.ZRef() >= dim.Z() / 2.0 ? result.ZRef() - dim.Z() : result.ZRef() < - dim.Z() / 2 ? result.ZRef() + dim.Z() : result.ZRef();

    const double cellVol = static_cast<double>(cell->volume);
    double cf = cellVol;
    if (isNewCell) cf /= (cellVol + 1.0);
    else cf /= cell->volume > 1 ? -(cellVol - 1.0) : -1.0;

    result.XRef() *= cf;
    result.YRef() *= cf;
    result.ZRef() *= cf;
    return result;
}

double PersistenceModel::changeEnergy(const Point3D& pt, const Coordinates3D<double>& copyVector, const bool& isNewCell) {
    if ((isNewCell && forceMode & PersistencePlugin::ForceMode::FORCETYPE_EXTENSION) || (!isNewCell && forceMode & PersistencePlugin::ForceMode::FORCETYPE_RETRACTION)) {
        auto disp = _calculateDisplacement(pt, copyVector, isNewCell);
        auto pvec = persistenceVector();
        return disp * pvec;
    }
    return 0.0;
}

const Coordinates3D<double> PersistenceModel::directionVector() {
    auto pd = plugin->getPersistenceData(cell);
    Coordinates3D<double> result{1.0, 0.0, 0.0};
    if (!pd) return result;

    PersistencePluginModel::applyEulerAnglesToRadians(result, pd->persistenceAngles.XRef(), pd->persistenceAngles.YRef(), pd->persistenceAngles.ZRef());
    return result;
}

const Coordinates3D<double> PersistenceModel::persistenceVector() {
    auto pd = plugin->getPersistenceData(cell);
    Coordinates3D<double> result{1.0, 0.0, 0.0};
    if (!pd) return result;

    PersistencePluginModel::applyEulerAnglesToRadians(result, pd->persistenceAngles.XRef(), pd->persistenceAngles.YRef(), pd->persistenceAngles.ZRef());
    result.XRef() *= pd->magnitude;
    result.YRef() *= pd->magnitude;
    result.ZRef() *= pd->magnitude;
    return result;
}

// SRPersistenceModel

void SRPersistenceModel::applyModelData(CC3DXMLElement* xmlData, const bool& initialize) {
    PersistenceModel::applyModelData(xmlData, initialize);

    if (!initialize) return;

    size_t period = 1;

    if (xmlData && xmlData->findAttribute("Period")) {
        auto a = xmlData->getAttributeAsInt("Period");
        if (a <= 0) throw CC3DException("Period must be positive");
        period = a;
    }

    const Coordinates3D<double> cellCOM{cell->xCOM, cell->yCOM, cell->zCOM};

    comHist = std::vector<Coordinates3D<double> >(period, cellCOM);
    comIdx = 0;
}

void SRPersistenceModel::update() {
    Coordinates3D<double> comCurr{cell->xCOM, cell->yCOM, cell->zCOM};

    if (!initialized) {
        initialized = true;
        for (size_t i = 0; i < comHist.size(); i++) 
            comHist[i] = comCurr;
    }

    if (looped) {
        auto disp = comCurr - comHist[comIdx];
        auto fieldDim = getFieldDim();
        const Coordinates3D<double> dim{static_cast<double>(fieldDim.x), static_cast<double>(fieldDim.y), static_cast<double>(fieldDim.z)};
        disp.XRef() = disp.XRef() >= dim.X() / 2.0 ? disp.XRef() - dim.X() : disp.XRef() < - dim.X() / 2 ? disp.XRef() + dim.X() : disp.XRef();
        disp.YRef() = disp.YRef() >= dim.Y() / 2.0 ? disp.YRef() - dim.Y() : disp.YRef() < - dim.Y() / 2 ? disp.YRef() + dim.Y() : disp.YRef();
        disp.ZRef() = disp.ZRef() >= dim.Z() / 2.0 ? disp.ZRef() - dim.Z() : disp.ZRef() < - dim.Z() / 2 ? disp.ZRef() + dim.Z() : disp.ZRef();

        auto pdata = plugin->getPersistenceData(cell);
        if (pdata) {
            const double dispLen2 = disp * disp;
            if (dispLen2 > 0.0) {
                pdata->persistenceAngles.XRef() = atan2(disp.YRef(), disp.XRef());
                pdata->persistenceAngles.YRef() = acos(disp.ZRef() / sqrt(dispLen2));
            }
        }

    }

    comHist[comIdx] = comCurr;
    const bool looping = (comIdx + 1) == comHist.size();
    looped |= looping;
    comIdx = looping ? 0 : comIdx + 1;
};

// ANPersistenceModel

void ANPersistenceModel::applyModelData(CC3DXMLElement* xmlData, const bool& initialize) {
    PersistenceModel::applyModelData(xmlData, initialize);

    if (xmlData && xmlData->findAttribute("Period")) {
        auto a = xmlData->getAttributeAsDouble("Period");
        if (a <= 0.0) throw CC3DException("Period must be positive");
        sqrtPeriod = sqrt(a);
    } 
    else sqrtPeriod = 1.0;

    if (xmlData && xmlData->findAttribute("StDev1")) {
        auto a = xmlData->getAttributeAsDouble("StDev1");
        if (a <= 0.0) throw CC3DException("Standard deviation must be positive");
        stDev1 = a;
    }
    else stDev1 = 0.0;

    if (xmlData && xmlData->findAttribute("StDev2")) {
        auto a = xmlData->getAttributeAsDouble("StDev2");
        if (a <= 0.0) throw CC3DException("Standard deviation must be positive");
        stDev2 = a;
    }
    else stDev2 = 0.0;

    if (xmlData && xmlData->findAttribute("StDev3")) {
        auto a = xmlData->getAttributeAsDouble("StDev3");
        if (a <= 0.0) throw CC3DException("Standard deviation must be positive");
        stDev3 = a;
    }
    else stDev3 = 0.0;
}

void ANPersistenceModel::update() {
    PersistenceData* pdata = plugin->getPersistenceData(cell);
    if (!pdata) 
        return;

    PersistencePluginModel::perturbZXZEuler(
        pdata->persistenceAngles,
        PersistencePluginModel::drawNormal(0.0, stDev1, getPRNG()) * sqrtPeriod,
        PersistencePluginModel::drawNormal(0.0, stDev2, getPRNG()) * sqrtPeriod,
        PersistencePluginModel::drawNormal(0.0, stDev3, getPRNG()) * sqrtPeriod
    );
};

// private methods

std::unordered_map<unsigned char, CC3DXMLElement*> PersistencePlugin::_pullTypeModelElements(CC3DXMLElement* _xmlData) {
    std::unordered_map<unsigned char, CC3DXMLElement*> result;

    if (!_xmlData) 
        _xmlData = xmlData;

    if (!_xmlData) return result;

    auto automaton = simulator->getPotts()->getAutomaton();
    if (!automaton)
        throw CC3DException("Cell type information is unavailable");

    for (auto typeId : automaton->getTypeIds()) 
        result[typeId] = NULL;
    
    CC3DXMLElementList modelVec = _xmlData->getElements("PersistenceModel");

    for (auto& el : modelVec) {
        if (!el->findAttribute("CellType")) 
            throw CC3DException("Missing CellType attribute in PersistenceModel specification");

        auto typeName = el->getAttribute("CellType");
        auto typeId = automaton->getTypeId(typeName);
        result[typeId] = el;
    }

    return result;
}

void PersistencePlugin::_applyTypeModelElements() {
    auto cellInventory = simulator->getPotts()->getCellInventory();

    CellG* cell;
    PersistenceModel* model;
    bool initializing;

    for (auto cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr) {
        cell = cInvItr->second;

        auto modelElementItr = cellTypeModelElements.find(cell->type);

        if (modelElementItr != cellTypeModelElements.end() && modelElementItr->second) {
            CC3DXMLElement* modelElement = modelElementItr->second;

            auto modelItr = modelInventory.find(cell->id);
            initializing = modelItr == modelInventory.end();
            if (initializing) {
                model = _persistenceModelConstructor(modelElement, simulator, this, cell);
                if (model) 
                    modelInventory[cell->id] = model;
            } 
            else model = modelItr->second;

            if (model) { 
                model->applyModelData(modelElement, initializing);
                _recordInitialized(cell->id);
            }
        }
    }
}
