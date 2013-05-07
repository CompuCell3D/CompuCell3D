#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <math.h>
#include "sbml/Model.h"
#include "sbml/SBMLDocument.h"
#include "sbml/math/FormulaFormatter.h"
#include "sbml/common/libsbml-version.h"
#include "rrLogger.h"
#include "rrNOMSupport.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrStringBuilder.h"
#include "rrSBMLSymbol.h"
#include "rrOSSpecifics.h"
//---------------------------------------------------------------------------

using namespace std;
namespace rr
{

NOMSupport::NOMSupport()
:
mSBMLDoc(NULL),
mModel(NULL),
STR_DoubleFormat("%.5G")
{}

NOMSupport::~NOMSupport()
{
//    delete mModel;
//    delete mSBMLDoc;
}


string	NOMSupport::getlibSBMLVersion()
{
	return getLibSBMLDottedVersion();
}

Model* NOMSupport::GetModel()
{
	return mModel;
}

SBMLDocument* NOMSupport::GetSBMLDocument()
{
	return mSBMLDoc;
}

void NOMSupport::reset()
{
    mSymbolTable.clear();
//    delete mModel;
//    delete mSBMLDoc;
//    mModel      = NULL;
//    mSBMLDoc    = NULL;
}

string NOMSupport::getNthCompartmentId(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nIndex < 0 || nIndex >= (int) mModel->getNumCompartments())
    {
        throw Exception("Invalid input - Argument should be >= 0 and should be less than total number of compartments in the model");

    }
    Compartment *oCompartment = mModel->getCompartment((int)nIndex);
    return GetId(*oCompartment);
}

double NOMSupport::getValue(const string& sId)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    Species *oSpecies = mModel->getSpecies(sId);
    if (oSpecies != NULL)
    {
        if (oSpecies->isSetInitialAmount())
        {
            return oSpecies->getInitialAmount();
        }
        else
        {
            return oSpecies->getInitialConcentration();
        }
    }

    Compartment *oCompartment = mModel->getCompartment(sId);
    if (oCompartment != NULL)
    {
        return oCompartment->getVolume();
    }

    Parameter *oParameter = mModel->getParameter(sId);
    if (oParameter != NULL)
    {
        return oParameter->getValue();
    }

    for (u_int i = 0; i < mModel->getNumReactions(); i++)
    {
        Reaction* reaction = mModel->getReaction(i);
        for (int j = 0; j < reaction->getNumReactants(); j++)
        {
            SpeciesReference *reference = reaction->getReactant(j);
            if (reference->isSetId() && reference->getId() == sId)
            {
                if (reference->isSetStoichiometry())
                {
                    return reference->getStoichiometry();
                }
                else
                {
                    return 1;
                }
            }
        }

        for (int j = 0; j < reaction->getNumProducts(); j++)
        {
            SpeciesReference *reference = reaction->getProduct(j);
            if (reference->isSetId() && reference->getId() == sId)
            {
                if (reference->isSetStoichiometry())
                {
                    return reference->getStoichiometry();
                }
                else
                {
                    return 1;
                }
            }
        }
    }


    throw Exception("Invalid string name. The id '" + sId + "' does not exist in the model");
}


//double NOMSupport::getValue(const string& id)
//{
//    double val;
//    if(::getValue(id.c_str(), &val))
//    {
//        //How to signal error..?
//        return -1;
//    }
//    return val;
//}

StringListContainer NOMSupport::getListOfBoundarySpecies()
{
    StringListContainer boundarySpeciesList;

    if (mModel == NULL)
    {
        throw NOMException("You need to load the model first");
    }

    for (int i = 0; i < mModel->getNumSpecies(); i++)
    {
        Species *oSpecies = mModel->getSpecies(i);
        if (oSpecies->getBoundaryCondition())
        {
            StringList oSpeciesValues;
            oSpeciesValues.Add(oSpecies->getId());
            double concentration = oSpecies->isSetInitialConcentration() ? oSpecies->getInitialConcentration() : oSpecies->getInitialAmount();
            oSpeciesValues.Add( ToString(concentration, STR_DoubleFormat) );
            oSpeciesValues.Add( ToString(oSpecies->isSetInitialConcentration()) );

            boundarySpeciesList.Add(oSpeciesValues);
        }
    }

    return boundarySpeciesList;
}

string NOMSupport::GetId(SBase& element)
{
    if (element.isSetId())
        return element.getId();
    return element.getName();
}

//        string NOMSupport::getMetaId(string sId)
//        {
//            if (mModel == NULL)
//            {
//                return "";
//                //throw new Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return oSpecies.getMetaId();
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter.getMetaId();
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment.getMetaId();
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                return oReaction.getMetaId();
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                return oRule.getMetaId();
//            }
//
//            if (mModel->getId() == sId)
//                return mModel->getMetaId();
//
//            return "";
//        }

string NOMSupport::GetName(SBase* element)
{
    if(!element)
    {
        return string("");
    }

    if (element->isSetName())
        return element->getName();
    return element->getId();
}

//        bool addMissingModifiers(Model oModel)
//        {
//            StringCollection _species = new StringCollection();
//            for (int i = 0; i < oModel.getNumSpecies(); i++)
//            {
//                _species.Add(GetId(oModel.getSpecies(i)));
//            }
//            int nReactions = (int)oModel.getNumReactions();
//            bool bReplaced = false;
//            for (int i = 0; i < nReactions; i++)
//            {
//                libsbmlcs.Reaction oReaction = oModel.getReaction(i);
//                KineticLaw oLaw = oReaction.getKineticLaw();
//                if (oLaw == NULL) continue;
//                StringCollection symbols = new StringCollection();
//
//                for (int j = 0; j < oReaction.getNumModifiers(); j++)
//                {
//                    symbols.Add(oReaction.getModifier(j).getSpecies());
//                }
//                for (int j = 0; j < oModel.getNumParameters(); j++)
//                {
//                    symbols.Add(GetId(oModel.getParameter(j)));
//                }
//                for (int j = 0; j < oModel.getNumCompartments(); j++)
//                {
//                    symbols.Add(GetId(oModel.getCompartment(j)));
//                }
//                for (int j = 0; j < oModel.getNumFunctionDefinitions(); j++)
//                {
//                    symbols.Add(GetId(oModel.getFunctionDefinition(j)));
//                }
//
//                if (oLaw != NULL)
//                {
//                    for (int j = 0; j < oLaw.getNumParameters(); j++)
//                    {
//                        symbols.Add(GetId(oLaw.getParameter(j)));
//                    }
//                }
//
//                for (int j = 0; j < oReaction.getNumReactants(); j++)
//                {
//                    symbols.Add(oReaction.getReactant(j).getSpecies());
//                }
//                for (int j = 0; j < oReaction.getNumProducts(); j++)
//                {
//                    symbols.Add(oReaction.getProduct(j).getSpecies());
//                }
//                ASTNode oRoot = oLaw.getMath();
//                StringCollection oMissingNames = new StringCollection();
//
//                // here the fancy function that discoveres themissing names and solves all problems
//                // magically ...
//                checkForMissingNames(oRoot, oMissingNames, symbols);
//                string sMissingName;
//                if (oMissingNames.Count > 0)
//                {
//                    bReplaced = true;
//                    for (int j = 0; j < oMissingNames.Count; j++)
//                    {
//                        sMissingName = oMissingNames[j];
//                        if (_species.Contains(sMissingName))
//                        {
//                            ModifierSpeciesReference reference = oReaction.createModifier();
//                            reference.setSpecies(sMissingName);
//                            oReaction.addModifier(reference);
//                        }
//                    }
//                }
//            }
//            return bReplaced;
//        }
//
//        string NOMSupport::addMissingModifiers(string sModel)
//        {
//
//            SBMLDocument d = libsbml.readSBMLFromString(sModel);
//            string sResult = sModel;
//            try
//            {
//                Model oModel = d.getModel();
//                if (oModel != NULL)
//                {
//                    bool bReplaced = addMissingModifiers(oModel);
//                    if (!bReplaced)
//                    {
//                        return sModel;
//                    }
//                }
//
//                sResult = libsbml.writeSBMLToString(d);
//            }
//            catch
//            {
//                throw new Exception("Exception occured while trying to modify the SBML file");
//            }
//
//            finally
//            {
//                if (d != NULL)
//                    d.Dispose();
//            }
//
//
//            return sResult;
//        }
//
//        void NOMSupport::checkForMissingNames(ASTNode node, StringCollection results, StringCollection symbols)
//        {
//            for (int i = 0; i < node.getNumChildren(); i++)
//            {
//                checkForMissingNames(node.getChild(i), results, symbols);
//            }
//            if (node.isName())
//            {
//                string sName = node.getName();
//                if (!symbols.Contains(sName) && !results.Contains(sName))
//                    results.Add(sName);
//            }
//        }
//
//        int NOMSupport::checkConsistency()
//        {
//            if (_oDoc == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//            return (int)_oDoc.checkConsistency();
//
//        }
//
//        string NOMSupport::convertLevel1ToLevel2Impl(string sSBML)
//        {
//            SBMLReader oReader = new SBMLReader();
//            SBMLDocument oDoc = oReader.readSBMLFromString(sSBML);
//            string sResult = sSBML;
//
//            try
//            {
//                oDoc.setLevelAndVersion(2, 1);
//                SBMLWriter oWriter = new SBMLWriter();
//                sResult = oWriter.writeToString(oDoc);
//            }
//            finally
//            {
//                if (oDoc != NULL)
//                    oDoc.Dispose();
//            }
//            return sResult;
//        }
//
//        string NOMSupport::convertLevel2ToLevel1Impl(string sSBML)
//        {
//            SBMLReader oReader = new SBMLReader();
//            SBMLDocument oDoc = oReader.readSBMLFromString(sSBML);
//            string sResult = sSBML;
//
//            try
//            {
//                oDoc.setLevelAndVersion(1, 2);
//                SBMLWriter oWriter = new SBMLWriter();
//                sResult = oWriter.writeToString(oDoc);
//            }
//            finally
//            {
//                if (oDoc != NULL)
//                    oDoc.Dispose();
//            }
//            return sResult;
//        }
//
string NOMSupport::convertMathMLToString(const string& sMathML)
{
    ASTNode* node = libsbml::readMathMLFromString(sMathML.c_str());
    string sResult = SBML_formulaToString(node);
    return sResult;
}
//
//        string NOMSupport::convertPowImpl(string sSBML)
//        {
//
//            SBMLDocument doc = libsbml.readSBMLFromString(sSBML);
//            try
//            {
//                Model model = doc.getModel();
//                if (model == NULL)
//                {
//                    throw new Exception("Error in sbml input. ");
//                }
//                string strKineticFormula;
//                for (int i = 0; i < model.getNumReactions(); i++)
//                {
//                    libsbmlcs.Reaction r = model.getReaction(i);
//                    KineticLaw kl = r.getKineticLaw();
//
//                    if (kl == NULL)
//                    {
//                        strKineticFormula = "";
//                    }
//                    else
//                    {
//                        strKineticFormula = kl.getFormula();
//                        if (strKineticFormula == NULL)
//                        {
//                            throw new Exception("The kinetic law has errors");
//                        }
//                    }
//                    ASTNode ast_Node = libsbml.parseFormula(strKineticFormula);
//                    changePow(ast_Node);
//                    kl.setMath(ast_Node);
//
//                }
//
//                doc.setLevelAndVersion(1, 2);
//                return libsbml.writeSBMLToString(doc);
//            }
//            finally
//            {
//                if (doc != NULL)
//                    doc.Dispose();
//            }
//        }
//
//        void NOMSupport::changePow(ASTNode node)
//        {
//            int c;
//
//            if (node.getType() == libsbml.AST_FUNCTION_POWER)
//            {
//                node.setType(libsbml.AST_POWER);
//            }
//
//            for (c = 0; c < node.getNumChildren(); c++)
//            {
//                changePow(node.getChild(c));
//            }
//        }
//
//        string NOMSupport::convertSBML(string sModel, int nLevel, int nVersion)
//        {
//            return convertSBML(sModel, nLevel, nVersion, true);
//        }
//
//        void NOMSupport::RemoveSpatialSizeUnitsFromSpecies(SBMLDocument doc)
//        {
//            if (doc == NULL) return;
//            if (doc.getModel() == NULL) return;
//            var model = doc.getModel();
//            for (int i = 0; i < model.getNumSpecies(); i++)
//            {
//                var species = model.getSpecies(i);
//                if (species.isSetSpatialSizeUnits())
//                    species.unsetSpatialSizeUnits();
//            }
//        }
//
//        void NOMSupport::RemoveTimeUnitsFromKineticLaws(SBMLDocument doc)
//        {
//            if (doc == NULL) return;
//            if (doc.getModel() == NULL) return;
//            var model = doc.getModel();
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                var reaction = model.getReaction(i);
//                if (reaction.isSetKineticLaw())
//                {
//                    var law = reaction.getKineticLaw();
//                    if (law.isSetTimeUnits())
//                        law.unsetTimeUnits();
//                }
//            }
//        }
//
//        void NOMSupport::RemoveSubstanceUnitsFromKineticLaws(SBMLDocument doc)
//        {
//            if (doc == NULL) return;
//            if (doc.getModel() == NULL) return;
//            var model = doc.getModel();
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                var reaction = model.getReaction(i);
//                if (reaction.isSetKineticLaw())
//                {
//                    var law = reaction.getKineticLaw();
//                    if (law.isSetSubstanceUnits())
//                        law.unsetSubstanceUnits();
//                }
//            }
//        }
//
//        void NOMSupport::AddMissingParameter(string parameterId, SBMLDocument doc)
//        {
//            if (doc == NULL) return;
//            var model = doc.getModel();
//            if (model == NULL) return;
//            var parameter = model.createParameter();
//            parameter.setId(parameterId);
//            parameter.setValue(0.1);
//            return;
//        }
//
//        void NOMSupport::UpgradeToL2V4IfNecessary(SBMLDocument doc)
//        {
//            if (doc.getLevel() == 1 || (doc.getLevel() == 2 && doc.getVersion() < 2))
//            {
//                Model oModel = doc.getModel();
//                addMissingModifiers(oModel);
//
//                if (oModel == NULL)
//                {
//                    SBMLErrorLog oLog = doc.getErrorLog();
//                    StringBuilder oBuilder = new StringBuilder();
//                    for (int i = 0; i < oLog.getNumErrors(); i++)
//                    {
//                        SBMLError error = oLog.getError(i);
//                        oBuilder.Append(String.Format("{0}: {1}{2}", error.getErrorId(), error.getMessage(), Environment.NewLine));
//                    }
//
//                    throw new Exception(oBuilder.ToString());
//                }
//
//                if (!doc.setLevelAndVersion(2, 4))
//                {
//
//                    SBMLErrorLog oLog = doc.getErrorLog();
//                    StringBuilder oBuilder = new StringBuilder();
//                    for (int i = 0; i < oLog.getNumErrors(); i++)
//                    {
//                        SBMLError error = oLog.getError(i);
//
//                        switch (error.getErrorId())
//                        {
//                            case 95004:
//                                {
//                                    RemoveSpatialSizeUnitsFromSpecies(doc);
//                                    break;
//                                }
//                            case 95002:
//                                {
//                                    RemoveTimeUnitsFromKineticLaws(doc);
//                                    break;
//                                }
//                            case 95003:
//                                {
//                                    RemoveSubstanceUnitsFromKineticLaws(doc);
//                                    break;
//                                }
//                            case 10215:
//                                {
//                                    var matches = new System.Text.RegularExpressions.Regex("'(?<id>.*?)'").Matches((error.getMessage()));
//                                    var thisMatch = matches[matches.Count - 1];
//                                    var parameterId = thisMatch.Groups[thisMatch.Groups.Count - 1].Value;
//                                    AddMissingParameter(parameterId, doc);
//                                    break;
//                                }
//                            default:
//                                break;
//                        }
//
//                    }
//                    oLog.clearLog();
//
//                    if (!doc.setLevelAndVersion(2, 4))
//                    {
//                        oLog = doc.getErrorLog();
//                        oBuilder = new StringBuilder();
//                        int numErrors = 0;
//                        for (int i = 0; i < oLog.getNumErrors(); i++)
//                        {
//                            SBMLError error = oLog.getError(i);
//                            if (//error.isError() ||
//                                error.isFatal())
//                                numErrors++;
//
//                            oBuilder.Append(error.getErrorId() + ": " + error.getMessage() + Environment.NewLine);
//                        }
//                        if (numErrors > 0)
//                            throw new Exception(oBuilder.ToString());
//                    }
//
//                }
//            }
//        }
//
//        string NOMSupport::getSBOCapableSBML(string sModel)
//        {
//            if (sModel == "")
//            {
//                throw new ArgumentException("The model cannot be empty");
//            }
//
//            SBMLDocument oSBMLDoc = libsbml.readSBMLFromString(sModel);
//            try
//            {
//
//                if (oSBMLDoc.getLevel() >= 2 && oSBMLDoc.getVersion() >= 2) return sModel;
//
//                UpgradeToL2V4IfNecessary(oSBMLDoc);
//
//
//                return libsbml.writeSBMLToString(oSBMLDoc);
//            }
//            finally
//            {
//                if (oSBMLDoc != NULL)
//                    oSBMLDoc.Dispose();
//            }
//        }
//
//        string NOMSupport::convertSBML(string sModel, int nLevel, int nVersion, bool throwError)
//        {
//            if (sModel == "")
//            {
//                throw new ArgumentException("The model cannot be empty");
//            }
//
//            SBMLDocument oSBMLDoc = libsbml.readSBMLFromString(sModel);
//            try
//            {
//                Model oModel = oSBMLDoc.getModel();
//
//                if (oModel == NULL)
//                {
//                    SBMLErrorLog oLog = oSBMLDoc.getErrorLog();
//                    StringBuilder oBuilder = new StringBuilder();
//                    for (int i = 0; i < oLog.getNumErrors(); i++)
//                    {
//                        SBMLError error = oLog.getError(i);
//                        oBuilder.Append(error.getErrorId() + ": " + error.getMessage() + Environment.NewLine);
//                    }
//
//                    throw new Exception(oBuilder.ToString());
//                }
//
//                oSBMLDoc.setLevelAndVersion((int)nLevel, (int)nVersion, false);
//
//                if (throwError && oSBMLDoc.getNumErrors() > 0)
//                {
//                    SBMLErrorLog oLog = oSBMLDoc.getErrorLog();
//                    bool fatal = false;
//                    StringBuilder oBuilder = new StringBuilder();
//                    for (int i = 0; i < oLog.getNumErrors(); i++)
//                    {
//                        SBMLError error = oLog.getError(i);
//                        if (error.getSeverity() == libsbml.LIBSBML_SEV_ERROR ||
//                         error.getSeverity() == libsbml.LIBSBML_SEV_FATAL)
//                        {
//                            fatal = true;
//                            oBuilder.Append(error.getErrorId() + ": " + error.getMessage() + Environment.NewLine);
//                        }
//                    }
//                    if (fatal)
//                    throw new Exception(oBuilder.ToString());
//                }
//
//                return libsbml.writeSBMLToString(oSBMLDoc);
//            }
//            finally
//            {
//                if (oSBMLDoc != NULL)
//                    oSBMLDoc.Dispose();
//            }
//        }
//
string NOMSupport::convertStringToMathML(const string& var0)
{
    ASTNode *node = SBML_parseFormula(var0.c_str());
    try
    {

        string sResult = writeMathMLToString(node);
        return sResult;
    }
    catch(const Exception& msg)
    {
        throw(msg);
    }
//    finally
//    {
//        if (node != NULL)
//            node.Dispose();
//    }
}

string NOMSupport::convertTime(const string& sArg, const string& sTimeSymbol)
{
    SBMLDocument* oSBMLDoc = NULL;
    Model* oModel = NULL;
    Log(lDebug4)<<"Entering function "<<__FUNC__<<" in file "<<__FILE__;
//    __try
//    {
        try
        {
            oSBMLDoc = readSBMLFromString(sArg.c_str());
            if(oSBMLDoc)
            {
                oModel = oSBMLDoc->getModel();
            }

            if (oModel == NULL)
            {
                throw NOMException("SBML Validation failed");
            }
            else
            {
                changeTimeSymbol(*oModel, sTimeSymbol);
                string sbml(writeSBMLToString(oSBMLDoc));
       	        delete oSBMLDoc;
                return sbml;
            }
        }
        catch(...)
        {
	        delete oSBMLDoc;
            throw NOMException("SBML Validation failed");
        }
//    }
//    __finally
//    {
//        if (oSBMLDoc != NULL)
//        {
//        	delete oSBMLDoc;//->Dispose();
//        }
//    }
    return string("");
}

//        void NOMSupport::ChangeConstantForRules(Model model)
//        {
//            var ruleTargets = new List<string>();
//            for (int i = 0; i < model.getNumRules(); i++)
//            {
//                var rule = model.getRule(i);
//                ruleTargets.Add(rule.getVariable());
//            }
//            for (int i = 0; i < model.getNumParameters(); i++)
//            {
//                var parameter = model.getParameter(i);
//                if (ruleTargets.Contains(parameter.getId()))
//                    parameter.setConstant(false);
//            }
//
//            for (int i = 0; i < model.getNumCompartments(); i++)
//            {
//                var compartment = model.getCompartment(i);
//                if (ruleTargets.Contains(compartment.getId()))
//                    compartment.setConstant(false);
//            }
//
//        }
//        /// <summary>
//        /// This function alters the given SBML model by fixing common errors:
//        ///
//        /// parameter "time", "avogadro" will be replaced with their respective CSymbol
//        /// missing modifiers will be added,
//        ///
//        /// also parameters with rules will be set to constant
//        /// </summary>
//        /// <param name="sbml">the sbml string to fix</param>
//        /// <returns></returns>
//        string NOMSupport::FixCommonIssues(string sbml)
//        {
//            return FixCommonIssues(sbml, NULL, NULL);
//        }
//
//        /// <summary>
//        /// This function alters the given SBML model by fixing common errors:
//        ///
//        /// parameter "time", "avogadro" will be replaced with their respective CSymbol
//        /// missing modifiers will be added,
//        ///
//        /// also parameters with rules will be set to constant
//        /// </summary>
//        /// <param name="sbml">the sbml string to fix</param>
//        /// <param name="programName">program name (or NULL in case of none)</param>
//        /// <param name="programVersion">program version</param>
//        /// <returns></returns>
//        string NOMSupport::FixCommonIssues(string sbml, string programName, string programVersion)
//        {
//            var doc = libsbml.readSBMLFromString(sbml);
//            var model = doc.getModel();
//            if (model == NULL)
//            {
//                throw new Exception("SBML Validation failed");
//            }
//            ChangeNameToCSymbol(model, "time", libsbml.AST_NAME_TIME);
//            ChangeNameToCSymbol(model, "avogadro", libsbml.AST_NAME_AVOGADRO);
//
//            addMissingModifiers(model);
//
//            ChangeConstantForRules(model);
//
//            using (var writer = new SBMLWriter())
//            {
//                if (!string.IsNullOrEmpty(programName))
//                {
//
//                    writer.setProgramName(programName);
//                    if (!string.IsNullOrEmpty(programVersion))
//                        writer.setProgramVersion(programVersion);
//                }
//
//                return writer.writeSBMLToString(doc);
//            }
//        }
//
//        string NOMSupport::convertTimeToCSymbol(string sArg, string sTimeSymbol)
//        {
//            SBMLDocument oSBMLDoc = NULL;
//            Model oModel = NULL;
//
//            try
//            {
//                oSBMLDoc = libsbml.readSBMLFromString(sArg);
//                oModel = oSBMLDoc.getModel();
//
//                if (oModel == NULL)
//                {
//                    throw new Exception("SBML Validation failed");
//                }
//                else
//                {
//                    ChangeNameToCSymbol(oModel, sTimeSymbol, libsbml.AST_NAME_TIME);
//                    return libsbml.writeSBMLToString(oSBMLDoc);
//                }
//            }
//            finally
//            {
//                if (oSBMLDoc != NULL)
//                    oSBMLDoc.Dispose();
//            }
//        }
//
void NOMSupport::changeSymbol(Model& oModel, const string& sTimeSymbol, const int& targetType)
{
    for (int i = 0; i < oModel.getNumReactions(); i++)
    {
        Reaction *r = oModel.getReaction(i);
        if(!r)
        {
            continue;
        }

        if (r->getKineticLaw() != NULL && r->getKineticLaw()->isSetMath())
        {
            const ASTNode* node = (ASTNode*) r->getKineticLaw()->getMath();
            const ASTNode* math = changeSymbol((ASTNode*) node, sTimeSymbol, targetType);
            r->getKineticLaw()->setMath(math);
        }
    }

    for (int i = 0; i < oModel.getNumRules(); i++)
    {
        Rule* r = oModel.getRule(i);
        if (r->isSetMath())
        {
            r->setMath(changeSymbol((ASTNode*) r->getMath(), sTimeSymbol, targetType));
        }
    }

    for (int i = 0; i < oModel.getNumInitialAssignments(); i++)
    {
        InitialAssignment *initialAssignment = oModel.getInitialAssignment(i);
        if (initialAssignment->isSetMath())
        {
            initialAssignment->setMath(changeSymbol((ASTNode*) initialAssignment->getMath(), sTimeSymbol, targetType));
        }
    }

    for (int i = 0; i < oModel.getNumEvents(); i++)
    {
        Event *oEvent = oModel.getEvent(i);
        if (oEvent->getTrigger()->isSetMath())
        {
            oEvent->getTrigger()->setMath((const ASTNode*) changeSymbol((ASTNode*) oEvent->getTrigger()->getMath(), sTimeSymbol, targetType));
        }

        if (oEvent->isSetDelay() && oEvent->getDelay()->isSetMath())
        {
            oEvent->getDelay()->setMath((ASTNode*) changeSymbol((ASTNode*) oEvent->getDelay()->getMath(), sTimeSymbol, targetType));
        }

        for (int j = 0; j < oEvent->getNumEventAssignments(); j++)
        {
            EventAssignment *assignment = oEvent->getEventAssignment(j);
            if (assignment->isSetMath())
            {
                assignment->setMath(changeSymbol((ASTNode*) assignment->getMath(), sTimeSymbol, targetType));
            }
        }
    }
}

//        void NOMSupport::ChangeNameToCSymbol(Model model, string name, int type)
//        {
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                libsbmlcs.Reaction r = model.getReaction(i);
//                if (r.getKineticLaw() != NULL && r.getKineticLaw().isSetMath())
//                    r.getKineticLaw().setMath(changeTimeToCSymbol(r.getKineticLaw().getMath(), name, type));
//            }
//            for (int i = 0; i < model.getNumRules(); i++)
//            {
//                Rule r = model.getRule(i);
//                if (r.isSetMath())
//                    r.setMath(changeTimeToCSymbol(r.getMath(), name, type));
//            }
//            for (int i = 0; i < model.getNumEvents(); i++)
//            {
//                Event oEvent = model.getEvent(i);
//                if (oEvent.getTrigger().isSetMath())
//                    oEvent.getTrigger().setMath(changeTimeToCSymbol(oEvent.getTrigger().getMath(), name, type));
//            }
//        }
//
const ASTNode* NOMSupport::changeSymbol(ASTNode* node, const string& time, const int& targetType)
{
    int c;
    if (node->getType() == targetType)
        node->setName(time.c_str());

    for (c = 0; c < node->getNumChildren(); c++)
        changeSymbol(node->getChild(c), time, targetType);
    return node;
}

//        ASTNode NOMSupport::ReplaceSymbol(ASTNode node, string oldId, string newId)
//        {
//            int c;
//            if (node.getType() == libsbml.AST_NAME && node.getName() == oldId)
//                node.setName(newId);
//
//            for (c = 0; c < node.getNumChildren(); c++)
//                ReplaceSymbol(node.getChild(c), oldId, newId);
//            return node;
//        }
//
//
//        ASTNode NOMSupport::changeTimeToCSymbol(ASTNode node, string name, int type)
//        {
//            int c;
//            if (node.getName() == name && node.getType() != type)
//                node.setType(type);
//            for (c = 0; c < node.getNumChildren(); c++)
//                changeTimeToCSymbol(node.getChild(c), name, type);
//            return node;
//        }
//
//        bool NOMSupport::exists(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return true;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return true;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return true;
//            }
//            return false;
//        }
//
//        string NOMSupport::getAnnotation(string sId)
//        {
//            if (mModel == NULL)
//            {
//                return "";
//                //throw new Exception("You need to load the model first");
//            }
//
//            string sResult = "";
//
//            if (mModel->getId() == sId || mModel->getName() == sId)
//            {
//                if (mModel->isSetAnnotation())
//                {
//                    sResult = mModel->getAnnotationString();
//                }
//                return sResult;
//
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                if (oSpecies.isSetAnnotation())
//                {
//                    sResult = oSpecies.getAnnotationString();
//                }
//                return sResult;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                if (oParameter.isSetAnnotation())
//                {
//                    sResult = oParameter.getAnnotationString();
//                }
//                return sResult;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                if (oCompartment.isSetAnnotation())
//                {
//                    sResult = oCompartment.getAnnotationString();
//                }
//                return sResult;
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                if (oReaction.isSetAnnotation())
//                {
//                    sResult = oReaction.getAnnotationString();
//                }
//                return sResult;
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                if (oRule.isSetAnnotation())
//                {
//                    sResult = oRule.getAnnotationString();
//                }
//                return sResult;
//            }
//
//            return "";
//            //throw new Exception("Invalid id. No element with the given id exists in the model.");
//        }
//
//        string[] NOMSupport::getBuiltinFunctionInfo(string var0)
//        {
//            for (int i = 0; i < _oPredefinedFunctions.Length; i++)
//            {
//                if (_oPredefinedFunctions[i][0] == var0)
//                    return _oPredefinedFunctions[i];
//            }
//
//            throw new Exception("Invalid string name. There is no inbuilt function with that name: " + var0);
//        }
//
//        string[] NOMSupport::getBuiltinFunctions()
//        {
//            string[] sResult = new string[_oPredefinedFunctions.Length];
//
//            int i;
//
//            for (i = 0; i < _oPredefinedFunctions.Length; i++)
//            {
//                sResult[i] = _oPredefinedFunctions[i][0];
//            }
//
//            return sResult;
//
//        }
//
//        string NOMSupport::getCompartmentIdBySpeciesId(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies == NULL)
//            {
//                throw new Exception("The model does not have a species corresponding to the Id provided");
//            }
//            return oSpecies.getCompartment();
//        }
//
//        ArrayList NOMSupport::getDerivedUnitDefinition(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                UnitDefinition oUnitDef = oSpecies.getDerivedUnitDefinition();
//                return returnUnitDefinition(oUnitDef);
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                UnitDefinition oUnitDef = oCompartment.getDerivedUnitDefinition();
//                return returnUnitDefinition(oUnitDef);
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                UnitDefinition oUnitDef = oParameter.getDerivedUnitDefinition();
//                return returnUnitDefinition(oUnitDef);
//            }
//            return new ArrayList();
//        }
//
//        ArrayList returnUnitDefinition(UnitDefinition oDefinition)
//        {
//            ArrayList oResultDef = new ArrayList();
//            for (int i = 0; i < oDefinition.getNumUnits(); i++)
//            {
//                Unit oUnit = oDefinition.getUnit(i);
//                if (oUnit != NULL)
//                {
//                    ArrayList oResult = new ArrayList();
//                    oResult.Add(libsbml.UnitKind_toString(oUnit.getKind()));
//                    oResult.Add(oUnit.getExponent());
//                    oResult.Add(oUnit.getMultiplier());
//                    oResult.Add(oUnit.getOffset());
//                    oResult.Add(oUnit.getScale());
//                    oResultDef.Add(oResult);
//                }
//            }
//            return oResultDef;
//        }
//
string NOMSupport::getKineticLaw(const int& index)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (mModel->getNumReactions() <= (int)index)
    {
        throw Exception("No Reaction for the provided index");
    }

    Reaction *r = mModel->getReaction((int)index);

    if (!r->isSetKineticLaw())
    {
        throw Exception("No Kinetic Law present");
    }

    KineticLaw *k = r->getKineticLaw();

    if (!k->isSetFormula())
    {
        throw Exception("No Formula present");
    }
    return k->getFormula();
}


//        ArrayList NOMSupport::getListOfBoundarySpeciesIds()
//        {
//            ArrayList boundarySpeciesIdList = new ArrayList();
//
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (oSpecies.getBoundaryCondition())
//                {
//                    boundarySpeciesIdList.Add(GetId(oSpecies));
//                }
//            }
//
//            return boundarySpeciesIdList;
//        }
//
//        ArrayList NOMSupport::getListOfErrors()
//        {
//            if (_oDoc == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//            int nErrors = (int)_oDoc.getNumErrors();
//
//            ArrayList oErrorList = new ArrayList();
//            for (int i = 0; i < nErrors; i++)
//            {
//                oErrorList.Add(getNthError((int)i));
//            }
//            return oErrorList;
//        }
//


//StringListContainer NOMSupport::GetFloatingSpecies()
//{
//    StringContainer floatingSpeciesList;
//    int nrOfSpecies = ::getNumFloatingSpecies();
//
//    for (int i = 0; i < nrOfSpecies; i++)
//    {
//        Species* aSpecies = mModel->getSpecies(i);
//        if( aSpecies != NULL && !aSpecies->getBoundaryCondition())
//        {
//            StringCollection oSpeciesValues;// = new ArrayList();
//            //oSpeciesValues.Add(GetId(aSpecies));
//            oSpeciesValues.Add(aSpecies->getId());
//            double concentration = aSpecies->isSetInitialConcentration() ? aSpecies->getInitialConcentration() : aSpecies->getInitialAmount();
//
//            oSpeciesValues.Add(ToString(concentration));
//            oSpeciesValues.Add(ToString(aSpecies->isSetInitialConcentration()));
//            floatingSpeciesList.Add(oSpeciesValues);
//        }
//    }
//
//    return floatingSpeciesList;
//}


StringListContainer NOMSupport::getListOfFloatingSpecies()
{
    StringListContainer floatingSpeciesList;

    if (mModel == NULL)
    {
        throw NOMException("You need to load the model first");
    }

    for (int i = 0; i < mModel->getNumSpecies(); i++)
    {
        Species *oSpecies = mModel->getSpecies(i);
        if (oSpecies && !oSpecies->getBoundaryCondition())
        {
            StringList oSpeciesValues;
            oSpeciesValues.Add( oSpecies->getId() );
            double concentration = oSpecies->isSetInitialConcentration() ? oSpecies->getInitialConcentration() : oSpecies->getInitialAmount();
            oSpeciesValues.Add( ToString(concentration, STR_DoubleFormat) );
            oSpeciesValues.Add( ToString(oSpecies->isSetInitialConcentration()));

            floatingSpeciesList.Add(oSpeciesValues);
        }
    }

    return floatingSpeciesList;
}

//        ArrayList NOMSupport::getListOfFloatingSpeciesIds()
//        {
//            ArrayList floatingSpeciesIdList = new ArrayList();
//
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (!oSpecies.getBoundaryCondition())
//                {
//                    floatingSpeciesIdList.Add(GetId(oSpecies));
//                }
//            }
//
//            return floatingSpeciesIdList;
//        }
//
ArrayList NOMSupport::getListOfParameters()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    ArrayList paramStrValueList;

    int numOfGlobalParameters = mModel->getNumParameters();

    for (int i = 0; i < numOfGlobalParameters; i++)
    {
        Parameter *parameter = mModel->getParameter(i);
        double paramValue;
        string paramStr = parameter->getId();
        StringList tempStrValueList;
        tempStrValueList.Add(paramStr);

        if ((parameter->isSetValue()))
        {
            paramValue = parameter->getValue();
        }
        else
        {
            paramValue = 0.0;
        }
        tempStrValueList.Add(ToString(paramValue, STR_DoubleFormat));

        paramStrValueList.Add(tempStrValueList);
    }

    int numOfReactions = mModel->getNumReactions();
    Reaction *r;
    KineticLaw *kl;
    for (int i = 0; i < numOfReactions; i++)
    {
        r = mModel->getReaction(i);
        kl = r->getKineticLaw();
        if (kl == NULL)
        {
            continue;
        }
        else
        {
            int numOfLocalParameters = kl->getNumParameters();
            for (int j = 0; j < numOfLocalParameters; j++)
            {
                Parameter *parameter = kl->getParameter(j);
                string paramStr = parameter->getId();
                StringList tempStrValueList;
                double paramValue;
                tempStrValueList.Add(paramStr);
                if (parameter->isSetValue())
                {
                    paramValue = parameter->getValue();
                }
                else
                {
                    paramValue = 0.0;
                }
                tempStrValueList.Add(ToString(paramValue, STR_DoubleFormat));
                paramStrValueList.Add(tempStrValueList);
            }
        }
    }
    return paramStrValueList;
}

//        string NOMSupport::getModelId()
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//            return GetId(mModel);
//        }
//
string NOMSupport::getModelName()
{
    if (mModel == NULL)
    {
        throw NOMException("You need to load the model first");
    }
    return GetName((SBase*) mModel);
}

//        string NOMSupport::getNotes(string sId)
//        {
//            if (mModel == NULL)
//            {
//                //throw new Exception("You need to load the model first");
//                return "";
//            }
//
//            string sResult = "";
//
//            if (mModel->getId() == sId || mModel->getName() == sId)
//            {
//                if (mModel->isSetNotes())
//                {
//                    sResult = mModel->getNotesString();
//                }
//                return sResult;
//
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                if (oSpecies.isSetNotes())
//                {
//                    sResult = oSpecies.getNotesString();
//                }
//                return sResult;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                if (oParameter.isSetNotes())
//                {
//                    sResult = oParameter.getNotesString();
//                }
//                return sResult;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                if (oCompartment.isSetNotes())
//                {
//                    sResult = oCompartment.getNotesString();
//                }
//                return sResult;
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                if (oReaction.isSetNotes())
//                {
//                    sResult = oReaction.getNotesString();
//                }
//                return sResult;
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                if (oRule.isSetNotes())
//                {
//                    sResult = oRule.getNotesString();
//                }
//                return sResult;
//            }
//
//            return "";
//            //throw new Exception("Invalid id. No element with the given id exists in the model.");
//        }
//
string NOMSupport::getNthBoundarySpeciesCompartmentName(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    int nCount = 0;
    for (int i = 0; i < mModel->getNumSpecies(); i++)
    {
        Species *oSpecies = mModel->getSpecies(i);
        if (oSpecies->getBoundaryCondition())
        {
            if (nCount == nIndex)
            {
                return oSpecies->getCompartment();
            }
            else
            {
                nCount++;
            }
        }
    }
    throw Exception("The model does not have a boundary species corresponding to the index provided");
}

//        string NOMSupport::getNthBoundarySpeciesId(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            int nCount = 0;
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (oSpecies.getBoundaryCondition())
//                {
//                    if (nCount == nIndex)
//                    {
//                        return GetId(oSpecies);
//                    }
//                    else
//                    {
//                        nCount++;
//                    }
//                }
//            }
//            throw new Exception("The model does not have a boundary species corresponding to the index provided");
//        }
//
//        string NOMSupport::getNthBoundarySpeciesName(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            int nCount = 0;
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (oSpecies.getBoundaryCondition())
//                {
//                    if (nCount == nIndex)
//                    {
//                        return GetName(oSpecies);
//                    }
//                    else
//                    {
//                        nCount++;
//                    }
//                }
//            }
//            throw new Exception("The model does not have a boundary species corresponding to the index provided");
//        }
//
//        string NOMSupport::getNthCompartmentId(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            if (nIndex < 0 || nIndex >= (int)mModel->getNumCompartments())
//            {
//                throw new Exception("Invalid input - Argument should be >= 0 and should be less than total number of compartments in the model");
//
//            }
//            Compartment oCompartment = mModel->getCompartment((int)nIndex);
//            return GetId(oCompartment);
//        }
//
//        string NOMSupport::getNthCompartmentName(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw new Exception("You need to load the model first");
//            }
//
//            if (nIndex < 0 || nIndex >= (int)mModel->getNumCompartments())
//            {
//                throw new Exception("Invalid input - Argument should be >= 0 and should be less than total number of compartments in the model");
//
//            }
//            Compartment oCompartment = mModel->getCompartment((int)nIndex);
//            return GetName(oCompartment);
//        }
//
ArrayList NOMSupport::getNthError(const int& nIndex)
{
    if (mSBMLDoc == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nIndex >= mSBMLDoc->getNumErrors())
    {
        throw Exception("Index out of Bounds.");
    }

    SBMLError *error = (SBMLError*) mSBMLDoc->getError(nIndex);
    ArrayList oResult;// = new ArrayList();

    switch (error->getSeverity())
    {
        default:
        case (int)LIBSBML_SEV_INFO: oResult.Add("Advisory"); break;
        case (int)LIBSBML_SEV_WARNING: oResult.Add("Warning"); break;
        case (int)LIBSBML_SEV_FATAL: oResult.Add("Fatal"); break;
        case (int)LIBSBML_SEV_ERROR: oResult.Add("Error"); break;
        case (int)LIBSBML_SEV_SCHEMA_ERROR: oResult.Add("Error"); break;
        case (int)LIBSBML_SEV_GENERAL_WARNING: oResult.Add("Warning"); break;
    }
    oResult.Add((int) error->getLine());
    oResult.Add((int) error->getColumn());
    oResult.Add((int) error->getErrorId());
    oResult.Add(error->getMessage());
    return oResult;
}

bool NOMSupport::getNthUseValuesFromTriggerTime(const int& arg)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    Event *oEvent = mModel->getEvent((int)arg);

    if (oEvent == NULL)
    {
        throw Exception("The model does not have a Event corresponding to the index provided");
    }
    return oEvent->getUseValuesFromTriggerTime();
}

ArrayList NOMSupport::getNthEvent(const int& arg)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    ArrayList triggerAssignmentsList;
    Event *oEvent = mModel->getEvent((int)arg);

    if (oEvent == NULL)
    {
        throw Exception("The model does not have a Event corresponding to the index provided");
    }

    string trigger = SBML_formulaToString(oEvent->getTrigger()->getMath());
    triggerAssignmentsList.Add(trigger);

    string delay;
    if (!oEvent->isSetDelay())
    {
        delay = "0";
    }
    else
    {
        Delay *oDelay = oEvent->getDelay();
        if (oDelay->isSetMath())
        {
            delay = SBML_formulaToString(oDelay->getMath());
        }
        else
        {
            delay = "0";
        }
    }

    triggerAssignmentsList.Add(delay);
    int numEventAssignments = (int)oEvent->getNumEventAssignments();

    for (int i = 0; i < numEventAssignments; i++)
    {
        StringList assignmentList;// = new ArrayList();

        EventAssignment *ea = oEvent->getEventAssignment(i);
        string lValue = ea->getVariable();
        string rValue = SBML_formulaToString(ea->getMath());

        assignmentList.Add(lValue);
        assignmentList.Add(rValue);
        triggerAssignmentsList.Add(assignmentList);
    }
    return triggerAssignmentsList;
}

string NOMSupport::getNthFloatingSpeciesCompartmentName(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw NOMException("You need to load the model first");
    }

    int nCount = 0;
    for (u_int i = 0; i < mModel->getNumSpecies(); i++)
    {
        Species *aSpecies = mModel->getSpecies(i);
        if (!aSpecies->getBoundaryCondition())
        {
            if (nCount == nIndex)
            {
                return aSpecies->getCompartment();
            }
            else
            {
                nCount++;
            }
        }
    }
    throw NOMException("The model does not have a floating species corresponding to the index provided");
}

//        string NOMSupport::getNthFloatingSpeciesId(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            int nCount = 0;
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (!oSpecies.getBoundaryCondition())
//                {
//                    if (nCount == nIndex)
//                    {
//                        return GetId(oSpecies);
//                    }
//                    else
//                    {
//                        nCount++;
//                    }
//                }
//            }
//            throw Exception("The model does not have a floating species corresponding to the index provided");
//        }
//
//        string NOMSupport::getNthFloatingSpeciesName(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            int nCount = 0;
//            for (int i = 0; i < mModel->getNumSpecies(); i++)
//            {
//                libsbmlcs.Species oSpecies = mModel->getSpecies(i);
//                if (!oSpecies.getBoundaryCondition())
//                {
//                    if (nCount == nIndex)
//                    {
//                        return GetName(oSpecies);
//                    }
//                    else
//                    {
//                        nCount++;
//                    }
//                }
//            }
//            throw Exception("The model does not have a floating species corresponding to the index provided");
//        }
//
ArrayList NOMSupport::getNthFunctionDefinition(const int& arg)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (arg < 0 || arg >= (int) mModel->getNumFunctionDefinitions())
    {
        throw Exception("Invalid input - Argument should be >= 0 and should be less than total number of Function Definitions in the model");
    }

    FunctionDefinition* fnDefn = mModel->getFunctionDefinition((int)arg);

    if (fnDefn == NULL)
    {
        throw Exception("The model does not have a Function Definition corresponding to the index provided");
    }

    string fnId = fnDefn->getId();
    string fnMath = SBML_formulaToString(fnDefn->getBody());

    ArrayList fnDefnList;
    fnDefnList.Add(fnId);

    int numArgs = (int) fnDefn->getNumArguments();

    StringList argList;
    for(int n = 0; n < numArgs; n++)
    {
        argList.Add(fnDefn->getArgument(n)->getName());
    }

    fnDefnList.Add(argList);
    fnDefnList.Add(fnMath);

    return fnDefnList;
}

//        string NOMSupport::getNthGlobalParameterId(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//
//            }
//
//            if (nIndex >= (int)mModel->getNumParameters())
//            {
//                throw Exception("There is no parameter corresponding to the index you provided");
//
//            }
//
//            Parameter oParameter = mModel->getParameter((int)nIndex);
//            if (oParameter == NULL)
//            {
//                throw Exception("There is no parameter corresponding to the index you provided");
//            }
//            return GetId(oParameter);
//        }
//
//        string NOMSupport::getNthGlobalParameterName(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//
//            }
//
//            if (nIndex >= (int)mModel->getNumParameters())
//            {
//                throw Exception("There is no parameter corresponding to the index you provided");
//
//            }
//
//            Parameter oParameter = mModel->getParameter((int)nIndex);
//            if (oParameter == NULL)
//            {
//                throw Exception("There is no parameter corresponding to the index you provided");
//            }
//            return GetName(oParameter);
//        }
//
//        ArrayList NOMSupport::getNthListOfModifiers(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            ArrayList modifierList = new ArrayList();
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            int numModifiers = (int)r.getNumModifiers();
//            for (int i = 0; i < numModifiers; i++)
//            {
//                modifierList.Add(r.getModifier(i).getSpecies());
//            }
//            return modifierList;
//        }
//
//        ArrayList NOMSupport::getNthListOfProducts(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            ArrayList productList = new ArrayList();
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            int numProducts = (int)r.getNumProducts();
//            for (int i = 0; i < numProducts; i++)
//            {
//                libsbmlcs.SpeciesReference product = r.getProduct(i);
//                string stoichiometryMath = "";
//                if (product.isSetStoichiometryMath() && product.getStoichiometryMath().isSetMath())
//                    stoichiometryMath = libsbml.formulaToString(product.getStoichiometryMath().getMath());
//                ArrayList oTemp = new ArrayList(); oTemp.Add(product.getSpecies()); oTemp.Add(product.getStoichiometry());
//                oTemp.Add(stoichiometryMath);
//                productList.Add(oTemp);
//            }
//            return productList;
//        }
//
//        ArrayList NOMSupport::getNthListOfReactants(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            ArrayList reactantList = new ArrayList();
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            int numReactants = (int)r.getNumReactants();
//            for (int i = 0; i < numReactants; i++)
//            {
//                libsbmlcs.SpeciesReference reactant = r.getReactant(i);
//                string stoichiometryMath = "";
//                if (reactant.isSetStoichiometryMath() && reactant.getStoichiometryMath().isSetMath())
//                    stoichiometryMath = libsbml.formulaToString(reactant.getStoichiometryMath().getMath());
//                ArrayList oTemp = new ArrayList(); oTemp.Add(reactant.getSpecies()); oTemp.Add(reactant.getStoichiometry());
//                oTemp.Add(stoichiometryMath);
//                reactantList.Add(oTemp);
//            }
//            return reactantList;
//        }
//
//        bool NOMSupport::getNthParameterHasValue(int nReactionIndex, int nParameterIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nReactionIndex < 0 || nReactionIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction((int)nReactionIndex);
//            KineticLaw kl = oReaction.getKineticLaw();
//
//            if (nParameterIndex < 0 || nParameterIndex >= (int)kl.getNumParameters())
//            {
//                throw Exception("Index exceeds the number of Parameters in the list");
//            }
//
//            return kl.getParameter((int)nParameterIndex).isSetValue();
//
//        }
//
string NOMSupport::getNthParameterId(const int& nReactionIndex, const int& nParameterIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nReactionIndex < 0 || nReactionIndex >= (int)mModel->getNumReactions())
    {
        throw Exception("There is no reaction corresponding to the index you provided");
    }

    Reaction *oReaction = mModel->getReaction((int)nReactionIndex);
    KineticLaw *kl = oReaction->getKineticLaw();

    if (nParameterIndex < 0 || nParameterIndex >= (int)kl->getNumParameters())
    {
        throw Exception("Index exceeds the number of Parameters in the list");
    }

    return kl->getParameter((int)nParameterIndex)->getId();
}

//        string NOMSupport::getNthParameterName(int nReactionIndex, int nParameterIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nReactionIndex < 0 || nReactionIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction((int)nReactionIndex);
//            KineticLaw kl = oReaction.getKineticLaw();
//
//            if (nParameterIndex < 0 || nParameterIndex >= (int)kl.getNumParameters())
//            {
//                throw Exception("Index exceeds the number of Parameters in the list");
//            }
//
//            return kl.getParameter((int)nParameterIndex).getName();
//        }
//
double NOMSupport::getNthParameterValue(const int& nReactionIndex, const int& nParameterIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nReactionIndex < 0 || nReactionIndex >= (int)mModel->getNumReactions())
    {
        throw Exception("There is no reaction corresponding to the index you provided");
    }

    Reaction *oReaction = mModel->getReaction((int)nReactionIndex);
    KineticLaw *kl = oReaction->getKineticLaw();

    if (nParameterIndex < 0 || nParameterIndex >= (int)kl->getNumParameters())
    {
        throw Exception("Index exceeds the number of Parameters in the list");
    }

    return kl->getParameter((int)nParameterIndex)->getValue();

}

//        string NOMSupport::getNthProductName(int nIndex, int nProduct)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getProduct((int)nProduct);
//            if (oRef == NULL)
//                throw Exception("No product for the provided index.");
//            return oRef.getSpecies();
//        }
//
//        int NOMSupport::getNthProductStoichiometry(int nIndex, int nProduct)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getProduct((int)nProduct);
//            if (oRef == NULL)
//                throw Exception("No product for the provided index.");
//            return (int)oRef.getStoichiometry();
//        }
//
//        double NOMSupport::getNthProductStoichiometryDouble(int nIndex, int nProduct)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getProduct((int)nProduct);
//            if (oRef == NULL)
//                throw Exception("No product for the provided index.");
//            return oRef.getStoichiometry();
//        }
//
//        string NOMSupport::getNthReactantName(int nIndex, int nReactant)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getReactant((int)nReactant);
//            if (oRef == NULL)
//                throw Exception("No reactant for the provided index.");
//            return oRef.getSpecies();
//        }
//
//        int NOMSupport::getNthReactantStoichiometry(int nIndex, int nReactant)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getReactant((int)nReactant);
//            if (oRef == NULL)
//                throw Exception("No reactant for the provided index.");
//            return (int)oRef.getStoichiometry();
//        }
//
//        double NOMSupport::getNthReactantStoichiometryDouble(int nIndex, int nReactant)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            libsbmlcs.SpeciesReference oRef = r.getReactant((int)nReactant);
//            if (oRef == NULL)
//                throw Exception("No reactant for the provided index.");
//            return oRef.getStoichiometry();
//        }
//
string NOMSupport::getNthReactionId(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nIndex >= (int)mModel->getNumReactions())
    {
        throw Exception("There is no reaction corresponding to the index you provided");
    }

    Reaction &r = *(mModel->getReaction((int)nIndex));
    return GetId(r);
}

string NOMSupport::getNthReactionName(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (nIndex >= (int)mModel->getNumReactions())
    {
        throw Exception("There is no reaction corresponding to the index you provided");
    }

    Reaction *r = mModel->getReaction((int)nIndex);
    return GetName(r);
}

pair<string, string> NOMSupport::getNthInitialAssignmentPair(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    InitialAssignment *oAssignment = mModel->getInitialAssignment((int)nIndex);

    if (oAssignment == NULL)
    {
        throw Exception("The model does not have an InitialAssignment corresponding to the index provided");
    }

    if (!oAssignment->isSetMath())
    {
        throw Exception("The InitialAssignment contains no math.");
    }
    string second = SBML_formulaToString(oAssignment->getMath());
    return pair<string, string> (oAssignment->getSymbol(), SBML_formulaToString(oAssignment->getMath()));
}

string NOMSupport::getNthInitialAssignment(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    InitialAssignment *oAssignment = mModel->getInitialAssignment((int)nIndex);
    if (oAssignment == NULL)
    {
        throw Exception("The model does not have an InitialAssignment corresponding to the index provided");
    }

    if (!oAssignment->isSetMath())
    {
        throw Exception("The InitialAssignment contains no math.");
    }

    return oAssignment->getSymbol() + " = " + SBML_formulaToString(oAssignment->getMath());

}

string NOMSupport::getNthConstraint(const int& nIndex, string& sMessage)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    Constraint *oConstraint = mModel->getConstraint((int)nIndex);

    if (oConstraint == NULL)
    {
        throw Exception("The model does not have a constraint corresponding to the index provided");
    }

    if (!oConstraint->isSetMath())
    {
        throw Exception("The constraint does not provide math.");
    }

    if (!oConstraint->isSetMessage())
    {
        sMessage = "Constraint: " + ToString(nIndex) + " was violated.";
    }
    else
    {
        XMLNode* node = (XMLNode*) oConstraint->getMessage();
        sMessage = node->toString();
    }

    return SBML_formulaToString(oConstraint->getMath());
}

string NOMSupport::getNthRule(const int& nIndex)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    Rule *aRule = mModel->getRule((int)nIndex);
    if (aRule == NULL)
    {
        throw Exception("The model does not have a Rule corresponding to the index provided");
    }
    Rule &oRule = *aRule;

    int type = oRule.getTypeCode();

    switch (type)
    {
        case SBML_PARAMETER_RULE:
        case SBML_SPECIES_CONCENTRATION_RULE:
        case SBML_COMPARTMENT_VOLUME_RULE:
        case SBML_ASSIGNMENT_RULE:
        case SBML_RATE_RULE:
            {
                string lValue = oRule.getVariable();
                string rValue = oRule.getFormula();

                return lValue + " = " + rValue;
            }
        case SBML_ALGEBRAIC_RULE:
            {
                string rValue = oRule.getFormula();
                return rValue + " = 0";
            }


        default:
            break;
    }

    return "";
}

string NOMSupport::getNthRuleType(const int& arg)
{
    string result = "";

    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    Rule *aRule = mModel->getRule((int)arg);

    if (aRule == NULL)
    {
        throw Exception("The model does not have a Rule corresponding to the index provided");
    }

    Rule& rule = *aRule;
    int type = rule.getTypeCode();
    if (type == SBML_PARAMETER_RULE)
    {
        result = "Parameter_Rule";
    }

    if (type == SBML_SPECIES_CONCENTRATION_RULE)
    {
        result = "Species_Concentration_Rule";
    }

    if (type == SBML_COMPARTMENT_VOLUME_RULE)
    {
        result = "Compartment_Volume_Rule";
    }

    if (type == libsbml::SBML_ASSIGNMENT_RULE)
    {
        result = "Assignment_Rule";
    }

    if (type == libsbml::SBML_ALGEBRAIC_RULE)
    {
        result = "Algebraic_Rule";
    }

    if (type == libsbml::SBML_RATE_RULE)
    {
        result = "Rate_Rule";
    }
    return result;
}

int NOMSupport::getNumBoundarySpecies()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumSpeciesWithBoundaryCondition();
}

int NOMSupport::getNumCompartments()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int) mModel->getNumCompartments();
}
//
//        int NOMSupport::getNumErrors()
//        {
//            if (mSBMLDoc == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//            return (int)mSBMLDoc.getNumErrors();
//        }
//
int NOMSupport::getNumEvents()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumEvents();
}

int NOMSupport::getNumFloatingSpecies()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumSpecies() - (int)mModel->getNumSpeciesWithBoundaryCondition();

}

int NOMSupport::getNumInitialAssignments()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumInitialAssignments();
}

int NOMSupport::getNumConstraints()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumConstraints();
}

int NOMSupport::getNumFunctionDefinitions()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumFunctionDefinitions();

}

//        int NOMSupport::getNumGlobalParameters()
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//            return (int)mModel->getNumParameters();
//
//        }
//
int NOMSupport::getNumParameters(const int& var0)
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    if (var0 > mModel->getNumReactions())
    {
        throw Exception("Reaction does not exist");
    }
    Reaction &r = *(mModel->getReaction(var0));
    if (!r.isSetKineticLaw())
    {
        return 0;
    }
    return (int)r.getKineticLaw()->getNumParameters();

}

//        int NOMSupport::getNumProducts(int var0)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//            if (var0 > mModel->getNumReactions())
//                throw Exception("Reaction does not exist");
//            libsbmlcs.Reaction r = mModel->getReaction((int)var0);
//            return (int)r.getNumProducts();
//        }
//
//        int NOMSupport::getNumReactants(int var0)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//            if (var0 > mModel->getNumReactions())
//                throw Exception("Reaction does not exist");
//            libsbmlcs.Reaction r = mModel->getReaction((int)var0);
//            return (int)r.getNumReactants();
//        }
//
int NOMSupport::getNumReactions()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int)mModel->getNumReactions();
}

int NOMSupport::getNumRules()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }
    return (int) mModel->getNumRules();
}

//        string NOMSupport::getOutsideCompartment(string var0)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//
//            }
//
//            Compartment oCompartment = mModel->getCompartment(var0);
//            if (oCompartment == NULL)
//            {
//                throw Exception("There is no compartment corresponding to the input argument.");
//            }
//            return oCompartment.getOutside();
//
//        }
//
string NOMSupport::getParamPromotedSBML(const string& sArg)
{
	SBMLDocument *oSBMLDoc = NULL;
	Model *oModel = NULL;

	oSBMLDoc = libsbml::readSBMLFromString(sArg.c_str());
    if (oSBMLDoc->getLevel() == 1)
    {
        oSBMLDoc->setLevelAndVersion(2, 1);
    }

    //oSBMLDoc.setLevelAndVersion(2, 1);
    oModel = oSBMLDoc->getModel();

    if (oModel == NULL)
    {
        throw Exception("SBML Validation failed");
    }
    else
    {
        modifyKineticLaws(*oSBMLDoc, *oModel);
        changeTimeSymbol(*oModel, "time");
        return libsbml::writeSBMLToString(oSBMLDoc);
    }
}

void NOMSupport::modifyKineticLawsForLocalParameters(KineticLaw& oLaw, const string& reactionId, Model& oModel)
{
	int numLocalParameters = (int)oLaw.getNumLocalParameters();
	if (numLocalParameters > 0)
    {
        StringCollection oList;// = new StringCollection();
        for (int j = numLocalParameters; j > 0; j--)
        {
            LocalParameter* localParameter = (LocalParameter*)oLaw.getLocalParameter(j - 1)->clone();
            string parameterId = localParameter->getId();// GetId(localParameter);
            string sPrefix = reactionId + "_";
            if (!oLaw.isSetMath())
            {
                if (oLaw.isSetFormula())
                {
                    ASTNode *node = readMathMLFromString(oLaw.getFormula().c_str());
                    ChangeParameterName(*node, parameterId, sPrefix);
                    string sNode = SBML_formulaToString(node);
                    oLaw.setFormula(sNode);
                }
            }
            else
            {
                ChangeParameterName(*((ASTNode*) oLaw.getMath()), parameterId, sPrefix);
            }

            Parameter *p = oModel.createParameter();
            p->setId(sPrefix + parameterId);
            p->setNotes(localParameter->getNotesString());
            p->setAnnotation(localParameter->getAnnotationString());
            p->setConstant(true);
            if (localParameter->isSetSBOTerm()) p->setSBOTerm(localParameter->getSBOTerm());
            if (localParameter->isSetName()) p->setName(localParameter->getName());
            if (localParameter->isSetMetaId()) p->setMetaId(localParameter->getMetaId());
            if (localParameter->isSetValue()) p->setValue(localParameter->getValue());
            if (localParameter->isSetUnits()) p->setUnits(localParameter->getUnits());

            LocalParameter* oTemp = (LocalParameter*) oLaw.getListOfLocalParameters()->remove(j - 1);
            //if (oTemp != NULL) oTemp.Dispose();

            oModel.addParameter(p);
            //if (localParameter != NULL) localParameter.Dispose();
        }
    }
}


void NOMSupport::modifyKineticLawsForReaction(KineticLaw& oLaw, const string& reactionId, Model& oModel)
{
    int numLocalParameters = (int)oLaw.getNumParameters();
    if (numLocalParameters > 0)
    {
//        StringCollection oList = new StringCollection();
        for (int j = numLocalParameters; j > 0; j--)
        {
            Parameter *parameter = (Parameter*) oLaw.getParameter(j - 1)->clone();
            if(!parameter)
            {
            	throw(NOMException("Null parameter pointer in modifyKineticLawsForReaction"));
            }
            string parameterId = GetId( *parameter);
            string sPrefix = reactionId + "_";
            if (!oLaw.isSetMath())
            {
                if (oLaw.isSetFormula())
                {
                    ASTNode *node = readMathMLFromString(oLaw.getFormula().c_str());
                    ChangeParameterName(*node, parameterId, sPrefix);
                    string sNode = SBML_formulaToString(node);
                    oLaw.setFormula(sNode);
                }
            }
            else
            {
                ChangeParameterName( *(ASTNode*)oLaw.getMath(), parameterId, sPrefix);
            }
            Parameter *oTemp = (Parameter*)oLaw.getListOfParameters()->remove(j - 1);
            if(!oTemp)
            {
            	throw(NOMException("Null parameter pointer in modifyKineticLawsForReaction"));
            }

            if (oTemp != NULL)
            {
                //    oTemp.Dispose();
            }
            parameter->setId(sPrefix + parameterId);
            //oModel.getListOfParameters().append(parameter);
            //oModel.getListOfParameters().appendAndOwn(parameter);
            oModel.addParameter(parameter);
            if (parameter != NULL)
            {
                //parameter->Dispose();
            }
        }
    }
}

void NOMSupport::modifyKineticLaws(SBMLDocument& oSBMLDoc, Model& oModel)
{
    int numOfReactions = (int)oModel.getNumReactions();
    for (int i = 0; i < numOfReactions; i++)
    {
        Reaction *oReaction = oModel.getReaction(i);
        string sId = GetId(*oReaction);
        KineticLaw *oLaw = oReaction->getKineticLaw();
        if (oLaw == NULL)
        {
            if (oReaction != NULL)
            {
                //oReaction->Dispose();
            }
            continue;
        }

        modifyKineticLawsForLocalParameters(*oLaw, sId, oModel);

        modifyKineticLawsForReaction(*oLaw, sId, oModel);

//        if (oLaw != NULL)
//        {
//            delete oLaw;
//            oLaw = NULL;
//        }
//
//        if (oReaction != NULL)
//        {
//            delete oReaction;
//            oReaction  = NULL;
//        }
    }
}

void NOMSupport::ChangeParameterName(ASTNode& node, const string& sParameterName, const string& sPrefix)
{
    int c;

    if (node.isName() && node.getName() == sParameterName)
    {
        node.setName( string(sPrefix + sParameterName).c_str());
    }

    for (c = 0; c < node.getNumChildren(); c++)
    {
        ChangeParameterName( *node.getChild(c), sParameterName, sPrefix);
    }
}

string NOMSupport::getSBML()
{
    if (mModel == NULL)
    {
        throw Exception("You need to load the model first");
    }

    //Todo: how to deal with parametersets...?
    //    if (_ParameterSets != NULL && mModel != NULL)
//    {
//        _ParameterSets.AddToModel(mModel);
//    }

    return libsbml::writeSBMLToString(mSBMLDoc);
}

//        int getSBOTerm(string sId)
//        {
//            if (mModel == NULL)
//            {
//                return 0;
//                //throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return oSpecies.getSBOTerm();
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter.getSBOTerm();
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment.getSBOTerm();
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                return oReaction.getSBOTerm();
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                return oRule.getSBOTerm();
//            }
//
//            if (mModel->getId() == sId)
//                return mModel->getSBOTerm();
//
//            return 0;
//            //throw Exception("Invalid id. No element with the given id exists in the model.");
//
//        }
//
//        void NOMSupport::TestASTTime()
//        {
//            var mathML = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n            <csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\"> time </csymbol>\n            </math>\n";
//            var node = libsbml::readMathMLFromString(mathML);
//
//            System.Diagnostics.Debug.WriteLine(
//                string.Format("Node Type: {0}, AST_NAME_TIME: {1}, AST_NAME: {2}, AST_NAME_AVOGADRO: {3}",
//                node.getType(),
//                libsbml::AST_NAME_TIME,
//                libsbml::AST_NAME,
//                libsbml::AST_NAME_AVOGADRO));
//
//
//        }
//
//
//        double NOMSupport::getValue(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                if (oSpecies.isSetInitialAmount())
//                    return oSpecies.getInitialAmount();
//                else
//                    return oSpecies.getInitialConcentration();
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment.getVolume();
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter.getValue();
//            }
//
//            for (int i = 0; i < mModel->getNumReactions(); i++)
//            {
//                var reaction = mModel->getReaction(i);
//                for (int j = 0; j < reaction.getNumReactants(); j++)
//                {
//                    var reference = reaction.getReactant(j);
//                    if (reference.isSetId() && reference.getId() == sId)
//                    {
//                        if (reference.isSetStoichiometry())
//                        {
//                            return reference.getStoichiometry();
//                        }
//                        else return 1;
//                    }
//                }
//                for (int j = 0; j < reaction.getNumProducts(); j++)
//                {
//                    var reference = reaction.getProduct(j);
//                    if (reference.isSetId() && reference.getId() == sId)
//                    {
//                        if (reference.isSetStoichiometry())
//                        {
//                            return reference.getStoichiometry();
//                        }
//                        else return 1;
//                    }
//                }
//            }
//
//
//            throw Exception("Invalid string name. The id '" + sId + "' does not exist in the model");
//        }
//
//        bool NOMSupport::hasInitialAmount(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//                return oSpecies.isSetInitialAmount();
//
//            throw Exception("Invalid string name. The name is not a valid id/name of a floating / boundary species.");
//
//        }
//
//        bool NOMSupport::hasInitialConcentration(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//                return oSpecies.isSetInitialConcentration();
//
//            throw Exception("Invalid string name. The name is not a valid id/name of a floating / boundary species.");
//        }
//
//        bool NOMSupport::hasSBOTerm(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return oSpecies.isSetSBOTerm();
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter.isSetSBOTerm();
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment.isSetSBOTerm();
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                return oReaction.isSetSBOTerm();
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                return oRule.isSetSBOTerm();
//            }
//
//            throw Exception("Invalid id. No element with the given id exists in the model.");
//
//        }
//
//        bool NOMSupport::hasValue(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return (oSpecies.isSetInitialAmount() || oSpecies.isSetInitialAmount());
//
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment.isSetVolume();
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter.isSetValue();
//            }
//
//            throw Exception("Invalid string name. The id '" + sId + "' does not exist in the model");
//        }
//
//        bool NOMSupport::isConstantImpl(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//                return oSpecies.getConstant();
//
//            throw Exception("Invalid string name. The name is not a valid id/name of a floating / boundary species.");
//        }
//
//        bool NOMSupport::isReactionReversible(int nIndex)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            ArrayList productList = new ArrayList();
//
//            if (nIndex >= (int)mModel->getNumReactions())
//            {
//                throw Exception("There is no reaction corresponding to the index you provided");
//            }
//
//            libsbmlcs.Reaction r = mModel->getReaction((int)nIndex);
//            return r.getReversible();
//        }

void NOMSupport::GetSymbols(ASTNode* aNode, StringList& list)
{
    if(!aNode)
    {
        return;
    }
    ASTNode& node = *aNode;

    if (node.isName())
    {
        string name = node.getName();
        if (!list.Contains(name))
        {
            list.Add(name);
        }
    }

    for (int i = 0; i < node.getNumChildren(); i++)
    {
        GetSymbols(node.getChild(i), list);
    }

}

/// <summary>
/// Returns the list of all names contained in the ASTNode
/// </summary>
/// <param name="math">ASTnode</param>
/// <returns>List of all symbols</returns>
StringList NOMSupport::GetSymbols(ASTNode* math)
{
    StringList result; //= new List<string>();
    if (math == NULL)
    {
        return result;
    }

    GetSymbols(math, result);
    return result;
}

/// <summary>
/// Reorders assignment rules. In SBML assignment rules does not have to appear in the correct order.
/// That is you could have an assignment rule A = B, and a rule B = C. Now the result would differ,
/// if the first rule is evaluated before the second. Thus the rules will be reordered such that
/// this will be taken care of.
/// </summary>
/// <param name="assignmentRules">assignment rules in original ordering</param>
/// <returns>assignment rules in independent order</returns>
deque<Rule> NOMSupport::ReorderAssignmentRules(deque<Rule>& assignmentRules)
{
    if (assignmentRules.size() < 2)
    {
        return assignmentRules;
    }

    //Todo: Need XML file to test this:
////            var result = new List<Rule>();
    deque<Rule> result;

//    var allSymbols = new Dictionary<int, List<string>>();
    map<int, StringList > allSymbols;

    //    var map = new Dictionary<string, List<string>>();
    map<string, StringList > map;
//    var idList = new List<string>();
    StringList idList;

    // read id list, initialize all symbols
    for (int index = 0; index < assignmentRules.size(); index++)
    {
        Rule aRule = assignmentRules[index];
        AssignmentRule *rule = (AssignmentRule*) &aRule;
        string variable = rule->getVariable();
        if (!rule->isSetMath())
        {
            allSymbols[index] = StringList();
        }
        else
        {
            allSymbols[index] = GetSymbols((ASTNode*) rule->getMath());
        }
        idList.Add(variable);
        map[variable] = StringList();//new List<string>();
    }

    // initialize order array
    vector<int> order;
    order.resize(assignmentRules.size()); // = new int[assignmentRules.size()];
    for (int i = 0; i < assignmentRules.size(); i++)
    {
        order[i] = i;
    }

    // build dependency graph
    vector<string>::iterator    id;
//    foreach (var id in idList)
    for(id = idList.begin(); id != idList.end(); id++)
    {
        for (int index = 0; index < assignmentRules.size(); index++)
        {
            if (allSymbols[index].Contains( (*id) ))
            {
                map[(assignmentRules[index]).getVariable()].Add( (*id) );
            }
        }
    }

    // print dependency graph
    //foreach (var id in idList)
    //{
    //    System.Diagnostics.Debug.Write(id + " depends on: ");
    //    foreach (var symbol in map[id])
    //    {
    //        System.Diagnostics.Debug.Write(symbol + ", ");
    //    }
    //    System.Diagnostics.Debug.WriteLine("");
    //}


    // sort
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int i = 0; i < order.size(); i++)
        {

            int first = order[i];
            for (int j = i + 1; j < order.size(); j++)
            {
                int second = order[j];

                string secondVar = assignmentRules[second].getVariable();
                string firstVar = assignmentRules[first].getVariable();

                if (map[firstVar].Contains(secondVar))
                {
                    // found dependency, swap and start over
                    order[i] = second;
                    order[j] = first;

                    changed = true;
                    break;
                }
            }

            // if swapped start over
            if (changed)
                break;
        }
    }

    // create new order
    for (int i = 0; i < order.size(); i++)
    {
        result.push_back(assignmentRules[order[i]]);
    }

    return result;
}

/// <summary>
/// Reorders the Rules of the model in such a way, that AssignmentRules are calculated first, followed by Rate Rules and Algebraic Rules.
/// </summary>
/// <param name="doc">the document to use</param>
/// <param name="model">the model to use</param>
void NOMSupport::ReorderRules(SBMLDocument& doc, Model& model)
{
    int numRules = (int) model.getNumRules();
    deque<Rule> assignmentRules;
    deque<Rule> rateRules;
    deque<Rule> algebraicRules;

    for (int i = numRules - 1; i >= 0; i--)
    {
        Rule* current = model.removeRule(i);   //Todo: The rule is removed here. Is this for a copy? of the model??
        switch (current->getTypeCode())
        {
            case SBML_ALGEBRAIC_RULE:
                algebraicRules.push_front(*current);
                break;
            case SBML_RATE_RULE:
                rateRules.push_front(*current);
                break;
            default:
            case SBML_ASSIGNMENT_RULE:
                assignmentRules.push_front(*current);
                break;
        }
    }

    //TODO: Need to load suitable XML file to test and convert following code..
    assignmentRules = ReorderAssignmentRules(assignmentRules);

    //Add rules back to the model..
    //    assignmentRules.ForEach(item => model.addRule(item));
    for(int i = 0; i < assignmentRules.size(); i++)
    {
        model.addRule( new Rule(assignmentRules[i]));
    }

    //    rateRules.ForEach(item => model.addRule(item));
    for(int i = 0; i < rateRules.size(); i++)
    {
        model.addRule(new Rule(rateRules[i]));
    }

    //    algebraicRules.ForEach(item => model.addRule(item));
    for(int i = 0; i < algebraicRules.size(); i++)
    {
        model.addRule(new Rule(algebraicRules[i]));
    }

}

void NOMSupport::loadSBML(const string& var0, const string& sTimeSymbol)
{
    loadSBML(var0);
    if(!mModel)
    {
        Log(lError)<<"No model is allocated in function "<<__FUNC__<<" file "<<__FILE__;
        return;
    }

    Model &aModel = *mModel;

    changeTimeSymbol(aModel, sTimeSymbol);
    changeSymbol(aModel, "avogadro", AST_NAME_AVOGADRO);
    SBMLDocument &sbmlDoc = *mSBMLDoc;

    modifyKineticLaws(sbmlDoc, aModel);
    ReorderRules(sbmlDoc, aModel);
    BuildSymbolTable();
}

void NOMSupport::changeTimeSymbol(Model& model, const string& timeSymbol)
{
    changeSymbol(model, timeSymbol, AST_NAME_TIME);
}

//        void NOMSupport::loadParameterPromotedSBML(string var0, string sTimeSymbol)
//        {
//            loadSBML(var0);
//            changeTimeSymbol(mModel, sTimeSymbol);
//            changeSymbol(mModel, "avogadro", libsbml::AST_NAME_AVOGADRO);
//            modifyKineticLaws(mSBMLDoc, mModel);
//
//            BuildSymbolTable();
//
//        }
//
//        void loadFromFile(string fileName)
//        {
//            loadSBML(File.ReadAllText(fileName));
//        }
//
//        static Hashtable _symbolTable = new Hashtable();
//
void NOMSupport::BuildSymbolTable()
{
    // Read CompartmentSymbols
    Log(lDebug5)<<"Building Symbol Table";
    for (int i = 0; i < mModel->getNumCompartments(); i++)
    {
        Compartment *temp = mModel->getCompartment(i);

        Log(lDebug1)<<"Processing compartment with ID: "<<temp->getId();
        SBMLSymbol symbol;
        symbol.mId = temp->getId();

        if (temp->isSetSize())
        {
            symbol.mValue = temp->getSize();
        }

        symbol.mInitialAssignment = GetInitialAssignmentFor(symbol.mId);
        symbol.mRule = GetRuleFor(symbol.mId);
        symbol.mType = stCompartment;

        mSymbolTable[symbol.mId] = symbol;
    }

    // Read Parameter Symbols
    for (int i = 0; i < mModel->getNumParameters(); i++)
    {
        Parameter *temp = mModel->getParameter(i);
        Log(lDebug1)<<"Processing parameter with ID:"<<temp->getId();
        SBMLSymbol symbol;
        symbol.mId = temp->getId();
        if (temp->isSetValue())
        {
            symbol.mValue = temp->getValue();
        }
        symbol.mInitialAssignment = GetInitialAssignmentFor(symbol.mId);
        symbol.mRule = GetRuleFor(symbol.mId);
        symbol.mType = stParameter;

        mSymbolTable[symbol.mId] = symbol;
    }

    // Read Species Symbols
    for (int i = 0; i < mModel->getNumSpecies(); i++)
    {
        Species *temp = mModel->getSpecies(i);
        Log(lDebug1)<<"Processing species with ID: "<<temp->getId();
        SBMLSymbol symbol;
        symbol.mId = temp->getId();
        if (temp->isSetInitialConcentration())
        {
            symbol.mConcentration = temp->getInitialConcentration();
        }

        if (temp->isSetInitialAmount())
        {
            symbol.mAmount = temp->getInitialAmount();
        }

        symbol.mInitialAssignment = GetInitialAssignmentFor(symbol.mId);
        symbol.mRule = GetRuleFor(symbol.mId);
        symbol.mType = stSpecies;

        mSymbolTable[symbol.mId] = symbol;
    }

    StringSymbolHashTable::iterator iter;
    Log(lDebug4)<<"========== Symbols read into Symbol Table ("<<mSymbolTable.size()<<") ==============";
    for (iter = mSymbolTable.begin(); iter != mSymbolTable.end(); iter++)//string sbmlId in mSymbolTable.Keys)
    {
        SBMLSymbol& aSymbol = (iter->second);
        Log(lDebug3)<<"Key = "<<iter->first<<endl<<aSymbol;
    }

    LookForDependencies();
}

void NOMSupport::LookForDependencies()
{
    Log(lDebug5)<<"In function "<<__FUNCTION__;

    // Go through each found Id, and test for dependencies
    StringSymbolHashTable::iterator iter;

    for (iter = mSymbolTable.begin(); iter != mSymbolTable.end(); iter++)//string sbmlId in mSymbolTable.Keys)
    {
        string sbmlId = (*iter).first;
        UpdateDependencies(sbmlId);
    }
}

void NOMSupport::UpdateDependencies(const string& sbmlId)
{
    SBMLSymbol& current = mSymbolTable[sbmlId];
    if (!current.mId.size())
    {
        return;
    }

    if (current.HasInitialAssignment())
    {
        StringList dependentSymbols = GetSymbols(current.mInitialAssignment);
        for(int i = 0; i < dependentSymbols.Count(); i++)
        {
            string dependency = dependentSymbols[i];
            if(dependency != current.mId)
            {
                SBMLSymbol *sym = &(mSymbolTable[dependency]); 
                current.AddDependency(sym);
            }
        }

//        foreach (string dependency in dependentSymbols)
//            if (dependency != current.Id)
//                current.Dependencies.Add((SBMLSymbol)mSymbolTable[dependency]);
    }

    if (current.HasRule())
    {
        StringList dependentSymbols = GetSymbols(current.mRule);
        for(int i = 0; i < dependentSymbols.Count(); i++)
        {
            string dependency = dependentSymbols[i];
            if(dependency != current.mId)
            {
                SBMLSymbol *sym = &(mSymbolTable[dependency]); 
                current.AddDependency(sym);
            }
        }
//        foreach (string dependency in dependentSymbols)
//            if (dependency != current.Id)
//                current.Dependencies.Add((SBMLSymbol)mSymbolTable[dependency]);
    }
}

StringList NOMSupport::GetSymbols(const string& formula)
{
    StringList sResult;
    if (IsNullOrEmpty(formula))
    {
        return sResult;
    }

    ASTNode *node = SBML_parseFormula(formula.c_str());

    addDependenciesToList(node, sResult);
    return sResult;
}

void NOMSupport::addDependenciesToList(const ASTNode *node, StringList& sResult)
{
    for (int i = 0; i < node->getNumChildren(); i++)
    {
        addDependenciesToList(node->getChild(i), sResult);
    }

    if (node->isName() && mSymbolTable.ContainsKey(node->getName()))
    {
        sResult.Add(node->getName());
    }
}

string NOMSupport::GetRuleFor(const string& sbmlId)
{
    for (int i = 0; i < mModel->getNumRules(); i++)
    {
        Rule* oRule = mModel->getRule(i);
        switch (oRule->getTypeCode())
        {
            case SBML_PARAMETER_RULE:
            case SBML_SPECIES_CONCENTRATION_RULE:
            case SBML_COMPARTMENT_VOLUME_RULE:
            case SBML_ASSIGNMENT_RULE:
                //case libsbml::SBML_RATE_RULE:
                {
                    if (sbmlId == oRule->getVariable())
                        return oRule->getFormula();
                }
            //case libsbml::SBML_ALGEBRAIC_RULE:
            //    {
            //        string rValue = oRule->getFormula();
            //        return rValue + " = 0";
            //    }

            default:
                break;
        }
    }

    return string("");
}

string NOMSupport::GetInitialAssignmentFor(const string& sbmlId)
{
    if(mModel)
    {
        for (int i = 0; i < mModel->getNumInitialAssignments(); i++)
        {
            InitialAssignment *oAssignment = mModel->getInitialAssignment(i);
            if (oAssignment->getSymbol() == sbmlId && oAssignment->isSetMath())
            {
                return SBML_formulaToString(oAssignment->getMath());
            }
        }
    }
    return string("");
}

//        List<string> mNamespaces;
//
//        List<string> Namespaces
//        {
//            get { return mNamespaces; }
//            set
//            {
//                mNamespaces = value;
//            }
//        }
//
void NOMSupport::loadSBML(const string& var0)
{
    //byte[] oBuffer = ASCIIEncoding.ASCII.GetBytes(var0.ToCharArray());      //Todo: Check what we can do in C++
    //System.IO.MemoryStream oStream = new System.IO.MemoryStream(oBuffer);
//    string sTemp = new System.IO.StreamReader(oStream).ReadToEnd();
    string sTemp = var0;
    if (mSBMLDoc != NULL)
    {
        try
        {
            if (mModel != NULL)
            {
                //delete mModel;//.Dispose();
                //mModel = NULL;
            }
            //delete mSBMLDoc;//mSBMLDoc.Dispose();
            //mSBMLDoc = NULL;
        }
        catch(...)
        {
            // never mind ....
        }

    }

    // we also need to collect all namespaces from the file, or rather
    // all registered prefixes:

    //Todo: How to deal with namespaces in SBML file???

//    //string regex=@"^.*?xmlns:(?<prefix>\w+?)^.*= "(?<namespace>.+?).*?$";
//    string regex = "xmlns:(?<prefix>\\w+?)\\s*=\\s*(?:\"(?<namespace>[^\"]*)\"|(?<namespace>\\S+))";
//    RegexOptions options = RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant;
//    string input = var0;
//
//    Namespaces = new List<string>();
//    List<string> prefixes = new List<string>();
//    MatchCollection matches = Regex.Matches(input, regex, options);
//    foreach (Match match in matches)
//    {
//        //Console.WriteLine(match.Value);
//        string prefix = match.Value.Substring(0, match.Value.IndexOf('='));
//        if (!prefixes.Contains(prefix) && !Namespaces.Contains(match.Value))
//        {
//            Namespaces.Add(match.Value);
//            prefixes.Add(prefix);
//        }
//
//        //Console.WriteLine("prefix:" + match.Groups["prefix"].Value);
//        //Console.WriteLine("namespace:" + match.Groups["namespace"].Value);
//
//    }

//    _ParameterSets = new ParameterSets(sTemp);

	//Who is to delete this document??
    mSBMLDoc = readSBMLFromString(sTemp.c_str());
    mModel = mSBMLDoc->getModel();
    if (mModel == NULL)
    {
        throw NOMException(validateSBML(sTemp));
    }
}

//        ParameterSets _ParameterSets;
//
//        ParameterSets ParameterSets
//        {
//            get { return _ParameterSets; }
//            set { _ParameterSets = value; }
//        }
//
//        void setAnnotation(string sId, string sAnnotation)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (mModel->getId() == sId || mModel->getName() == sId)
//            {
//                mModel->setAnnotation(sAnnotation);
//                return;
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                oSpecies.setAnnotation(sAnnotation);
//                return;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                oParameter.setAnnotation(sAnnotation);
//                return;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                oCompartment.setAnnotation(sAnnotation);
//                return;
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                oReaction.setAnnotation(sAnnotation);
//                return;
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                oRule.setAnnotation(sAnnotation);
//                return;
//            }
//
//            throw Exception("Invalid id. No element with the given id exists in the model.");
//        }
//
//        void NOMSupport::setModelId(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//            mModel->setId(sId);
//        }
//
//        void NOMSupport::setNotes(string sId, string sNotes)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            if (mModel->getId() == sId || mModel->getName() == sId)
//            {
//                mModel->setNotes(sNotes);
//                return;
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                oSpecies.setNotes(sNotes);
//                return;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                oParameter.setNotes(sNotes);
//                return;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                oCompartment.setNotes(sNotes);
//                return;
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                oReaction.setNotes(sNotes);
//                return;
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                oRule.setNotes(sNotes);
//                return;
//            }
//
//            throw Exception("Invalid id. No element with the given id exists in the model.");
//        }
//
//        void NOMSupport::setSBOTerm(string sId, int nSBOTerm)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                oSpecies.setSBOTerm(nSBOTerm);
//                return;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                oParameter.setSBOTerm(nSBOTerm);
//                return;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                oCompartment.setSBOTerm(nSBOTerm);
//                return;
//            }
//
//            libsbmlcs.Reaction oReaction = mModel->getReaction(sId);
//            if (oReaction != NULL)
//            {
//                oReaction.setSBOTerm(nSBOTerm);
//                return;
//            }
//
//            Rule oRule = mModel->getRule(sId);
//            if (oRule != NULL)
//            {
//                oRule.setSBOTerm(nSBOTerm);
//                return;
//            }
//
//            throw Exception("Invalid id. No element with the given id exists in the model.");
//
//        }
//
void NOMSupport::setValue(Model* model, const string& id, const double& value, const bool& throwIfNotFound)
{
    if (model == NULL)
    {
        throw Exception("You need to load the model first");
    }

    libsbml::Species* oSpecies = model->getSpecies(id);
    if (oSpecies != NULL)
    {
        if (oSpecies->isSetInitialAmount())
            oSpecies->setInitialAmount(value);
        else
            oSpecies->setInitialConcentration(value);
        return;
    }

    Compartment* oCompartment = model->getCompartment(id);
    if (oCompartment != NULL)
    {
        oCompartment->setVolume(value); return;
    }

    Parameter* oParameter = model->getParameter(id);
    if (oParameter != NULL)
    {
        oParameter->setValue(value);
        return;
    }

    for (int i = 0; i < mModel->getNumReactions(); i++)
    {
        Reaction* reaction = mModel->getReaction(i);
        for (int j = 0; j < reaction->getNumReactants(); j++)
        {
            SpeciesReference* reference = reaction->getReactant(j);
            if (reference->isSetId() && reference->getId() == id)
            {
                reference->setStoichiometry(value);
                return;
            }
        }
        for (int j = 0; j < reaction->getNumProducts(); j++)
        {
            SpeciesReference* reference = reaction->getProduct(j);
            if (reference->isSetId() && reference->getId() == id)
            {
                reference->setStoichiometry(value);
                return;
            }
        }
    }

    if (throwIfNotFound)
    {
        throw Exception(Format("Invalid string name. The id '{0}' does not exist in the model", id));
    }
}

void NOMSupport::setValue(const string& sId, const double& dValue)
{
    setValue(mModel, sId, dValue, true);
}

string NOMSupport::validateSBML(const string& sModel)
{
    SBMLDocument *oDoc = readSBMLFromString(sModel.c_str());
    if (oDoc->getNumErrors() > 0)
    {
        StringBuilder oBuilder;// = new StringBuilder();
        for (int i = 0; i < oDoc->getNumErrors(); i++)
        {
            ArrayList oList = getNthError(i);

            //oBuilder.Append(oList[0] + ": (" + oList[1] + ":" + oList[2] + "[" + oList[3] + "]) " + oList[4] + Environment.NewLine);
        }
        throw Exception("SBML Validation failed: " + oBuilder.ToString());
    }
    return "Validation Successfull";
}

//        string NOMSupport::validateWithConsistency(string sModel)
//        {
//            SBMLDocument oDoc = libsbml::readSBMLFromString(sModel);
//            if (oDoc.getNumErrors() + oDoc.checkConsistency() > 0)
//            {
//                StringBuilder oBuilder = new StringBuilder();
//                for (int i = 0; i < oDoc.getNumErrors(); i++)
//                {
//                    ArrayList oList = getNthError(i);
//                    oBuilder.Append(oList[0] + ": (" + oList[1] + ":" + oList[2] + "[" + oList[3] + "]) " + oList[4] + Environment.NewLine);
//                }
//                throw Exception("Validation failed: " + Environment.NewLine + oBuilder.ToString());
//            }
//            return "Validation Successfull";
//        }
//
//        #region Pre Defined Functions
//
//        static string[][] _oPredefinedFunctions = new string[][]
//
//    {
//        new string[]{ "massi1", "Irreversible Mass Action Kinetics for 1 substrate", "S", "k", "k * S" },
//        new string[]{ "massi2", "Irreversible Mass Action Kinetics for 2 substrates", "S1", "S2", "k", "k * S1 * S2" },
//        new string[]{ "massi3", "Irreversible Mass Action Kinetics for 3 substrates", "S1", "S2", "S3", "k", "k * S1 * S2 * S3" },
//        new string[]{
//                        "massr11", "Reversible Mass Action Kinetics for 1 substrate and 1 product", "S", "P", "k_1", "k_2",
//                        "k_1 * S - k_2 * P"},
//        new string[]{
//                        "massr12", "Reversible Mass Action Kinetics for 1 substrate and 2 products", "S", "P1", "P2", "k_1", "k_2",
//                        "k_1 * S - k_2 * P1 * P2"},
//        new string[]{
//                        "massr13", "Reversible Mass Action Kinetics for 1 substrate and 3 products", "S", "P1", "P2", "P3", "k_1", "k_2",
//                        "k_1 * S - k_2 * P1 * P2 * P3"},
//        new string[]{
//                        "massr21", "Reversible Mass Action Kinetics for 2 substrates and 1 product", "S1", "S2", "P", "k_1", "k_2",
//                        "k_1 * S1 * S2 - k_2 * P"},
//        new string[]{
//                        "massr22", "Reversible Mass Action Kinetics for 2 substrates and 2 products", "S1", "S2", "P1", "P2", "k_1", "k_2",
//                        "k_1 * S1 * S2 - k_2 * P1 * P2"},
//        new string[]{
//                        "massr23", "Reversible Mass Action Kinetics for 2 substrates and 3 products", "S1", "S2", "P1", "P2", "P3", "k_1", "k_2",
//                        "k_1 * S1 * S2 - k_2 * P1 * P2 * P3"},
//        new string[]{
//                        "massr31", "Reversible Mass Action Kinetics for 3 substrates and 1 product", "S1", "S2", "S3", "P", "k_1", "k_2",
//                        "k_1 * S1 * S2 * S3 - k_2 * P"},
//        new string[]{
//                        "massr32", "Reversible Mass Action Kinetics for 3 substrates and 2 products", "S1", "S2", "S3", "P1", "P2", "k_1", "k_2",
//                        "k_1 * S1 * S2 * S3 - k_2 * P1 * P2"},
//        new string[]{
//                        "massr33", "Reversible Mass Action Kinetics for 3 substrates and 3 products", "S1", "S2", "S3", "P1", "P2", "P3", "k_1", "k_2",
//                        "k_1 * S1 * S2 * S3 - k_2 * P1 * P2 * P3"},
//        new string[]{ "uui", "Irreversible Simple Michaelis-Menten ", "S", "V_m", "K_m", "(V_m * S)/(K_m + S)" },
//        new string[]{
//                        "uur", "Uni-Uni Reversible Simple Michaelis-Menten", "S", "P", "V_f", "V_r", "K_ms", "K_mp",
//                        "(V_f * S / K_ms - V_r * P / K_mp)/(1 + S / K_ms +  P / K_mp)"},
//        new string[]{
//                        "uuhr", "Uni-Uni Reversible Simple Michaelis-Menten with Haldane adjustment", "S", "P", "V_f", "K_m1", "K_m2", "K_eq",
//                        "( V_f / K_m1 * (S - P / K_eq ))/(1 + S / K_m1 + P / K_m2)"},
//        new string[]{
//                        "isouur", "Iso Uni-Uni", "S", "P", "V_f", "K_ms", "K_mp", "K_ii", "K_eq",
//                        "(V_f * (S - P / K_eq ))/(S * (1 + P / K_ii ) + K_ms * (1 + P / K_mp))"},
//        new string[]{ "hilli", "Hill Kinetics", "S", "V", "S_0_5", "h", "(V * pow(S,h))/(pow(S_0_5,h) + pow(S,h))"},
//        new string[]{
//                        "hillr", "Reversible Hill Kinetics", "S", "P", "V_f", "S_0_5", "P_0_5", "h", "K_eq",
//                        "(V_f * (S / S_0_5) * (1 - P / (S * K_eq) ) * pow(S / S_0_5 + P / P_0_5, h-1))/(1 + pow(S / S_0_5 + P / P_0_5, h))"},
//        new string[]{
//                        "hillmr", "Reversible Hill Kinetics with One Modifier", "S", "M", "P", "V_f", "K_eq", "k", "h", "alpha",
//                        "(V_f * (S / S_0_5) * (1 - P / (S * K_eq) ) * pow(S / S_0_5 + P / P_0_5, h-1))/( pow(S / S_0_5 + P / P_0_5, h) + (1 + pow(M / M_0_5, h))/(1 + alpha * pow(M/M_0_5,h)))"},
//        new string[]{
//                        "hillmmr", "Reversible Hill Kinetics with Two Modifiers", "S", "P", "M", "V_f", "K_eq", "k", "h", "a", "b", "alpha_1", "alpha_2", "alpha_12",
//                        "(V_f * (S / S_0_5) * (1 - P / (S * K_eq) ) * pow(S / S_0_5 + P / P_0_5, h-1)) / (pow(S / S_0_5 + P / P_0_5, h) + ((1 + pow(Ma/Ma_0_5,h) + pow(Mb/Mb_0_5,h))/( 1 + alpha_1 * pow(Ma/Ma_0_5,h) + alpha_2 * pow(Mb/Mb_0_5,h) + alpha_1 * alpha_2 * alpha_12 * pow(Ma/Ma_0_5,h) * pow(Mb/Mb_0_5,h))))"},
//        new string[]{ "usii", "Substrate Inhibition Kinetics (Irreversible)", "S", "V", "K_m", "K_i", "V*(S/K_m)/(1 + S/K_m + sqr(S)/K_i)"},
//        new string[]{
//                        "usir", "Substrate Inhibition Kinetics (Reversible)", "S", "P", "V_f", "V_r", "K_ms", "K_mp", "K_i",
//                        "(V_f*S/K_ms + V_r*P/K_mp)/(1 + S/K_ms + P/K_mp + sqr(S)/K_i)"},
//        new string[]{ "usai", "Substrate Activation", "S", "V", "K_sa", "K_sc", "V * sqr(S/K_sa)/(1 + S/K_sc + sqr(S/K_sa) + S/K_sa)"},
//        new string[]{ "ucii", "Competitive Inhibition (Irreversible)", "S", "V", "K_m", "K_i", "(V * S/K_m)/(1 + S/K_m + I/K_i)"},
//        new string[]{
//                        "ucir", "Competitive Inhibition (Reversible)", "S", "P", "V_f", "V_r", "K_ms", "K_mp", "K_i",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + S/K_ms + P/K_mp + I/K_i)"},
//        new string[]{ "unii", "Noncompetitive Inhibition (Irreversible)", "S", "I", "V", "K_m", "K_i", "(V*S/K_m)/(1 + I/K_i + (S/K_m)*(1 + I/K_i))"},
//        new string[]{
//                        "unir", "Noncompetitive Inhibition (Reversible)", "S", "P", "I", "V_f", "K_ms", "K_mp", "K_i",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + I/K_i + (S/K_ms + P/K_mp )*(1 + I/K_i))"},
//        new string[]{ "uuci", "Uncompetitive Inhibition (Irreversible)", "S", "I", "V", "K_m", "K_i", "(V*S/K_m)/(1 + (S/K_m)*(1 + I/K_i))"},
//        new string[]{
//                        "uucr", "Uncompetitive Inhibition (Reversible)", "S", "P", "I", "V_f", "V_r", "K_ms", "K_mp", "K_i",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + ( S/K_ms + P/K_mp )*( 1 + I/K_i))"},
//        new string[]{
//                        "umi", "Mixed Inhibition Kinetics (Irreversible)", "S", "I", "V", "K_m", "K_is", "K_ic",
//                        "(V*S/K_m)/(1 + I/K_is + (S/K_m)*(1 + I/K_ic))"},
//        new string[]{
//                        "umr", "Mixed Inhibition Kinetics (Reversible)", "S", "P", "I", "V_f", "V_r", "K_ms", "K_mp", "K_is", "K_ic",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + I/K_is + ( S/K_ms + P/K_mp )*( 1 + I/K_ic ))"},
//        new string[]{ "uai", "Specific Activation Kinetics - irreversible", "S", "A_c", "V", "K_m", "K_a", "(V*S/K_m)/(1 + S/K_m + K_a/A_c)"},
//        new string[]{
//                        "uar", "Specific Activation Kinetics (Reversible)", "S", "P", "A_c", "V_f", "V_r", "K_ms", "K_mp", "K_a",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + S/K_ms + P/K_mp + K_a/A_c)"},
//        new string[]{ "ucti", "Catalytic Activation (Irreversible)", "S", "A_c", "V", "K_m", "K_a", "(V*S/K_m)/(1 + K_a/A_c + (S/K_m)*(1 + K_a/A_c))"},
//        new string[]{
//                        "uctr", "Catalytic Activation (Reversible)", "S", "P", "A_c", "V_f", "V_r", "K_ms", "K_mp", "K_a",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + K_a/A_c + (S/K_ms + P/K_mp)*(1 + K_a/A_c))"},
//        new string[]{
//                        "umai", "Mixed Activation Kinetics (Irreversible)", "S", "A_c", "V", "K_m", "Kas", "Kac",
//                        "(V*S/K_m)/(1 + Kas/A_c + (S/K_m)*(1 + Kac/A_c))"},
//        new string[]{
//                        "umar", "Mixed Activation Kinetics (Reversible)", "S", "P", "A_c", "V_f", "V_r", "K_ms", "K_mp", "K_as", "K_ac",
//                        "(V_f*S/K_ms - V_r*P/K_mp)/(1 + K_as/A_c + (S/K_ms + P/K_mp)*(1 + K_ac/A_c))"},
//        new string[]{
//                        "uhmi", "General Hyperbolic Modifier Kinetics (Irreversible)", "S", "M", "V", "K_m", "K_d", "a", "b",
//                        "(V*(S/K_m)*(1 + b * M / (a*K_d)))/(1 + M/K_d + (S/K_m)*(1 + M/(a*K_d)))"},
//        new string[]{
//                        "uhmr", "General Hyperbolic Modifier Kinetics (Reversible)", "S", "P", "M", "V_f", "V_r", "K_ms", "K_mp", "K_d", "a", "b",
//                        "((V_f*S/K_ms - V_r*P/K_mp)*(1 + b*M/(a*K_d)))/(1 + M/K_d + (S/K_ms + P/K_mp)*(1 + M/(a*K_d)))"},
//        new string[]{
//                        "ualii", "Allosteric inhibition (Irreversible)", "S", "I", "V", "K_s", "K_ii", "n", "L",
//                        "(V*pow(1 + S/K_s, n-1))/(L*pow(1 + I/K_ii,n) + pow(1 + S/K_s,n))"},
//        new string[]{
//                        "ordubr", "Ordered Uni Bi Kinetics", "A", "P", "Q", "V_f", "V_r", "K_ma", "K_mq", "K_mp", "K_ip", "K_eq",
//                        "(V_f*( A - P*Q/K_eq))/(K_ma + A*(1 + P/K_ip) + (V_f/(V_r*K_eq))*(K_mq*P + K_mp*Q + P*Q))"},
//        new string[]{
//                        "ordbur", "Ordered Bi Uni Kinetics", "A", "B", "P", "V_f", "V_r", "K_ma", "Kmb", "K_mp", "K_ia", "K_eq",
//                        "(V_f*(A*B - P/K_eq))/(A*B + K_ma*B + Kmb*A + (V_f/(V_r*K_eq))*(K_mp + P*(1 + A/K_ia)))"},
//        new string[]{
//                        "ordbbr", "Ordered Bi Bi Kinetics", "A", "B", "P", "Q", "V_f", "K_ma", "K_mb", "K_mp", "K_ia", "K_ib", "K_ip", "K_eq",
//                        "(V_f*(A*B - P*Q/K_eq))/(A*B*(1 + P/K_ip) + K_mb*(A + K_ia) + K_ma*B + ((V_f / (V_r*K_eq)) * (K_mq*P*( 1 + A/K_ia) + Q*(K_mp*( 1 + (K_ma*B)/(K_ia*K_mb) + P*(1 + B/K_ib))))))"},
//        new string[]{
//                        "ppbr", "Ping Pong Bi Bi Kinetics", "A", "B", "P", "Q", "V_f", "V_r", "K_ma", "K_mb", "K_mp", "K_mq", "K_ia", "K_iq", "K_eq",
//                        "(V_f*(A*B - P*Q/K_eq))/(A*B + K_mb*A + K_ma*B*(1 + Q/K_iq) + ((V_f/(V_r*K_eq))*(K_mq*P*(1 + A/K_ia) + Q*(K_mp + P))))"}};
//
//
//
//        SBMLDocument SbmlDocument
//        {
//            get
//            {
//                return mSBMLDoc;
//            }
//        }
//
//
//        Model SbmlModel
//        {
//            get
//            {
//                return mModel;
//            }
//        }
//        #endregion
//
/// <summary>
/// Checks whether the element for the given SBML id is a compartment
/// </summary>
/// <param name="sId">the id to check</param>
/// <returns>true if element is a compartment, false otherwise</returns>
bool NOMSupport::IsCompartment(const string& sId)
{
    if(!mModel)
    {
        //Todo: should this throw.?
        return false;
    }

    Compartment* temp = mModel->getCompartment(sId);
    if (temp != NULL)
    {
        return true;
    }
    return false;
}

//        /// <summary>
//        /// Checks whether the element for the given SBML id is a species
//        /// </summary>
//        /// <param name="sId">the id to check</param>
//        /// <returns>true if element is a species, false otherwise</returns>
//        bool NOMSupport::IsSpecies (string sId)
//        {
//            var temp = mModel->getSpecies(sId);
//            if (temp != NULL) return true;
//            return false;
//        }
//
//        /// <summary>
//        /// Checks whether the element for the given SBML id is a floating species
//        /// </summary>
//        /// <param name="sId">the id to check</param>
//        /// <returns>true if element is a floating species, false otherwise</returns>
//        bool NOMSupport::IsFloating(string sId)
//        {
//            var temp = mModel->getSpecies(sId);
//            if (temp != NULL && temp.getBoundaryCondition() == false) return true;
//            return false;
//        }
//
//        /// <summary>
//        /// Returns the element for the given sId, or NULL if not present
//        /// </summary>
//        /// <param name="sId">the sbml id for the element to find</param>
//        /// <returns>the element with the given sbml id</returns>
//        SBase NOMSupport::GetElement(string sId)
//        {
//            if (mModel == NULL)
//            {
//                throw Exception("You need to load the model first");
//            }
//
//            libsbmlcs.Species oSpecies = mModel->getSpecies(sId);
//            if (oSpecies != NULL)
//            {
//                return oSpecies;
//            }
//
//            Compartment oCompartment = mModel->getCompartment(sId);
//            if (oCompartment != NULL)
//            {
//                return oCompartment;
//            }
//
//            Parameter oParameter = mModel->getParameter(sId);
//            if (oParameter != NULL)
//            {
//                return oParameter;
//            }
//
//            for (int i = 0; i < mModel->getNumReactions(); i++)
//            {
//                var reaction = mModel->getReaction(i);
//
//                if (reaction.isSetId() && reaction.getId() == sId)
//                    return reaction;
//
//                for (int j = 0; j < reaction.getNumReactants(); j++)
//                {
//                    var reference = reaction.getReactant(j);
//                    if (reference.isSetId() && reference.getId() == sId)
//                    {
//                        return reference;
//                    }
//                }
//                for (int j = 0; j < reaction.getNumProducts(); j++)
//                {
//                    var reference = reaction.getProduct(j);
//                    if (reference.isSetId() && reference.getId() == sId)
//                    {
//                        return reference;
//                    }
//                }
//            }
//
//
//
//
//            throw Exception("Invalid string name. The id '" + sId + "' does not exist in the model");
//        }
//
//        /// <summary>
//        /// Checks whether the element for the given SBML id is a boundary species
//        /// </summary>
//        /// <param name="sId">the id to check</param>
//        /// <returns>true if element is a boundary species, false otherwise</returns>
//        bool NOMSupport::IsBoundary(string sId)
//        {
//            var temp = mModel->getSpecies(sId);
//            if (temp != NULL && temp.getBoundaryCondition() == true) return true;
//            return false;
//        }
//
bool NOMSupport::MultiplyCompartment(const string& sbmlId, string& compartmentId)
{
    compartmentId = "";

    if(!mModel)
    {
        return false;
    }

    Species* temp = mModel->getSpecies(sbmlId);
    if (temp != NULL &&
        //temp.isSetInitialAmount() &&
        temp->isSetCompartment() &&
        !temp->getHasOnlySubstanceUnits())
    {

        compartmentId = temp->getCompartment();

        Compartment* comp = mModel->getCompartment(compartmentId);
        if (comp == NULL || comp->getSpatialDimensions() == 0)
        {
            return false;
        }

        return true;
    }
    return false;
}

/// <summary>
/// This should return an initialization for the given sbmlId that is sideeffect free
/// </summary>
/// <param name="sbmlId"></param>
/// <returns></returns>
stack<string> NOMSupport::GetMatchForSymbol(const string& sbmlId)
{
    stack<string> result;

    //SBMLSymbol *symbol = &(mSymbolTable[sbmlId]);

    FillStack(result, mSymbolTable[sbmlId]);
    return result;
}

void NOMSupport::FillStack(stack<string>& stack, SBMLSymbol& symbol)
{
    Log(lDebug5)<<"In "<<__FUNCTION__<<" Filling stack with symbol: "<<(symbol);
    if (!symbol.mId.size())
    {
        return;
    }

    if (symbol.HasRule())
    {
        stack.push(symbol.mId + " = " + symbol.mRule);
    }
    if (symbol.HasInitialAssignment())
    {
        stack.push(symbol.mId + " = " + symbol.mInitialAssignment);
    }
    if (symbol.HasValue())
    {
        stack.push(symbol.mId + " = " + ToString(symbol.mValue, STR_DoubleFormat));
    }

    for(int i = 0; i < symbol.NumberOfDependencies(); i++)
    {
        SBMLSymbol dependency = symbol.GetDependency(i);
        FillStack(stack, dependency); //hmm recursive.. Todo: ...?
    }
}

//        string NOMSupport::addSourceSinkNodes(string sbml)
//        {
//            SBMLDocument doc = libsbml::readSBMLFromString(sbml);
//
//            UpgradeToL2V4IfNecessary(doc);
//
//            Model model = doc.getModel();
//
//            if (NeedSourceNode(model))
//            {
//                Species source = model.getSpecies("source");
//                if (source == NULL)
//                {
//                    source = model.createSpecies();
//                    source.setId("source");
//                    source.setName(" ");
//                    //source.setName("Source");
//                    source.setSBOTerm(291);
//                    source.setCompartment(model.getCompartment(0).getId());
//                    source.setBoundaryCondition(true);
//                    source.setInitialAmount(0);
//                }
//
//                for (int i = 0; i < model.getNumReactions(); i++)
//                {
//                    libsbmlcs.Reaction r = model.getReaction(i);
//                    if (r.getNumReactants() == 0)
//                    {
//                        libsbmlcs.SpeciesReference reference = r.createReactant();
//                        reference.setSpecies(source.getId());
//                    }
//                }
//
//            }
//            if (NeedSinkNode(model))
//            {
//                Species sink = model.getSpecies("sink");
//                if (sink == NULL)
//                {
//                    sink = model.createSpecies();
//                    sink.setId("sink");
//                    //sink.setName("Sink");
//                    sink.setName(" ");
//                    sink.setSBOTerm(291);
//                    sink.setCompartment(model.getCompartment(0).getId());
//                    sink.setBoundaryCondition(true);
//                    sink.setInitialAmount(0);
//                }
//
//                for (int i = 0; i < model.getNumReactions(); i++)
//                {
//                    libsbmlcs.Reaction r = model.getReaction(i);
//                    if (r.getNumProducts() == 0)
//                    {
//                        libsbmlcs.SpeciesReference reference = r.createProduct();
//                        reference.setSpecies(sink.getId());
//                    }
//                }
//
//            }
//
//            return libsbml::writeSBMLToString(doc);
//        }
//
//        bool NOMSupport::NeedSourceNode(Model model)
//        {
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                libsbmlcs.Reaction r = model.getReaction(i);
//                if (r.getNumReactants() == 0) return true;
//            }
//
//            return false;
//        }
//
//        bool NOMSupport::NeedSinkNode(Model model)
//        {
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                libsbmlcs.Reaction r = model.getReaction(i);
//                if (r.getNumProducts() == 0) return true;
//            }
//
//            return false;
//        }
//
//        bool NOMSupport::NeedEmptySetNode(Model model)
//        {
//            for (int i = 0; i < model.getNumReactions(); i++)
//            {
//                libsbmlcs.Reaction r = model.getReaction(i);
//                if (r.getNumReactants() == 0 || r.getNumProducts() == 0) return true;
//            }
//
//            return false;
//        }
//
//        string NOMSupport::addEmptySetNodes(string sbml)
//        {
//            SBMLDocument doc = libsbml::readSBMLFromString(sbml);
//
//            UpgradeToL2V4IfNecessary(doc);
//
//            Model model = doc.getModel();
//
//            int nCount = 0;
//
//            while (model.getSpecies("empty_" + nCount) != NULL)
//            {
//                nCount++;
//            }
//
//            if (model != NULL)
//            {
//                for (int i = 0; i < model.getNumReactions(); i++)
//                {
//                    libsbmlcs.Reaction r = model.getReaction(i);
//                    if (r.getNumReactants() == 0)
//                    {
//
//                        Species species = model.createSpecies();
//                        nCount++;
//                        species.setId("empty_" + nCount);
//                        //species.setName("EmptySet");
//                        species.setName(" ");
//                        species.setSBOTerm(291);
//                        species.setCompartment(model.getCompartment(0).getId());
//                        species.setBoundaryCondition(true);
//                        species.setInitialAmount(0);
//
//                        libsbmlcs.SpeciesReference reference = r.createReactant();
//                        reference.setSpecies(species.getId());
//                    }
//                    if (r.getNumProducts() == 0)
//                    {
//                        Species species = model.createSpecies();
//                        nCount++;
//                        species.setId("empty_" + nCount);
//                        //species.setName("EmptySet");
//                        species.setName(" ");
//                        species.setSBOTerm(291);
//                        species.setCompartment(model.getCompartment(0).getId());
//                        species.setBoundaryCondition(true);
//                        species.setInitialAmount(0);
//
//                        libsbmlcs.SpeciesReference reference = r.createProduct();
//                        reference.setSpecies(species.getId());
//                    }
//                }
//            }
//
//
//            return libsbml::writeSBMLToString(doc);
//        }
//
//        string NOMSupport::addEmptySetNode(string sbml)
//        {
//            SBMLDocument doc = libsbml::readSBMLFromString(sbml);
//
//            UpgradeToL2V4IfNecessary(doc);
//
//            Model model = doc.getModel();
//
//            if (NeedEmptySetNode(model))
//            {
//                Species source = model.getSpecies("emptySet");
//                if (source == NULL)
//                {
//                    source = model.createSpecies();
//                    source.setId("emptySet");
//                    //source.setName("EmptySet");
//                    source.setName(" ");
//                    source.setSBOTerm(291);
//                    source.setCompartment(model.getCompartment(0).getId());
//                    source.setBoundaryCondition(true);
//                    source.setInitialAmount(0);
//                }
//
//                for (int i = 0; i < model.getNumReactions(); i++)
//                {
//                    libsbmlcs.Reaction r = model.getReaction(i);
//                    if (r.getNumReactants() == 0)
//                    {
//                        libsbmlcs.SpeciesReference reference = r.createReactant();
//                        reference.setSpecies(source.getId());
//                    }
//                    if (r.getNumProducts() == 0)
//                    {
//                        libsbmlcs.SpeciesReference reference = r.createProduct();
//                        reference.setSpecies(source.getId());
//                    }
//                }
//
//            }
//
//
//            return libsbml::writeSBMLToString(doc);
//        }
//
//        string NOMSupport::RemoveJD2Layout(string sSBML)
//        {
//            int jdStart = sSBML.IndexOf("<jd2:JDesignerLayout");
//            string endTag = "</jd2:JDesignerLayout>";
//            int jdEnd = sSBML.IndexOf(endTag);
//
//            if (jdEnd != -1)
//                return sSBML.Substring(0, jdStart) + sSBML.Substring(jdEnd + endTag.Length);
//            return sSBML;
//        }
//
//        string NOMSupport::RemoveJD1Layout(string sSBML)
//        {
//            XmlDocument doc = new XmlDocument();
//            doc.LoadXml(sSBML);
//            var nodes = doc.DocumentElement.GetElementsByTagName("annotation");
//            foreach (var item in nodes)
//            {
//                XmlElement node = item as XmlElement;
//                for (int i = node.ChildNodes.Count - 1; i >= 0; i--)
//                {
//                    XmlNode child = node.ChildNodes[i];
//                    if (child.Prefix == "jd")
//                        node.RemoveChild(child);
//                }
//            }
//
//
//            string result;
//            using (MemoryStream stream = new MemoryStream())
//            {
//                XmlWriterSettings settingsVariable = new XmlWriterSettings();
//                settingsVariable.Indent = true;
//                XmlWriter write = XmlWriter.Create(stream, settingsVariable);
//                doc.WriteTo(write);
//                write.Close();
//                stream.Flush();
//                stream.SetLength(stream.Length);
//                stream.Close();
//
//                result = ASCIIEncoding.UTF8.GetString(stream.GetBuffer()).Trim();
//            }
//
//
//            return result.Replace("UTF-16", "utf-8");
//        }
//
//        string NOMSupport::RemoveLayoutInformation(string sSBML)
//        {
//            sSBML = RemoveJD2Layout(sSBML);
//            sSBML = RemoveJD1Layout(sSBML);
//
//            return sSBML;
//        }
//
//    }
//
//}

}//namespace rr
