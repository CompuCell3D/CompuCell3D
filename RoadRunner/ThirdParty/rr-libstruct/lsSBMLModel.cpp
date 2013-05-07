#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <string>
#include <vector>
#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/SBMLReader.h>
#include "lsSBMLModel.h"
#include "lsUtils.h"

//---------------------------------------------------------------------------
using namespace std;

namespace ls
{

SBMLmodel::SBMLmodel()
	:
	_Document(NULL),
    _Model(NULL)
{
}

SBMLmodel::SBMLmodel(string &sSBML)
	:
    _Document(NULL),
    _Model(NULL)
{
    InitializeFromSBML(sSBML);
}

SBMLmodel::~SBMLmodel(void)
{
    delete _Document;
}

SBMLmodel* SBMLmodel::FromFile(string &sFileName)
{
    SBMLmodel *oResult = new SBMLmodel();
    oResult->InitializeFromFile(sFileName);
    return oResult;
}

SBMLmodel* SBMLmodel::FromSBML(string &sSBML)
{
    SBMLmodel *oResult = new SBMLmodel();
    oResult->InitializeFromSBML(sSBML);
    return oResult;
}

void SBMLmodel::InitializeFromSBML(std::string &sSBML)
{
    SBMLReader oReader;
    _Document = oReader.readSBMLFromString(sSBML);
    _Model = _Document->getModel();
    if (_Model == NULL)
        throw new ApplicationException("Invalid SBML Model", "The SBML model was invalid. Please validate it using a SBML validator such as: http://sys-bio.org/validate.");

}
void SBMLmodel::InitializeFromFile(std::string &sFileName)
{
    SBMLReader oReader;
    _Document = oReader.readSBML(sFileName);
    _Model = _Document->getModel();
    if (_Model == NULL)
        throw new ApplicationException("Invalid SBML Model", "The SBML model was invalid. Please validate it using a SBML validator such as: http://sys-bio.org/validate.");
}

Model* SBMLmodel::getModel()
{
	return _Model;
}

int SBMLmodel::numFloatingSpecies()
{
    return (int) _Model->getNumSpecies() - _Model->getNumSpeciesWithBoundaryCondition();
}

int SBMLmodel::numReactions()
{
    return (int) _Model->getNumReactions();
}

const Species* SBMLmodel::getNthFloatingSpecies(int n)
{
    int nCount = 0;
    for (unsigned int i = 0; i < _Model->getNumSpecies(); i++)
    {
        if (!_Model->getSpecies(i)->getBoundaryCondition())
        {
            if (nCount == n)
                return _Model->getSpecies(i);
            nCount ++;
        }
    }
    return NULL;
}

const Species* SBMLmodel::getNthBoundarySpecies(int n)
{
    int nCount = 0;
    for (unsigned int i = 0; i < _Model->getNumSpecies(); i++)
    {
        if (_Model->getSpecies(i)->getBoundaryCondition())
        {
            if (nCount == n)
                return _Model->getSpecies(i);
            nCount ++;
        }
    }
    return NULL;
}

const Reaction* SBMLmodel::getNthReaction(int n)
{
    return _Model->getReaction(n);
}


}//namespace ls

