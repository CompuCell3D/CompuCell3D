/**
 * @file    Model.cpp
 * @brief   Implementation of Model.
 * @author  Ben Bornstein
 * 
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2009-2012 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/annotation/RDFAnnotation.h>
#include <sbml/math/ASTNode.h>

#include <sbml/SBMLDocument.h>
#include <sbml/SBO.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/SBMLError.h>
#include <sbml/KineticLaw.h>

#include <sbml/Model.h>

#include <sbml/ModifierSpeciesReference.h>
#include <sbml/SimpleSpeciesReference.h>

#include <sbml/AlgebraicRule.h>
#include <sbml/AssignmentRule.h>
#include <sbml/RateRule.h>


#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBasePlugin.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

Model::Model (unsigned int level, unsigned int version) :
   SBase ( level, version )
 , mId               ( "" )
 , mName             ( "" )
 , mSubstanceUnits   ( "" )
 , mTimeUnits        ( "" )
 , mVolumeUnits      ( "" )
 , mAreaUnits        ( "" )
 , mLengthUnits      ( "" )
 , mExtentUnits      ( "" )
 , mConversionFactor ( "" )
 , mFunctionDefinitions (level,version)
 , mUnitDefinitions     (level,version)
 , mCompartmentTypes    (level,version)
 , mSpeciesTypes        (level,version)
 , mCompartments        (level,version)
 , mSpecies             (level,version)
 , mParameters          (level,version)
 , mInitialAssignments  (level,version)
 , mRules               (level,version)
 , mConstraints         (level,version)
 , mReactions           (level,version)
 , mEvents              (level,version)
 , mFormulaUnitsData ( NULL  )
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();

  connectToChild();
}


Model::Model (SBMLNamespaces * sbmlns) :
   SBase             ( sbmlns )
 , mId               ( "" )
 , mName             ( "" )
 , mSubstanceUnits   ( "" )
 , mTimeUnits        ( "" )
 , mVolumeUnits      ( "" )
 , mAreaUnits        ( "" )
 , mLengthUnits      ( "" )
 , mExtentUnits      ( "" )
 , mConversionFactor ( "" )
 , mFunctionDefinitions (sbmlns)
 , mUnitDefinitions     (sbmlns)
 , mCompartmentTypes    (sbmlns)
 , mSpeciesTypes        (sbmlns)
 , mCompartments        (sbmlns)
 , mSpecies             (sbmlns)
 , mParameters          (sbmlns)
 , mInitialAssignments  (sbmlns)
 , mRules               (sbmlns)
 , mConstraints         (sbmlns)
 , mReactions           (sbmlns)
 , mEvents              (sbmlns)
 , mFormulaUnitsData ( NULL  )
{
  if (!hasValidLevelVersionNamespaceCombination())
  {
    throw SBMLConstructorException(getElementName(), sbmlns);
  }

  connectToChild();
  loadPlugins(sbmlns);

}


/*
 * Destroys this Model.
 */
Model::~Model ()
{
  if (mFormulaUnitsData != NULL)
  {  
    unsigned int size = mFormulaUnitsData->getSize();
    while (size--) 
      delete static_cast<FormulaUnitsData*>( mFormulaUnitsData->remove(0) );
    delete mFormulaUnitsData;
  }
}


/*
 * Copy constructor.
 */
Model::Model(const Model& orig) :
       SBase                (orig                    )
     , mFunctionDefinitions (orig.mFunctionDefinitions)
     , mUnitDefinitions     (orig.mUnitDefinitions)
     , mCompartmentTypes    (orig.mCompartmentTypes)
     , mSpeciesTypes        (orig.mSpeciesTypes)
     , mCompartments        (orig.mCompartments)
     , mSpecies             (orig.mSpecies)
     , mParameters          (orig.mParameters)
     , mInitialAssignments  (orig.mInitialAssignments)
     , mRules               (orig.mRules)
     , mConstraints         (orig.mConstraints)
     , mReactions           (orig.mReactions)
     , mEvents              (orig.mEvents)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mId                   = orig.mId;
    mName                 = orig.mName;
    mSubstanceUnits       = orig.mSubstanceUnits ;
    mTimeUnits            = orig.mTimeUnits ;
    mVolumeUnits          = orig.mVolumeUnits ;
    mAreaUnits            = orig.mAreaUnits ;
    mLengthUnits          = orig.mLengthUnits ;
    mExtentUnits          = orig.mExtentUnits ;
    mConversionFactor     = orig.mConversionFactor ;

    if(orig.mFormulaUnitsData != NULL)
    {
      this->mFormulaUnitsData  = new List();
      unsigned int i,iMax = orig.mFormulaUnitsData->getSize();
      for(i = 0; i < iMax; ++i)
      {
        this->mFormulaUnitsData
          ->add(static_cast<FormulaUnitsData*>
                                  (orig.mFormulaUnitsData->get(i))->clone());
      }
    }
    else
    {
      this->mFormulaUnitsData = NULL;
    }
  }
  connectToChild();
}


/*
 * Assignment operator
 */
Model& Model::operator=(const Model& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    this->SBase::operator = (rhs);
    mId = rhs.mId;
    mName = rhs.mName;
    mSubstanceUnits       = rhs.mSubstanceUnits ;
    mTimeUnits            = rhs.mTimeUnits ;
    mVolumeUnits          = rhs.mVolumeUnits ;
    mAreaUnits            = rhs.mAreaUnits ;
    mLengthUnits          = rhs.mLengthUnits ;
    mExtentUnits          = rhs.mExtentUnits ;
    mConversionFactor     = rhs.mConversionFactor ;
    mFunctionDefinitions  = rhs.mFunctionDefinitions;
    mUnitDefinitions      = rhs.mUnitDefinitions;
    mCompartmentTypes     = rhs.mCompartmentTypes;
    mSpeciesTypes         = rhs.mSpeciesTypes;
    mCompartments         = rhs.mCompartments;
    mSpecies              = rhs.mSpecies;
    mParameters           = rhs.mParameters;
    mInitialAssignments   = rhs.mInitialAssignments;
    mRules                = rhs.mRules;
    mConstraints          = rhs.mConstraints;
    mReactions            = rhs.mReactions;
    mEvents               = rhs.mEvents;


    if (this->mFormulaUnitsData  != NULL)
    {
      unsigned int size = this->mFormulaUnitsData->getSize();
      while (size--)
        delete static_cast<FormulaUnitsData*>( 
                          this->mFormulaUnitsData->remove(0) );
      delete this->mFormulaUnitsData;
    }

    if(rhs.mFormulaUnitsData != NULL)
    {
      this->mFormulaUnitsData  = new List();
      unsigned int i,iMax = rhs.mFormulaUnitsData->getSize();
      for(i = 0; i < iMax; ++i)
      {
        this->mFormulaUnitsData
          ->add(static_cast<FormulaUnitsData*>
                                   (rhs.mFormulaUnitsData->get(i))->clone());
      }
    }
    else
    {
      this->mFormulaUnitsData = NULL;
    }
  }

  connectToChild();

  return *this;
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
Model::accept (SBMLVisitor& v) const
{
  v.visit(*this);

  mFunctionDefinitions.accept(v);
  mUnitDefinitions    .accept(v);
  mCompartmentTypes   .accept(v);
  mSpeciesTypes       .accept(v);
  mCompartments       .accept(v);
  mSpecies            .accept(v);
  mParameters         .accept(v);
  mInitialAssignments .accept(v);
  mRules              .accept(v);
  mConstraints        .accept(v);
  mReactions          .accept(v);
  mEvents             .accept(v);

  v.leave(*this);

  return true;
}


/*
 * @return a (deep) copy of this Model.
 */
Model*
Model::clone () const
{
  return new Model(*this);
}

SBase* 
Model::getElementBySId(std::string id)
{
  if (id.empty()) return NULL;
  SBase* obj = mFunctionDefinitions.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mUnitDefinitions.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mCompartmentTypes.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mSpeciesTypes.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mCompartments.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mSpecies.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mParameters.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mReactions.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mInitialAssignments.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mRules.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mConstraints.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mEvents.getElementBySId(id);
  if (obj != NULL) return obj;

  return getElementFromPluginsBySId(id);
}


SBase*
Model::getElementByMetaId(std::string metaid)
{
  if (metaid.empty()) return NULL;
  if (mFunctionDefinitions.getMetaId()==metaid) return &mFunctionDefinitions;
  if (mUnitDefinitions.getMetaId()==metaid) return &mUnitDefinitions;
  if (mCompartmentTypes.getMetaId()==metaid) return &mCompartmentTypes;
  if (mSpeciesTypes.getMetaId()==metaid) return &mSpeciesTypes;
  if (mCompartments.getMetaId()==metaid) return &mCompartments;
  if (mSpecies.getMetaId()==metaid) return &mSpecies;
  if (mParameters.getMetaId()==metaid) return &mParameters;
  if (mInitialAssignments.getMetaId()==metaid) return &mInitialAssignments;
  if (mRules.getMetaId()==metaid) return &mRules;
  if (mConstraints.getMetaId()==metaid) return &mConstraints;
  if (mReactions.getMetaId()==metaid) return &mReactions;
  if (mEvents.getMetaId()==metaid) return &mEvents;

  SBase* obj = mFunctionDefinitions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mUnitDefinitions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mCompartmentTypes.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mSpeciesTypes.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mCompartments.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mSpecies.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mParameters.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mInitialAssignments.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mRules.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mConstraints.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mReactions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mEvents.getElementByMetaId(metaid);
  if (obj != NULL) return obj;

  return getElementFromPluginsByMetaId(metaid);
}


List*
Model::getAllElements()
{
  List* ret = new List();
  List* sublist = NULL;
  if (mFunctionDefinitions.size() > 0) {
    ret->add(&mFunctionDefinitions);
    sublist = mFunctionDefinitions.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mUnitDefinitions.size() > 0) {
    ret->add(&mUnitDefinitions);
    sublist = mUnitDefinitions.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mCompartmentTypes.size() > 0) {
    ret->add(&mCompartmentTypes);
    sublist = mCompartmentTypes.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mSpeciesTypes.size() > 0) {
    ret->add(&mSpeciesTypes);
    sublist = mSpeciesTypes.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mCompartments.size() > 0) {
    ret->add(&mCompartments);
    sublist = mCompartments.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mSpecies.size() > 0) {
    ret->add(&mSpecies);
    sublist = mSpecies.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mParameters.size() > 0) {
    ret->add(&mParameters);
    sublist = mParameters.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mInitialAssignments.size() > 0) {
    ret->add(&mInitialAssignments);
    sublist = mInitialAssignments.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mRules.size() > 0) {
    ret->add(&mRules);
    sublist = mRules.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mConstraints.size() > 0) {
    ret->add(&mConstraints);
    sublist = mConstraints.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mReactions.size() > 0) {
    ret->add(&mReactions);
    sublist = mReactions.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mEvents.size() > 0) {
    ret->add(&mEvents);
    sublist = mEvents.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }

  sublist = getAllElementsFromPlugins();
  ret->transferFrom(sublist);
  delete sublist;
  return ret;
}

/*
 * @return the id of this SBML object.
 */
const string&
Model::getId () const
{
  return mId;
}


/*
 * @return the name of this SBML object.
 */
const string&
Model::getName () const
{
  return (getLevel() == 1) ? mId : mName;
}


/*
 * Returns the value of the "substanceUnits" attribute of this Model.
 */
const std::string& 
Model::getSubstanceUnits () const
{
  return mSubstanceUnits;
}


/*
 * Returns the value of the "timeUnits" attribute of this Model.
 */
const std::string& 
Model::getTimeUnits () const
{
  return mTimeUnits;
}

/*
 * Returns the value of the "volumeUnits" attribute of this Model.
 */
const std::string& 
Model::getVolumeUnits () const
{
  return mVolumeUnits;
}


/*
 * Returns the value of the "areaUnits" attribute of this Model.
 */
const std::string& 
Model::getAreaUnits () const
{
  return mAreaUnits;
}


/*
 * Returns the value of the "lengthUnits" attribute of this Model.
 */
const std::string& 
Model::getLengthUnits () const
{
  return mLengthUnits;
}


/*
 * Returns the value of the "extentUnits" attribute of this Model.
 */
const std::string& 
Model::getExtentUnits () const
{
  return mExtentUnits;
}


/*
 * Returns the value of the "conversionFactor" attribute of this Model.
 */
const std::string& 
Model::getConversionFactor () const
{
  return mConversionFactor;
}



/*
 * @return true if the id of this SBML object has been set, false
 * otherwise.
 */
bool
Model::isSetId () const
{
  return (mId.empty() == false);
}


/*
 * @return true if the name of this SBML object is set, false
 * otherwise.
 */
bool
Model::isSetName () const
{
  return (getLevel() == 1) ? (mId.empty() == false) : 
                            (mName.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "substanceUnits" attribute is set.
 */
bool 
Model::isSetSubstanceUnits () const
{
  return (mSubstanceUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "timeUnits" attribute is set.
 */
bool 
Model::isSetTimeUnits () const
{
  return (mTimeUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "volumeUnits" attribute is set.
 */
bool 
Model::isSetVolumeUnits () const
{
  return (mVolumeUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "areaUnits" attribute is set.
 */
bool 
Model::isSetAreaUnits () const
{
  return (mAreaUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "lengthUnits" attribute is set.
 */
bool 
Model::isSetLengthUnits () const
{
  return (mLengthUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "extentUnits" attribute is set.
 */
bool 
Model::isSetExtentUnits () const
{
  return (mExtentUnits.empty() == false);
}


/*
 * Predicate returning @c true if this
 * Model's "conversionFactor" attribute is set.
 */
bool 
Model::isSetConversionFactor () const
{
  return (mConversionFactor.empty() == false);
}


/*
 * Sets the id of this SBML object to a copy of sid.
 */
int
Model::setId (const std::string& sid)
{
  /* since the setId function has been used as an
   * alias for setName we cant require it to only
   * be used on a L2 model
   */
/*  if (getLevel() == 1)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
*/
  if (&(sid) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (!(SyntaxChecker::isValidInternalSId(sid)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mId = sid;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the name of this SBML object to a copy of name.
 */
int
Model::setName (const std::string& name)
{
  /* if this is setting an L2 name the type is string
   * whereas if it is setting an L1 name its type is SId
   */
  if (&(name) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() == 1)
  {
    if (!(SyntaxChecker::isValidInternalSId(name)))
    {
      return LIBSBML_INVALID_ATTRIBUTE_VALUE;
    }
    else
    {
      mId = name;
      return LIBSBML_OPERATION_SUCCESS;
    }
  }
  else
  {
    mName = name;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setSubstanceUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mSubstanceUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setTimeUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mTimeUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setVolumeUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mVolumeUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setAreaUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mAreaUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setLengthUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mLengthUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setExtentUnits (const std::string& units)
{
  if (&(units) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalUnitSId(units)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mExtentUnits = units;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the substanceUnits of this SBML object.
 */
int
Model::setConversionFactor (const std::string& id)
{
  if (&(id) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalSId(id)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mConversionFactor = id;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Unsets the id of this SBML object.
 */
int
Model::unsetId ()
{
  mId.erase();

  if (mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the name of this SBML object.
 */
int
Model::unsetName ()
{
  if (getLevel() == 1) 
  {
    mId.erase();
  }
  else 
  {
    mName.erase();
  }

  if (getLevel() == 1 && mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (mName.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}



/*
 * Unsets the SubstanceUnits of this SBML object.
 */
int
Model::unsetSubstanceUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mSubstanceUnits.erase();

  if (mSubstanceUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the TimeUnits of this SBML object.
 */
int
Model::unsetTimeUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mTimeUnits.erase();

  if (mTimeUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the VolumeUnits of this SBML object.
 */
int
Model::unsetVolumeUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mVolumeUnits.erase();

  if (mVolumeUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the AreaUnits of this SBML object.
 */
int
Model::unsetAreaUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mAreaUnits.erase();

  if (mAreaUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the LengthUnits of this SBML object.
 */
int
Model::unsetLengthUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mLengthUnits.erase();

  if (mLengthUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the LengthUnits of this SBML object.
 */
int
Model::unsetExtentUnits ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mExtentUnits.erase();

  if (mExtentUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the ConversionFactor of this SBML object.
 */
int
Model::unsetConversionFactor ()
{
  /* only in L3 */
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  
  mConversionFactor.erase();

  if (mConversionFactor.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Adds a copy of the given FunctionDefinition to this Model.
 */
int
Model::addFunctionDefinition (const FunctionDefinition* fd)
{
  if (fd == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(fd->hasRequiredAttributes()) || !(fd->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != fd->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != fd->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(fd)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getFunctionDefinition(fd->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mFunctionDefinitions.append(fd);
   
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given UnitDefinition to this Model.
 */
int
Model::addUnitDefinition (const UnitDefinition* ud)
{
  if (ud == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(ud->hasRequiredAttributes()) || !(ud->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != ud->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != ud->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(ud)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getUnitDefinition(ud->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mUnitDefinitions.append(ud);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given CompartmentType to this Model.
 */
int
Model::addCompartmentType (const CompartmentType* ct)
{
  if (ct == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(ct->hasRequiredAttributes()) || !(ct->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != ct->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != ct->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(ct)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getCompartmentType(ct->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mCompartmentTypes.append(ct);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given SpeciesType to this Model.
 */
int
Model::addSpeciesType (const SpeciesType* st)
{
  if (st == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(st->hasRequiredAttributes()) || !(st->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != st->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != st->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(st)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getSpeciesType(st->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mSpeciesTypes.append(st);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Compartment to this Model.
 */
int
Model::addCompartment (const Compartment* c)
{
  if (c == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(c->hasRequiredAttributes()) || !(c->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != c->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != c->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(c)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getCompartment(c->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mCompartments.append(c);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Species to this Model.
 */
int
Model::addSpecies (const Species* s)
{
  if (s == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(s->hasRequiredAttributes()) || !(s->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != s->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != s->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(s)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getSpecies(s->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mSpecies.append(s);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Parameter to this Model.
 */
int
Model::addParameter (const Parameter* p)
{
  if (p == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(p->hasRequiredAttributes()) || !(p->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != p->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != p->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(p)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getParameter(p->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    /* hack so that this will accept a local parameter !! */
    if (p->getTypeCode() == SBML_LOCAL_PARAMETER)
    {
      Parameter *p1 = new Parameter(*p);
      mParameters.append(p1);
    }
    else
    {
      mParameters.append(p);
    }

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given InitialAssignment to this Model.
 */
int
Model::addInitialAssignment (const InitialAssignment* ia)
{
  if (ia == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(ia->hasRequiredAttributes()) || !(ia->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != ia->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != ia->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(ia)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getInitialAssignment(ia->getSymbol()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mInitialAssignments.append(ia);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Rule to this Model.
 */
int
Model::addRule (const Rule* r)
{
  if (r == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(r->hasRequiredAttributes()) || !(r->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != r->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != r->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(r)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (!r->isAlgebraic() 
         && getRule(r->getVariable()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mRules.append(r);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Constraint to this Model.
 */
int
Model::addConstraint (const Constraint* c)
{
  if (c == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(c->hasRequiredAttributes()) || !(c->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != c->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != c->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(c)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else
  {
    mConstraints.append(c);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Reaction to this Model.
 */
int
Model::addReaction (const Reaction* r)
{
  if (r == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(r->hasRequiredAttributes()) || !(r->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != r->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != r->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(r)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getReaction(r->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mReactions.append(r);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Adds a copy of the given Event to this Model.
 */
int
Model::addEvent (const Event* e)
{
  if (e == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(e->hasRequiredAttributes()) || !(e->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != e->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != e->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(e)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (e->isSetId() && getEvent(e->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mEvents.append(e);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Creates a new FunctionDefinition inside this Model and returns it.
 */
FunctionDefinition*
Model::createFunctionDefinition ()
{
  FunctionDefinition* fd = NULL;

  try
  {
    fd = new FunctionDefinition(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (fd != NULL) mFunctionDefinitions.appendAndOwn(fd);

  return fd;
}


/*
 * Creates a new UnitDefinition inside this Model and returns it.
 */
UnitDefinition*
Model::createUnitDefinition ()
{
  UnitDefinition* ud = NULL;

  try
  {
    ud = new UnitDefinition(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ud != NULL) mUnitDefinitions.appendAndOwn(ud);

  return ud;
}


/*
 * Creates a new Unit inside this Model and returns a pointer to it.  The
 * Unit is added to the last UnitDefinition created.
 *
 * If a UnitDefinitions does not exist for this model, a new Unit is not
 * created and NULL is returned.
 */
Unit*
Model::createUnit ()
{
  unsigned int size = getNumUnitDefinitions();
  return (size > 0) ? getUnitDefinition(size - 1)->createUnit() : NULL;
}


/*
 * Creates a new CompartmentType inside this Model and returns it.
 */
CompartmentType*
Model::createCompartmentType ()
{
  CompartmentType* ct = NULL;

  try
  {
    ct = new CompartmentType(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ct != NULL) mCompartmentTypes.appendAndOwn(ct);

  return ct;
}


/*
 * Creates a new SpeciesType inside this Model and returns it.
 */
SpeciesType*
Model::createSpeciesType ()
{
  SpeciesType* st = NULL;

  try
  {
    st = new SpeciesType(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (st != NULL) mSpeciesTypes.appendAndOwn(st);

  return st;
}


/*
 * Creates a new Compartment inside this Model and returns it.
 */
Compartment*
Model::createCompartment ()
{
  Compartment* c = NULL;

  try
  {
    c = new Compartment(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (c != NULL) mCompartments.appendAndOwn(c);

  return c;
}


/*
 * Creates a new Species inside this Model and returns it.
 */
Species*
Model::createSpecies ()
{
  Species* s = NULL;

  try
  {
    s = new Species(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (s != NULL) mSpecies.appendAndOwn(s);

  return s;
}


/*
 * Creates a new Parameter inside this Model and returns.
 */
Parameter*
Model::createParameter ()
{
  Parameter* p = NULL;

  try
  {
    p = new Parameter(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (p != NULL) mParameters.appendAndOwn(p);

  return p;
}


/*
 * Creates a new InitialAssignment inside this Model and returns it.
 */
InitialAssignment*
Model::createInitialAssignment ()
{
  InitialAssignment* ia = NULL;

  try
  {
    ia = new InitialAssignment(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ia != NULL) mInitialAssignments.appendAndOwn(ia);

  return ia;
}


/*
 * Creates a new AlgebraicRule inside this Model and returns it.
 */
AlgebraicRule*
Model::createAlgebraicRule ()
{
  AlgebraicRule* ar = NULL;

  try
  {
    ar = new AlgebraicRule(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ar != NULL) mRules.appendAndOwn(ar);

  return ar;
}


/*
 * Creates a new AssignmentRule inside this Model and returns it.
 */
AssignmentRule*
Model::createAssignmentRule ()
{
  AssignmentRule* ar = NULL;

  try
  {
    ar = new AssignmentRule(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ar != NULL) mRules.appendAndOwn(ar);

  return ar;
}


/*
 * Creates a new RateRule inside this Model and returns it.
 */
RateRule*
Model::createRateRule ()
{
  RateRule* rr = NULL;

  try
  {
    rr = new RateRule(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (rr != NULL) mRules.appendAndOwn(rr);

  return rr;
}


/*
 * Creates a new Constraint inside this Model and returns it.
 */
Constraint*
Model::createConstraint ()
{
  Constraint* c = NULL;

  try
  {
    c = new Constraint(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (c != NULL) mConstraints.appendAndOwn(c);

  return c;
}


/*
 * Creates a new Reaction inside this Model and returns it.
 */
Reaction*
Model::createReaction ()
{
  Reaction* r = NULL;

  try
  {
    r = new Reaction(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (r != NULL) mReactions.appendAndOwn(r);

  return r;
}


/*
 * Creates a new Reactant (ie SpeciesReference) inside this Model and
 * returns a pointer to it.  The SpeciesReference is added to the reactants
 * of the last Reaction created.
 *
 * If a Reaction does not exist for this model, a new SpeciesReference is
 * not created and NULL is returned.
 */
SpeciesReference*
Model::createReactant ()
{
  unsigned int size = getNumReactions();
  return (size > 0) ? getReaction(size - 1)->createReactant() : NULL;
}


/*
 * Creates a new Product (ie SpeciesReference) inside this Model and
 * returns a pointer to it.  The SpeciesReference is added to the products
 * of the last Reaction created.
 *
 * If a Reaction does not exist for this model, a new SpeciesReference is
 * not created and NULL is returned.
 */
SpeciesReference*
Model::createProduct ()
{
  unsigned int size = getNumReactions();
  return (size > 0) ? getReaction(size - 1)->createProduct() : NULL;
}


/*
 * Creates a new Modifer (ie ModifierSpeciesReference) inside this Model
 * and returns a pointer to it.  The ModifierSpeciesReference is added to
 * the modifiers of the last Reaction created.
 *
 * If a Reaction does not exist for this model, a new
 * ModifierSpeciesReference is not created and NULL is returned.
 */
ModifierSpeciesReference*
Model::createModifier ()
{
  unsigned int size = getNumReactions();
  return (size > 0) ? getReaction(size - 1)->createModifier() : NULL;
}


/*
 * Creates a new KineticLaw inside this Model and returns a pointer to it.
 * The KineticLaw is associated with the last Reaction created.
 *
 * If a Reaction does not exist for this model, or a Reaction does exist,
 * but already has a KineticLaw, a new KineticLaw is not created and NULL
 * is returned.
 */
KineticLaw*
Model::createKineticLaw ()
{
  unsigned int size = getNumReactions();
  return (size > 0) ? getReaction(size - 1)->createKineticLaw() : NULL;
}


/*
 * Creates a new Parameter (of a KineticLaw) inside this Model and returns
 * a pointer to it.  The Parameter is associated with the KineticLaw of the
 * last Reaction created.
 *
 * If a Reaction does not exist for this model, or a KineticLaw for the
 * Reaction, a new Parameter is not created and NULL is returned.
 */
Parameter*
Model::createKineticLawParameter ()
{
  unsigned int size = getNumReactions();

  if (size > 0)
  {
    KineticLaw* kl = getReaction(size - 1)->getKineticLaw();
    if (kl != NULL) return kl->createParameter();
  }

  return NULL;
}


/*
 * Creates a new Parameter (of a KineticLaw) inside this Model and returns
 * a pointer to it.  The Parameter is associated with the KineticLaw of the
 * last Reaction created.
 *
 * If a Reaction does not exist for this model, or a KineticLaw for the
 * Reaction, a new Parameter is not created and NULL is returned.
 */
LocalParameter*
Model::createKineticLawLocalParameter ()
{
  unsigned int size = getNumReactions();

  if (size > 0)
  {
    KineticLaw* kl = getReaction(size - 1)->getKineticLaw();
    if (kl != NULL) return kl->createLocalParameter();
  }

  return NULL;
}


/*
 * Creates a new Event inside this Model and returns it.
 */
Event*
Model::createEvent ()
{
  Event* e = NULL;

  try
  {
    e = new Event(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (e != NULL) mEvents.appendAndOwn(e);

  return e;
}


/*
 * Creates a new EventAssignment inside this Model and returns a pointer to
 * it.  The EventAssignment is added to the the last Event created.
 *
 * If an Event does not exist for this model, a new EventAssignment is not
 * created and NULL is returned.
 */
EventAssignment*
Model::createEventAssignment ()
{
  unsigned int size = getNumEvents();
  return (size > 0) ? getEvent(size - 1)->createEventAssignment() : NULL;
}


/*
 * Creates a new Trigger inside this Model and returns a pointer to
 * it.  The Trigger is added to the the last Event created.
 *
 * If an Event does not exist for this model, a new Trigger is not
 * created and NULL is returned.
 */
Trigger*
Model::createTrigger ()
{
  unsigned int size = getNumEvents();
  return (size > 0) ? getEvent(size - 1)->createTrigger() : NULL;
}


/*
 * Creates a new Delay inside this Model and returns a pointer to
 * it.  The Delay is added to the the last Event created.
 *
 * If an Event does not exist for this model, a new Delay is not
 * created and NULL is returned.
 */
Delay*
Model::createDelay ()
{
  unsigned int size = getNumEvents();
  return (size > 0) ? getEvent(size - 1)->createDelay() : NULL;
}


/*
 * Sets the annotation of this SBML object to a copy of annotation.
 */
int
Model::setAnnotation (const XMLNode* annotation)
{
  int success = SBase::setAnnotation(annotation);

  if (success == 0)
  {
    //
    // delete existing mHistory
    //
    // existing mHistory (if any) needs to be deleted at any rate, otherwise
    // unsetAnnotation() ( setAnnotation(NULL) ) doesn't work as expected.
    // (These functions must clear all elements in an annotation.)
    //
    delete mHistory;
    mHistory = NULL;

    if(mAnnotation != NULL 
       && RDFAnnotationParser::hasHistoryRDFAnnotation(mAnnotation))
    {
      // parse mAnnotation (if any) and set mHistory
      mHistory = RDFAnnotationParser::parseRDFAnnotation(mAnnotation);
      mHistoryChanged = true;

    }
  }

  return success;
}


/*
 * Sets the annotation (by string) of this SBML object to a copy of annotation.
 */
int
Model::setAnnotation (const std::string& annotation)
{
  int success = LIBSBML_OPERATION_FAILED;
  if(annotation.empty())
  {
    unsetAnnotation();
    return LIBSBML_OPERATION_SUCCESS;
  }

  XMLNode* annt_xmln;
  if (getSBMLDocument())
  {
    XMLNamespaces* xmlns = getSBMLDocument()->getNamespaces();
    annt_xmln = XMLNode::convertStringToXMLNode(annotation,xmlns);
  }
  else
  {
    annt_xmln = XMLNode::convertStringToXMLNode(annotation);
  }

  if(annt_xmln != NULL)
  {
    success = setAnnotation(annt_xmln);
    delete annt_xmln;
  }
  return success;
}


/*
 * Appends annotation to the existing annotations.
 * This allows other annotations to be preserved whilst
 * adding additional information.
 */
int
Model::appendAnnotation (const XMLNode* annotation)
{
  // take out any attempt to merge RDF

  return SBase::appendAnnotation(annotation);

//  int success = LIBSBML_OPERATION_FAILED;
//  if(annotation == NULL) return LIBSBML_OPERATION_SUCCESS;
//
//  XMLNode* new_annotation = NULL;
//  const string&  name = annotation->getName();
//
//  // check for annotation tags and add if necessary
//  if (name != "annotation")
//  {
//    XMLToken ann_t = XMLToken(XMLTriple("annotation", "", ""), XMLAttributes());
//    new_annotation = new XMLNode(ann_t);
//    new_annotation->addChild(*annotation);
//  }
//  else
//  {
//    new_annotation = annotation->clone();
//  }
//
//  // parse new_annotation and reset mHistory 
//  if (RDFAnnotationParser::hasHistoryRDFAnnotation(new_annotation))
//  {
//    ModelHistory* new_mhistory = RDFAnnotationParser::parseRDFAnnotation(new_annotation);
//    if(new_mhistory != NULL)
//    {
//      delete mHistory;
//      mHistory = new_mhistory;
////      mHistoryChanged = true;
//    }
//  }
//
//  success = SBase::appendAnnotation(new_annotation);
//
//  delete new_annotation;
//
//  return success;
}


/*
 * Appends annotation (by string) to the existing annotations.
 * This allows other annotations to be preserved whilst
 * adding additional information.
 */
int
Model::appendAnnotation (const std::string& annotation)
{
  int success = LIBSBML_OPERATION_FAILED;
  XMLNode* annt_xmln;
  if (getSBMLDocument() != NULL)
  {
    XMLNamespaces* xmlns = getSBMLDocument()->getNamespaces();
    annt_xmln = XMLNode::convertStringToXMLNode(annotation,xmlns);
  }
  else
  {
    annt_xmln = XMLNode::convertStringToXMLNode(annotation);
  }

  if(annt_xmln != NULL)
  {
    success = appendAnnotation(annt_xmln);
    delete annt_xmln;
  }

  return success;
}


/** @cond doxygen-libsbml-internal */
/*
 * Synchronizes the annotation of this SBML object.
 */
void
Model::syncAnnotation ()
{
  SBase::syncAnnotation();

  //if (mAnnotationChanged == false)
  //  return;

  //if (mHistoryChanged == false && mCVTermsChanged == false)
  //  return;
  //else
  //  reconstructRDFAnnotation();

  //bool hasRDF = false;
  //bool hasAdditionalRDF = false;
  //// determine status of existing annotation before doing anything
  //if (mAnnotation != NULL)
  //{
  //  hasRDF = RDFAnnotationParser::hasRDFAnnotation(mAnnotation);
  //  hasAdditionalRDF = 
  //    RDFAnnotationParser::hasAdditionalRDFAnnotation(mAnnotation);
  //}

  //XMLNode * history = RDFAnnotationParser::parseModelHistory(this);

  //if(mAnnotation != NULL && hasRDF && mHistoryChanged == true)
  //{
  //  XMLNode* new_annotation = RDFAnnotationParser::deleteRDFAnnotation(mAnnotation);
  //  if(new_annotation == NULL)
  //  {
  //    XMLToken ann_token = XMLToken(XMLTriple("annotation", "", ""), XMLAttributes());
  //    new_annotation = new XMLNode(ann_token);
  //    new_annotation->addChild(*mAnnotation);
  //  }
  //  *mAnnotation = *new_annotation;
  //  delete new_annotation;
  //}

  //if (history != NULL)
  //{
  //  if (mHistoryChanged == true)
  //  {
  //    if (mAnnotation == NULL)
  //    {
  //      mAnnotation = history;
  //    }
  //    else
  //    {
  //      if (mAnnotation->isEnd())
  //      {
  //        mAnnotation->unsetEnd();
  //      }
  //      if (hasAdditionalRDF)
  //      {
  //        //need to insert the history into existing RDF
  //        unsigned int n = 0;
  //        while (n < mAnnotation->getNumChildren())
  //        {
  //          if (mAnnotation->getChild(n).getName() == "RDF")
  //          {
  //            mAnnotation->getChild(n).insertChild(0, 
  //              history->getChild(0).getChild(0));
  //            break;
  //          }
  //          n++;
  //        }
  //      }
  //      else
  //      {
  //        mAnnotation->addChild(history->getChild(0));
  //      }
  //      delete history;
  //    }
  //  }
  //}
  //else
  //{
  //  // Annotations for CVTerm are added by the above RDFAnnotationParser::parseModelHistory(this)
  //  // if and only if mHistory is not NULL.
  //  // Thus, annotations for CVTerm (if any) needs to be added here if history (mHistory) is NULL.
  //  SBase::syncAnnotation();
  //}

}
/** @endcond */


/*
 * @return the list of FunctionDefinitions for this Model.
 */
const ListOfFunctionDefinitions*
Model::getListOfFunctionDefinitions () const
{
  return &mFunctionDefinitions;
}


/*
 * @return the list of FunctionDefinitions for this Model.
 */
ListOfFunctionDefinitions*
Model::getListOfFunctionDefinitions ()
{
  return &mFunctionDefinitions;
}


/*
 * @return the list of UnitDefinitions for this Model.
 */
const ListOfUnitDefinitions*
Model::getListOfUnitDefinitions () const
{
  return &mUnitDefinitions;
}


/*
 * @return the list of UnitDefinitions for this Model.
 */
ListOfUnitDefinitions*
Model::getListOfUnitDefinitions ()
{
  return &mUnitDefinitions;
}


/*
 * @return the list of CompartmentTypes for this Model.
 */
const ListOfCompartmentTypes*
Model::getListOfCompartmentTypes () const
{
  return &mCompartmentTypes;
}


/*
 * @return the list of CompartmentTypes for this Model.
 */
ListOfCompartmentTypes*
Model::getListOfCompartmentTypes ()
{
  return &mCompartmentTypes;
}


/*
 * @return the list of SpeciesTypes for this Model.
 */
const ListOfSpeciesTypes*
Model::getListOfSpeciesTypes () const
{
  return &mSpeciesTypes;
}

 
/*
 * @return the list of SpeciesTypes for this Model.
 */
ListOfSpeciesTypes*
Model::getListOfSpeciesTypes ()
{
  return &mSpeciesTypes;
}


/*
 * @return the list of Compartments for this Model.
 */
const ListOfCompartments*
Model::getListOfCompartments () const
{
  return &mCompartments;
}


/*
 * @return the list of Compartments for this Model.
 */
ListOfCompartments*
Model::getListOfCompartments ()
{
  return &mCompartments;
}


/*
 * @return the list of Species for this Model.
 */
const ListOfSpecies*
Model::getListOfSpecies () const
{
  return &mSpecies;
}


/*
 * @return the list of Species for this Model.
 */
ListOfSpecies*
Model::getListOfSpecies ()
{
  return &mSpecies;
}


/*
 * @return the list of Parameters for this Model.
 */
const ListOfParameters*
Model::getListOfParameters () const
{
  return &mParameters;
}


/*
 * @return the list of Parameters for this Model.
 */
ListOfParameters*
Model::getListOfParameters ()
{
  return &mParameters;
}


/*
 * @return the list of InitialAssignments for this Model.
 */
const ListOfInitialAssignments*
Model::getListOfInitialAssignments () const
{
  return &mInitialAssignments;
}


/*
 * @return the list of InitialAssignment for this Model.
 */
ListOfInitialAssignments*
Model::getListOfInitialAssignments ()
{
  return &mInitialAssignments;
}


/*
 * @return the list of Rules for this Model.
 */
const ListOfRules*
Model::getListOfRules () const
{
  return &mRules;
}


/*
 * @return the list of Rules for this Model.
 */
ListOfRules*
Model::getListOfRules ()
{
  return &mRules;
}


/*
 * @return the list of Constraints for this Model.
 */
const ListOfConstraints*
Model::getListOfConstraints () const
{
  return &mConstraints;
}

 
/*
 * @return the list of Constraints for this Model.
 */
ListOfConstraints*
Model::getListOfConstraints ()
{
  return &mConstraints;
}


/*
 * @return the list of Reactions for this Model.
 */
const ListOfReactions*
Model::getListOfReactions () const
{
  return &mReactions;
}


/*
 * @return the list of Reactions for this Model.
 */
ListOfReactions*
Model::getListOfReactions ()
{
  return &mReactions;
}


/*
 * @return the list of Events for this Model.
 */
const ListOfEvents*
Model::getListOfEvents () const
{
  return &mEvents;
}


/*
 * @return the list of Events for this Model.
 */
ListOfEvents*
Model::getListOfEvents ()
{
  return &mEvents;
}


/*
 * @return the nth FunctionDefinition of this Model.
 */
const FunctionDefinition*
Model::getFunctionDefinition (unsigned int n) const
{
  return static_cast<const FunctionDefinition*>( mFunctionDefinitions.get(n) );
}


/*
 * @return the nth FunctionDefinition of this Model.
 */
FunctionDefinition*
Model::getFunctionDefinition (unsigned int n)
{
  return static_cast<FunctionDefinition*>( mFunctionDefinitions.get(n) );
}


/*
 * @return the FunctionDefinition in this Model with the given id or @c NULL
 * if no such FunctionDefinition exists.
 */
const FunctionDefinition*
Model::getFunctionDefinition (const std::string& sid) const
{
  return static_cast<const FunctionDefinition*>(mFunctionDefinitions.get(sid));
}


/*
 * @return the FunctionDefinition in this Model with the given id or @c NULL
 * if no such FunctionDefinition exists.
 */
FunctionDefinition*
Model::getFunctionDefinition (const std::string& sid)
{
  return static_cast<FunctionDefinition*>( mFunctionDefinitions.get(sid) );
}


/*
 * @return the nth UnitDefinition of this Model.
 */
const UnitDefinition*
Model::getUnitDefinition (unsigned int n) const
{
  return static_cast<const UnitDefinition*>( mUnitDefinitions.get(n) );
}


/*
 * @return the nth UnitDefinition of this Model.
 */
UnitDefinition*
Model::getUnitDefinition (unsigned int n)
{
  return static_cast<UnitDefinition*>( mUnitDefinitions.get(n) );
}


/*
 * @return the UnitDefinition in this Model with the given id or @c NULL if no
 * such UnitDefinition exists.
 */
const UnitDefinition*
Model::getUnitDefinition (const std::string& sid) const
{
  return static_cast<const UnitDefinition*>( mUnitDefinitions.get(sid) );
}


/*
 * @return the UnitDefinition in this Model with the given id or @c NULL if no
 * such UnitDefinition exists.
 */
UnitDefinition*
Model::getUnitDefinition (const std::string& sid)
{
  return static_cast<UnitDefinition*>( mUnitDefinitions.get(sid) );
}


/*
 * @return the nth CompartmentType of this Model.
 */
const CompartmentType*
Model::getCompartmentType (unsigned int n) const
{
  return static_cast<const CompartmentType*>( mCompartmentTypes.get(n) );
}


/*
 * @return the nth CompartmentType of this Model.
 */
CompartmentType*
Model::getCompartmentType (unsigned int n)
{
  return static_cast<CompartmentType*>( mCompartmentTypes.get(n) );
}


/*
 * @return the CompartmentType in this Model with the given id or @c NULL if
 * no such CompartmentType exists.
 */
const CompartmentType*
Model::getCompartmentType (const std::string& sid) const
{
  return static_cast<const CompartmentType*>( mCompartmentTypes.get(sid) );
}


/*
 * @return the CompartmentType in this Model with the given id or @c NULL if
 * no such CompartmentType exists.
 */
CompartmentType*
Model::getCompartmentType (const std::string& sid)
{
  return static_cast<CompartmentType*>( mCompartmentTypes.get(sid) );
}


/*
 * @return the nth SpeciesType of this Model.
 */
const SpeciesType*
Model::getSpeciesType (unsigned int n) const
{
  return static_cast<const SpeciesType*>( mSpeciesTypes.get(n) );
}


/*
 * @return the nth SpeciesType of this Model.
 */
SpeciesType*
Model::getSpeciesType (unsigned int n)
{
  return static_cast<SpeciesType*>( mSpeciesTypes.get(n) );
}


/*
 * @return the SpeciesType in this Model with the given id or @c NULL if
 * no such SpeciesType exists.
 */
const SpeciesType*
Model::getSpeciesType (const std::string& sid) const
{
  return static_cast<const SpeciesType*>( mSpeciesTypes.get(sid) );
}


/*
 * @return the SpeciesType in this Model with the given id or @c NULL if
 * no such SpeciesType exists.
 */
SpeciesType*
Model::getSpeciesType (const std::string& sid)
{
  return static_cast<SpeciesType*>( mSpeciesTypes.get(sid) );
}


/*
 * @return the nth Compartment of this Model.
 */
const Compartment*
Model::getCompartment (unsigned int n) const
{
  return static_cast<const Compartment*>( mCompartments.get(n) );
}


/*
 * @return the nth Compartment of this Model.
 */
Compartment*
Model::getCompartment (unsigned int n)
{
  return static_cast<Compartment*>( mCompartments.get(n) );
}


/*
 * @return the Compartment in this Model with the given id or @c NULL if no
 * such Compartment exists.
 */
const Compartment*
Model::getCompartment (const std::string& sid) const
{
  return static_cast<const Compartment*>( mCompartments.get(sid) );
}


/*
 * @return the Compartment in this Model with the given id or @c NULL if no
 * such Compartment exists.
 */
Compartment*
Model::getCompartment (const std::string& sid)
{
  return static_cast<Compartment*>( mCompartments.get(sid) );
}


/*
 * @return the nth Species of this Model.
 */
const Species*
Model::getSpecies (unsigned int n) const
{
  return static_cast<const Species*>( mSpecies.get(n) );
}


/*
 * @return the nth Species of this Model.
 */
Species*
Model::getSpecies (unsigned int n)
{
  return static_cast<Species*>( mSpecies.get(n) );
}


/*
 * @return the Species in this Model with the given id or @c NULL if no such
 * Species exists.
 */
const Species*
Model::getSpecies (const std::string& sid) const
{
  return static_cast<const Species*>( mSpecies.get(sid) );
}


/*
 * @return the Species in this Model with the given id or @c NULL if no such
 * Species exists.
 */
Species*
Model::getSpecies (const std::string& sid)
{
  return static_cast<Species*>( mSpecies.get(sid) );
}


/*
 * @return the nth Parameter of this Model.
 */
const Parameter*
Model::getParameter (unsigned int n) const
{
  return static_cast<const Parameter*>( mParameters.get(n) );
}


/*
 * @return the nth Parameter of this Model.
 */
Parameter*
Model::getParameter (unsigned int n)
{
  return static_cast<Parameter*>( mParameters.get(n) );
}


/*
 * @return the Parameter in this Model with the given id or @c NULL if no such
 * Parameter exists.
 */
const Parameter*
Model::getParameter (const std::string& sid) const
{
  return static_cast<const Parameter*>( mParameters.get(sid) );
}


/*
 * @return the Parameter in this Model with the given id or @c NULL if no such
 * Parameter exists.
 */
Parameter*
Model::getParameter (const std::string& sid)
{
  return static_cast<Parameter*>( mParameters.get(sid) );
}


/*
 * @return the nth InitialAssignment of this Model.
 */
const InitialAssignment*
Model::getInitialAssignment (unsigned int n) const
{
  return static_cast<const InitialAssignment*>( mInitialAssignments.get(n) );
}


/*
 * @return the nth InitialAssignment of this Model.
 */
InitialAssignment*
Model::getInitialAssignment (unsigned int n)
{
  return static_cast<InitialAssignment*>( mInitialAssignments.get(n) );
}


/*
 * @return the InitialAssignment in this Model with the given symbol or
 * @c NULL if no such InitialAssignment exists.
 */
const InitialAssignment*
Model::getInitialAssignment (const std::string& symbol) const
{
  return static_cast<const InitialAssignment*>
  (
    mInitialAssignments.get(symbol)
  );
}

 
/*
 * @return the InitialAssignment in this Model with the given symbol or
 * @c NULL if no such InitialAssignment exists.
 */
InitialAssignment*
Model::getInitialAssignment (const std::string& symbol)
{
  return static_cast<InitialAssignment*>( mInitialAssignments.get(symbol) );
}


/*
 * @return the nth Rule of this Model.
 */
const Rule*
Model::getRule (unsigned int n) const
{
  return static_cast<const Rule*>( mRules.get(n) );
}


/*
 * @return the nth Rule of this Model.
 */
Rule*
Model::getRule (unsigned int n)
{
  return static_cast<Rule*>( mRules.get(n) );
}


/*
 * @return the Rule in this Model with the given variable or @c NULL if no
 * such Rule exists.
 */
const Rule*
Model::getRule (const std::string& variable) const
{
  return static_cast<const Rule*>( mRules.get(variable) );
}

 
/*
 * @return the Rule in this Model with the given symbol or @c NULL if no
 * such Rule exists.
 */
Rule*
Model::getRule (const std::string& variable)
{
  return static_cast<Rule*>( mRules.get(variable) );
}


/*
 * @return the nth Constraint of this Model.
 */
const Constraint*
Model::getConstraint (unsigned int n) const
{
  return static_cast<const Constraint*>( mConstraints.get(n) );
}


/*
 * @return the nth Constraint of this Model.
 */
Constraint*
Model::getConstraint (unsigned int n)
{
  return static_cast<Constraint*>( mConstraints.get(n) );
}


/*
 * @return the nth Reaction of this Model.
 */
const Reaction*
Model::getReaction (unsigned int n) const
{
  return static_cast<const Reaction*>( mReactions.get(n) );
}


/*
 * @return the nth Reaction of this Model.
 */
Reaction*
Model::getReaction (unsigned int n)
{
  return static_cast<Reaction*>( mReactions.get(n) );
}


/*
 * @return the Reaction in this Model with the given id or @c NULL if no such
 * Reaction exists.
 */
const Reaction*
Model::getReaction (const std::string& sid) const
{
  return static_cast<const Reaction*>( mReactions.get(sid) );
}


/*
 * @return the Reaction in this Model with the given id or @c NULL if no such
 * Reaction exists.
 */
Reaction*
Model::getReaction (const std::string& sid)
{
  return static_cast<Reaction*>( mReactions.get(sid) );
}


/**
 * Get a SpeciesReference object based on its identifier.
 */
SpeciesReference* 
Model::getSpeciesReference (const std::string& sid)
{
  SpeciesReference* sr = NULL;
  for (unsigned int i = 0; i < getNumReactions(); i++)
  {
    sr = getReaction(i)->getReactant(sid);
    if (sr != NULL) break;

    sr = getReaction(i)->getProduct(sid);
    if (sr != NULL) break;
  }

  return sr;
}

const SpeciesReference* 
Model::getSpeciesReference (const std::string& sid) const
{
  const SpeciesReference* sr = NULL;
  for (unsigned int i = 0; i < getNumReactions(); i++)
  {
    sr = getReaction(i)->getReactant(sid);
    if (sr != NULL) break;

    sr = getReaction(i)->getProduct(sid);
    if (sr != NULL) break;
  }

  return sr;
}


/*
 * @return the nth Event of this Model.
 */
const Event*
Model::getEvent (unsigned int n) const
{
  return static_cast<const Event*>( mEvents.get(n) );
}


/*
 * @return the nth Event of this Model.
 */
Event*
Model::getEvent (unsigned int n)
{
  return static_cast<Event*>( mEvents.get(n) );
}


/*
 * @return the Event in this Model with the given id or @c NULL if no such
 * Event exists.
 */
const Event*
Model::getEvent (const std::string& sid) const
{
  return static_cast<const Event*>( mEvents.get(sid) );
}


/*
 * @return the Event in this Model with the given id or @c NULL if no such
 * Event exists.
 */
Event*
Model::getEvent (const std::string& sid)
{
  return static_cast<Event*>( mEvents.get(sid) );
}


/*
 * @return the number of FunctionDefinitions in this Model.
 */
unsigned int
Model::getNumFunctionDefinitions () const
{
  return mFunctionDefinitions.size();
}


/*
 * @return the number of UnitDefinitions in this Model.
 */
unsigned int
Model::getNumUnitDefinitions () const
{
  return mUnitDefinitions.size();
}


/*
 * @return the number of CompartmentTypes in this Model.
 */
unsigned int
Model::getNumCompartmentTypes () const
{
  return mCompartmentTypes.size();
}


/*
 * @return the number of SpeciesTypes in this Model.
 */
unsigned int
Model::getNumSpeciesTypes () const
{
  return mSpeciesTypes.size();
}


/*
 * @return the number of Compartments in this Model.
 */
unsigned int
Model::getNumCompartments () const
{
  return mCompartments.size();
}


/*
 * @return the number of Species in this Model.
 */
unsigned int
Model::getNumSpecies () const
{
  return mSpecies.size();
}


/*
 * @return the number of Species in this Model with boundaryCondition set
 * to true.
 */
unsigned int
Model::getNumSpeciesWithBoundaryCondition () const 
{
  unsigned int count = 0;

  for(unsigned int i = 0; i < mSpecies.size(); i++)
  {
    if (getSpecies(i)->getBoundaryCondition())
      count++;
  }
  return count;
}


/*
 * @return the number of Parameters in this Model.  Parameters defined in
 * KineticLaws are not included.
 */
unsigned int
Model::getNumParameters () const
{
  return mParameters.size();
}


/*
 * @return the number of InitialAssignments in this Model.
 */
unsigned int
Model::getNumInitialAssignments () const
{
  return mInitialAssignments.size();
}


/*
 * @return the number of Rules in this Model.
 */
unsigned int
Model::getNumRules () const
{
  return mRules.size();
}


/*
 * @return the number of Constraints in this Model.
 */
unsigned int
Model::getNumConstraints () const
{
  return mConstraints.size();
}


/*
 * @return the number of Reactions in this Model.
 */
unsigned int
Model::getNumReactions () const
{
  return mReactions.size();
}


/*
 * @return the number of Events in this Model.
 */
unsigned int
Model::getNumEvents () const
{
  return mEvents.size();
}

int Model::removeFromParentAndDelete()
{
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) return LIBSBML_OPERATION_FAILED;
  SBMLDocument* parentDoc= static_cast<SBMLDocument*>(parent);
  if (parentDoc== NULL) return LIBSBML_OPERATION_FAILED;
  return parentDoc->setModel(NULL);
}


/** @cond doxygen-libsbml-internal */
/*
 * @return true if the given ASTNode is a boolean.  Often times, this
 * question can be answered with the ASTNode's own isBoolean() method,
 * but if the AST is an expression that calls a function defined in the
 * Model's ListOf FunctionDefinitions, the model is needed for lookup
 * context.
 */
LIBSBML_EXTERN
bool
Model::isBoolean (const ASTNode* node) const
{
  if (node == NULL)
  {
    return false;
  }

  else if ( node->isBoolean() )
  {
    return true;
  }

  else if (node->getType() == AST_FUNCTION)
  {
    const FunctionDefinition* fd = getFunctionDefinition( node->getName() );

    if (fd != NULL && fd->isSetMath())
    {
      return isBoolean( fd->getMath()->getRightChild() );
    }
    else
    {
      return false;
    }
  }

  else if (node->getType() == AST_FUNCTION_PIECEWISE)
  {
    for (unsigned int c = 0; c < node->getNumChildren(); c += 2)
    {
      if ( !isBoolean( node->getChild(c) ) ) return false;
    }

    return true;
  }

  return false;
}
/** @endcond */

/** @cond doxygen-libsbml-internal */

/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Model::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mFunctionDefinitions.setSBMLDocument(d);
  mUnitDefinitions    .setSBMLDocument(d);
  mCompartmentTypes   .setSBMLDocument(d);
  mSpeciesTypes       .setSBMLDocument(d);
  mCompartments       .setSBMLDocument(d);
  mSpecies            .setSBMLDocument(d);
  mParameters         .setSBMLDocument(d);
  mInitialAssignments .setSBMLDocument(d);
  mRules              .setSBMLDocument(d);
  mConstraints        .setSBMLDocument(d);
  mReactions          .setSBMLDocument(d);
  mEvents             .setSBMLDocument(d);
  
}

/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
  */
void
Model::connectToChild()
{
	  mFunctionDefinitions.connectToParent(this);
	  mUnitDefinitions    .connectToParent(this);
	  mCompartmentTypes   .connectToParent(this);
	  mSpeciesTypes       .connectToParent(this);
	  mCompartments       .connectToParent(this);
	  mSpecies            .connectToParent(this);
	  mParameters         .connectToParent(this);
	  mInitialAssignments .connectToParent(this);
	  mRules              .connectToParent(this);
	  mConstraints        .connectToParent(this);
	  mReactions          .connectToParent(this);
	  mEvents             .connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePackage function)
 */
void 
Model::enablePackageInternal(const std::string& pkgURI, 
                             const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mFunctionDefinitions.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mUnitDefinitions    .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mCompartmentTypes   .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mSpeciesTypes       .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mCompartments       .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mSpecies            .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mParameters         .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mInitialAssignments .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mRules              .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mConstraints        .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mReactions          .enablePackageInternal(pkgURI,pkgPrefix,flag);
  mEvents             .enablePackageInternal(pkgURI,pkgPrefix,flag);
}

/** @endcond */


/*
 * @return the typecode (int) of this SBML object or SBML_UNKNOWN
 * (default).
 *
 * @see getElementName()
 */
int
Model::getTypeCode () const
{
  return SBML_MODEL;
}


/*
 * @return the name of this element ie "model".
 
 */
const string&
Model::getElementName () const
{
  static const string name = "model";
  return name;
}


bool 
Model::hasRequiredElements() const
{
  bool allPresent = true;

  /* required attributes for model: compart(L1); species (L1V1)
   * reaction (L1V1)
   */

  if (getLevel() == 1)
  {
    if (getNumCompartments() == 0)
      allPresent = false;

    if (getVersion() == 1)
    {
      if (getNumSpecies() == 0)
        allPresent = false;
      if (getNumReactions() == 0)
        allPresent = false;
    }
  }
  return allPresent;
}


/**
 * Removes the nth FunctionDefinition object from this Model object and
 * returns a pointer to it.
 */
FunctionDefinition* 
Model::removeFunctionDefinition (unsigned int n)
{
  return mFunctionDefinitions.remove(n);
}


/**
 * Removes the FunctionDefinition object with the given identifier from this Model
 * object and returns a pointer to it.
 */
FunctionDefinition* 
Model::removeFunctionDefinition (const std::string& sid)
{
  return mFunctionDefinitions.remove(sid);
}


/**
 * Removes the nth UnitDefinition object from this Model object and
 * returns a pointer to it.
 */
UnitDefinition*
Model::removeUnitDefinition (unsigned int n)
{
  return mUnitDefinitions.remove(n);
}


/**
 * Removes the UnitDefinition object with the given identifier from this Model
 * object and returns a pointer to it.
 */
UnitDefinition*
Model::removeUnitDefinition (const std::string& sid)
{
  return mUnitDefinitions.remove(sid);
}


/**
 * Removes the nth CompartmentType object from this Model object and
 * returns a pointer to it.
 */
CompartmentType*
Model::removeCompartmentType (unsigned int n)
{
  return mCompartmentTypes.remove(n);
}


/**
 * Removes the CompartmentType object with the given identifier from this Model
 * object and returns a pointer to it.
 */
CompartmentType*
Model::removeCompartmentType (const std::string& sid)
{
  return mCompartmentTypes.remove(sid);
}


/**
 * Removes the nth SpeciesType object from this Model object and
 * returns a pointer to it.
 */
SpeciesType*
Model::removeSpeciesType (unsigned int n)
{
  return mSpeciesTypes.remove(n);
}


/**
 * Removes the SpeciesType object with the given identifier from this Model
 * object and returns a pointer to it.
 */
SpeciesType*
Model::removeSpeciesType (const std::string& sid)
{
  return mSpeciesTypes.remove(sid);
}


/**
 * Removes the nth Compartment object from this Model object and
 * returns a pointer to it.
 */
Compartment*
Model::removeCompartment (unsigned int n)
{
  return mCompartments.remove(n);
}


/**
 * Removes the Compartment object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Compartment*
Model::removeCompartment (const std::string& sid)
{
  return mCompartments.remove(sid);
}


/**
 * Removes the nth Species object from this Model object and
 * returns a pointer to it.
 */
Species*
Model::removeSpecies (unsigned int n)
{
  return mSpecies.remove(n);
}


/**
 * Removes the Species object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Species*
Model::removeSpecies (const std::string& sid)
{
  return mSpecies.remove(sid);
}


/**
 * Removes the nth Parameter object from this Model object and
 * returns a pointer to it.
 */
Parameter*
Model::removeParameter (unsigned int n)
{
  return mParameters.remove(n);
}


/**
 * Removes the Parameter object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Parameter*
Model::removeParameter (const std::string& sid)
{
  return mParameters.remove(sid);
}


/**
 * Removes the nth InitialAssignment object from this Model object and
 * returns a pointer to it.
 */
InitialAssignment*
Model::removeInitialAssignment (unsigned int n)
{
  return mInitialAssignments.remove(n);
}


/**
 * Removes the InitialAssignment object with the given identifier from this Model
 * object and returns a pointer to it.
 */
InitialAssignment*
Model::removeInitialAssignment (const std::string& sid)
{
  return mInitialAssignments.remove(sid);
}


/**
 * Removes the nth Rule object from this Model object and
 * returns a pointer to it.
 */
Rule*
Model::removeRule (unsigned int n)
{
  return mRules.remove(n);
}


/**
 * Removes the Rule object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Rule*
Model::removeRule (const std::string& sid)
{
  return mRules.remove(sid);
}


/**
 * Removes the nth Constraint object from this Model object and
 * returns a pointer to it.
 */
Constraint*
Model::removeConstraint (unsigned int n)
{
  return mConstraints.remove(n);
}


/**
 * Removes the nth Reaction object from this Model object and
 * returns a pointer to it.
 */
Reaction*
Model::removeReaction (unsigned int n)
{
  return mReactions.remove(n);
}


/**
 * Removes the Reaction object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Reaction*
Model::removeReaction (const std::string& sid)
{
  return mReactions.remove(sid);
}


/**
 * Removes the nth Event object from this Model object and
 * returns a pointer to it.
 */
Event*
Model::removeEvent (unsigned int n)
{
  return mEvents.remove(n);
}


/**
 * Removes the Event object with the given identifier from this Model
 * object and returns a pointer to it.
 */
Event*
Model::removeEvent (const std::string& sid)
{
  return mEvents.remove(sid);
}

int 
Model::appendFrom(const Model* model)
{
  int ret = LIBSBML_OPERATION_SUCCESS;
  ret = mFunctionDefinitions.appendFrom(&model->mFunctionDefinitions);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mUnitDefinitions    .appendFrom(&model->mUnitDefinitions);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mCompartmentTypes   .appendFrom(&model->mCompartmentTypes);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mSpeciesTypes       .appendFrom(&model->mSpeciesTypes);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mCompartments       .appendFrom(&model->mCompartments);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mSpecies            .appendFrom(&model->mSpecies);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mParameters         .appendFrom(&model->mParameters);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mInitialAssignments .appendFrom(&model->mInitialAssignments);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mRules              .appendFrom(&model->mRules);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mConstraints        .appendFrom(&model->mConstraints);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mReactions          .appendFrom(&model->mReactions);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  ret = mEvents             .appendFrom(&model->mEvents);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;

  for (size_t i=0; i < mPlugins.size(); i++) {
    ret = mPlugins[i]->appendFrom(model);
    if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  }
  return ret;
}

void
Model::renameSIdRefs(std::string oldid, std::string newid)
{
  if (isSetConversionFactor()) {
    if (getConversionFactor()==oldid) {
      setConversionFactor(newid);
    }
  }
}

void 
Model::renameUnitSIdRefs(std::string oldid, std::string newid)
{
  if (mSubstanceUnits == oldid) mSubstanceUnits = newid;
  if (mTimeUnits == oldid)      mTimeUnits = newid;
  if (mVolumeUnits == oldid)    mVolumeUnits = newid;
  if (mAreaUnits == oldid)      mAreaUnits = newid;
  if (mLengthUnits == oldid)    mLengthUnits = newid;
  if (mExtentUnits == oldid)    mExtentUnits = newid;
}

/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read (and store) XHTML,
 * MathML, etc. directly from the XMLInputStream.
 *
 * @return true if the subclass read from the stream, false otherwise.
 */
bool
Model::readOtherXML (XMLInputStream& stream)
{
  bool          read = false;
  const string& name = stream.peek().getName();

  // This has to do additional work for reading annotations, so the code
  // here is copied and expanded from SBase::readNotes().

  if (name == "annotation")
  {
//    XMLNode* new_annotation = NULL;
    /* if annotation already exists then it is an error 
     */
    if (mAnnotation != NULL)
    {
      if (getLevel() < 3) 
      {
        logError(NotSchemaConformant, getLevel(), getVersion(),
	        "Only one <annotation> element is permitted inside a "
	        "particular containing element.");
      }
      else
      {
        logError(MultipleAnnotations, getLevel(), getVersion());
      }
    }

    delete mAnnotation;
    mAnnotation = new XMLNode(stream);
    checkAnnotation();
    if (mCVTerms != NULL)
    {
      unsigned int size = mCVTerms->getSize();
      while (size--) delete static_cast<CVTerm*>( mCVTerms->remove(0) );
      delete mCVTerms;
    }
    mCVTerms = new List();
    delete mHistory;
    if (RDFAnnotationParser::hasHistoryRDFAnnotation(mAnnotation))
    {
      mHistory = RDFAnnotationParser::parseRDFAnnotation(mAnnotation, 
                                            getMetaId().c_str(), &(stream));

      if (mHistory != NULL && mHistory->hasRequiredAttributes() == false)
      {
        logError(RDFNotCompleteModelHistory, getLevel(), getVersion(),
          "An invalid ModelHistory element has been stored.");
      }
      setModelHistory(mHistory);
    }
    else
      mHistory = NULL;
    if (RDFAnnotationParser::hasCVTermRDFAnnotation(mAnnotation))
      RDFAnnotationParser::parseRDFAnnotation(mAnnotation, mCVTerms, 
                                                getMetaId().c_str(), &(stream));
//    new_annotation = RDFAnnotationParser::deleteRDFAnnotation(mAnnotation);
//    delete mAnnotation;
//    mAnnotation = new_annotation;
	
    read = true;
  }

  /* ------------------------------
   *
   *   (EXTENSION)
   *
   * ------------------------------ */
  if ( SBase::readOtherXML(stream) )
    read = true;

  return read;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or @c NULL if the token was not recognized.
 */
SBase*
Model::createObject (XMLInputStream& stream)
{
  const string& name   = stream.peek().getName();
  SBase*        object = NULL;
  unsigned int level = getLevel();
  unsigned int version = getVersion();

  /* dont create objects for wrong levels/versions */
  if (name == "listOfFunctionDefinitions")
  {
    if (level == 1)
    {
      return NULL;
    }

    if (mFunctionDefinitions.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mFunctionDefinitions;
  }

  else if ( name == "listOfUnitDefinitions"    ) 
  {
    if (mUnitDefinitions.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mUnitDefinitions;
  }

  else if ( name == "listOfCompartmentTypes"   ) 
  {
    if (level == 1  
      || (level == 2 && version == 1)
      || level == 3)
    {
      return NULL;
    }
    if (mCompartmentTypes.size() != 0)
    {
      logError(NotSchemaConformant);
    }
    object = &mCompartmentTypes;
  }

  else if ( name == "listOfSpeciesTypes"       ) 
  {
    if (level == 1  
      || (level == 2 && version == 1)
      || level == 3)
    {
      return NULL;
    }
    if (mSpeciesTypes.size() != 0)
    {
      logError(NotSchemaConformant);
    }
    object = &mSpeciesTypes;
  }

  else if ( name == "listOfCompartments"       ) 
  {
    if (mCompartments.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mCompartments;
  }
  
  else if ( name == "listOfSpecies"            ) 
  {
    if (mSpecies.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mSpecies;
  }

  else if ( name == "listOfParameters"         ) 
  {
    if (mParameters.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mParameters;
  }

  else if ( name == "listOfInitialAssignments" ) 
  {
    if (level == 1  || (level == 2 && version == 1))
    {
      return NULL;
    }
    if (mInitialAssignments.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mInitialAssignments;
  }

  else if ( name == "listOfRules"              ) 
  {
    if (mRules.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mRules;
  }

  else if ( name == "listOfConstraints"        ) 
  {
    if (level == 1  || (level == 2 && version == 1))
    {
      return NULL;
    }
    if (mConstraints.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mConstraints;
  }

  else if ( name == "listOfReactions"          ) 
  {
    if (mReactions.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mReactions;
  }

  else if ( name == "listOfEvents"             ) 
  {
    if (level == 1)
    {
      return NULL;
    }
    if (mEvents.size() != 0)
    {
      if (getLevel() < 3)
      {
        logError(NotSchemaConformant);
      }
      else
      {
        logError(OneOfEachListOf);
      }
    }
    object = &mEvents;
  }

  else if ( level == 1 && version == 1 )
  {
    if (name == "listOfSpecie") 
    {
      if (mSpecies.size() != 0)
      {
        logError(NotSchemaConformant);
      }
      object = &mSpecies;
    }
  }

  return object;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
Model::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  switch (level)
  {
  case 1:
    attributes.add("name");
    break;
  case 2:
    attributes.add("name");
    attributes.add("id");
    if (version == 2)
    {
      attributes.add("sboTerm");
    }
    break;
  case 3:
  default:
    attributes.add("name");
    attributes.add("id");
    attributes.add("substanceUnits");
    attributes.add("timeUnits");
    attributes.add("volumeUnits");
    attributes.add("areaUnits");
    attributes.add("lengthUnits");
    attributes.add("extentUnits");
    attributes.add("conversionFactor");
    break;
  }
}

/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Model::readAttributes (const XMLAttributes& attributes,
                       const ExpectedAttributes& expectedAttributes)
{
  const unsigned int level   = getLevel  ();

  SBase::readAttributes(attributes, expectedAttributes);

  switch (level)
  {
  case 1:
    readL1Attributes(attributes);
    break;
  case 2:
    readL2Attributes(attributes);
    break;
  case 3:
  default:
    readL3Attributes(attributes);
    break;
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Model::readL1Attributes (const XMLAttributes& attributes)
{
  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  //
  // name: SName  { use="optional" }  (L1v1, L1v2)
  //   id: SId    { use="optional" }  (L2v1 -> )
  //
  bool assigned = attributes.readInto("name", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.size() == 0)
  {
    logEmptyString("name", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);

}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Model::readL2Attributes (const XMLAttributes& attributes)
{
  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  //
  //   id: SId    { use="optional" }  (L2v1 -> )
  //
  bool assigned = attributes.readInto("id", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.size() == 0)
  {
    logEmptyString("id", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);

  //
  // name: string  { use="optional" }  (L2v1 ->)
  //
  attributes.readInto("name", mName, getErrorLog(), false, getLine(), getColumn());

  //
  // sboTerm: SBOTerm { use="optional" }  (L2v2 ->)
  //
  if (version == 2) 
    mSBOTerm = SBO::readTerm(attributes, this->getErrorLog(), level, version,
				getLine(), getColumn());
  
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Model::readL3Attributes (const XMLAttributes& attributes)
{
  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  //
  //   id: SId    { use="optional" }  (L2v1 -> )
  //
  bool assigned = attributes.readInto("id", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.size() == 0)
  {
    logEmptyString("id", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);


  //
  // name: string  { use="optional" }  (L2v1 ->)
  //
  attributes.readInto("name", mName, getErrorLog(), false, getLine(), getColumn());

  //
  // substanceUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("substanceUnits", mSubstanceUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mSubstanceUnits.size() == 0)
  {
    logEmptyString("substanceUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mSubstanceUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // timeUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("timeUnits", mTimeUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mTimeUnits.size() == 0)
  {
    logEmptyString("timeUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mTimeUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // volumeUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("volumeUnits", mVolumeUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mVolumeUnits.size() == 0)
  {
    logEmptyString("volumeUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mVolumeUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // areaUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("areaUnits", mAreaUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mAreaUnits.size() == 0)
  {
    logEmptyString("areaUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mAreaUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // lengthUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("lengthUnits", mLengthUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mLengthUnits.size() == 0)
  {
    logEmptyString("lengthUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mLengthUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // extentUnits: string  { use="optional" }  (L3v1 ->)
  //
  assigned = attributes.readInto("extentUnits", mExtentUnits, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mExtentUnits.size() == 0)
  {
    logEmptyString("extentUnits", level, version, "<model>");
  }
  if (!SyntaxChecker::isValidInternalUnitSId(mExtentUnits))
  {
    logError(InvalidUnitIdSyntax);
  }

  //
  // conversionFactor: string  { use="optional" }  (L3v1 ->)
  //
  attributes.readInto("conversionFactor", mConversionFactor, getErrorLog(), false, getLine(), getColumn());
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream.  Be sure to call your parents implementation
 * of this method as well.
 */
void
Model::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  //
  // sboTerm: SBOTerm { use="optional" }  (L2v2->)
  //
  // sboTerm for L2V3 or later is written in SBase::writeAttributes()
  //
  if ( (level == 2) && (version == 2) )
  {
    SBO::writeTerm(stream, mSBOTerm);
  }

  //
  // name: SName   { use="required" }  (L1v1, L1v2)
  //   id: SId     { use="required" }  (L2v1->)
  //
  const string id = (level == 1) ? "name" : "id";
  stream.writeAttribute(id, mId);

  if (level > 1)
  {
    //
    // name: string  { use="optional" }  (L2v1->)
    //
    stream.writeAttribute("name", mName);
  }

  if (level > 2)
  {
    //
    // substanceUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("substanceUnits", mSubstanceUnits);

    //
    // timeUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("timeUnits", mTimeUnits);

    //
    // volumeUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("volumeUnits", mVolumeUnits);

    //
    // areaUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("areaUnits", mAreaUnits);

    //
    // lengthUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("lengthUnits", mLengthUnits);

    //
    // extentUnits: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("extentUnits", mExtentUnits);

    //
    // conversionFactor: string  { use="optional" }  (L3v1 ->)
    //
    stream.writeAttribute("conversionFactor", mConversionFactor);

  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
Model::writeElements (XMLOutputStream& stream) const
{
  if ( mNotes ) stream << *mNotes;
  Model * m = const_cast <Model *> (this);
  m->syncAnnotation();
  if ( mAnnotation ) stream << *mAnnotation;

  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  if (level > 1 && getNumFunctionDefinitions() > 0)
  {
    mFunctionDefinitions.write(stream);
  }

  if ( getNumUnitDefinitions() > 0 ) mUnitDefinitions.write(stream);

  if (level == 2 && version > 1)
  {
    if ( getNumCompartmentTypes() > 0 ) mCompartmentTypes.write(stream);
    if ( getNumSpeciesTypes    () > 0 ) mSpeciesTypes    .write(stream);
  }

  if ( getNumCompartments() > 0 ) mCompartments.write(stream);
  if ( getNumSpecies     () > 0 ) mSpecies     .write(stream);
  if ( getNumParameters  () > 0 ) mParameters  .write(stream);

  if (level > 2 || (level == 2 && version > 1))
  {
    if ( getNumInitialAssignments() > 0 ) mInitialAssignments.write(stream);
  }

  if ( getNumRules() > 0 ) mRules.write(stream);

  if (level > 2 || (level == 2 && version > 1))
  {
    if ( getNumConstraints() > 0 ) mConstraints.write(stream);
  }

  if ( getNumReactions() > 0 ) mReactions.write(stream);

  if (level > 1 && getNumEvents () > 0 )
  {
    mEvents.write(stream);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the ordinal position of the element with respect to its siblings
 * or -1 (default) to indicate the position is not significant.
 */
int
Model::getElementPosition () const
{
  return 2;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * Populates the ListFormulaDataUnits with the units of each 
  * set of math encountered in the model
  */
void
Model::populateListFormulaUnitsData()
{
  /* remove list if it already exists */
  if (mFormulaUnitsData != NULL)
  {  
    unsigned int size = mFormulaUnitsData->getSize();
    while (size--) 
      delete static_cast<FormulaUnitsData*>( mFormulaUnitsData->remove(0) );
    delete mFormulaUnitsData;
    mFormulaUnitsData = NULL;
  }

  unsigned int n, j;
  char newId[12];
  std::string eaId;
  std::string newID;
  unsigned int countAlg = 0, countEvents = 0;
  
  Compartment * c;
  Species * s;
  Parameter * p;
  Rule * r;
  InitialAssignment * ia;
  Event * e;
  EventAssignment * ea;
  Reaction * react;
  SpeciesReference * sr;

  UnitFormulaFormatter *unitFormatter = new UnitFormulaFormatter(this);
  FormulaUnitsData *fud;
  UnitDefinition *ud = new UnitDefinition(getSBMLNamespaces());
  Unit *u;
  Unit *uFromModel;

/* need to put in default units or global units assigned */
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("substance");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    // unless substance has been overridden
    if (getUnitDefinition("substance") != NULL)
    {
      for (n = 0; n < getUnitDefinition("substance")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("substance")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_MOLE);
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string substanceUnits = getSubstanceUnits();
    if (UnitKind_isValidUnitKindString(substanceUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(substanceUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(substanceUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(substanceUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(substanceUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);

/* volume */
  ud = new UnitDefinition(getSBMLNamespaces());
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("volume");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    // unless volume has been overridden
    if (getUnitDefinition("volume") != NULL)
    {
      for (n = 0; n < getUnitDefinition("volume")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("volume")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_LITRE);
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string volumeUnits = getVolumeUnits();
    if (UnitKind_isValidUnitKindString(volumeUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(volumeUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(volumeUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(volumeUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(volumeUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);


/* area */
  ud = new UnitDefinition(getSBMLNamespaces());
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("area");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    // unless area has been overridden
    if (getUnitDefinition("area") != NULL)
    {
      for (n = 0; n < getUnitDefinition("area")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("area")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_METRE);
      u->initDefaults();
      u->setExponent(2);
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string areaUnits = getAreaUnits();
    if (UnitKind_isValidUnitKindString(areaUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(areaUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(areaUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(areaUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(areaUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);


/* length */
  ud = new UnitDefinition(getSBMLNamespaces());
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("length");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    // unless length has been overridden
    if (getUnitDefinition("length") != NULL)
    {
      for (n = 0; n < getUnitDefinition("length")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("length")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_METRE);
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string lengthUnits = getLengthUnits();
    if (UnitKind_isValidUnitKindString(lengthUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(lengthUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(lengthUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(lengthUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(lengthUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);


/* time */
  ud = new UnitDefinition(getSBMLNamespaces());
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("time");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    // unless time has been overridden
    if (getUnitDefinition("time") != NULL)
    {
      for (n = 0; n < getUnitDefinition("time")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("time")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_SECOND);
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string timeUnits = getTimeUnits();
    if (UnitKind_isValidUnitKindString(timeUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(timeUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(timeUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(timeUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(timeUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);


/* extent */
  ud = new UnitDefinition(getSBMLNamespaces());
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("extent");
  fud->setComponentTypecode(SBML_MODEL);
  if (getLevel() < 3)
  {
    fud->setContainsParametersWithUndeclaredUnits(true);
    fud->setCanIgnoreUndeclaredUnits(false);
    //// unless extent has been overridden
    //if (getUnitDefinition("extent") != NULL)
    //{
    //  for (n = 0; n < getUnitDefinition("extent")->getNumUnits(); n++)
    //  {
    //    // need to prevent level/version mismatches
    //    // ud will have default level and veersion
    //    uFromModel = getUnitDefinition("extent")->getUnit(n);
    //    if (uFromModel  != NULL)
    //    {
    //      u = new Unit(uFromModel->getSBMLNamespaces());
    //      u->setKind(uFromModel->getKind());
    //      u->setExponent(uFromModel->getExponent());
    //      u->setScale(uFromModel->getScale());
    //      u->setMultiplier(uFromModel->getMultiplier());
    //      ud->addUnit(u);
    //      delete u;
    //    }
    //  }
    //}
    //else
    //{
    //  u = new Unit(getSBMLNamespaces());
    //  u->setKind(UNIT_KIND_LITRE);
    //  u->initDefaults();
    //  ud->addUnit(u);
    //  delete u;
    //}
  }
  else
  {
    /* in L3 the units will be possibly not declared at all !
     */
    std::string extentUnits = getExtentUnits();
    if (UnitKind_isValidUnitKindString(extentUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(extentUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(extentUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(extentUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(extentUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);


  /* put in the dafult units of substance per time */
  /* but not in L3 */
  fud = createFormulaUnitsData();
  fud->setUnitReferenceId("subs_per_time");
  fud->setComponentTypecode(SBML_UNKNOWN);
  ud = new UnitDefinition(getSBMLNamespaces());

  if (getLevel() < 3)
  {
    // unless substance has been overridden
    if (getUnitDefinition("substance") != NULL)
    {
      for (n = 0; n < getUnitDefinition("substance")->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition("substance")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_MOLE);
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }

    if (getUnitDefinition("time") != NULL)
    {
      for (n = 0; n < getUnitDefinition("time")->getNumUnits(); n++)
      {
        uFromModel = getUnitDefinition("time")->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          u->setExponent(u->getExponent() * -1);
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UNIT_KIND_SECOND);
      u->initDefaults();
      u->setExponent(-1);
      ud->addUnit(u);
      delete u;
    }
  }
  else
  {
    /* in L3 the units will be extent per time
     * or possibly not declared at all !
     */
    std::string extentUnits = getExtentUnits();
    if (UnitKind_isValidUnitKindString(extentUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(extentUnits.c_str()));
      u->initDefaults();
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(extentUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(extentUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(extentUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent());
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }

    std::string timeUnits = getTimeUnits();
    if (UnitKind_isValidUnitKindString(timeUnits.c_str(), 
                                       getLevel(), getVersion()))
    {
      u = new Unit(getSBMLNamespaces());
      u->setKind(UnitKind_forName(timeUnits.c_str()));
      u->initDefaults();
      u->setExponent(-1);
      ud->addUnit(u);
      delete u;
    }
    else if (getUnitDefinition(timeUnits) != NULL)
    {
      for (n = 0; n < getUnitDefinition(timeUnits)->getNumUnits(); n++)
      {
        // need to prevent level/version mismatches
        // ud will have default level and veersion
        uFromModel = getUnitDefinition(timeUnits)->getUnit(n);
        if (uFromModel  != NULL)
        {
          u = new Unit(uFromModel->getSBMLNamespaces());
          u->setKind(uFromModel->getKind());
          u->setExponent(uFromModel->getExponent() * -1);
          u->setScale(uFromModel->getScale());
          u->setMultiplier(uFromModel->getMultiplier());
          ud->addUnit(u);
          delete u;
        }
      }
    }
    else
    {
      fud->setContainsParametersWithUndeclaredUnits(true);
      fud->setCanIgnoreUndeclaredUnits(false);
    }
  }
  fud->setUnitDefinition(ud);

  /* get unit data from each compartment 
   * this is necessary for validation
   */
  for (n = 0; n < getNumCompartments(); n++)
  {
    c = getCompartment(n);
    fud = createFormulaUnitsData();
    fud->setUnitReferenceId(c->getId());
    fud->setComponentTypecode(SBML_COMPARTMENT);
    ud = unitFormatter->getUnitDefinitionFromCompartment(c);
    fud->setUnitDefinition(ud);

    if (ud != NULL)
    {
      /* in L3 this will be different */
      if (getLevel() < 3)
      {
        ud = new UnitDefinition(getSBMLNamespaces());
        for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
          ud->addUnit(fud->getUnitDefinition()->getUnit(j));
        // unless time has been overridden
        if (getUnitDefinition("time") != NULL)
        {
          for (unsigned int ii = 0; ii < getUnitDefinition("time")->getNumUnits(); ii++)
          {
            // need to prevent level/version mismatches
            // ud will have default level and veersion
            uFromModel = getUnitDefinition("time")->getUnit(ii);
            if (uFromModel  != NULL)
            {
              u = new Unit(uFromModel->getSBMLNamespaces());
              u->setKind(uFromModel->getKind());
              u->setExponent(-1 * uFromModel->getExponent());
              u->setScale(uFromModel->getScale());
              u->setMultiplier(uFromModel->getMultiplier());
              ud->addUnit(u);
              delete u;
            }
          }
        }
        else
        {
          u = new Unit(getSBMLNamespaces());
          u->setKind(UNIT_KIND_SECOND);
          u->initDefaults();
          u->setExponent(-1);
          ud->addUnit(u);
          delete u;
        }
        UnitDefinition::simplify(ud);
        //u = new Unit(getSBMLNamespaces());
        //u->setKind(UNIT_KIND_SECOND);
        //u->initDefaults();
        //u->setExponent(-1);
        //ud->addUnit(u);
        //delete u;
      }
      else
      {
        /* in L3 there are only defined time units IF the model
         * has the timeUnits attribute set
         */
        if (!isSetTimeUnits())
        {
          ud = new UnitDefinition(getSBMLNamespaces());
        }
        else
        {
          std::string units = getTimeUnits();
          UnitDefinition *tempUD = new UnitDefinition(getSBMLNamespaces());

          /* either unit name or unitDefinition id */
          if (UnitKind_isValidUnitKindString(units.c_str(), 
                                             getLevel(), getVersion()))
          {
            u = new Unit(getSBMLNamespaces());
            u->setKind(UnitKind_forName(units.c_str()));
            u->initDefaults();
            u->setExponent(-1);

            tempUD->addUnit(u);
            delete u;
          }
          else
          {
            for (j = 0; j < getNumUnitDefinitions(); j++)
            {
              if (!strcmp(units.c_str(), getUnitDefinition(j)->getId().c_str()))
              {           
                for (unsigned int pp = 0; 
                  pp < getUnitDefinition(j)->getNumUnits(); pp++)
                {
                  u = new Unit(getSBMLNamespaces());
                  u->setKind(getUnitDefinition(j)->getUnit(pp)->getKind());
                  u->setMultiplier(getUnitDefinition(j)->getUnit(pp)
                                                   ->getMultiplier());
                  u->setScale(getUnitDefinition(j)->getUnit(pp)->getScale());
                  u->setExponent(-1 * (getUnitDefinition(j)->getUnit(pp)
                                       ->getExponent()));

                  tempUD->addUnit(u);

                  delete u;
                }
              }
            }
          }


          ud = new UnitDefinition(getSBMLNamespaces());
          
          for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
            ud->addUnit(fud->getUnitDefinition()->getUnit(j));
            
          for (j = 0; j < tempUD->getNumUnits(); j++)
            ud->addUnit(tempUD->getUnit(j));
            
          UnitDefinition::simplify(ud);
          delete tempUD;
        }

      }
      fud->setPerTimeUnitDefinition(ud);
    }
  }

  /* get unit data from each species 
   * this is necessary for validation
   */
  for (n=0; n < getNumSpecies(); n++)
  {
    s = getSpecies(n);
    fud = createFormulaUnitsData();
    fud->setUnitReferenceId(s->getId());
    fud->setComponentTypecode(SBML_SPECIES);
    /* if the species has not been given a compartment
     * this will blow up although it is caught by another rule
     */
    if (getCompartment(s->getCompartment()) == NULL)
      ud = NULL;
    else
      ud = unitFormatter->getUnitDefinitionFromSpecies(s);
    fud->setUnitDefinition(ud);
    
    if (ud != NULL)
    {
      if (getLevel() < 3)
      {
        ud = new UnitDefinition(getSBMLNamespaces());
        for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
          ud->addUnit(fud->getUnitDefinition()->getUnit(j));
        // unless time has been overridden
        if (getUnitDefinition("time") != NULL)
        {
          for (unsigned int ii = 0; ii < getUnitDefinition("time")->getNumUnits(); ii++)
          {
            // need to prevent level/version mismatches
            // ud will have default level and veersion
            uFromModel = getUnitDefinition("time")->getUnit(ii);
            if (uFromModel  != NULL)
            {
              u = new Unit(uFromModel->getSBMLNamespaces());
              u->setKind(uFromModel->getKind());
              u->setExponent(-1 * uFromModel->getExponent());
              u->setScale(uFromModel->getScale());
              u->setMultiplier(uFromModel->getMultiplier());
              ud->addUnit(u);
              delete u;
            }
          }
        }
        else
        {
          u = new Unit(getSBMLNamespaces());
          u->setKind(UNIT_KIND_SECOND);
          u->initDefaults();
          u->setExponent(-1);
          ud->addUnit(u);
          delete u;
        }
        UnitDefinition::simplify(ud);
        //u = new Unit(getSBMLNamespaces());
        //u->setKind(UNIT_KIND_SECOND);
        //u->initDefaults();
        //u->setExponent(-1);
        //ud->addUnit(u);
        //delete u;
      }
      else
      {
        /* in L3 there are only defined time units IF the model
         * has the timeUnits attribute set
         */
        if (!isSetTimeUnits())
        {
          ud = new UnitDefinition(getSBMLNamespaces());
        }
        else
        {
          std::string units = getTimeUnits();
          UnitDefinition *tempUD = new UnitDefinition(getSBMLNamespaces());

          /* either unit name or unitDefinition id */
          if (UnitKind_isValidUnitKindString(units.c_str(), getLevel(),
                                                            getVersion()))
          {
            u = new Unit(getSBMLNamespaces());
            u->setKind(UnitKind_forName(units.c_str()));
            u->initDefaults();
            u->setExponent(-1);

            tempUD->addUnit(u);
            delete u;
          }
          else
          {
            for (j = 0; j < getNumUnitDefinitions(); j++)
            {
              if (!strcmp(units.c_str(), getUnitDefinition(j)->getId().c_str()))
              {           
                for (unsigned int pp = 0; 
                  pp < getUnitDefinition(j)->getNumUnits(); pp++)
                {
                  u = new Unit(getSBMLNamespaces());
                  u->setKind(getUnitDefinition(j)->getUnit(pp)->getKind());
                  u->setMultiplier(getUnitDefinition(j)->getUnit(pp)
                                                   ->getMultiplier());
                  u->setScale(getUnitDefinition(j)->getUnit(pp)->getScale());
                  u->setExponent(-1 * (getUnitDefinition(j)->getUnit(pp)
                                       ->getExponent()));

                  tempUD->addUnit(u);

                  delete u;
                }
              }
            }
          }


          ud = new UnitDefinition(getSBMLNamespaces());
          
          for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
            ud->addUnit(fud->getUnitDefinition()->getUnit(j));
            
          for (j = 0; j < tempUD->getNumUnits(); j++)
            ud->addUnit(tempUD->getUnit(j));
            
          UnitDefinition::simplify(ud);
          delete tempUD;
        }

      }
      fud->setPerTimeUnitDefinition(ud);
    }
  }

  
  /* for L3 we need additional information relating to 
   * extent and conversion factors
   */
  if (getLevel() > 2)
  {
    for (n=0; n < getNumSpecies(); n++)
    {
      s = getSpecies(n);
      fud = createFormulaUnitsData();
      fud->setUnitReferenceId(s->getId()+"subs");
      fud->setComponentTypecode(SBML_SPECIES);
      ud = unitFormatter->getSpeciesSubstanceUnitDefinition(s);
      
      if (ud != NULL)
      {
        fud->setSpeciesSubstanceUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      }
    }
    for (n=0; n < getNumSpecies(); n++)
    {
      s = getSpecies(n);
      fud = createFormulaUnitsData();
      fud->setUnitReferenceId(s->getId()+"extent");
      fud->setComponentTypecode(SBML_SPECIES);
      ud = unitFormatter->getSpeciesExtentUnitDefinition(s);
      
      if (ud != NULL)
      {
        fud->setSpeciesExtentUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      }
    }

  }
  /* get unit data from each parameter    */
  for (n=0; n < getNumParameters(); n++)
  {
    p = getParameter(n);
    fud = createFormulaUnitsData();
    fud->setUnitReferenceId(p->getId());
    fud->setComponentTypecode(SBML_PARAMETER);
    unitFormatter->resetFlags();
    ud = unitFormatter->getUnitDefinitionFromParameter(p);
    fud->setUnitDefinition(ud);
    fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
    fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());

    if (ud != NULL)
    {
      if (getLevel() < 3)
      {
        ud = new UnitDefinition(getSBMLNamespaces());
        for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
          ud->addUnit(fud->getUnitDefinition()->getUnit(j));
        // unless time has been overridden
        if (getUnitDefinition("time") != NULL)
        {
          for (unsigned int ii = 0; ii < getUnitDefinition("time")->getNumUnits(); ii++)
          {
            // need to prevent level/version mismatches
            // ud will have default level and veersion
            uFromModel = getUnitDefinition("time")->getUnit(ii);
            if (uFromModel  != NULL)
            {
              u = new Unit(uFromModel->getSBMLNamespaces());
              u->setKind(uFromModel->getKind());
              u->setExponent(-1 * uFromModel->getExponent());
              u->setScale(uFromModel->getScale());
              u->setMultiplier(uFromModel->getMultiplier());
              ud->addUnit(u);
              delete u;
            }
          }
        }
        else
        {
          u = new Unit(getSBMLNamespaces());
          u->setKind(UNIT_KIND_SECOND);
          u->initDefaults();
          u->setExponent(-1);
          ud->addUnit(u);
          delete u;
        }
        //u = new Unit(getSBMLNamespaces());
        //u->setKind(UNIT_KIND_SECOND);
        //u->initDefaults();
        //u->setExponent(-1);
        //ud->addUnit(u);
        UnitDefinition::simplify(ud);
      }
      else
      {
        /* in L3 there are only defined time units IF the model
         * has the timeUnits attribute set
         */
        if (!isSetTimeUnits())
        {
          ud = new UnitDefinition(getSBMLNamespaces());
        }
        else
        {
          std::string units = getTimeUnits();
          UnitDefinition *tempUD = new UnitDefinition(getSBMLNamespaces());

          /* either unit name or unitDefinition id */
          if (UnitKind_isValidUnitKindString(units.c_str(), getLevel(), 
                                                            getVersion()))
          {
            u = new Unit(getSBMLNamespaces());
            u->setKind(UnitKind_forName(units.c_str()));
            u->initDefaults();
            u->setExponent(-1);

            tempUD->addUnit(u);
            delete u;
          }
          else
          {
            for (j = 0; j < getNumUnitDefinitions(); j++)
            {
              if (!strcmp(units.c_str(), getUnitDefinition(j)->getId().c_str()))
              {           
                for (unsigned int pp = 0; 
                  pp < getUnitDefinition(j)->getNumUnits(); pp++)
                {
                  u = new Unit(getSBMLNamespaces());
                  u->setKind(getUnitDefinition(j)->getUnit(pp)->getKind());
                  u->setMultiplier(getUnitDefinition(j)->getUnit(pp)
                                                   ->getMultiplier());
                  u->setScale(getUnitDefinition(j)->getUnit(pp)->getScale());
                  u->setExponent(-1 * (getUnitDefinition(j)->getUnit(pp)
                                       ->getExponent()));

                  tempUD->addUnit(u);

                  delete u;
                }
              }
            }
          }


          ud = new UnitDefinition(getSBMLNamespaces());
          
          for (j = 0; j < fud->getUnitDefinition()->getNumUnits(); j++)
            ud->addUnit(fud->getUnitDefinition()->getUnit(j));
            
          for (j = 0; j < tempUD->getNumUnits(); j++)
            ud->addUnit(tempUD->getUnit(j));
            
          UnitDefinition::simplify(ud);
          delete tempUD;
        }

      }
      fud->setPerTimeUnitDefinition(ud);
   }
  }

   /* get units returned by the formula given for each initial assignment
   */
  for (n=0; n < getNumInitialAssignments(); n++)
  {
    ia = getInitialAssignment(n);
    fud = createFormulaUnitsData();
    fud->setUnitReferenceId(ia->getSymbol());
    fud->setComponentTypecode(SBML_INITIAL_ASSIGNMENT);
    unitFormatter->resetFlags();
    if (ia->isSetMath())
    {
      ud = unitFormatter->getUnitDefinition(ia->getMath());
      fud->setUnitDefinition(ud);
      fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
      fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
    }
    else
    {
      ud = NULL;
      fud->setUnitDefinition(ud);
    }
  }
 /* get units returned by the formula given for each rule
   */
  for (n=0; n < getNumRules(); n++)
  {
    r = getRule(n);
    fud = createFormulaUnitsData();
    if (r->getTypeCode() == SBML_ALGEBRAIC_RULE)
    {
      sprintf(newId, "alg_rule_%u", countAlg);
      newID.assign(newId);
      fud->setUnitReferenceId(newID);
      r->setInternalId(newID);
      static_cast <AlgebraicRule *> (r)->setInternalIdOnly();
      countAlg++;
    }
    else
    {
      fud->setUnitReferenceId(r->getVariable());
    }
    fud->setComponentTypecode(r->getTypeCode());
    unitFormatter->resetFlags();
    if (r->isSetMath())
    {
      ud = unitFormatter->getUnitDefinition(r->getMath());
      fud->setUnitDefinition(ud);
      fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
      fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
    }
    else
    {
      ud = NULL;
      fud->setUnitDefinition(ud);
    }
  }


  /**
   * math may occur in reactions in kineticLaw or as stoichiometryMath 
   * on reactants or products
   */
  for (n=0; n < getNumReactions(); n++)
  {
    react = getReaction(n);

    /* get units returned by kineticLaw formula */
    if (react->isSetKineticLaw())
    {
      fud = createFormulaUnitsData();
      fud->setUnitReferenceId(react->getId());

      /* set the id of the kinetic law 
       * normally a kinetic law doesnt have an id
       * but since it is an sbase object it can
       * so we set it to be the reaction id so 
       * that searching the listFormulaUnitsData can find it
       */
      react->getKineticLaw()->setInternalId(react->getId());

      fud->setComponentTypecode(SBML_KINETIC_LAW);
      unitFormatter->resetFlags();
      if(react->getKineticLaw()->isSetMath())
      {
        ud = unitFormatter->getUnitDefinition
                                  (react->getKineticLaw()->getMath(), true, n);
        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                 (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                   (unitFormatter->canIgnoreUndeclaredUnits());
      }
      else
      {
        ud = NULL;
        fud->setUnitDefinition(ud);
      }
    }

    /* get units returned by any stoichiometryMath set */
    for (j = 0; j < react->getNumReactants(); j++)
    {
      sr = react->getReactant(j);

      if (sr->isSetStoichiometryMath())
      {
        fud = createFormulaUnitsData();
        fud->setUnitReferenceId(sr->getSpecies());
        sr->getStoichiometryMath()->setInternalId(sr->getSpecies());
        fud->setComponentTypecode(SBML_STOICHIOMETRY_MATH);
        unitFormatter->resetFlags();
        ud = unitFormatter->getUnitDefinition
                                      (sr->getStoichiometryMath()->getMath());
        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      }

      /* for l3 the id may represent the stoichiometry */
      if (sr->getLevel() > 2 && sr->isSetId())
      {
        fud = createFormulaUnitsData();
        fud->setUnitReferenceId(sr->getId());
        fud->setComponentTypecode(SBML_SPECIES_REFERENCE);
        /* units will be dimensionless */
        ud = new UnitDefinition(getSBMLNamespaces());
        u = new Unit(getSBMLNamespaces());
        u->setKind(UNIT_KIND_DIMENSIONLESS);
        u->initDefaults();

        ud->addUnit(u);
        delete u;

        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits(false);
        fud->setCanIgnoreUndeclaredUnits(true);


        /* in L3 there are only defined time units IF the model
         * has the timeUnits attribute set
         */
        if (!isSetTimeUnits())
        {
          ud = new UnitDefinition(getSBMLNamespaces());
          fud->setContainsParametersWithUndeclaredUnits(true);
          fud->setCanIgnoreUndeclaredUnits(false);
        }
        else
        {
          std::string units = getTimeUnits();
          ud = new UnitDefinition(getSBMLNamespaces());

          /* either unit name or unitDefinition id */
          if (UnitKind_isValidUnitKindString(units.c_str(), getLevel(), 
                                                            getVersion()))
          {
            u = new Unit(getSBMLNamespaces());
            u->setKind(UnitKind_forName(units.c_str()));
            u->initDefaults();
            u->setExponent(-1);

            ud->addUnit(u);
            delete u;
          }
          else
          {
            for (j = 0; j < getNumUnitDefinitions(); j++)
            {
              if (!strcmp(units.c_str(), getUnitDefinition(j)->getId().c_str()))
              {           
                for (unsigned int pp = 0; 
                  pp < getUnitDefinition(j)->getNumUnits(); pp++)
                {
                  u = new Unit(getSBMLNamespaces());
                  u->setKind(getUnitDefinition(j)->getUnit(pp)->getKind());
                  u->setMultiplier(getUnitDefinition(j)->getUnit(pp)
                                                   ->getMultiplier());
                  u->setScale(getUnitDefinition(j)->getUnit(pp)->getScale());
                  u->setExponent(-1 * (getUnitDefinition(j)->getUnit(pp)
                                       ->getExponent()));

                  ud->addUnit(u);

                  delete u;
                }
              }
            }
          }
        }

        
        fud->setPerTimeUnitDefinition(ud);
      }
    }

    for (j = 0; j < react->getNumProducts(); j++)
    {
      sr = react->getProduct(j);

      if (sr->isSetStoichiometryMath())
      {
        fud = createFormulaUnitsData();
        sr->getStoichiometryMath()->setInternalId(sr->getSpecies());
        fud->setComponentTypecode(SBML_STOICHIOMETRY_MATH);
        unitFormatter->resetFlags();
        ud = unitFormatter->getUnitDefinition
                                      (sr->getStoichiometryMath()->getMath());
        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      }

      /* for l3 the id may represent the stoichiometry */
      if (sr->getLevel() > 2 && sr->isSetId())
      {
        fud = createFormulaUnitsData();
        fud->setUnitReferenceId(sr->getId());
        fud->setComponentTypecode(SBML_SPECIES_REFERENCE);
        /* units will be dimensionless */
        ud = new UnitDefinition(getSBMLNamespaces());
        u = new Unit(getSBMLNamespaces());
        u->setKind(UNIT_KIND_DIMENSIONLESS);
        u->initDefaults();

        ud->addUnit(u);
        delete u;

        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits(false);
        fud->setCanIgnoreUndeclaredUnits(true);
        /* in L3 there are only defined time units IF the model
         * has the timeUnits attribute set
         */
        if (!isSetTimeUnits())
        {
          ud = new UnitDefinition(getSBMLNamespaces());
          fud->setContainsParametersWithUndeclaredUnits(true);
          fud->setCanIgnoreUndeclaredUnits(false);
        }
        else
        {
          std::string units = getTimeUnits();
          ud = new UnitDefinition(getSBMLNamespaces());

          /* either unit name or unitDefinition id */
          if (UnitKind_isValidUnitKindString(units.c_str(), getLevel(), 
                                                            getVersion()))
          {
            u = new Unit(getSBMLNamespaces());
            u->setKind(UnitKind_forName(units.c_str()));
            u->initDefaults();
            u->setExponent(-1);

            ud->addUnit(u);
            delete u;
          }
          else
          {
            for (j = 0; j < getNumUnitDefinitions(); j++)
            {
              if (!strcmp(units.c_str(), getUnitDefinition(j)->getId().c_str()))
              {           
                for (unsigned int pp = 0; 
                  pp < getUnitDefinition(j)->getNumUnits(); pp++)
                {
                  u = new Unit(getSBMLNamespaces());
                  u->setKind(getUnitDefinition(j)->getUnit(pp)->getKind());
                  u->setMultiplier(getUnitDefinition(j)->getUnit(pp)
                                                   ->getMultiplier());
                  u->setScale(getUnitDefinition(j)->getUnit(pp)->getScale());
                  u->setExponent(-1 * (getUnitDefinition(j)->getUnit(pp)
                                       ->getExponent()));

                  ud->addUnit(u);

                  delete u;
                }
              }
            }
          }
        }

        
        fud->setPerTimeUnitDefinition(ud);
      }
    }
  }

  /**
   * math may occur in events as the trigger, the delay or 
   * in the eventAssignment
   */
  for (n=0; n < getNumEvents(); n++)
  {
    e = getEvent(n);

    if (e->isSetId())
    {
      newID = e->getId();//sprintf(newId, "%s", e->getId());
    }
    else
    {
      sprintf(newId, "event_%u", countEvents);
      newID.assign(newId);
    }
    countEvents++;
    if (!e->isSetId())
    {
      e->setId(newID);
      e->setInternalIdOnly();
    }

    /* dont need units returned by trigger formula - 
     * should be boolean
     */
    
    /* get units returned by dely */
    if (e->isSetDelay())
    {
      Delay * d = e->getDelay();
      fud = createFormulaUnitsData();
        
      fud->setUnitReferenceId(newID);
      d->setInternalId(newID);

      fud->setComponentTypecode(SBML_EVENT);
      unitFormatter->resetFlags();
      ud = unitFormatter->getUnitDefinition(e->getDelay()->getMath());
      fud->setUnitDefinition(ud);
      fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
      fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      
      /* get event time definition */
      unitFormatter->resetFlags();
      ud = unitFormatter->getUnitDefinitionFromEventTime(e);
      if (ud->getNumUnits() == 0)
      {
        fud->setContainsParametersWithUndeclaredUnits(true);
        fud->setCanIgnoreUndeclaredUnits(false);
      }
      fud->setEventTimeUnitDefinition(ud);
    }


    /* get units returned by priority */
    if (e->isSetPriority())
    {
      Priority * p = e->getPriority();
      fud = createFormulaUnitsData();
        
      fud->setUnitReferenceId(newID);
      p->setInternalId(newID);

      fud->setComponentTypecode(SBML_PRIORITY);
      unitFormatter->resetFlags();
      ud = unitFormatter->getUnitDefinition(p->getMath());
      fud->setUnitDefinition(ud);
      fud->setContainsParametersWithUndeclaredUnits
                                (unitFormatter->getContainsUndeclaredUnits());
      fud->setCanIgnoreUndeclaredUnits
                                  (unitFormatter->canIgnoreUndeclaredUnits());
      
    }

    /* get units returned by any event assignments */
    for (j = 0; j < e->getNumEventAssignments(); j++)
    {
      ea = e->getEventAssignment(j);

      eaId = ea->getVariable() + newID;
      if (ea->isSetMath())    
      {
        fud = createFormulaUnitsData();
        fud->setUnitReferenceId(eaId);
        fud->setComponentTypecode(SBML_EVENT_ASSIGNMENT);
        unitFormatter->resetFlags();
        ud = unitFormatter->getUnitDefinition(ea->getMath());
        fud->setUnitDefinition(ud);
        fud->setContainsParametersWithUndeclaredUnits
                                 (unitFormatter->getContainsUndeclaredUnits());
        fud->setCanIgnoreUndeclaredUnits
                                   (unitFormatter->canIgnoreUndeclaredUnits());
      }
    }
  }
  delete unitFormatter;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Adds a copy of the given FormulaUnitsData to this Model.
 */
void
Model::addFormulaUnitsData (const FormulaUnitsData* fud)
{
  if (mFormulaUnitsData == NULL)
  {
    mFormulaUnitsData = new List();
    mFormulaUnitsData->add((void *) fud->clone());
  }
  else
  {
    mFormulaUnitsData->add((void *)fud->clone());
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * Creates a new FormulaUnitsData inside this Model and returns it.
  */
FormulaUnitsData*
Model::createFormulaUnitsData ()
{
  FormulaUnitsData* fud = new FormulaUnitsData;
  if (mFormulaUnitsData == NULL)
  {
    mFormulaUnitsData = new List();
  }
  mFormulaUnitsData->add(fud);

  return fud;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the nth FormulaUnitsData of this Model.
 */
const FormulaUnitsData*
Model::getFormulaUnitsData (unsigned int n) const
{
  return static_cast<const FormulaUnitsData*>( mFormulaUnitsData->get(n) );
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the nth FormulaUnitsData of this Model.
 */
FormulaUnitsData*
Model::getFormulaUnitsData (unsigned int n)
{
  return static_cast<FormulaUnitsData*>( mFormulaUnitsData->get(n) );
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the FormulaUnitsData in this Model with the given id 
 * or @c NULL if no such
 * FormulaUnitsData exists.
 */
const FormulaUnitsData*
Model::getFormulaUnitsData (const std::string& sid, 
                            int typecode) const
{
  return const_cast<Model*>(this)->getFormulaUnitsData(sid,typecode);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the FormulaUnitsData in this Model with the given id  and typecode 
 * or @c NULL if no such FormulaUnitsData exists.
 */
FormulaUnitsData*
Model::getFormulaUnitsData (const std::string& sid, int typecode)
{
  FormulaUnitsData * fud;

  for (unsigned int n = 0; n < getNumFormulaUnitsData(); n++)
  {
    fud = static_cast <FormulaUnitsData*> (mFormulaUnitsData->get(n));
    if (!strcmp(fud->getUnitReferenceId().c_str(), sid.c_str()))
    {
      if (fud->getComponentTypecode() == typecode)
      {
        return fud;
      }
    }
  }
  return NULL;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the number of FormulaUnitsDatas in this Model.
 */
unsigned int
Model::getNumFormulaUnitsData () const
{
  return mFormulaUnitsData->getSize();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * Get the list of FormulaUnitsData object in this Model.
  * 
  * @return the list of FormulaUnitsData for this Model.
  */
List* 
Model::getListFormulaUnitsData ()
{
  return mFormulaUnitsData;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * Get the list of FormulaUnitsData object in this Model.
  * 
  * @return the list of FormulaUnitsData for this Model.
  */
const List* 
Model::getListFormulaUnitsData () const
{
  return mFormulaUnitsData;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * returns true if the list has been populated, false otherwise
 */
bool
Model::isPopulatedListFormulaUnitsData()
{
  if (mFormulaUnitsData != NULL)
    return true;
  else
    return false;
}
/** @endcond */



/** @cond doxygen-c-only */


/**
 * Creates a new Model_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * Model
 *
 * @param version an unsigned int, the SBML Version to assign to this
 * Model
 *
 * @return a pointer to the newly created Model_t structure.
 *
 * @note Once a Model has been added to an SBMLDocument, the @p
 * level and @p version for the document @em override those used to create
 * the Model.  Despite this, the ability to supply the values at
 * creation time is an important aid to creating valid SBML.  Knowledge of
 * the intended SBML Level and Version  determine whether it is valid to
 * assign a particular value to an attribute, or whether it is valid to add
 * an object to an existing SBMLDocument.
 */
LIBSBML_EXTERN
Model_t *
Model_create (unsigned int level, unsigned int version)
{
  try
  {
    Model* obj = new Model(level,version);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Creates a new Model_t structure using the given
 * SBMLNamespaces_t structure.
 *
 * @param sbmlns SBMLNamespaces, a pointer to an SBMLNamespaces structure
 * to assign to this Model
 *
 * @return a pointer to the newly created Model_t structure.
 *
 * @note Once a Model has been added to an SBMLDocument, the
 * @p sbmlns namespaces for the document @em override those used to create
 * the Model.  Despite this, the ability to supply the values at creation time
 * is an important aid to creating valid SBML.  Knowledge of the intended SBML
 * Level and Version determine whether it is valid to assign a particular value
 * to an attribute, or whether it is valid to add an object to an existing
 * SBMLDocument.
 */
LIBSBML_EXTERN
Model_t *
Model_createWithNS (SBMLNamespaces_t* sbmlns)
{
  try
  {
    Model* obj = new Model(sbmlns);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Creates and returns a deep copy of a given Model_t structure.
 *
 * @param m the Model_t structure to copy
 * 
 * @return a (deep) copy of this Model_t structure.
 */
LIBSBML_EXTERN
Model_t *
Model_clone (const Model_t *m)
{
  return (m != NULL) ? static_cast<Model*>( m->clone() ) : NULL;
}


/**
 * Frees the given Model_t structure.
 *
 * @param m the Model_structure to free
 */
LIBSBML_EXTERN
void
Model_free (Model_t *m)
{
  if (m != NULL)
  delete m;
}


/**
 * Returns a list of XMLNamespaces_t associated with this Model_t
 * structure.
 *
 * @param m the Model_t structure
 * 
 * @return pointer to the XMLNamespaces_t structure associated with 
 * this SBML object
 */
LIBSBML_EXTERN
const XMLNamespaces_t *
Model_getNamespaces(Model_t *m)
{
  return (m != NULL) ? m->getNamespaces() : NULL;
}


/**
 * Get the identifier of the given Model_t structure.
 *
 * @param m the Model_t structure
 * 
 * @return the id of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getId (const Model_t *m)
{
  return (m != NULL && m->isSetId()) ? m->getId().c_str() : NULL;
}


/**
 * Get the name of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the name of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getName (const Model_t *m)
{
  return (m != NULL && m->isSetName()) ? m->getName().c_str() : NULL;
}


/**
 * Get the substanceUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the substanceUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getSubstanceUnits (const Model_t *m)
{
  return (m != NULL && m->isSetSubstanceUnits()) ? 
                       m->getSubstanceUnits().c_str() : NULL;
}


/**
 * Get the timeUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the timeUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getTimeUnits (const Model_t *m)
{
  return (m != NULL && m->isSetTimeUnits()) ? m->getTimeUnits().c_str() : NULL;
}


/**
 * Get the volumeUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the volumeUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getVolumeUnits (const Model_t *m)
{
  return (m != NULL && m->isSetVolumeUnits()) ? 
                       m->getVolumeUnits().c_str() : NULL;
}


/**
 * Get the areaUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the areaUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getAreaUnits (const Model_t *m)
{
  return (m != NULL && m->isSetAreaUnits()) ? m->getAreaUnits().c_str() : NULL;
}


/**
 * Get the lengthUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the lengthUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getLengthUnits (const Model_t *m)
{
  return (m != NULL && m->isSetLengthUnits()) ? 
                       m->getLengthUnits().c_str() : NULL;
}


/**
 * Get the extentUnits of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the extentUnits of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getExtentUnits (const Model_t *m)
{
  return (m != NULL && m->isSetExtentUnits()) ? 
                       m->getExtentUnits().c_str() : NULL;
}


/**
 * Get the conversionFactor of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * 
 * @return the conversionFactor of this Model_t structure.
 */
LIBSBML_EXTERN
const char *
Model_getConversionFactor (const Model_t *m)
{
  return (m != NULL && m->isSetConversionFactor()) ? 
                       m->getConversionFactor().c_str() : NULL;
}


/**
 * Predicate for testing whether the identifier of a given Model_t
 * structure is assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "id" attribute of this Model_t structure is 
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetId (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetId() ) : 0;
}


/**
 * Predicate for testing whether the name of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "name" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetName (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetName() ) : 0;
}


/**
 * Predicate for testing whether the substanceUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "substanceUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetSubstanceUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetSubstanceUnits() ) : 0;
}


/**
 * Predicate for testing whether the timeUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "timeUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetTimeUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetTimeUnits() ) : 0;
}


/**
 * Predicate for testing whether the volumeUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "volumeUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetVolumeUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetVolumeUnits() ) : 0;
}


/**
 * Predicate for testing whether the areaUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "areaUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetAreaUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetAreaUnits() ) : 0;
}


/**
 * Predicate for testing whether the lengthUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "lengthUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetLengthUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetLengthUnits() ) : 0;
}


/**
 * Predicate for testing whether the extentUnits of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "extentUnits" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetExtentUnits (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetExtentUnits() ) : 0;
}


/**
 * Predicate for testing whether the conversionFactor of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the "conversionFactor" attribute of this Model_t structure is
 * set, zero (0) otherwise.
 */
LIBSBML_EXTERN
int
Model_isSetConversionFactor (const Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetConversionFactor() ) : 0;
}


/**
 * Set the identifier of a given Model_t structure.
 *
 * This copies the string in @p sid.
 * 
 * @param m the Model_t structure
 * @param sid the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with an id of NULL is equivalent to
 * unsetting the "id" attribute.
 */
LIBSBML_EXTERN
int
Model_setId (Model_t *m, const char *sid)
{
  if (m != NULL)
    return (sid == NULL) ? m->unsetId() : m->setId(sid);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the identifier of the given Model_t structure.
 * 
 * This copies the string in @p name.
 *
 * @param m the Model_t structure
 * @param name the name string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "name" attribute.
 */
LIBSBML_EXTERN
int
Model_setName (Model_t *m, const char *name)
{
  if (m != NULL)
    return (name == NULL) ? m->unsetName() : m->setName(name);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the substanceUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "substanceUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setSubstanceUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetSubstanceUnits() : 
                                     m->setSubstanceUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the timeUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "timeUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setTimeUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetTimeUnits() : 
                                     m->setTimeUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the volumeUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "volumeUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setVolumeUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetVolumeUnits() : 
                                     m->setVolumeUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the areaUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "areaUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setAreaUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetAreaUnits() : 
                                     m->setAreaUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the lengthUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "lengthUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setLengthUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetLengthUnits() : 
                                     m->setLengthUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the extentUnits attribute of a given Model_t structure.
 *
 * This copies the string in @p units.
 * 
 * @param m the Model_t structure
 * @param units the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with units NULL is equivalent to
 * unsetting the "extentUnits" attribute.
 */
LIBSBML_EXTERN
int
Model_setExtentUnits (Model_t *m, const char *units)
{
  if (m != NULL)
    return (units == NULL) ? m->unsetExtentUnits() : 
                                     m->setExtentUnits(units);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Set the conversionFactor attribute of a given Model_t structure.
 *
 * This copies the string in @p sid.
 * 
 * @param m the Model_t structure
 * @param sid the identifier string
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with sid NULL is equivalent to
 * unsetting the "conversionFactor" attribute.
 */
LIBSBML_EXTERN
int
Model_setConversionFactor (Model_t *m, const char *sid)
{
  if (m != NULL)
    return (sid == NULL) ? m->unsetConversionFactor() : 
                                     m->setConversionFactor(sid);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "id" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetId (Model_t *m)
{
  if (m != NULL)
    return m->unsetId();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "name" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetName (Model_t *m)
{
  if (m != NULL)
    return m->unsetName();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "substanceUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetSubstanceUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetSubstanceUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "timeUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetTimeUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetTimeUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "volumeUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetVolumeUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetVolumeUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "areaUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetAreaUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetAreaUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "lengthUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetLengthUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetLengthUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "extentUnits" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetExtentUnits (Model_t *m)
{
  if (m != NULL)
    return m->unsetExtentUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "conversionFactor" attribute of the given Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_unsetConversionFactor (Model_t *m)
{
  if (m != NULL)
    return m->unsetConversionFactor();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Returns the ModelHistory of the given Model_t structure.
 *
 * @return the ModelHistory of the given Model_t structure.
 * 
 * @param m the Model_t structure
 */
LIBSBML_EXTERN
ModelHistory_t * 
Model_getModelHistory(Model_t *m)
{
  return (m != NULL) ? m->getModelHistory() : NULL;
}

/**
 * Predicate for testing whether the ModelHistory of a given Model_t structure is
 * assigned.
 * 
 * @param m the Model_t structure
 * 
 * @return nonzero if the ModelHistory of this Model_t structure is
 * set, zero (0) otherwise.
 */LIBSBML_EXTERN
int 
Model_isSetModelHistory(Model_t *m)
{
  return (m != NULL) ? static_cast<int>( m->isSetModelHistory() ) : 0;
}


/**
 * Set the ModelHistory of the given Model_t structure.
 * 
 * @param m the Model_t structure
 * @param history the ModelHistory_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
Model_setModelHistory(Model_t *m, ModelHistory_t *history)
{
  return (m != NULL) ? m->setModelHistory(history) : LIBSBML_INVALID_OBJECT;
}

/**
 * Unsets the ModelHistory of the given Model_t structure.
 * 
 * @param m the Model_t structure
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int 
Model_unsetModelHistory(Model_t *m)
{
  return (m != NULL) ? m->unsetModelHistory() : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a FunctionDefinition_t structure to a given Model_t
 * structure.
 *
 * @param m the Model_t structure
 * @param fd the FunctionDefinition_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addFunctionDefinition (Model_t *m, const FunctionDefinition_t *fd)
{
  return  (m != NULL) ? m->addFunctionDefinition(fd) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a UnitDefinition_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param ud the UnitDefinition_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addUnitDefinition (Model_t *m, const UnitDefinition_t *ud)
{
  return (m != NULL) ? m->addUnitDefinition(ud) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a CompartmentType_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param ct the CompartmentType_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addCompartmentType (Model_t *m, const CompartmentType_t *ct)
{
  return (m != NULL) ? m->addCompartmentType(ct) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a SpeciesType_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param st the SpeciesType_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addSpeciesType (Model_t *m, const SpeciesType_t *st)
{
  return (m != NULL) ? m->addSpeciesType(st) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Compartment_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param c the Compartment_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addCompartment (Model_t *m, const Compartment_t *c)
{
  return (m != NULL) ? m->addCompartment(c) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Species_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param s the Species_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addSpecies (Model_t *m, const Species_t *s)
{
  return (m != NULL) ? m->addSpecies(s) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Parameter_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param p the Parameter_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addParameter (Model_t *m, const Parameter_t *p)
{
  return (m != NULL) ? m->addParameter(p) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a InitialAssignment_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param ia the InitialAssignment_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addInitialAssignment (Model_t *m, const InitialAssignment_t *ia)
{
  return (m != NULL) ? m->addInitialAssignment(ia) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Rule_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param r the Rule_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addRule (Model_t *m, const Rule_t *r)
{
  return (m != NULL) ? m->addRule(r) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Constraint_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param c the Constraint_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addConstraint (Model_t *m, const Constraint_t *c)
{
  return (m != NULL) ? m->addConstraint(c) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Reaction_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param r the Reaction_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addReaction (Model_t *m, const Reaction_t *r)
{
  return (m != NULL) ? m->addReaction(r) : LIBSBML_INVALID_OBJECT;
}


/**
 * Adds a copy of a Event_t structure to a given Model_t structure.
 *
 * @param m the Model_t structure
 * @param e the Event_t structure to copy and add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Model_addEvent (Model_t *m, const Event_t *e)
{
  return (m != NULL) ? m->addEvent(e) : LIBSBML_INVALID_OBJECT;
}


/**
 * Creates a new FunctionDefinition_t structure inside the given Model_t
 * and returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the new FunctionDefinition_t structure
 */
LIBSBML_EXTERN
FunctionDefinition_t *
Model_createFunctionDefinition (Model_t *m)
{
  return (m != NULL) ? m->createFunctionDefinition() : NULL;
}


/**
 * Creates a new UnitDefinition_t structure inside the given Model_t and
 * returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the new UnitDefinition_t structure
 */
LIBSBML_EXTERN
UnitDefinition_t *
Model_createUnitDefinition (Model_t *m)
{
  return (m != NULL) ? m->createUnitDefinition() : NULL;
}


/**
 * Creates a new Unit_t structure inside the last UnitDefinition_t
 * structure created in this model and returns a pointer to it.
 *
 * The mechanism by which the UnitDefinition_t structure was created is not
 * significant.  If a UnitDefinition_t does not exist in this model, a new
 * Unit_t structure is @em not created and NULL is returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the Unit_t structure created, or @c NULL.
 */
LIBSBML_EXTERN
Unit_t *
Model_createUnit (Model_t *m)
{
  return (m != NULL) ? m->createUnit() : NULL;
}


/**
 * Creates a new CompartmentType_t structure inside the given Model_t and
 * returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the CompartmentType_t structure created
 */
LIBSBML_EXTERN
CompartmentType_t *
Model_createCompartmentType (Model_t *m)
{
  return (m != NULL) ? m->createCompartmentType() : NULL;
}


/**
 * Creates a new SpeciesType_t structure inside the given Model_t and
 * returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the SpeciesType_t structure created
 */
LIBSBML_EXTERN
SpeciesType_t *
Model_createSpeciesType (Model_t *m)
{
  return (m != NULL) ? m->createSpeciesType() : NULL;
}


/**
 * Creates a new Compartment_t structure inside the given Model_t and
 * returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the Compartment_t structure created
 */
LIBSBML_EXTERN
Compartment_t *
Model_createCompartment (Model_t *m)
{
  return (m != NULL) ? m->createCompartment() : NULL;
}


/**
 * Creates a new Species_t structure inside the given Model_t and returns a
 * pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the Species_t structure created
 */
LIBSBML_EXTERN
Species_t *
Model_createSpecies (Model_t *m)
{
  return (m != NULL) ? m->createSpecies() : NULL;
}


/**
 * Creates a new Parameter_t structure inside the given Model_t and returns
 * a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the Parameter_t structure created
 */
LIBSBML_EXTERN
Parameter_t *
Model_createParameter (Model_t *m)
{
  return (m != NULL) ? m->createParameter() : NULL;
}


/**
 * Creates a new InitialAssignment_t structure inside the given Model_t
 * structure and returns it.
 *
 * @param m the Model_t structure
 *
 * @return the InitialAssignment_t structure created
 */
LIBSBML_EXTERN
InitialAssignment_t *
Model_createInitialAssignment (Model_t *m)
{
  return (m != NULL) ? m->createInitialAssignment() : NULL;
}


/**
 * Creates a new AlgebraicRule_t structure inside the given Model_t
 * structure and returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the AlgebraicRule_t structure created.
 */
LIBSBML_EXTERN
Rule_t *
Model_createAlgebraicRule (Model_t *m)
{
  return (m != NULL) ? m->createAlgebraicRule() : NULL;
}


/**
 * Creates a new AssignmentRule_t structure inside the given Model_t
 * structure and returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the AssignmentRule_t structure created
 */
LIBSBML_EXTERN
Rule_t *
Model_createAssignmentRule (Model_t *m)
{
  return (m != NULL) ? m->createAssignmentRule() : NULL;
}


/**
 * Creates a new RateRule_t structure inside the given Model_t structure
 * and returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the RateRule_t structure created.
 */
LIBSBML_EXTERN
Rule_t *
Model_createRateRule (Model_t *m)
{
  return (m != NULL) ? m->createRateRule() : NULL;
}


/**
 * Creates a new Constraint_t structure inside the given Model_t structure
 * and returns it.
 *
 * @param m the Model_t structure
 *
 * @return the Constraint_t structure created.
 */
LIBSBML_EXTERN
Constraint_t *
Model_createConstraint (Model_t *m)
{
  return (m != NULL) ? m->createConstraint() : NULL;
}

/**
 * Creates a new Reaction_t structure inside the given Model_t structure
 * and returns a pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return the Reaction_t structure created.
 */
LIBSBML_EXTERN
Reaction_t *
Model_createReaction (Model_t *m)
{
  return (m != NULL) ? m->createReaction() : NULL;
}


/**
 * Creates a new SpeciesReference_t structure for a reactant inside the
 * last Reaction_t structure in the given Model_t structure, and returns a
 * pointer to it.
 *
 * The mechanism by which the last Reaction_t structure was created and
 * added to @p m is not significant.  It could have been created in a
 * variety of ways, for example using Model_createReaction().  If a
 * Reaction_t structure does not exist, a new SpeciesReference_t structure
 * is @em not created and NULL is returned instead.
 *
 * @param m the Model_t structure
 * 
 * @return the SpeciesReference object created
 */
LIBSBML_EXTERN
SpeciesReference_t *
Model_createReactant (Model_t *m)
{
  return (m != NULL) ? m->createReactant() : NULL;
}


/**
 * Creates a new SpeciesReference_t structure for a product inside the
 * last Reaction_t structure in the given Model_t structure, and returns a
 * pointer to it.
 *
 * The mechanism by which the last Reaction_t structure was created and
 * added to @p m is not significant.  It could have been created in a
 * variety of ways, for example using Model_createReaction().  If a
 * Reaction_t structure does not exist, a new SpeciesReference_t structure
 * is @em not created and NULL is returned instead.
 *
 * @param m the Model_t structure
 * 
 * @return the SpeciesReference object created
 */
LIBSBML_EXTERN
SpeciesReference_t *
Model_createProduct (Model_t *m)
{
  return (m != NULL) ? m->createProduct() : NULL;
}


/**
 * Creates a new ModifierSpeciesReference_t structure for a reactant inside
 * the last Reaction_t structure in the given Model_t structure, and
 * returns a pointer to it.
 *
 * The mechanism by which the last Reaction_t structure was created and
 * added to @p m is not significant.  It could have been created in a
 * variety of ways, for example using Model_createReaction().  If a
 * Reaction_t structure does not exist, a new ModifierSpeciesReference_t
 * structure is @em not created and NULL is returned instead.
 *
 * @param m the Model_t structure
 * 
 * @return the ModifierSpeciesReference object created
 */
LIBSBML_EXTERN
SpeciesReference_t *
Model_createModifier (Model_t *m)
{
  return (m != NULL) ? 
          static_cast<SpeciesReference_t*>( m->createModifier() ) : NULL;
}


/**
 * Creates a new KineticLaw_t structure inside the last Reaction_t
 * structure in the given Model_t structure, and returns a pointer to it.
 *
 * The mechanism by which the last Reaction_t structure was created and
 * added to @p m is not significant.  It could have been created in a
 * variety of ways, for example using Model_createReaction().  If a
 * Reaction_t structure does not exist for the model, or a Reaction_t
 * structure exists but already contains a KineticLaw_t structure, a new
 * KineticLaw_t is @em not created and NULL is returned instead.
 *
 * @param m the Model_t structure
 * 
 * @return the KineticLaw object created
 */
LIBSBML_EXTERN
KineticLaw_t *
Model_createKineticLaw (Model_t *m)
{
  return (m != NULL) ? m->createKineticLaw() : NULL;
}


/**
 * Creates a new local Parameter_t structure inside the KineticLaw_t
 * structure of the last Reaction_t structure created inside the given
 * model, and returns a pointer to it.
 *
 * The last KineticLaw_t structure could have been created in a variety of
 * ways.  For example, it could have been added using
 * Model_createKineticLaw(), or it could be the result of using
 * Reaction_createKineticLaw() on the Reaction_t structure created by a
 * Model_createReaction().  If a Reaction_t structure does not exist for
 * this model, or the last Reaction_t structure does not contain a
 * KineticLaw_t structure, a new Parameter_t is @em not created and NULL is
 * returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the Parameter object created
 */
LIBSBML_EXTERN
Parameter_t *
Model_createKineticLawParameter (Model_t *m)
{
  return (m != NULL) ? m->createKineticLawParameter() : NULL;
}


/**
 * Creates a new LocalParameter_t structure inside the KineticLaw_t
 * structure of the last Reaction_t structure created inside the given
 * model, and returns a pointer to it.
 *
 * The last KineticLaw_t structure could have been created in a variety of
 * ways.  For example, it could have been added using
 * Model_createKineticLaw(), or it could be the result of using
 * Reaction_createKineticLaw() on the Reaction_t structure created by a
 * Model_createReaction().  If a Reaction_t structure does not exist for
 * this model, or the last Reaction_t structure does not contain a
 * KineticLaw_t structure, a new Parameter_t is @em not created and NULL is
 * returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the LocalParameter object created
 */
LIBSBML_EXTERN
LocalParameter_t *
Model_createKineticLawLocalParameter (Model_t *m)
{
  return (m != NULL) ? m->createKineticLawLocalParameter() : NULL;
}


/**
 * Creates a new Event inside the given Model_t structure and returns a
 * pointer to it.
 *
 * @param m the Model_t structure
 *
 * @return a new Event_t structure
 */
LIBSBML_EXTERN
Event_t *
Model_createEvent (Model_t *m)
{
  return (m != NULL) ? m->createEvent() : NULL;
}


/**
 * Creates a new EventAssignment_t structure inside the last Event_t
 * structure created in the given Model_t structure, and returns a pointer
 * to it.
 *
 * The mechanism by which the last Event_t structure was created is not
 * significant.  It could have been created in a variety of ways, for
 * example by using Model_createEvent().  If an Event_t structure does not
 * exist, a new EventAssignment_t structure is @em not created and NULL is
 * returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the new EventAssignment_t structure
 */
LIBSBML_EXTERN
EventAssignment_t *
Model_createEventAssignment (Model_t *m)
{
  return (m != NULL) ? m->createEventAssignment() : NULL;
}


/**
 * Creates a new Trigger_t structure inside the last Event_t
 * structure created in the given Model_t structure, and returns a pointer
 * to it.
 *
 * The mechanism by which the last Event_t structure was created is not
 * significant.  It could have been created in a variety of ways, for
 * example by using Model_createEvent().  If an Event_t structure does not
 * exist, a new Trigger_t structure is @em not created and NULL is
 * returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the new Trigger_t structure
 */
LIBSBML_EXTERN
Trigger_t *
Model_createTrigger (Model_t *m)
{
  return (m != NULL) ? m->createTrigger() : NULL;
}


/**
 * Creates a new Delay_t structure inside the last Event_t
 * structure created in the given Model_t structure, and returns a pointer
 * to it.
 *
 * The mechanism by which the last Event_t structure was created is not
 * significant.  It could have been created in a variety of ways, for
 * example by using Model_createEvent().  If an Event_t structure does not
 * exist, a new Delay_t structure is @em not created and NULL is
 * returned instead.
 *
 * @param m the Model_t structure
 *
 * @return the new Delay_t structure
 */
LIBSBML_EXTERN
Delay_t *
Model_createDelay (Model_t *m)
{
  return (m != NULL) ? m->createDelay() : NULL;
}


/**
 * Get the list of FunctionDefinition_t structures contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of FunctionDefinition_t structures
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfFunctionDefinitions (Model_t *m)
{
  return (m != NULL) ? m->getListOfFunctionDefinitions() : NULL;
}


/**
 * Get the list of UnitDefinition_t structures contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of UnitDefinition_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfUnitDefinitions (Model_t *m)
{
  return (m != NULL) ? m->getListOfUnitDefinitions() : NULL;
}


/**
 * Get the list of CompartmentType_t structures contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of CompartmentType_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfCompartmentTypes (Model_t *m)
{
  return (m != NULL) ? m->getListOfCompartmentTypes() : NULL;
}


/**
 * Get the list of SpeciesType_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of SpeciesType_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfSpeciesTypes (Model_t *m)
{
  return (m != NULL) ? m->getListOfSpeciesTypes() : NULL;
}


/**
 * Get the list of Compartment_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Compartment_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfCompartments (Model_t *m)
{
  return (m != NULL) ? m->getListOfCompartments() : NULL;
}


/**
 * Get the list of Species_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Specie_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfSpecies (Model_t *m)
{
  return (m != NULL) ? m->getListOfSpecies() : NULL;
}


/**
 * Get the list of Parameter_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Parameter_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfParameters (Model_t *m)
{
  return (m != NULL) ? m->getListOfParameters() : NULL;
}


/**
 * Get the list of InitialAssignment_t structures contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of InitialAssignment_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfInitialAssignments (Model_t* m)
{
  return (m != NULL) ? m->getListOfInitialAssignments() : NULL;
}


/**
 * Get the list of Rule_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Rule_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfRules (Model_t *m)
{
  return (m != NULL) ? m->getListOfRules() : NULL;
}


/**
 * Get the list of Constraint_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Constraint_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfConstraints (Model_t* m)
{
  return (m != NULL) ? m->getListOfConstraints() : NULL;
}


/**
 * Get the list of Reaction_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Reaction_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfReactions (Model_t *m)
{
  return (m != NULL) ? m->getListOfReactions() : NULL;
}


/**
 * Get the list of Event_t structures contained in the given Model_t
 * structure.
 *
 * @param m the Model_t structure
 *
 * @return the list of Event_t structures.
 */
LIBSBML_EXTERN
ListOf_t *
Model_getListOfEvents (Model_t *m)
{
  return (m != NULL) ? m->getListOfEvents() : NULL;
}


/**
 * Get the nth FunctionDefinition_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the FunctionDefinition_t sought
 *
 * @return the FunctionDefinition_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
FunctionDefinition_t *
Model_getFunctionDefinition (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getFunctionDefinition(n) : NULL;
}


/**
 * Get the FunctionDefinition_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the FunctionDefinition_t structure sought
 *
 * @return the FunctionDefinition_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
FunctionDefinition_t *
Model_getFunctionDefinitionById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getFunctionDefinition(sid) : NULL;
}


/**
 * Get the nth UnitDefinition_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the UnitDefinition_t sought
 *
 * @return the UnitDefinition_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
UnitDefinition_t *
Model_getUnitDefinition (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getUnitDefinition(n) : NULL;
}


/**
 * Get the UnitDefinition_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the UnitDefinition_t structure sought
 *
 * @return the UnitDefinition_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
UnitDefinition_t *
Model_getUnitDefinitionById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getUnitDefinition(sid) : NULL;
}


/**
 * Get the nth CompartmentType_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the CompartmentType_t sought
 *
 * @return the CompartmentType_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
CompartmentType_t *
Model_getCompartmentType (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getCompartmentType(n) : NULL;
}


/**
 * Get the CompartmentType_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the CompartmentType_t structure sought
 *
 * @return the CompartmentType_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
CompartmentType_t *
Model_getCompartmentTypeById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getCompartmentType(sid) : NULL;
}


/**
 * Get the nth SpeciesType_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the SpeciesType_t sought
 *
 * @return the SpeciesType_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
SpeciesType_t *
Model_getSpeciesType (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getSpeciesType(n) : NULL;
}


/**
 * Get the SpeciesType_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the SpeciesType_t structure sought
 *
 * @return the SpeciesType_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
SpeciesType_t *
Model_getSpeciesTypeById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getSpeciesType(sid) : NULL;
}


/**
 * Get the nth Compartment_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Compartment_t sought
 *
 * @return the Compartment_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Compartment_t *
Model_getCompartment (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getCompartment(n) : NULL;
}


/**
 * Get the Compartment_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the Compartment_t structure sought
 *
 * @return the Compartment_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Compartment_t *
Model_getCompartmentById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getCompartment(sid) : NULL;
}


/**
 * Get the nth Species_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Species_t sought
 *
 * @return the Species_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Species_t *
Model_getSpecies (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getSpecies(n) : NULL;
}


/**
 * Get the Species_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the Species_t structure sought
 *
 * @return the Species_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Species_t *
Model_getSpeciesById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getSpecies(sid) : NULL;
}


/**
 * Get the nth Parameter_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Parameter_t sought
 *
 * @return the Parameter_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Parameter_t *
Model_getParameter (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getParameter(n) : NULL;
}


/**
 * Get the Parameter_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the Parameter_t structure sought
 *
 * @return the Parameter_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Parameter_t *
Model_getParameterById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getParameter(sid) : NULL;
}


/**
 * Get the nth InitialAssignment_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the InitialAssignment_t sought
 *
 * @return the InitialAssignment_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
InitialAssignment_t *
Model_getInitialAssignment (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getInitialAssignment(n) : NULL;
}


/**
 * @return the InitialAssignment in this Model with the given symbol or
 * @c NULL if no such InitialAssignment exists.
 *
 * @param m the Model_t structure
 */
LIBSBML_EXTERN
InitialAssignment_t *
Model_getInitialAssignmentBySym (Model_t *m, const char *symbol)
{
  return (symbol != NULL) ? m->getInitialAssignment(symbol) : NULL;
}


/**
 * Get the nth Rule_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Rule_t sought
 *
 * @return the Rule_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Rule_t *
Model_getRule (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getRule(n) : NULL;
}


/**
 * @return the Rule in this Model with the given symbol or @c NULL if no
 * such Rule exists.
 *
 * @param m the Model_t structure
 */
LIBSBML_EXTERN
Rule_t *
Model_getRuleByVar (Model_t *m, const char *variable)
{
  return (variable != NULL) ? m->getRule(variable) : NULL;
}


/**
 * Get the nth Constraint_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Constraint_t sought
 *
 * @return the Constraint_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Constraint_t *
Model_getConstraint (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getConstraint(n) : NULL;
}


/**
 * Get the nth Reaction_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Reaction_t sought
 *
 * @return the Reaction_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Reaction_t *
Model_getReaction (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getReaction(n) : NULL;
}


/**
 * Get the Reaction_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the Reaction_t structure sought
 *
 * @return the Reaction_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Reaction_t *
Model_getReactionById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getReaction(sid) : NULL;
}

/**
 * Get the SpeciesReference_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the SpeciesReference_t structure sought
 *
 * @return the SpeciesReference_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
SpeciesReference_t *
Model_getSpeciesReferenceById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getSpeciesReference(sid) : NULL;
}



/**
 * Get the nth Event_t structure contained in the given
 * Model_t structure.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Event_t sought
 *
 * @return the Event_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Event_t *
Model_getEvent (Model_t *m, unsigned int n)
{
  return (m != NULL) ? m->getEvent(n) : NULL;
}


/**
 * Get the Event_t structure whose identifier is @p sid in the
 * given Model_t structure.
 *
 * @param m the Model_t structure
 * @param sid the identifier of the Event_t structure sought
 *
 * @return the Event_t if found, or @c NULL if not found
 */
LIBSBML_EXTERN
Event_t *
Model_getEventById (Model_t *m, const char *sid)
{
  return (m != NULL && sid != NULL) ? m->getEvent(sid) : NULL;
}


/**
 * Get the number of FunctionDefinition_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of FunctionDefinition_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumFunctionDefinitions (const Model_t *m)
{
  return (m != NULL) ? m->getNumFunctionDefinitions() : SBML_INT_MAX;
}


/**
 * Get the number of UnitDefinition_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of UnitDefinition_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumUnitDefinitions (const Model_t *m)
{
  return (m != NULL) ? m->getNumUnitDefinitions() : SBML_INT_MAX;
}


/**
 * Get the number of CompartmentType_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of CompartmentType_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumCompartmentTypes (const Model_t *m)
{
  return (m != NULL) ? m->getNumCompartmentTypes() : SBML_INT_MAX;
}


/**
 * Get the number of SpeciesType_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of SpeciesType_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumSpeciesTypes (const Model_t *m)
{
  return (m != NULL) ? m->getNumSpeciesTypes() : SBML_INT_MAX;
}


/**
 * Get the number of Compartment_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Compartment_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumCompartments (const Model_t *m)
{
  return (m != NULL) ? m->getNumCompartments() : SBML_INT_MAX;
}


/**
 * Get the number of Specie_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Specie_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumSpecies (const Model_t *m)
{
  return (m != NULL) ? m->getNumSpecies() : SBML_INT_MAX;
}


/**
 * Get the number of Species_t structure in this Model_t structure having
 * nonzero values for their "boundaryCondition" attribute.
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Species_t structures
 */
LIBSBML_EXTERN
unsigned int
Model_getNumSpeciesWithBoundaryCondition (const Model_t *m)
{
  return (m != NULL) ? m->getNumSpeciesWithBoundaryCondition() : SBML_INT_MAX;
}


/**
 * Get the number of Parameter_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Parameter_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumParameters (const Model_t *m)
{
  return (m != NULL) ? m->getNumParameters() : SBML_INT_MAX;
}


/**
 * Get the number of InitialAssignment_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of InitialAssignment_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumInitialAssignments (const Model_t *m)
{
  return (m != NULL) ? m->getNumInitialAssignments() : SBML_INT_MAX;
}


/**
 * Get the number of Rule_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Rule_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumRules (const Model_t *m)
{
  return (m != NULL) ? m->getNumRules() : SBML_INT_MAX;
}


/**
 * Get the number of Constraint_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Constraint_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumConstraints (const Model_t *m)
{
  return (m != NULL) ? m->getNumConstraints() : SBML_INT_MAX;
}

/**
 * Get the number of Reaction_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Reaction_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumReactions (const Model_t *m)
{
  return (m != NULL) ? m->getNumReactions() : SBML_INT_MAX;
}


/**
 * Get the number of Event_t structures in the given
 * Model_t structure
 *
 * @param m the Model_t structure
 *
 * @return an unsigned integer as the count of Event_t
 * structures in @p m
 */
LIBSBML_EXTERN
unsigned int
Model_getNumEvents (const Model_t *m)
{
  return (m != NULL) ? m->getNumEvents() : SBML_INT_MAX;
}


 /*
  * Populates the list of FormulaDataUnits with the units derived 
  * for the model. The list contains elements of class
  * FormulaUnitsData. 
  *
  * The first element of the list refers to the default units
  * of 'substance per time' derived from the model and has the
  * unitReferenceId 'subs_per_time'. This facilitates the comparison of units
  * derived from mathematical formula with the expected units.
  * 
  * The next elements of the list record the units of the 
  * compartments and species established from either explicitly
  * declared or default units.
  *
  * The next elements record the units of any parameters.
  *
  * Subsequent elements of the list record the units derived for
  * each mathematical expression encountered within the model.
  *
  * @param m the Model_t structure
  *
  * @note This function is utilised by the Unit Consistency Validator.
  * The List is populated prior to running the validation and thus
  * the consistency of units can be checked by accessing the members
  * of the list and comparing the appropriate data.
  */
LIBSBML_EXTERN
void 
Model_populateListFormulaUnitsData(Model_t *m)
{
  if (m != NULL) 
    m->populateListFormulaUnitsData();
}


 /*
  * Predicate returning @c true or @c false depending on whether 
  * the list of FormulaUnitsData is populated.
  *
  * @param m the Model_t structure
  * 
  * @return @c true if the list of FormulaUnitsData is populated, 
  * @c false otherwise.
  */
LIBSBML_EXTERN
int 
Model_isPopulatedListFormulaUnitsData(Model_t *m)
{
  return (m != NULL) ? 
    static_cast<int>( m->isPopulatedListFormulaUnitsData()) : 0;
}


/**
 * Removes the nth FunctionDefinition_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the FunctionDefinition_t sought
 *
 * @return the FunctionDefinition_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
FunctionDefinition_t*
Model_removeFunctionDefinition (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeFunctionDefinition(n);
  else
    return NULL;
}


/**
 * Removes the FunctionDefinition_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the FunctionDefinition_t sought
 *
 * @return the FunctionDefinition_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no FunctionDefinition_t
 * object with the identifier exists in this Model_t object.
 *
 */
LIBSBML_EXTERN
FunctionDefinition_t*
Model_removeFunctionDefinitionById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeFunctionDefinition(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth UnitDefinition_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the UnitDefinition_t sought
 *
 * @return the UnitDefinition_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
UnitDefinition_t*
Model_removeUnitDefinition (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeUnitDefinition(n);
  else
    return NULL;
}


/**
 * Removes the UnitDefinition_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the UnitDefinition_t sought
 *
 * @return the UnitDefinition_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no UnitDefinition_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
UnitDefinition_t*
Model_removeUnitDefinitionById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeUnitDefinition(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth CompartmentType_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the CompartmentType_t sought
 *
 * @return the CompartmentType_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
CompartmentType_t*
Model_removeCompartmentType (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeCompartmentType(n);
  else
    return NULL;
}


/**
 * Removes the CompartmentType_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the CompartmentType_t sought
 *
 * @return the CompartmentType_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no CompartmentType_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
CompartmentType_t*
Model_removeCompartmentTypeById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeCompartmentType(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth SpeciesType_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the SpeciesType_t sought
 *
 * @return the SpeciesType_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
SpeciesType_t*
Model_removeSpeciesType (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeSpeciesType(n);
  else
    return NULL;
}


/**
 * Removes the SpeciesType_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the SpeciesType_t sought
 *
 * @return the SpeciesType_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no SpeciesType_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
SpeciesType_t*
Model_removeSpeciesTypeById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeSpeciesType(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Compartment_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Compartment_t sought
 *
 * @return the Compartment_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Compartment_t*
Model_removeCompartment (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeCompartment(n);
  else
    return NULL;
}


/**
 * Removes the Compartment_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the Compartment_t sought
 *
 * @return the Compartment_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Compartment_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
Compartment_t*
Model_removeCompartmentById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeCompartment(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Species_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Species_t sought
 *
 * @return the Species_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Species_t*
Model_removeSpecies (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeSpecies(n);
  else
    return NULL;
}


/**
 * Removes the Species_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the Species_t sought
 *
 * @return the Species_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Species_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
Species_t*
Model_removeSpeciesById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeSpecies(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Parameter_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Parameter_t sought
 *
 * @return the Parameter_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Parameter_t*
Model_removeParameter (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeParameter(n);
  else
    return NULL;
}


/**
 * Removes the Parameter_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the Parameter_t sought
 *
 * @return the Parameter_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Parameter_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
Parameter_t*
Model_removeParameterById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeParameter(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth InitialAssignment_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the InitialAssignment_t sought
 *
 * @return the InitialAssignment_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
InitialAssignment_t*
Model_removeInitialAssignment (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeInitialAssignment(n);
  else
    return NULL;
}


/**
 * Removes the InitialAssignment_t object with the given "symbol" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "symbol" attribute of the InitialAssignment_t sought
 *
 * @return the InitialAssignment_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no InitialAssignment_t
 * object with the "symbol" attribute exists in this Model_t object.
 */
LIBSBML_EXTERN
InitialAssignment_t*
Model_removeInitialAssignmentBySym (Model_t *m, const char* symbol)
{
  if (m != NULL)
    return (symbol != NULL) ? m->removeInitialAssignment(symbol) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Rule_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Rule_t sought
 *
 * @return the Rule_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Rule_t*
Model_removeRule (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeRule(n);
  else
    return NULL;
}


/**
 * Removes the Rule_t object with the given "variable" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "variable" attribute of the Rule_t sought
 *
 * @return the Rule_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Rule_t
 * object with the "variable" attribute exists in this Model_t object.
 */
LIBSBML_EXTERN
Rule_t*
Model_removeRuleByVar (Model_t *m, const char* variable)
{
  if (m != NULL)
    return (variable != NULL) ? m->removeRule(variable) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Constraint_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Constraint_t sought
 *
 * @return the Constraint_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Constraint_t*
Model_removeConstraint (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeConstraint(n);
  else
    return NULL;
}


/**
 * Removes the nth Reaction_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Reaction_t sought
 *
 * @return the Reaction_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Reaction_t*
Model_removeReaction (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeReaction(n);
  else
    return NULL;
}


/**
 * Removes the Reaction_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the Reaction_t sought
 *
 * @return the Reaction_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Reaction_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
Reaction_t*
Model_removeReactionById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeReaction(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes the nth Event_t object from this Model_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param n the integer index of the Event_t sought
 *
 * @return the Event_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
Event_t*
Model_removeEvent (Model_t *m, unsigned int n)
{
  if (m != NULL)
    return m->removeEvent(n);
  else
    return NULL;
}


/**
 * Removes the Event_t object with the given "id" attribute
 * from this Model_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Model_t structure
 * @param sid the string of the "id" attribute of the Event_t sought
 *
 * @return the Event_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no Event_t
 * object with the identifier exists in this Model_t object.
 */
LIBSBML_EXTERN
Event_t*
Model_removeEventById (Model_t *m, const char* sid)
{
  if (m != NULL)
    return (sid != NULL) ? m->removeEvent(sid) : NULL;
  else
    return NULL;
}


/* NOT YET USED but leave in case of future need 

  * Adds a copy of the given FormulaUnitsData object to this Model.
  *
  * @param m the Model_t structure
  * @param fud the FormulaUnitsData_t structure to add

LIBSBML_EXTERN
void 
Model_addFormulaUnitsData (Model_t *m, FormulaUnitsData_t* fud)
{
  m->addFormulaUnitsData(fud);
}



 * Creates a new FormulaUnitsData inside this Model and returns it.
 *
 * @param m the Model_t structure
 *
 * @return the FormulaUnitsData_t structure created

LIBSBML_EXTERN
FormulaUnitsData_t* 
Model_createFormulaUnitsData (Model_t *m)
{
  return m->createFormulaUnitsData();
}


 * Get the nth FormulaUnitsData object in this Model.
 *
 * @param m the Model_t structure
 * 
 * @return the nth FormulaUnitsData of this Model.

LIBSBML_EXTERN
FormulaUnitsData_t* 
Model_getFormulaUnitsData (Model_t *m, unsigned int n)
{
  return m->getFormulaUnitsData(n);
}



  * Get a FormulaUnitsData object based on its unitReferenceId and typecode.
  *
  * @param m the Model_t structure
  * 
  * @return the FormulaUnitsData in this Model with the unitReferenceId @p sid 
  * and the typecode (int) @p typecode or @c NULL
  * if no such FormulaUnitsData exists.
  *
  * @note The typecode (int) parameter is necessary as the unitReferenceId
  * of the FormulaUnitsData need not be unique. For example if a Species
  * with id 's' is assigned by an AssignmentRule there will be two 
  * elements of the FormulaUnitsData List with the unitReferenceId 's'; 
  * one with
  * typecode 'SBML_SPECIES' referring to the units related to the species, 
  * the other with typecode 'SBML_ASSIGNMENT_RULE' referring to the units
  * derived from the math element of the AssignmentRule.

LIBSBML_EXTERN
FormulaUnitsData_t* 
Model_getFormulaUnitsDataById(Model_t *m, const char* sid, 
                                          int typecode)
{
  return m->getFormulaUnitsData(sid, typecode);
}


  * Get the number of FormulaUnitsData objects in this Model.
  * 
  * @param m the Model_t structure
  * 
  * @return the number of FormulaUnitsData in this Model.

LIBSBML_EXTERN
unsigned int 
Model_getNumFormulaUnitsData (Model_t *m)
{
  return m->getNumFormulaUnitsData();
}



  * Get the list of FormulaUnitsData object in this Model.
  * 
  * @param m the Model_t structure
  * 
 * @return the list of FormulaUnitsData for this Model.

LIBSBML_EXTERN
List_t* 
Model_getListFormulaUnitsData (Model_t *m)
{
  return m->getListFormulaUnitsData();
}


*/

/** @endcond */
LIBSBML_CPP_NAMESPACE_END
