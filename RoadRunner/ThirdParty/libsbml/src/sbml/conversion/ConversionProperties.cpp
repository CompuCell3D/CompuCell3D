/**
* @file    ConversionProperties.cpp
* @brief   Implemenentation of ConversionProperties, the class encapsulating conversion configuration.
* @author  Frank Bergmann
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
* ------------------------------------------------------------------------ -->
*/

#ifdef __cplusplus

#include <sbml/conversion/ConversionProperties.h>
#include <sbml/conversion/ConversionOption.h>
#include <sbml/util/util.h>
#include <sbml/SBase.h>

#include <algorithm>
#include <string>
#include <sstream>

using namespace std;
LIBSBML_CPP_NAMESPACE_BEGIN



ConversionProperties::ConversionProperties(SBMLNamespaces* targetNS) : mTargetNamespaces(NULL)
{
  if (targetNS != NULL) mTargetNamespaces = targetNS->clone();
}

ConversionProperties::ConversionProperties(const ConversionProperties& orig)
{
  
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {    
    if (orig.mTargetNamespaces != NULL)
      mTargetNamespaces = orig.mTargetNamespaces->clone();
    else 
      mTargetNamespaces = NULL;

    map<string, ConversionOption*>::const_iterator it;
    for (it = orig.mOptions.begin(); it != orig.mOptions.end(); it++)
    {
      mOptions.insert(pair<string, ConversionOption*>
        ( it->second->getKey(), it->second->clone()));
    }
  }
}

ConversionProperties& 
ConversionProperties::operator=(const ConversionProperties& rhs)
{
    if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else
  {
    if (rhs.mTargetNamespaces != NULL)
      mTargetNamespaces = rhs.mTargetNamespaces->clone();
    else 
      mTargetNamespaces = NULL;

    map<string, ConversionOption*>::const_iterator it;
    for (it = rhs.mOptions.begin(); it != rhs.mOptions.end(); it++)
    {
      mOptions.insert(pair<string, ConversionOption*>
        ( it->second->getKey(), it->second->clone()));
    }

  }
    return *this;
}

ConversionProperties* 
ConversionProperties::clone() const
{
  return new ConversionProperties(*this);
}

ConversionProperties::~ConversionProperties()
{
  if (mTargetNamespaces != NULL)
  {
    delete mTargetNamespaces;
    mTargetNamespaces = NULL;
  }

  map<string, ConversionOption*>::iterator it;
  for (it = mOptions.begin(); it != mOptions.end(); it++)
  {
    if (it->second != NULL) 
    { 
      delete it->second;
      it->second=NULL;
    }
  }

}

SBMLNamespaces * 
ConversionProperties::getTargetNamespaces() const
{
  return mTargetNamespaces;
}

bool 
ConversionProperties::hasTargetNamespaces() const
{
  return mTargetNamespaces != NULL;
}


void 
ConversionProperties::setTargetNamespaces(SBMLNamespaces *targetNS)
{
  if (mTargetNamespaces != NULL) 
  {
      delete mTargetNamespaces;
      mTargetNamespaces = NULL;
  }
  if (targetNS == NULL) return;
  
  mTargetNamespaces = targetNS->clone();
}

std::string 
ConversionProperties::getDescription(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getDescription();

  return "";
}

ConversionOptionType_t 
ConversionProperties::getType(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getType();

  return CNV_TYPE_STRING;
}


ConversionOption* 
ConversionProperties::getOption(std::string key) const
{

  map<string, ConversionOption*>::const_iterator it;
  for (it = mOptions.begin(); it != mOptions.end(); it++)
  {
    if (it->second != NULL && it->second->getKey() == key)
      return it->second;
  }
  return NULL;
}

void 
ConversionProperties::addOption(const ConversionOption &option)
{
  if (&option == NULL) return;
  mOptions.insert(pair<string, ConversionOption*>( option.getKey(), option.clone()));
}

void 
ConversionProperties::addOption(std::string key, std::string value, 
    ConversionOptionType_t type, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, type, description) ));
}
void 
ConversionProperties::addOption(std::string key, const char* value, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, description) ));
}
void 
ConversionProperties::addOption(std::string key, bool value, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, description) ));
}
void 
ConversionProperties::addOption(std::string key, double value, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, description) ));
}
void 
ConversionProperties::addOption(std::string key, float value, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, description) ));
}
void 
ConversionProperties::addOption(std::string key, int value, 
    std::string description)
{
  mOptions.insert(pair<string, ConversionOption*>( key, new ConversionOption(key, value, description) ));
}

ConversionOption* 
ConversionProperties::removeOption(std::string key)
{
  ConversionOption* result = getOption(key);
  if (result != NULL)
    mOptions.erase(key);
  return result;
}

bool 
ConversionProperties::hasOption(std::string key) const
{
  return (getOption(key) != NULL);
}

std::string 
ConversionProperties::getValue(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getValue();
  return "";
}

void 
ConversionProperties::setValue(std::string key, std::string value)
{
  ConversionOption *option = getOption(key);
  if (option != NULL) option->setValue(value);
}


bool 
ConversionProperties::getBoolValue(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getBoolValue();
  return false;
}

void 
ConversionProperties::setBoolValue(std::string key, bool value)
{
  ConversionOption *option = getOption(key);
  if (option != NULL) option->setBoolValue(value);
}

double 
ConversionProperties::getDoubleValue(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getDoubleValue();
  return std::numeric_limits<double>::quiet_NaN();
}

void 
ConversionProperties::setDoubleValue(std::string key, double value)
{
  ConversionOption *option = getOption(key);
  if (option != NULL) option->setDoubleValue(value);
}

float 
ConversionProperties::getFloatValue(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getFloatValue();
  return std::numeric_limits<float>::quiet_NaN();
}

void 
ConversionProperties::setFloatValue(std::string key, float value)
{
  ConversionOption *option = getOption(key);
  if (option != NULL) option->setFloatValue(value);

}

int 
ConversionProperties::getIntValue(std::string key) const
{
  ConversionOption *option = getOption(key);
  if (option != NULL) return option->getIntValue();
  return -1;
}

void 
ConversionProperties::setIntValue(std::string key, int value)
{
  ConversionOption *option = getOption(key);
  if (option != NULL) option->setIntValue(value);

}

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */



