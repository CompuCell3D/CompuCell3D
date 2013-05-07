/**
 * @file    ConversionOption.cpp
 * @brief   Implementation of ConversionOption, the class encapsulating conversion options.
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

#include <sbml/conversion/ConversionOption.h>
#include <sbml/SBase.h>

#include <algorithm>
#include <string>
#include <sstream>

using namespace std;
LIBSBML_CPP_NAMESPACE_BEGIN

ConversionOption::ConversionOption(string key, string value, 
    ConversionOptionType_t type, 
    string description) : 
    mKey(key)
  , mValue(value)
  , mType(type)
  , mDescription(description)
{
}

ConversionOption::ConversionOption(std::string key, const char* value, 
  std::string description) : 
    mKey(key)
  , mValue(value)
  , mType(CNV_TYPE_STRING)
  , mDescription(description)
{
}

ConversionOption::ConversionOption(std::string key, bool value, 
  std::string description) : 
    mKey(key)
  , mValue("")
  , mType(CNV_TYPE_STRING)
  , mDescription(description)
{
  setBoolValue(value);
}

ConversionOption::ConversionOption(std::string key, double value, 
  std::string description): 
    mKey(key)
  , mValue("")
  , mType(CNV_TYPE_STRING)
  , mDescription(description)
{
  setDoubleValue(value);
}

ConversionOption::ConversionOption(std::string key, float value, 
  std::string description) : 
    mKey(key)
  , mValue("")
  , mType(CNV_TYPE_STRING)
  , mDescription(description)
{
  setFloatValue(value);
}

ConversionOption::ConversionOption(std::string key, int value, 
  std::string description) : 
    mKey(key)
  , mValue("")
  , mType(CNV_TYPE_STRING)
  , mDescription(description)
{
      setIntValue(value);
}


ConversionOption::ConversionOption
  (const ConversionOption& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mDescription = orig.mDescription;
    mKey = orig.mKey;
    mType = orig.mType;
    mValue = orig.mValue;
  }
}



ConversionOption& 
ConversionOption::operator=(const ConversionOption& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else
  {
    mDescription = rhs.mDescription;
    mKey = rhs.mKey;
    mType = rhs.mType;
    mValue = rhs.mValue;
  }
   return *this;
}

ConversionOption* 
ConversionOption::clone() const
{
  return new ConversionOption(*this);
}

ConversionOption::~ConversionOption() {}

string 
ConversionOption::getKey() const
{
  return mKey;
}

void 
ConversionOption::setKey(string key)
{
  mKey = key;
}

string 
ConversionOption::getValue() const
{
  return mValue;
}

void 
ConversionOption::setValue(string value)
{
  mValue = value;
}

string 
ConversionOption::getDescription() const
{
  return mDescription;
}

void 
ConversionOption::setDescription(string description)
{
  mDescription = description;
}

ConversionOptionType_t 
ConversionOption::getType() const
{
  return mType;
}

void 
ConversionOption::setType(ConversionOptionType_t type)
{
  mType = type;
}

bool 
ConversionOption::getBoolValue() const
{
  string value = mValue;
#ifdef __BORLANDC__
   std::transform(value.begin(), value.end(), value.begin(),  (int(*)(int))
std::tolower);
#else
   std::transform(value.begin(), value.end(), value.begin(), ::tolower);
#endif
  if (value == "true") return true;
  if (value == "false") return false;

  stringstream str; str << mValue;
  bool result; str >> result;
  return result;
}

void 
ConversionOption::setBoolValue(bool value)
{  
  mValue = (value ? "true" : "false");
  setType(CNV_TYPE_BOOL);
}

double 
ConversionOption::getDoubleValue() const
{
  stringstream str; str << mValue;
  double result; str >> result;
  return result;
}
 
void 
ConversionOption::setDoubleValue(double value)
{
  stringstream str; str << value;
  mValue = str.str();
  setType(CNV_TYPE_DOUBLE);
}

 
float 
ConversionOption::getFloatValue() const
{
  stringstream str; str << mValue;
  float result; str >> result;
  return result;
}

void 
ConversionOption::setFloatValue(float value)
{
  stringstream str; str << value;
  mValue = str.str();
  setType(CNV_TYPE_SINGLE);
}

 
int 
ConversionOption::getIntValue() const
{
  stringstream str; str << mValue;
  int result; str >> result;
  return result;
}
 
void 
ConversionOption::setIntValue(int value)
{
  stringstream str; str << value;
  mValue = str.str();
  setType(CNV_TYPE_INT);
}

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */



