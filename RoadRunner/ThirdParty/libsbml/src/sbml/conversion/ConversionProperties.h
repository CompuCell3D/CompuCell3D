/**
 * @file    ConversionProperties.h
 * @brief   Definition of ConversionProperties, the class encapsulating conversion configuration.
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
 *
 * @class ConversionProperties
 * @brief Class of object that encapsulates the properties of an SBML converter.
 * 
 * @htmlinclude libsbml-facility-only-warning.html
 * 
 * The properties of SBML converters are communicated using objects of
 * class ConversionProperties, and within such objects, individual options
 * are encapsulated using ConversionOption objects.  The ConversionProperties
 * class provides numerous methods for setting and getting options.
 *
 * ConversionProperties objects are also used to determine the target SBML
 * namespace when an SBML converter's behavior depends on the intended
 * Level+Version combination of SBML.  In addition, it is conceivable that
 * conversions may be affected by SBML Level&nbsp;3 packages being used
 * by an SBML document.  These, too, are communicated by the values of
 * the SBML namespaces set on a ConversionProperties object.
 *
 * @see ConversionOption
 * @see SBMLNamespaces
 */

#ifndef ConversionProperties_h
#define ConversionProperties_h


#include <sbml/common/extern.h>
#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/ConversionOption.h>


#ifdef __cplusplus

#include <map>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN ConversionProperties
{
public:

  /** 
   * Constructor that initializes the conversion properties
   * with a specific SBML target namespace.
   * 
   * @param targetNS the target namespace to convert to
   */
  ConversionProperties(SBMLNamespaces* targetNS=NULL);


  /** 
   * Copy constructor.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  ConversionProperties(const ConversionProperties& orig);

  
  /**
   * Assignment operator for conversion properties.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  ConversionProperties& operator=(const ConversionProperties& rhs);


  /** 
   * Creates and returns a deep copy of this ConversionProperties object.
   * 
   * @return a (deep) copy of this ConversionProperties object.
   */
  virtual ConversionProperties* clone() const; 


  /**
   * Destructor.
   */
  virtual ~ConversionProperties();


  /**
   * Returns the current target SBML namespace.
   *
   * @return the SBMLNamepaces object expressing the target namespace.
   */ 
  virtual SBMLNamespaces * getTargetNamespaces() const;


  /**
   * Returns @c true if the target SBML namespace has been set.
   * 
   * @return @c true if the target namespace has been set, @c false
   * otherwise.
   */
  virtual bool hasTargetNamespaces() const;


  /** 
   * Sets the target namespace.
   * 
   * @param targetNS the target namespace to use.
   */
  virtual void setTargetNamespaces(SBMLNamespaces *targetNS);


  /**
   * Returns the description string for a given option in this properties
   * object.
   * 
   * @param key the key for the option.
   * 
   * @return the description text of the option with the given key.
   */
  virtual std::string getDescription(std::string key) const;


  /**
   * Returns the type of a given option in this properties object.
   * 
   * @param key the key for the option.
   * 
   * @return the type of the option with the given key.
   */
  virtual ConversionOptionType_t  getType(std::string key) const;


  /**
   * Returns the ConversionOption object for a given key.
   * 
   * @param key the key for the option.
   * 
   * @return the option with the given key.
   */
  virtual ConversionOption* getOption(std::string key) const;  


  /**
   * Adds a copy of the given option to this properties object.
   * 
   * @param option the option to add
   */
  virtual void addOption(const ConversionOption& option);


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value (optional) the value of that option
   * @param type (optional) the type of the option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, std::string value="", 
                         ConversionOptionType_t type=CNV_TYPE_STRING, 
                         std::string description="");


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value the string value of that option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, const char* value, 
                         std::string description="");


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value the boolean value of that option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, bool value, 
    std::string description="");


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value the double value of that option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, double value, 
                         std::string description="");


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value the float value of that option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, float value, 
                         std::string description="");


  /**
   * Adds a new ConversionOption object with the given parameters.
   * 
   * @param key the key for the new option
   * @param value the integer value of that option
   * @param description (optional) the description for the option
   */
  virtual void addOption(std::string key, int value, 
                         std::string description="");


  /**
   * Removes the option with the given key from this properties object.
   * 
   * @param key the key for the new option to remove
   * @return the removed option
   */
  virtual ConversionOption* removeOption(std::string key);


  /** 
   * Returns @c true if this properties object contains an option with
   * the given key.
   * 
   * @param key the key of the option to find.
   * 
   * @return @c true if an option with the given @p key exists in
   * this properties object, @c false otherwise.
   */
  virtual bool hasOption(std::string key) const;  
  

  /**
   * Returns the value of the given option as a string.
   * 
   * @param key the key for the option.
   * 
   * @return the string value of the option with the given key.
   */
  virtual std::string getValue(std::string key) const;  


  /**
   * Sets the value of the given option to a string.
   * 
   * @param key the key for the option
   * @param value the new value
   */
  virtual void setValue(std::string key, std::string value);  
  

  /**
   * Returns the value of the given option as a Boolean.
   * 
   * @param key the key for the option.
   * 
   * @return the boolean value of the option with the given key.
   */
  virtual bool getBoolValue(std::string key) const;


  /**
   * Sets the value of the given option to a Boolean.
   * 
   * @param key the key for the option.
   * 
   * @param value the new Boolean value.
   */
  virtual void setBoolValue(std::string key, bool value);

  
  /**
   * Returns the value of the given option as a @c double.
   * 
   * @param key the key for the option.
   * 
   * @return the double value of the option with the given key.
   */
  virtual double getDoubleValue(std::string key) const;


  /**
   * Sets the value of the given option to a @c double.
   * 
   * @param key the key for the option.
   * 
   * @param value the new double value.
   */
  virtual void setDoubleValue(std::string key, double value);

  
  /**
   * Returns the value of the given option as a @c float.
   * 
   * @param key the key for the option.
   * 
   * @return the float value of the option with the given key.
   */
  virtual float getFloatValue(std::string key) const;


  /**
   * Sets the value of the given option to a @c float.
   * 
   * @param key the key for the option.
   * 
   * @param value the new float value.
   */
  virtual void setFloatValue(std::string key, float value);

  
  /**
   * Returns the value of the given option as an integer.
   * 
   * @param key the key for the option.
   * 
   * @return the int value of the option with the given key.
   */
  virtual int getIntValue(std::string key) const;


  /**
   * Sets the value of the given option to an integer.
   * 
   * @param key the key for the option.
   * 
   * @param value the new integer value.
   */
  virtual void setIntValue(std::string key, int value);


protected:
  /** @cond doxygen-libsbml-internal */

  SBMLNamespaces *mTargetNamespaces;
  std::map<std::string, ConversionOption*> mOptions;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif /* !ConversionProperties_h */

