/**
 * @file    ConversionOption.h
 * @brief   Definition of ConversionOption, the class encapsulating conversion options.
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
 * @class ConversionOption
 * @brief Class of object that encapsulates a conversion option.
 * 
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * LibSBML provides a number of converters that can perform transformations
 * on SBML documents.  These converters allow their behaviors to be
 * controlled by setting property values.  Converter properties are
 * communicated using objects of class ConversionProperties, and within
 * such objects, individual options are encapsulated using ConversionOption
 * objects.
 *
 * A ConversionOption object consists of four parts:
 * @li A @em key, acting as the name of the option;
 * @li A @em value of this option;
 * @li A @em type for the value; this is chosen from  the enumeration type
 * <a class="el" href="#ConversionOptionType_t">ConversionOptionType_t</a>; and
 * @li A @em description consisting of a text string that describes the
 * option in some way.
 *
 * There are no constraints on the values of keys or descriptions;
 * authors of SBML converters are free to choose them as they see fit.
 *
 * @section ConversionOptionType_t Conversion option data types
 *
 * An option in ConversionOption must have a data type declared, to
 * indicate whether it is a string value, an integer, and so forth.  The
 * possible types of values are taken from the enumeration <a
 * class="el" href="#ConversionOptionType_t">ConversionOptionType_t</a>.
 * The following are the possible values:
 * 
 * <p>
 * <center>
 * <table width="90%" cellspacing="1" cellpadding="1" border="0" class="normal-font">
 *  <tr style="background: lightgray" class="normal-font">
 *      <td><strong>Enumerator</strong></td>
 *      <td><strong>Meaning</strong></td>
 *  </tr>
 * <tr>
 * <td><code>@link ConversionOptionType_t#CNV_TYPE_BOOL CNV_TYPE_BOOL@endlink</code></td>
 * <td>Indicates the value type is a Boolean.</td>
 * </tr>
 * <tr>
 * <td><code>@link ConversionOptionType_t#CNV_TYPE_DOUBLE CNV_TYPE_DOUBLE@endlink</code></td>
 * <td>Indicates the value type is a double-sized float.</td>
 * </tr>
 * <tr>
 * <td><code>@link ConversionOptionType_t#CNV_TYPE_INT CNV_TYPE_INT@endlink</code></td>
 * <td>Indicates the value type is an integer.</td>
 * </tr>
 * <tr>
 * <td><code>@link ConversionOptionType_t#CNV_TYPE_SINGLE CNV_TYPE_SINGLE@endlink</code></td>
 * <td>Indicates the value type is a float.</td>
 * </tr>
 * <tr>
 * <td><code>@link ConversionOptionType_t#CNV_TYPE_STRING CNV_TYPE_STRING@endlink</code></td>
 * <td>Indicates the value type is a string.</td>
 * </tr>
 * </table>
 * </center>
 *
 * @see ConversionProperties
 */

#ifndef ConversionOption_h
#define ConversionOption_h


#include <sbml/common/extern.h>


LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * @enum  ConversionOptionType_t
 * @brief ConversionOptionType_t is the enumeration of possible option types.
 */
typedef enum
{
    CNV_TYPE_BOOL     /*!< The Boolean option value type. */
  , CNV_TYPE_DOUBLE   /*!< The double-sized float option value type. */
  , CNV_TYPE_INT      /*!< The integer option value type. */
  , CNV_TYPE_SINGLE   /*!< The float option value type. */
  , CNV_TYPE_STRING   /*!< The string option value type. */
} ConversionOptionType_t;

LIBSBML_CPP_NAMESPACE_END


#ifdef __cplusplus
#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN ConversionOption
{
public:
  /**
   * Creates a new ConversionOption.
   *
   * This is the general constructor, taking arguments for all aspects of
   * an option.  Other constructors exist with different arguments.
   * 
   * @param key the key for this option
   * @param value an optional value for this option
   * @param type the type of this option
   * @param description the description for this option
   */
  ConversionOption(std::string key, std::string value="", 
                   ConversionOptionType_t type=CNV_TYPE_STRING, 
                   std::string description="");

  
  /**
   * Creates a new ConversionOption specialized for string-type options.
   * 
   * @param key the key for this option
   * @param value the value for this option
   * @param description an optional description
   */
  ConversionOption(std::string key, const char* value, 
                   std::string description="");


  /**
   * Creates a new ConversionOption specialized for Boolean-type options.
   * 
   * @param key the key for this option
   * @param value the value for this option
   * @param description an optional description
   */
  ConversionOption(std::string key, bool value, 
                   std::string description="");


  /**
   * Creates a new ConversionOption specialized for double-type options.
   * 
   * @param key the key for this option
   * @param value the value for this option
   * @param description an optional description
   */
  ConversionOption(std::string key, double value, 
                   std::string description="");


  /**
   * Creates a new ConversionOption specialized for float-type options.
   * 
   * @param key the key for this option
   * @param value the value for this option
   * @param description an optional description
   */
  ConversionOption(std::string key, float value, 
                   std::string description="");


  /**
   * Creates a new ConversionOption specialized for integer-type options.
   * 
   * @param key the key for this option
   * @param value the value for this option
   * @param description an optional description
   */
  ConversionOption(std::string key, int value, 
                   std::string description="");


  /**
   * Copy constructor; creates a copy of an ConversionOption object.
   *
   * @param orig the ConversionOption object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  ConversionOption(const ConversionOption& orig);


  /**
   * Assignment operator for ConversionOption.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  ConversionOption& operator=(const ConversionOption& rhs);


  /**
   * Destroys this object.
   */ 
  virtual ~ConversionOption();


  /** 
   * Creates and returns a deep copy of this ConversionOption object.
   * 
   * @return a (deep) copy of this ConversionOption object.
   */
  virtual ConversionOption* clone() const;


  /**
   * Returns the key for this option.
   * 
   * @return the key, as a string.
   */
  virtual std::string getKey() const; 


  /**
   * Sets the key for this option.
   * 
   * @param key a string representing the key to set.
   */
  virtual void setKey(std::string key);


  /**
   * Returns the value of this option.
   * 
   * @return the value of this option, as a string.
   */
  virtual std::string getValue() const;


  /**
   * Sets the value for this option.
   * 
   * @param value the value to set, as a string.
   */
  virtual void setValue(std::string value);


  /**
   * Returns the description string for this option.
   * 
   * @return the description of this option.
   */
  virtual std::string getDescription() const;


  /**
   * Sets the description text for this option.
   * 
   * @param description the description to set for this option.
   */
  virtual void setDescription(std::string description);


  /**
   * Returns the type of this option
   * 
   * @return the type of this option.
   */
  virtual ConversionOptionType_t getType() const;


  /**
   * Sets the type of this option.
   * 
   * @param type the type value to use.
   */
  virtual void setType(ConversionOptionType_t type);


  /**
   * Returns the value of this option as a Boolean.
   * 
   * @return the value of this option.
   */   
  virtual bool getBoolValue() const;


  /** 
   * Set the value of this option to a given Boolean value.
   *
   * Invoking this method will also set the type of the option to
   * @link ConversionOptionType_t#CNV_TYPE_BOOL CNV_TYPE_BOOL@endlink.
   * 
   * @param value the Boolean value to set
   */
  virtual void setBoolValue(bool value);


  /**
   * Returns the value of this option as a @c double.
   * 
   * @return the value of this option.
   */   
  virtual double getDoubleValue() const;


  /** 
   * Set the value of this option to a given @c double value.
   *
   * Invoking this method will also set the type of the option to
   * @link ConversionOptionType_t#CNV_TYPE_DOUBLE CNV_TYPE_DOUBLE@endlink.
   * 
   * @param value the value to set
   */
  virtual void setDoubleValue(double value);


  /**
   * Returns the value of this option as a @c float.
   * 
   * @return the value of this option as a float
   */   
  virtual float getFloatValue() const;


  /** 
   * Set the value of this option to a given @c float value.
   *
   * Invoking this method will also set the type of the option to
   * @link ConversionOptionType_t#CNV_TYPE_SINGLE CNV_TYPE_SINGLE@endlink.
   * 
   * @param value the value to set
   */
  virtual void setFloatValue(float value);


  /**
   * Returns the value of this option as an @c integer.
   * 
   * @return the value of this option, as an int
   */   
  virtual int getIntValue() const;


  /** 
   * Set the value of this option to a given @c int value.
   *
   * Invoking this method will also set the type of the option to
   * @link ConversionOptionType_t#CNV_TYPE_INT CNV_TYPE_INT@endlink.
   * 
   * @param value the value to set
   */
  virtual void setIntValue(int value);


protected: 
  /** @cond doxygen-libsbml-internal */

  std::string mKey;
  std::string mValue;
  ConversionOptionType_t mType;
  std::string mDescription;

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
#endif /* !ConversionOption */

