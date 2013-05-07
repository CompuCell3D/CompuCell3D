/**
 * @cond doxygen-libsbml-internal
 * 
 * @file    SBMLExternalValidator.h
 * @brief   Definition of SBMLExternalValidator, a validator calling external programs
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

#ifndef SBMLExternalValidator_h
#define SBMLExternalValidator_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/validator/SBMLValidator.h>
#ifndef LIBSBML_USE_STRICT_INCLUDES
#include <sbml/SBMLTypes.h>
#endif


#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLExternalValidator : public SBMLValidator
{
public:

  /** @cond doxygen-libsbml-internal */
  

  /** @endcond */


  /**
   * Constructor.
   */
  SBMLExternalValidator();


  /**
   * Copy constructor.
   */
  SBMLExternalValidator(const SBMLExternalValidator&);

  /**
   * Creates and returns a deep copy of this converter.
   * 
   * @return a (deep) copy of this converter.
   */
  virtual SBMLValidator* clone() const;

  
  /** 
   * the actual conversion 
   * 
   * @return status code represeting success/failure/conversion impossible
   */
  virtual unsigned int validate();


  /** 
   * @return the program name of the validator to be run
   */
  const std::string& getProgram()  const;


  /**
   * Sets the name of the program to run
   *
   * @param program the program to be started
   */
  void setProgram (std::string program);


  /**
   * @return the output file name (this is the file the external program will write)
   */
  const std::string& getOutputFileName()  const;


  /**
   * Sets the output file name
   *
   * @param outputFileName the name of the output XML file
   */
  void setOutputFileName(std::string outputFileName);


  /** 
   * @return the name of the SBML file (the document of this validator will be written to it)
   */
  const std::string& getSBMLFileName()  const;


  /**
   * Sets the filename for the temporary file to be created
   *
   * @param sbmlFileName the temporary name
   */
  void setSBMLFileName(std::string sbmlFileName);


  /**
   * Clear all additional arguments
   */
  void clearArguments();


  /** 
   * Adds the given argument to the list of additional arguments 
   *
   * @param arg the argument
   */
  void addArgument(std::string arg);

  /** 
   * @return the number of arguments. 
   */
  unsigned int getNumArguments() const;

  /** 
   * Returns the argument for the given index. 
   * 
   * @param n the zero based index of the argument. 
   *
   * @return the argument at the given index. 
   */
  std::string getArgument(unsigned int n) const;

  /**
   * @return all arguments
   */
  const std::vector<std::string>& getArguments() const;


  /** 
   * Sets the additional arguments
   *
   * @param args teh additional arguments
   */
  void setArguments(std::vector<std::string> args);

protected:

  std::string mProgram;
  std::vector<std::string> mArguments;
  std::string mSBMLFileName;
  std::string mOutputFileName;


private:
  /** @cond doxygen-libsbml-internal */

  /** @endcond */


};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

  
#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* SBMLExternalValidator_h */


/** @endcond */
