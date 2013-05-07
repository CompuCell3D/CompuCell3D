/**
 * @file    SBMLWriter.h
 * @brief   Writes an SBML Document to file or in-memory string
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
 * ------------------------------------------------------------------------ -->
 *
 * @class SBMLWriter
 * @brief Methods for writing SBML to files and text strings.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The SBMLWriter class is the converse of SBMLReader, and provides the
 * main interface for serializing SBML models into XML and writing the
 * result to files and text strings.  The methods for writing SBML all take
 * an SBMLDocument object and a destination.  They return a boolean or
 * integer value to indicate success or failure.
 *
 * @section compression Support for writing compressed files
 *
 * LibSBML provides support for writing (as well as reading) compressed
 * SBML files.  The process is transparent to the calling
 * application&mdash;the application does not need to do anything
 * deliberate to invoke the functionality.  If a given SBML filename ends
 * with an extension for the @em gzip, @em zip or @em bzip2 compression
 * formats (respectively, <code>&quot;.gz&quot;</code>,
 * <code>&quot;.zip&quot;</code>, or <code>&quot;.bz2&quot;</code>),
 * then the methods
 * SBMLWriter::writeSBML(@if java SBMLDocument d, String filename@endif)
 * and SBMLReader::readSBML(@if java String filename@endif)
 * will automatically compress and decompress the file while writing and
 * reading it.  If the filename has no such extension, it
 * will be written and read uncompressed as normal.
 *
 * The compression feature requires that the @em zlib (for @em gzip and @em
 * zip formats) and/or @em bzip2 (for @em bzip2 format) be available on the
 * system running libSBML, and that libSBML was configured with their
 * support compiled-in.  Please see the libSBML @if clike <a href="libsbml-installation.html">installation instructions</a>@endif@if python <a href="libsbml-installation.html">installation instructions</a>@endif@if java  <a href="../../../libsbml-installation.html">installation instructions</a>@endif@~ for 
 * more information about this.  The methods
 * SBMLWriter::hasZlib() and
 * SBMLWriter::hasBzip2()
 * can be used by an application to query at run-time whether support
 * for the compression libraries is available in the present copy of
 * libSBML.
 *
 * Support for compression is not mandated by the SBML standard, but
 * applications may find it helpful, particularly when large SBML models
 * are being communicated across data links of limited bandwidth.
 */

#ifndef SBMLWriter_h
#define SBMLWriter_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus


#include <iosfwd>
#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLDocument;


class LIBSBML_EXTERN SBMLWriter
{
public:

  /**
   * Creates a new SBMLWriter.
   *
   * The libSBML SBMLWriter objects offer methods for writing SBML in
   * XML form to files and text strings.
   */
  SBMLWriter  ();


  /**
   * Destroys this SBMLWriter.
   */
  ~SBMLWriter ();


  /**
   * Sets the name of this program, i.e., the program that is about to
   * write out the SBMLDocument.
   *
   * If the program name and version are set (see
   * SBMLWriter::setProgramVersion(@if java String version@endif)), the
   * following XML comment, intended for human consumption, will be written
   * at the beginning of the XML document:
   * @verbatim
 <!-- Created by <program name> version <program version>
 on yyyy-MM-dd HH:mm with libSBML version <libsbml version>. -->
 @endverbatim
   *
   * If the program name and version are not set at some point before
   * calling the writeSBML() methods, no such comment is written out.
   *
   * @param name the name of this program (where "this program" refers to
   * program in which libSBML is embedded, not libSBML itself!)
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * 
   * @see setProgramVersion(const std::string& version)
   */
  int setProgramName (const std::string& name);


  /**
   * Sets the version of this program, i.e., the program that is about to
   * write out the SBMLDocument.
   *
   * If the program version and name are set (see
   * SBMLWriter::setProgramName(@if java String name@endif)), the
   * following XML comment, intended for human consumption, will be written
   * at the beginning of the document:
   * @verbatim
 <!-- Created by <program name> version <program version>
 on yyyy-MM-dd HH:mm with libSBML version <libsbml version>. -->
 @endverbatim
   *
   * If the program version and name are not set at some point before
   * calling the writeSBML() methods, no such comment is written out.
   *
   * @param version the version of this program (where "this program"
   * refers to program in which libSBML is embedded, not libSBML itself!)
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   *
   * @see setProgramName(const std::string& name)
   */
  int setProgramVersion (const std::string& version);


  /**
   * Writes the given SBML document to filename.
   *
   * @htmlinclude assuming-compressed-file.html
   *
   * @param d the SBML document to be written
   *
   * @param filename the name or full pathname of the file where the SBML
   * is to be written. 
   *
   * @return @c true on success and @c false if the filename could not be
   * opened for writing.
   *
   * @note @htmlinclude note-writing-zipped-files.html
   * 
   * @see setProgramVersion(const std::string& version)
   * @see setProgramName(const std::string& name)
   */
  bool writeSBML (const SBMLDocument* d, const std::string& filename);


  /**
   * Writes the given SBML document to the output stream.
   *
   * @param d the SBML document to be written
   *
   * @param stream the stream object where the SBML is to be written.
   *
   * @return @c true on success and @c false if one of the underlying
   * parser components fail (rare).
   * 
   * @see setProgramVersion(const std::string& version)
   * @see setProgramName(const std::string& name)
   */
  bool writeSBML (const SBMLDocument* d, std::ostream& stream);


  /** @cond doxygen-libsbml-internal */

  /**
   * Writes the given SBML document to an in-memory string and returns a
   * pointer to it.
   *
   * The string is owned by the caller and should be freed (with @c free())
   * when no longer needed.
   *
   * @param d the SBML document to be written
   *
   * @return the string on success and @c 0 if one of the underlying parser
   * components fail.
   * 
   * @see setProgramVersion(const std::string& version)
   * @see setProgramName(const std::string& name)
   */
  char* writeToString (const SBMLDocument* d);

  /** @endcond */

  /**
   * Writes the given SBML document to filename.
   *
   * @htmlinclude assuming-compressed-file.html
   *
   * @param d the SBML document to be written
   *
   * @param filename the name or full pathname of the file where the SBML
   * is to be written. 
   *
   * @return @c true on success and @c false if the filename could not be
   * opened for writing.
   *
   * @note @htmlinclude note-writing-zipped-files.html
   * 
   * @see setProgramVersion(const std::string& version)
   * @see setProgramName(const std::string& name)
   */
  bool writeSBMLToFile (const SBMLDocument* d, const std::string& filename);


  /**
   * Writes the given SBML document to an in-memory string and returns a
   * pointer to it.
   *
   * The string is owned by the caller and should be freed (with @c free())
   * when no longer needed.
   *
   * @param d the SBML document to be written
   *
   * @return the string on success and @c 0 if one of the underlying parser
   * components fail.
   * 
   * @see setProgramVersion(const std::string& version)
   * @see setProgramName(const std::string& name)
   */
  char* writeSBMLToString (const SBMLDocument* d);


  /**
   * Predicate returning @c true if this copy of libSBML has been linked
   * with the <em>zlib</em> library.
   *
   * LibSBML supports reading and writing files compressed with either
   * bzip2 or zip/gzip compression.  The facility depends on libSBML having
   * been compiled with the necessary support libraries.  This method
   * allows a calling program to inquire whether that is the case for the
   * copy of libSBML it is using.
   *
   * @if notclike @note Because this is a @em static method, the
   * non-C++ language interfaces for libSBML will contain two variants.  One
   * will be a static method on the class (i.e., SBMLWriter), and the other
   * will be a standalone top-level function with the name
   * SBMLWriter_hasZlib(). They are functionally identical. @endif@~
   *
   * @return @c true if libSBML is linked with zlib, @c false otherwise.
   *
   * @see @if clike hasBzip2() @else SBMLWriter::hasBzip2() @endif@~
   */
  static bool hasZlib();


  /**
   * Predicate returning @c true if this copy of libSBML has been linked
   * with the <em>bzip2</em> library.
   *
   * LibSBML supports reading and writing files compressed with either
   * bzip2 or zip/gzip compression.  The facility depends on libSBML having
   * been compiled with the necessary support libraries.  This method
   * allows a calling program to inquire whether that is the case for the
   * copy of libSBML it is using.
   *
   * @if notclike @note Because this is a @em static method, the
   * non-C++ language interfaces for libSBML will contain two variants.  One
   * will be a static method on the class (i.e., SBMLWriter), and the other
   * will be a standalone top-level function with the name
   * SBMLWriter_hasZlib(). They are functionally identical. @endif@~
   *
   * @return @c true if libSBML is linked with bzip2, @c false otherwise.
   *
   * @see @if clike hasZlib() @else SBMLWriter::hasZlib() @endif@~
   */
  static bool hasBzip2();


 protected:
  /** @cond doxygen-libsbml-internal */

  std::string mProgramName;
  std::string mProgramVersion;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


#ifndef SWIG


/**
 * Creates a new SBMLWriter and returns a pointer to it.
 */
LIBSBML_EXTERN
SBMLWriter_t *
SBMLWriter_create (void);

/**
 * Frees the given SBMLWriter.
 */
LIBSBML_EXTERN
void
SBMLWriter_free (SBMLWriter_t *sw);

/**
 * Sets the name of this program. i.\ e.\ the one about to write out the
 * SBMLDocument.  If the program name and version are set
 * (setProgramVersion()), the following XML comment, intended for human
 * consumption, will be written at the beginning of the document:
 *
 *   <!-- Created by <program name> version <program version>
 *   on yyyy-MM-dd HH:mm with libSBML version <libsbml version>. -->
 */
LIBSBML_EXTERN
int
SBMLWriter_setProgramName (SBMLWriter_t *sw, const char *name);

/**
 * Sets the version of this program. i.\ e.\ the one about to write out the
 * SBMLDocument.  If the program version and name are set
 * (setProgramName()), the following XML comment, intended for human
 * consumption, will be written at the beginning of the document:
 *
 *   <!-- Created by <program name> version <program version>
 *   on yyyy-MM-dd HH:mm with libSBML version <libsbml version>. -->
 */
LIBSBML_EXTERN
int
SBMLWriter_setProgramVersion (SBMLWriter_t *sw, const char *version);

/**
 * Writes the given SBML document to filename.
 *
 * @htmlinclude assuming-compressed-file.html
 *
 * @return non-zero on success and zero if the filename could not be opened
 * for writing.
 *
 * @note To write a gzip/zip file, libSBML needs to be configured and
 * linked with the <a target="_blank" href="http://www.zlib.net/">zlib</a> library at
 * compile time.  It also needs to be linked with the <a target="_blank"
 * href="">bzip2</a> library to write files in @em bzip2 format.  (Both
 * of these are the default configurations for libSBML.)  Errors about
 * unreadable files will be logged and this method will return @c false
 * if a compressed filename is given and libSBML was @em not linked with
 * the corresponding required library.
 *
 * @note SBMLReader::hasZlib() and SBMLReader::hasBzip2() can be used to
 * check whether libSBML has been linked with each library.
 */
LIBSBML_EXTERN
int
SBMLWriter_writeSBML ( SBMLWriter_t         *sw,
                       const SBMLDocument_t *d,
                       const char           *filename );

LIBSBML_EXTERN
int
SBMLWriter_writeSBMLToFile ( SBMLWriter_t         *sw,
                       const SBMLDocument_t *d,
                       const char           *filename );


/**
 * Writes the given SBML document to an in-memory string and returns a
 * pointer to it.  The string is owned by the caller and should be freed
 * (with free()) when no longer needed.
 *
 * @return the string on success and @c NULL if one of the underlying parser
 * components fail (rare).
 */
LIBSBML_EXTERN
char *
SBMLWriter_writeSBMLToString (SBMLWriter_t *sw, const SBMLDocument_t *d);


/**
 * Predicate returning @c non-zero or @c zero depending on whether
 * libSBML is linked with zlib at compile time.
 *
 * @return @c non-zero if zlib is linked, @c zero otherwise.
 */
LIBSBML_EXTERN
int
SBMLWriter_hasZlib ();


/**
 * Predicate returning @c non-zero or @c zero depending on whether
 * libSBML is linked with bzip2 at compile time.
 *
 * @return @c non-zero if bzip2 is linked, @c zero otherwise.
 */
LIBSBML_EXTERN
int
SBMLWriter_hasBzip2 ();

#endif  /* !SWIG */


/**
 * Writes the given SBML document @p d to the file named by @p filename.
 *
 * This function is identical to @if clike SBMLWriter::writeSBMLToFile (const SBMLDocument_t *d, const char *filename)@endif@if java <a href="#writeSBMLToFile(org.sbml.libsbml.SBMLDocument, java.lang.String)"><code>writeSBMLToFile(SBMLDocument d, String filename)</code></a>@endif.
 * 
 * @htmlinclude assuming-compressed-file.html
 *
 * @param d the SBMLDocument object to be written out in XML format
 * 
 * @param filename a string giving the path to a file where the XML
 * content is to be written.
 *
 * @return @c 1 on success and @c 0 (zero) if @p filename could not be
 * written.  Some possible reasons for failure include (a) being unable to
 * open the file, and (b) using a filename that indicates a compressed SBML
 * file (i.e., a filename ending in <code>&quot;.zip&quot;</code> or
 * similar) when the compression functionality has not been enabled in
 * the underlying copy of libSBML.
 *
 * @if clike @warning Note that the string is owned by the caller and
 * should be freed (with the normal string <code>free()</code> C++
 * function) after it is no longer needed.@endif@~
 *
 * @see SBMLWriter::hasZlib()
 * @see SBMLWriter::hasBzip2()
 */
LIBSBML_EXTERN
int
writeSBML (const SBMLDocument_t *d, const char *filename);


/**
 * Writes the given SBML document @p d to an in-memory string and
 * returns the string.
 *
 * This is a convenience function that uses the
 * SBMLWriter::writeSBMLToString(const SBMLDocument* d) method internally,
 * yet does not require the caller to create an SBMLWriter object first.
 *
 * @param d an SBMLDocument object to be written out in XML format
 *
 * @return the string on success and @c NULL if one of the underlying parser
 * components fail.
 *
 * @if clike @warning Note that the string is owned by the caller and
 * should be freed (with the normal string <code>free()</code> C++
 * function) after it is no longer needed.@endif@~
 */
LIBSBML_EXTERN
char *
writeSBMLToString (const SBMLDocument_t *d);


/**
 * Writes the given SBML document @p d to the file @p filename.
 *
 * This is a convenience function that uses the
 * SBMLWriter::writeSBMLToFile(const SBMLDocument* d, const std::string&
 * filename) method internally, yet does not require the caller to create
 * an SBMLWriter object first.
 *
 * @htmlinclude assuming-compressed-file.html
 * 
 * @param d an SBMLDocument object to be written out in XML format
 *
 * @param filename a string giving the path to a file where the XML
 * content is to be written.
 *
 * @return @c 1 on success and @c 0 (zero) if @p filename could not be
 * written.  Some possible reasons for failure include (a) being unable to
 * open the file, and (b) using a filename that indicates a compressed SBML
 * file (i.e., a filename ending in <code>&quot;.zip&quot;</code> or
 * similar) when the compression functionality has not been enabled in
 * the underlying copy of libSBML.
 *
 * @if clike @warning Note that the string is owned by the caller and
 * should be freed (with the normal string <code>free()</code> C++
 * function) after it is no longer needed.@endif@~
 *
 * @see SBMLWriter::hasZlib()
 * @see SBMLWriter::hasBzip2()
 */
LIBSBML_EXTERN
int
writeSBMLToFile (const SBMLDocument_t *d, const char *filename);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* SBMLWriter_h */
