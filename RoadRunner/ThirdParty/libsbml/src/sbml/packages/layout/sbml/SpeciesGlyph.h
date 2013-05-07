/**
 * Filename    : SpeciesGlyph.h
 * Description : SBML Layout SpeciesGlyph C++ Header
 * Organization: European Media Laboratories Research gGmbH
 * Created     : 2004-07-15
 *
 * Copyright 2004 European Media Laboratories Research gGmbH
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * European Media Laboratories Research gGmbH have no obligations to
 * provide maintenance, support, updates, enhancements or modifications.
 * In no event shall the European Media Laboratories Research gGmbH be
 * liable to any party for direct, indirect, special, incidental or
 * consequential damages, including lost profits, arising out of the use of
 * this software and its documentation, even if the European Media
 * Laboratories Research gGmbH have been advised of the possibility of such
 * damage.  See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Ralph Gauges
 *     Bioinformatics Group
 *     European Media Laboratories Research gGmbH
 *     Schloss-Wolfsbrunnenweg 31c
 *     69118 Heidelberg
 *     Germany
 *
 *     http://www.eml-research.de/english/Research/BCB/
 *     mailto:ralph.gauges@eml-r.villa-bosch.de
 *
 * Contributor(s):
 *
 *     Akiya Jouraku <jouraku@bio.keio.ac.jp>
 *     Modified this file for package extension in libSBML5
 *
 */


#ifndef SpeciesGlyph_H__
#define SpeciesGlyph_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus


#include <string>
#include <sbml/packages/layout/sbml/GraphicalObject.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SpeciesGlyph : public GraphicalObject
{
protected:

  std::string mSpecies;        


public:

  /**
   * Creates a new SpeciesGlyph with the given SBML level, version, and package version
   * and the id of the associated species set to the empty string.
   */        
  
  SpeciesGlyph (unsigned int level      = LayoutExtension::getDefaultLevel(),
                unsigned int version    = LayoutExtension::getDefaultVersion(),
                unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  /**
   * Ctor.
   */
  SpeciesGlyph(LayoutPkgNamespaces* layoutns);


  /**
   * Creates a new SpeciesGlyph with the given id. 
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */   
  SpeciesGlyph (LayoutPkgNamespaces* layoutns, const std::string& id);

  /**
   * Creates a new SpeciesGlyph with the given id and the id of the
   * associated species object set to the second argument.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */   
  SpeciesGlyph (LayoutPkgNamespaces* layoutns, const std::string& id, const std::string& speciesId);
        

  /**
   * Creates a new SpeciesGlyph from the given XMLNode
   */
  SpeciesGlyph(const XMLNode& node, unsigned int l2version=4);

  /**
   * Copy constructor.
   */
   SpeciesGlyph(const SpeciesGlyph& source);

  /**
   * Assignment operator.
   */
   virtual SpeciesGlyph& operator=(const SpeciesGlyph& source);

  /**
   * Destructor.
   */ 
  
  virtual ~SpeciesGlyph ();
        
  /**
   * Returns the id of the associated species object.
   */ 
  
  const std::string& getSpeciesId () const;
        
  /**
   * Sets the id of the associated species object.
   */ 
  
  void setSpeciesId (const std::string& id);
        
  /**
   * Returns true if the id of the associated species object is not the
   * empty string.
   */ 
  
  bool isSetSpeciesId () const;    

  /**
   * Calls initDefaults from GraphicalObject.
   */ 
  
  void initDefaults ();

  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.  For example:
   *
   *   SBase::writeElements(stream);
   *   mReactans.write(stream);
   *   mProducts.write(stream);
   *   ...
   */
  virtual void writeElements (XMLOutputStream& stream) const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const ;

  /**
   * @return a (deep) copy of this SpeciesGlyph.
   */
  virtual SpeciesGlyph* clone () const;


  /**
   * Returns the libSBML type code for this object.
   *
   * This method MAY return the typecode of this SBML object or it MAY
   * return SBML_UNKNOWN.  That is, subclasses of SBase are not required to
   * implement this method to return a typecode.  This method is meant
   * primarily for the LibSBML C interface where class and subclass
   * information is not readily available.
   *
   * @note In libSBML 5, the type of return value has been changed from
   *       SBMLTypeCode_t to int. The return value is one of enum values defined
   *       for each package. For example, return values will be one of
   *       SBMLTypeCode_t if this object is defined in SBML core package,
   *       return values will be one of SBMLLayoutTypeCode_t if this object is
   *       defined in Layout extension (i.e. similar enum types are defined in
   *       each pacakge extension for each SBase subclass)
   *       The value of each typecode can be duplicated between those of
   *       different packages. Thus, to distinguish the typecodes of different
   *       packages, not only the return value of getTypeCode() but also that of
   *       getPackageName() must be checked.
   *
   * @return the typecode (int value) of this SBML object or SBML_UNKNOWN
   * (default).
   *
   * @see getElementName()
   * @see getPackageName()
   */
  virtual int getTypeCode () const;


   /**
    * Creates an XMLNode object from this.
    */
    virtual XMLNode toXML() const;
    
protected:
  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase*
  createObject (XMLInputStream& stream);

  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes, 
                               const ExpectedAttributes& expectedAttributes);

  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.  For example:
   *
   *   SBase::writeAttributes(stream);
   *   stream.writeAttribute( "id"  , mId   );
   *   stream.writeAttribute( "name", mName );
   *   ...
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;


};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Creates a new SpeciesGlyph and returns the pointer to it.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
SpeciesGlyph_create (void);

/**
 * Create a new SpeciesGlyph object from a template.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
SpeciesGlyph_createFrom (const SpeciesGlyph_t *temp);



/**
 * Creates a new SpeciesGlyph with the given id
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
SpeciesGlyph_createWith (const char *id);

/**
 * Creates a new SpeciesGlyph referencing with the give species id.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
SpeciesGlyph_createWithSpeciesId (const char *id, const char *speciesId);

/**
 * Frees the memory taken by the given compartment glyph.
 */
LIBSBML_EXTERN
void
SpeciesGlyph_free (SpeciesGlyph_t *sg);

/**
 * Sets the associated species id. 
 */
LIBSBML_EXTERN
void
SpeciesGlyph_setSpeciesId (SpeciesGlyph_t *sg, const char *id);

/**
 * Gets the the id of the associated species.
 */
LIBSBML_EXTERN
const char *
SpeciesGlyph_getSpeciesId (const SpeciesGlyph_t *sg);

/**
 * Returns 0 if the  id of the associated species is the empty string.
 * otherwise.
 */
LIBSBML_EXTERN
int
SpeciesGlyph_isSetSpeciesId (const SpeciesGlyph_t *sg);

/**
 * Calls initDefaults from GraphicalObject.
 */ 
LIBSBML_EXTERN
void
SpeciesGlyph_initDefaults (SpeciesGlyph_t *sg);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
SpeciesGlyph_clone (const SpeciesGlyph_t *m);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* SpeciesGlyph_H__ */
