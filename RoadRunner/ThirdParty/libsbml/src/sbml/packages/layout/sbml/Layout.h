/**
 * Filename    : Layout.h
 * Description : SBML Layout Layout C++ Header
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


#ifndef Layout_H__
#define Layout_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>
#include <sbml/packages/layout/sbml/Dimensions.h>
#include <sbml/packages/layout/sbml/CompartmentGlyph.h>
#include <sbml/packages/layout/sbml/SpeciesGlyph.h>
#include <sbml/packages/layout/sbml/ReactionGlyph.h>
#include <sbml/packages/layout/sbml/TextGlyph.h>
#include <sbml/packages/layout/sbml/GraphicalObject.h>
#include <sbml/packages/layout/sbml/SpeciesReferenceGlyph.h>

#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN ListOfCompartmentGlyphs : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfCompartmentGlyphs.
   */
  virtual ListOfCompartmentGlyphs* clone () const;


  /**
   * Ctor.
   */
   ListOfCompartmentGlyphs(unsigned int level      = LayoutExtension::getDefaultLevel(),
                           unsigned int version    = LayoutExtension::getDefaultVersion(),
                           unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfCompartmentGlyphs(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;


  /**
   * Get a CompartmentGlyph from the ListOfCompartmentGlyphs.
   *
   * @param n the index number of the CompartmentGlyph to get.
   * 
   * @return the nth CompartmentGlyph in this ListOfCompartmentGlyphs.
   *
   * @see size()
   */
  virtual CompartmentGlyph * get(unsigned int n); 


  /**
   * Get a CompartmentGlyph from the ListOfCompartmentGlyphs.
   *
   * @param n the index number of the CompartmentGlyph to get.
   * 
   * @return the nth CompartmentGlyph in this ListOfCompartmentGlyphs.
   *
   * @see size()
   */
  virtual const CompartmentGlyph * get(unsigned int n) const; 

  /**
   * Get a CompartmentGlyph from the ListOfCompartmentGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the CompartmentGlyph to get.
   * 
   * @return CompartmentGlyph in this ListOfCompartmentGlyphs
   * with the given id or NULL if no such
   * CompartmentGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual CompartmentGlyph* get (const std::string& sid);


  /**
   * Get a CompartmentGlyph from the ListOfCompartmentGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the CompartmentGlyph to get.
   * 
   * @return CompartmentGlyph in this ListOfCompartmentGlyphs
   * with the given id or NULL if no such
   * CompartmentGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const CompartmentGlyph* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfCompartmentGlyphs items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual CompartmentGlyph* remove (unsigned int n);


  /**
   * Removes item in this ListOfCompartmentGlyphs items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual CompartmentGlyph* remove (const std::string& sid);

    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
};

class LIBSBML_EXTERN ListOfSpeciesGlyphs : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfSpeciesGlyphs.
   */
  virtual ListOfSpeciesGlyphs* clone () const;


  /**
   * Ctor.
   */
   ListOfSpeciesGlyphs(unsigned int level      = LayoutExtension::getDefaultLevel(),
                       unsigned int version    = LayoutExtension::getDefaultVersion(),
                       unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfSpeciesGlyphs(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a SpeciesGlyph from the ListOfSpeciesGlyphs.
   *
   * @param n the index number of the SpeciesGlyph to get.
   * 
   * @return the nth SpeciesGlyph in this ListOfSpeciesGlyphs.
   *
   * @see size()
   */
  virtual SpeciesGlyph * get(unsigned int n); 


  /**
   * Get a SpeciesGlyph from the ListOfSpeciesGlyphs.
   *
   * @param n the index number of the SpeciesGlyph to get.
   * 
   * @return the nth SpeciesGlyph in this ListOfSpeciesGlyphs.
   *
   * @see size()
   */
  virtual const SpeciesGlyph * get(unsigned int n) const; 

  /**
   * Get a SpeciesGlyph from the ListOfSpeciesGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the SpeciesGlyph to get.
   * 
   * @return SpeciesGlyph in this ListOfSpeciesGlyphs
   * with the given id or NULL if no such
   * SpeciesGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual SpeciesGlyph* get (const std::string& sid);


  /**
   * Get a SpeciesGlyph from the ListOfSpeciesGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the SpeciesGlyph to get.
   * 
   * @return SpeciesGlyph in this ListOfSpeciesGlyphs
   * with the given id or NULL if no such
   * SpeciesGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const SpeciesGlyph* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfSpeciesGlyphs items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual SpeciesGlyph* remove (unsigned int n);


  /**
   * Removes item in this ListOfSpeciesGlyphs items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual SpeciesGlyph* remove (const std::string& sid);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
};

class LIBSBML_EXTERN ListOfReactionGlyphs : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfReactionGlyphs.
   */
  virtual ListOfReactionGlyphs* clone () const;


  /**
   * Ctor.
   */
   ListOfReactionGlyphs(unsigned int level      = LayoutExtension::getDefaultLevel(),
                        unsigned int version    = LayoutExtension::getDefaultVersion(),
                        unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfReactionGlyphs(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a ReactionGlyph from the ListOfReactionGlyphs.
   *
   * @param n the index number of the ReactionGlyph to get.
   * 
   * @return the nth ReactionGlyph in this ListOfReactionGlyphs.
   *
   * @see size()
   */
  virtual ReactionGlyph * get(unsigned int n); 


  /**
   * Get a ReactionGlyph from the ListOfReactionGlyphs.
   *
   * @param n the index number of the ReactionGlyph to get.
   * 
   * @return the nth ReactionGlyph in this ListOfReactionGlyphs.
   *
   * @see size()
   */
  virtual const ReactionGlyph * get(unsigned int n) const; 

  /**
   * Get a ReactionGlyph from the ListOfReactionGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the ReactionGlyph to get.
   * 
   * @return ReactionGlyph in this ListOfReactionGlyphs
   * with the given id or NULL if no such
   * ReactionGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual ReactionGlyph* get (const std::string& sid);


  /**
   * Get a ReactionGlyph from the ListOfReactionGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the ReactionGlyph to get.
   * 
   * @return ReactionGlyph in this ListOfReactionGlyphs
   * with the given id or NULL if no such
   * ReactionGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const ReactionGlyph* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfReactionGlyphs items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual ReactionGlyph* remove (unsigned int n);


  /**
   * Removes item in this ListOfReactionGlyphs items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual ReactionGlyph* remove (const std::string& sid);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
};

class LIBSBML_EXTERN ListOfTextGlyphs : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfTextGlyphs.
   */
  virtual ListOfTextGlyphs* clone () const;


  /**
   * Ctor.
   */
   ListOfTextGlyphs(unsigned int level      = LayoutExtension::getDefaultLevel(),
                    unsigned int version    = LayoutExtension::getDefaultVersion(),
                    unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfTextGlyphs(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a TextGlyph from the ListOfTextGlyphs.
   *
   * @param n the index number of the TextGlyph to get.
   * 
   * @return the nth TextGlyph in this ListOfTextGlyphs.
   *
   * @see size()
   */
  virtual TextGlyph * get(unsigned int n); 


  /**
   * Get a TextGlyph from the ListOfTextGlyphs.
   *
   * @param n the index number of the TextGlyph to get.
   * 
   * @return the nth TextGlyph in this ListOfTextGlyphs.
   *
   * @see size()
   */
  virtual const TextGlyph * get(unsigned int n) const; 

  /**
   * Get a TextGlyph from the ListOfTextGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the TextGlyph to get.
   * 
   * @return TextGlyph in this ListOfTextGlyphs
   * with the given id or NULL if no such
   * TextGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual TextGlyph* get (const std::string& sid);


  /**
   * Get a TextGlyph from the ListOfTextGlyphs
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the TextGlyph to get.
   * 
   * @return TextGlyph in this ListOfTextGlyphs
   * with the given id or NULL if no such
   * TextGlyph exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const TextGlyph* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfTextGlyphs items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual TextGlyph* remove (unsigned int n);


  /**
   * Removes item in this ListOfTextGlyphs items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual TextGlyph* remove (const std::string& sid);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
};

class LIBSBML_EXTERN Layout : public SBase
{
protected:

  std::string mId;

  Dimensions mDimensions;
  ListOfCompartmentGlyphs mCompartmentGlyphs;
  ListOfSpeciesGlyphs mSpeciesGlyphs;
  ListOfReactionGlyphs mReactionGlyphs;
  ListOfTextGlyphs mTextGlyphs;
  ListOfGraphicalObjects mAdditionalGraphicalObjects;

  
  GraphicalObject*
  removeObjectWithId (ListOf* list, const std::string& id);

  
  const GraphicalObject*
  getObjectWithId (const ListOf* list, const std::string& id) const;

  GraphicalObject*
  getObjectWithId (ListOf* list, const std::string& id) ;


public:

  /**
   * Creates a new Layout with the given level, version, and package version.
   */
   Layout(unsigned int level      = LayoutExtension::getDefaultLevel(),
          unsigned int version    = LayoutExtension::getDefaultVersion(),
          unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Creates a new Layout with the given LayoutPkgNamespaces object.
   */
   Layout(LayoutPkgNamespaces* layoutns);


  /**
   * Creates a new Layout with the given id and dimensions.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  Layout (LayoutPkgNamespaces* layoutns, const std::string& id, const Dimensions* dimensions);


  /**
   * Creates a new Layout from the given XMLNode
   * (only for SBML Level2)
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  Layout (const XMLNode& node, unsigned int l2version = 4);

  /**
   * Copy constructor.
   */
   Layout(const Layout& source);

  /**
   * Assignment operator.
   */
   Layout& operator=(const Layout& source);



  /**
   * Destructor.
   */ 
  
  virtual ~Layout ();


  /**
   * Does nothing since no defaults are defined for Layout.
   */ 
  
  void initDefaults ();    

        
  /**
   * Returns the value of the "id" attribute of this Layout.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Layout's "id" attribute has been set.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the value of the "id" attribute of this Layout.
   */
  virtual int setId (const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this Layout.
   */
  virtual int unsetId ();


  /**
   * Returns the dimensions of the layout.
   */ 
  
  const Dimensions* getDimensions () const;

  /**
   * Returns the dimensions of the layout.
   */ 
  
  Dimensions* getDimensions ();

  /**
   * Sets the dimensions of the layout.
   */ 
    
  void setDimensions (const Dimensions* dimensions);


  /**
   * Returns the ListOf object that holds all compartment glyphs.
   */ 
  
  const ListOfCompartmentGlyphs* getListOfCompartmentGlyphs () const;

  /**
   * Returns the ListOf object that holds all species glyphs.
   */ 
   
  const ListOfSpeciesGlyphs* getListOfSpeciesGlyphs () const;

  /**
   * Returns the ListOf object that holds all reaction glyphs.
   */ 
   
  const ListOfReactionGlyphs* getListOfReactionGlyphs () const;

  /**
   * Returns the ListOf object that holds all text glyphs.
   */ 
   
  const ListOfTextGlyphs* getListOfTextGlyphs () const;

  /**
   * Returns the ListOf object that holds all additonal graphical objects.
   */ 
   
  const ListOfGraphicalObjects* getListOfAdditionalGraphicalObjects () const;
  
  /**
   * Returns the ListOf object that holds all compartment glyphs.
   */ 
  
  ListOfCompartmentGlyphs* getListOfCompartmentGlyphs ();

  /**
   * Returns the ListOf object that holds all species glyphs.
   */ 
   
  ListOfSpeciesGlyphs* getListOfSpeciesGlyphs ();

  /**
   * Returns the ListOf object that holds all reaction glyphs.
   */ 
   
  ListOfReactionGlyphs* getListOfReactionGlyphs ();

  /**
   * Returns the ListOf object that holds all text glyphs.
   */ 
   
  ListOfTextGlyphs* getListOfTextGlyphs ();

  /**
   * Returns the ListOf object that holds all additional graphical objects.
   */ 
   
  ListOfGraphicalObjects* getListOfAdditionalGraphicalObjects ();


  /**
   * Returns the compartment glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  const CompartmentGlyph* getCompartmentGlyph (unsigned int index) const;

  /**
   * Returns the compartment glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  CompartmentGlyph* getCompartmentGlyph (unsigned int index) ;

  /**
   * Returns the species glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  SpeciesGlyph* getSpeciesGlyph (unsigned int index) ;

  /**
   * Returns the species glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  const SpeciesGlyph* getSpeciesGlyph (unsigned int index) const;

  /**
   * Returns the reaction glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  const ReactionGlyph* getReactionGlyph (unsigned int index) const;

  /**
   * Returns the reaction glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  ReactionGlyph* getReactionGlyph (unsigned int index) ;

  /**
   * Returns the text glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  const TextGlyph* getTextGlyph (unsigned int index) const;

  /**
   * Returns the text glyph with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  TextGlyph* getTextGlyph (unsigned int index) ;

  /**
   * Returns the additional graphical object with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  const GraphicalObject* getAdditionalGraphicalObject (unsigned int index) const;

  /**
   * Returns the additional graphical object with the given index.
   * If the index is invalid, NULL is returned.
   */ 
  GraphicalObject* getAdditionalGraphicalObject (unsigned int index) ;

  /**
   * Returns the general glyph with the given index.
   * If the index is invalid, NULL is returned.
   */
  GeneralGlyph* getGeneralGlyph (unsigned int index);
  
  /**
   * Returns the general glyph with the given index.
   * If the index is invalid, NULL is returned.
   */
  const GeneralGlyph* getGeneralGlyph (unsigned int index) const;


  /**
   * Returns the compartment glyph that has the given id, or NULL if no
   * compartment glyph has the id.
   */
  
  const CompartmentGlyph* getCompartmentGlyph (const std::string& id) const;

  /**
   * Returns the species glyph that has the given id, or NULL if no species
   * glyph has the id.
   */
  
  const SpeciesGlyph* getSpeciesGlyph (const std::string& id) const;
        
  /**
   * Returns the reaction glyph that has the given id, or NULL if no
   * reaction glyph has the id.
   */
  
  const ReactionGlyph* getReactionGlyph (const std::string& id) const;

  /**
   * Returns the text glyph that has the given id, or NULL if no text glyph
   * has the id.
   */
  
  const TextGlyph* getTextGlyph (const std::string& id) const;

  /**
   * Returns the additional graphical object that has the given id, or NULL
   * if no graphical object has the id.
   */
  
  const GraphicalObject* getAdditionalGraphicalObject (const std::string& id) const;

  /**
   * Returns the general glyph that has the given id, or NULL
   * if no graphical object has the id.
   */
  
  const GeneralGlyph* getGeneralGlyph (const std::string& id) const;


  /**
   * Returns the compartment glyph that has the given id, or NULL if no
   * compartment glyph has the id.
   */
  
  CompartmentGlyph* getCompartmentGlyph (const std::string& id) ;

  /**
   * Returns the species glyph that has the given id, or NULL if no species
   * glyph has the id.
   */
  
  SpeciesGlyph* getSpeciesGlyph (const std::string& id) ;
        
  /**
   * Returns the reaction glyph that has the given id, or NULL if no
   * reaction glyph has the id.
   */
  
  ReactionGlyph* getReactionGlyph (const std::string& id) ;

  /**
   * Returns the text glyph that has the given id, or NULL if no text glyph
   * has the id.
   */
  
  TextGlyph* getTextGlyph (const std::string& id) ;

  /**
   * Returns the additional graphical object that has the given id, or NULL
   * if no graphical object has the id.
   */
  
  GraphicalObject* getAdditionalGraphicalObject (const std::string& id) ;

  /**
   * Returns the general glyph that has the given id, or NULL
   * if no graphical object has the id.
   */
  GeneralGlyph* getGeneralGlyph (const std::string& id);

  /**
   * Adds a new compartment glyph.
   */
  
  void addCompartmentGlyph (const CompartmentGlyph* glyph);

  /**
   * Adds a new species glyph.
   */
  
  void addSpeciesGlyph (const SpeciesGlyph* glyph);

  /**
   * Adds a new reaction glyph.
   */
  
  void addReactionGlyph (const ReactionGlyph* glyph);

  /**
   * Adds a new text glyph.
   */
  
  void addTextGlyph (const TextGlyph* glyph);

  /**
   * Adds a new additional graphical object glyph.
   */
  
  void addAdditionalGraphicalObject (const GraphicalObject* glyph);

  /**
   * Adds a new general glyph.
   */
  void addGeneralGlyph (const GeneralGlyph* glyph);


  /**
   * Returns the number of compartment glyphs for the layout.
   */
  
  unsigned int getNumCompartmentGlyphs () const;

  /**
   * Returns the number of species glyphs for the layout.
   */
   
  unsigned int getNumSpeciesGlyphs () const;

  /**
   * Returns the number of reaction glyphs for the layout.
   */
  
  unsigned int getNumReactionGlyphs () const;

  /**
   * Returns the number of text glyphs for the layout.
   */
   
  unsigned int getNumTextGlyphs () const;

  /**
   * Returns the number of additional graphical objects for the layout.
   */
  unsigned int getNumAdditionalGraphicalObjects () const;

  /**
   * Returns the number of general glyphs for the layout.
   */
  unsigned int getNumGeneralGlyphs () const;
  

  /**
   * Creates a CompartmentGlyph object, adds it to the end of the
   * compartment glyph objects list and returns a pointer to the newly
   * created object.
   */
  
  CompartmentGlyph* createCompartmentGlyph ();

  /**
   * Creates a SpeciesGlyph object, adds it to the end of the species glyph
   * objects list and returns a pointer to the newly created object.
   */
  
  SpeciesGlyph* createSpeciesGlyph ();

  /**
   * Creates a ReactionGlyph object, adds it to the end of the reaction
   * glyph objects list and returns a pointer to the newly created
   * object.
   */
  
  ReactionGlyph* createReactionGlyph ();

  /**
   * Creates a GeneralGlyph object, adds it to the end of the additional 
   * objects list and returns a reference to the newly created object.
   */
  GeneralGlyph*  createGeneralGlyph();

  /**
   * Creates a TextGlyph object, adds it to the end of the text glyph
   * objects list and returns a pointer to the newly created object.
   */
  
  TextGlyph* createTextGlyph ();

  /**
   * Creates a GraphicalObject object, adds it to the end of the additional
   * graphical objects list and returns a pointer to the newly created
   * object.
   */
  
  GraphicalObject* createAdditionalGraphicalObject ();

  /**
   * Creates a new SpeciesReferenceGlyph for the last ReactionGlyph and
   * adds it to its list of SpeciesReferenceGlyph objects.  A pointer to
   * the newly created object is returned.
   */
  
  SpeciesReferenceGlyph* createSpeciesReferenceGlyph();


  /**
   * Creates a new LineSegment for the Curve object of the last
   * ReactionGlyph or the last SpeciesReferenceGlyph in the last
   * ReactionGlyph and adds it to its list of SpeciesReferenceGlyph
   * objects.  A pointer to the newly created object is returned.
   */
  
  LineSegment* createLineSegment ();

  /**
   * Creates a new CubicBezier for the Curve object of the last
   * ReactionGlyph or the last SpeciesReferenceGlyph in the last
   * ReactionGlyph and adds it to its list of SpeciesReferenceGlyph
   * objects.  A pointer to the newly created object is returned.
   */
  
  CubicBezier* createCubicBezier ();

  /**
   * Removes the compartment glyph with the given index from the layout.
   * A pointer to the compartment glyph that was removed is returned.
   * If no compartment glyph has been removed, NULL is returned.
   */
  
  CompartmentGlyph* removeCompartmentGlyph(unsigned int index);

  /**
   * Removes the species glyph with the given index from the layout.
   * A pointer to the species glyph that was removed is returned.
   * If no species glyph has been removed, NULL is returned.
   */
  
  SpeciesGlyph* removeSpeciesGlyph(unsigned int index);
  
  /**
   * Removes the reaction glyph with the given index from the layout.
   * A pointer to the reaction glyph that was removed is returned.
   * If no reaction glyph has been removed, NULL is returned.
   */
  
  ReactionGlyph* removeReactionGlyph(unsigned int index);
  
  /**
   * Removes the text glyph with the given index from the layout.
   * A pointer to the text glyph that was removed is returned.
   * If no text glyph has been removed, NULL is returned.
   */
  
  TextGlyph* removeTextGlyph(unsigned int index);
  
  /**
   * Removes the graphical object with the given index from the layout.
   * A pointer to the graphical object that was removed is returned.
   * If no graphical object has been removed, NULL is returned.
   */
  
  GraphicalObject* removeAdditionalGraphicalObject(unsigned int index);

  /**
   * Remove the compartment glyph with the given id.
   * A pointer to the removed compartment glyph is returned.
   * If no compartment glyph has been removed, NULL is returned.
   */
  
  CompartmentGlyph*
  removeCompartmentGlyph(const std::string id);


  /**
   * Remove the species glyph with the given id.
   * A pointer to the removed species glyph is returned.
   * If no species glyph has been removed, NULL is returned.
   */
  
  SpeciesGlyph*
  removeSpeciesGlyph(const std::string id);


  /**
   * Remove the reaction glyph with the given id.
   * A pointer to the removed reaction glyph is returned.
   * If no reaction glyph has been removed, NULL is returned.
   */
  
  ReactionGlyph*
  removeReactionGlyph(const std::string id);


  /**
   * Remove the species reference glyph with the given id.
   * A pointer to the removed species reference glyph is returned.
   * If no species reference glyph has been removed, NULL is returned.
   */
  
  SpeciesReferenceGlyph*
  removeSpeciesReferenceGlyph(const std::string id);


  /**
   * Remove the text glyph with the given id.
   * A pointer to the removed text glyph is returned.
   * If no text glyph has been removed, NULL is returned.
   */
  
  TextGlyph*
  removeTextGlyph(const std::string id);


  /**
   * Remove the graphical object with the given id.
   * A pointer to the removed graphical object is returned.
   * If no graphical object has been removed, NULL is returned.
   */
  
  GraphicalObject*
  removeAdditionalGraphicalObject(const std::string id);

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
   * @return a (deep) copy of this Layout.
   */
  virtual Layout* clone () const;


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
   * Accepts the given SBMLVisitor.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether or not the Visitor would like to visit the SBML object's next
   * sibling object (if available).
   */
  virtual bool accept (SBMLVisitor& v) const;
   

   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument object to use
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor, assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();


  /**
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePakcage function)
   *
   * @note Subclasses in which one or more child elements are defined
   * must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);
  /** @endcond */

    
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


class LIBSBML_EXTERN ListOfLayouts : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfLayouts.
   */
  virtual ListOfLayouts* clone () const;


  /**
   * Ctor.
   */
   ListOfLayouts(unsigned int level      = LayoutExtension::getDefaultLevel(), 
                 unsigned int version    = LayoutExtension::getDefaultVersion(), 
                 unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfLayouts(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a Layout from the ListOfLayouts.
   *
   * @param n the index number of the Layout to get.
   * 
   * @return the nth Layout in this ListOfLayouts.
   *
   * @see size()
   */
  virtual Layout * get(unsigned int n); 


  /**
   * Get a Layout from the ListOfLayouts.
   *
   * @param n the index number of the Layout to get.
   * 
   * @return the nth Layout in this ListOfLayouts.
   *
   * @see size()
   */
  virtual const Layout * get(unsigned int n) const; 

  /**
   * Get a Layout from the ListOfLayouts
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Layout to get.
   * 
   * @return Layout in this ListOfLayouts
   * with the given id or NULL if no such
   * Layout exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Layout* get (const std::string& sid);


  /**
   * Get a Layout from the ListOfLayouts
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Layout to get.
   * 
   * @return Layout in this ListOfLayouts
   * with the given id or NULL if no such
   * Layout exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Layout* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfLayouts items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual Layout* remove (unsigned int n);


  /**
   * Removes item in this ListOfLayouts items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual Layout* remove (const std::string& sid);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;


    void resetElementNamespace(const std::string& uri);
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);

  virtual void writeXMLNS (XMLOutputStream& stream) const;
};


LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Creates a new Layout and returns a pointer to it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_create (void);

/**
 * Creates a new Layout with the given id and returns a pointer to it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWith (const char *sid);

/**
 * Creates a Layout object from a template.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createFrom (const Layout_t *temp);

/**
 * Creates a new Layout with the given width, height and depth and returns
 * a pointer to it.  The depth value defaults to 0.0.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWithSize (const char *id,
                       double width, double height, double depth);

/**
 * Creates a new Layout with the given Dimensions and returns a pointer to
 * it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWithDimensions (const char *id, const Dimensions_t *dimensions);

/** 
 * Frees the memory for the given layout.
 */
LIBSBML_EXTERN
void 
Layout_free (Layout_t *l);


LIBSBML_EXTERN
void
Layout_setDimensions (Layout_t *l, const Dimensions_t *dimensions);

/**
 * Adds a new compartment glyph to the list of compartment glyphs.
 */
LIBSBML_EXTERN
void
Layout_addCompartmentGlyph (Layout_t *l, CompartmentGlyph_t *cg);

/**
 * Adds a new species glyph to the list of species glyphs.
 */
LIBSBML_EXTERN
void
Layout_addSpeciesGlyph (Layout_t *l, SpeciesGlyph_t *sg);

/**
 * Adds a new reaction glyph to the list of reaction glyphs.
 */
LIBSBML_EXTERN
void
Layout_addReactionGlyph (Layout_t *l, ReactionGlyph_t *rg);

/**
 * Adds a new GraphicalObject to the list of additional graphical objects.
 */
LIBSBML_EXTERN
void
Layout_addAdditionalGraphicalObject (Layout_t *l, GraphicalObject_t *go);

/**
 * Adds a new TextGlyph to the list of text glyphs.
 */
LIBSBML_EXTERN
void
Layout_addTextGlyph (Layout_t *l, TextGlyph_t *go);


/**
 * Returns a pointer to the CompartmentGlyph with the given index.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_getCompartmentGlyph (Layout_t *l, unsigned int index);

/**
 * Returns a pointer to the SpeciesGlyph with the given index.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_getSpeciesGlyph (Layout_t *l, unsigned int index);


/**
 * Returns a pointer to the ReactionGlyph with the given index.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_getReactionGlyph (Layout_t *l, unsigned int index);


/**
 * Returns a pointer to the AdditionalGraphicalObject with the given index.
 */
LIBSBML_EXTERN
GraphicalObject_t *
Layout_getAdditionalGraphicalObject (Layout_t *l, unsigned int index);

/**
 * Returns a pointer to the GraphicalObject with the given index.
 */
LIBSBML_EXTERN
TextGlyph_t *
Layout_getTextGlyph (Layout_t *l, unsigned int index);


/**
 * Returns a pointer to the list of CompartmentGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfCompartmentGlyphs (Layout_t *l);

/**
 * Returns a pointer to the list of SpeciesGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfSpeciesGlyphs (Layout_t *l);


/**
 * Returns a pointer to the list of ReactionGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfReactionGlyphs (Layout_t *l);


/**
 * Returns a pointer to the list of additional GraphicalObjects.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfAdditionalGraphicalObjects (Layout_t *l);

/**
 * Returns a pointer to the list of TextGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfTextGlyphs (Layout_t *l);


/**
 * Returns a Dimensions_t pointer from the layout.
 */
LIBSBML_EXTERN
Dimensions_t*
Layout_getDimensions(Layout_t *l);

/**
 * Returns the number of CompartmentGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumCompartmentGlyphs (const Layout_t *l);

/**
 * Returns the number of SpeciesGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumSpeciesGlyphs (const Layout_t *l);


/**
 * Returns the number of ReactionGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumReactionGlyphs (const Layout_t *l);


/**
 * Returns the number of additional GraphicalObjects.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumAdditionalGraphicalObjects (const Layout_t *l);

/**
 * Returns the number of TextGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumTextGlyphs (const Layout_t *l);



/**
 * Removes the compartment glyph with the given index.  If the index is
 * invalid, nothing is deleted.
 */ 
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_removeCompartmentGlyph (Layout_t *l, unsigned int index);

/**
 * Removes the species glyph with the given index.  If the index is
 * invalid, nothing is deleted.
 */ 
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_removeSpeciesGlyph (Layout_t *l, unsigned int index);

/**
 * Removes the reaction glyph with the given index.  If the index is
 * invalid, nothing is deleted.
 */ 
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_removeReactionGlyph (Layout_t *l, unsigned int index);
 
/**
 * Removes the text glyph with the given index.  If the index is invalid,
 * nothing is deleted.
 */ 
LIBSBML_EXTERN
TextGlyph_t *
Layout_removeTextGlyph (Layout_t *l, unsigned int index);
 
/**
 * Removes the graphical object with the given index.  If the index is
 * invalid, nothing is deleted.
 */ 
LIBSBML_EXTERN
GraphicalObject_t *
Layout_removeAdditionalGraphicalObject (Layout_t *l, unsigned int index);

/**
 * Removes the compartment glyph with the given id.  If the id is
 * not found, nothing is deleted.
 */ 
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_removeCompartmentGlyphWithId (Layout_t *l, const char* id);

/**
 * Removes the species glyph with the given id.  If the id is
 * not found, nothing is deleted.
 */ 
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_removeSpeciesGlyphWithId (Layout_t *l, const char* id);

/**
 * Removes the species reference glyph with the given id.  If the id is
 * not found, nothing is deleted.
 */ 
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
Layout_removeSpeciesReferenceGlyphWithId (Layout_t *l, const char* id);

/**
 * Removes the reaction glyph with the given id.  If the id is
 * not found, nothing is deleted.
 */ 
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_removeReactionGlyphWithId (Layout_t *l, const char* id);
 
/**
 * Removes the text glyph with the given id.  If the id is not found,
 * nothing is deleted.
 */ 
LIBSBML_EXTERN
TextGlyph_t *
Layout_removeTextGlyphWithId (Layout_t *l, const char* id);
 
/**
 * Removes the graphical object with the given id.  If the id is
 * not found, nothing is deleted.
 */ 
LIBSBML_EXTERN
GraphicalObject_t *
Layout_removeAdditionalGraphicalObjectWithId (Layout_t *l, const char* );
    
/**
 * Does nothing since no defaults are defined for Layout.
 */ 
LIBSBML_EXTERN
void
Layout_initDefaults (Layout_t *l);


/**
 * Creates a ComparmentGlyph_t object, adds it to the end of the
 * compartment glyphs objects list and returns a pointer to the newly
 * created object.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_createCompartmentGlyph (Layout_t *);

/**
 * Creates a SpeciesGlyph object, adds it to the end of the species glyphs
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_createSpeciesGlyph (Layout_t *);


/**
 * Creates a ReactionGlyph_t object, adds it to the end of the reaction
 * glyphs objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_createReactionGlyph (Layout_t *);

/**
 * Creates a GeneralGlyph_t object, adds it to the end of the additional
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
Layout_createGeneralGlyph (Layout_t *);


/**
 * Creates a TextGlyph_t object, adds it to the end of the text glyphs
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
TextGlyph_t *
Layout_createTextGlyph (Layout_t *);


/**
 * Creates a GraphicalObject object, adds it to the end of the additional
 * graphical objects list and returns a pointer to the newly created
 * object.
 */
LIBSBML_EXTERN
GraphicalObject_t *
Layout_createAdditionalGraphicalObject (Layout_t *);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Layout_t *
Layout_clone (const Layout_t *m);


/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Layout_t *
Layout_clone (const Layout_t *m);


LIBSBML_EXTERN
int
Layout_isSetId (const Layout_t *l);

LIBSBML_EXTERN
const char *
Layout_getId (const Layout_t *l);


LIBSBML_EXTERN
int
Layout_setId (Layout_t *l, const char *sid);


LIBSBML_EXTERN
void
Layout_unsetId (Layout_t *l);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* Layout_H__ */
