/**
 * @file rr_c_types.h
 * @brief roadRunner C API 2012
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerwork/ for more details.
 *
 * Copyright (C) 2012
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/

#ifndef rrc_typesH
#define rrc_typesH

#if defined(__cplusplus)
namespace rrc {
extern "C"
{
#else
    #if defined(_MSC_VER)
    	typedef int bool;
    	#define false 0
    	#define true 1
    #else
    	#include "stdbool.h"
    #endif
#endif

/*!@brief Void pointer to RRHandle */
typedef void* RRHandle; /*! Void pointer to RRHandle */

/*!@brief Void pointer to T(hread) Handle */
typedef void* RRJobHandle; /*! Void pointer to RRJobHandle */

/*!@brief Void pointer to T(hread)P(ool)Handle */
typedef void* RRJobsHandle; /*! Void pointer to RRJobsHandle */

/*!@struct*/
/*!@brief Structure for a set of RoadRunner handles */
typedef struct RRInstanceList
{
    int             Count;  	/*!< The number of elements in the vector */
    RRHandle*       Handle;   	/*!< Points to an array of double items */
    void* 			RRList;
} *RRInstanceListHandle;          			/*!< Pointer to RRVectorHandle struct */

/*!@struct*/
/*!@brief Structure for a simple vector of doubles */
typedef struct RRVector
{
    int             Count;  /*!< The number of elements in the vector */
    double*         Data;   /*!< Points to an array of double items */
} *RRVectorHandle;          /*!< Pointer to RRVectorHandle struct */

/*!@struct*/
/*!@brief Structure for a simple vector of strings */
typedef struct RRStringArray
{
    int             Count;  /*!< The number of elements in the string array */
    char**          String; /*!< Points to an array of string items */
} *RRStringArrayHandle;      /*!< Pointer to RRStringListHandle struct */

/*!@struct*/
/*!@brief Structure for a simple Matrix type */
typedef struct RRMatrix
{
    int             RSize;  /*!< The number of rows in the matrix */
    int             CSize;  /*!< The number of columns in the matrix */
    double*         Data;   /*!< Items in the matrix stored as a linear array. Access an element using Data[row*CSize + col],
							  where i,j represent the row and column numberof the element. Indexing is from zero */
} *RRMatrixHandle;          /*!< Pointer to RRMatrixHandle struct */

/*!@struct*/
/*!@brief Structure for the result type from the simulate calls */
typedef struct RRResult
{
    int             RSize;  /*!< The number of rows in the result matrix */
    int             CSize;  /*!< The number of columns in the result matrix */
    double*         Data;   /*!< A pointer to the data stored in the matrix. Access an element using Data[i*CSize + j] */
    char**          ColumnHeaders;   /*!< Pointer to an array of column header strings */
} *RRResultHandle;          /*!< Pointer to RRResultHandle struct */

/*!@struct*/
/*!@brief Convenient structure for storing the header and main body source for the generate simulation C code */
typedef struct RRCCode
{
    char*   Header;  /*!< Header file *.h */
    char*   Source;  /*!< Main source code, *.c */

} *RRCCodeHandle;    /*!< Pointer to RRCCodeHandle struct */

/*!@enum*/
/*!@brief The list type supports strings, integers, double and lists */
enum ListItemType {litString, litInteger, litDouble, litList};

// The above enums correspond to the currently supported types in an RRArrayList
// char, int, double, RRArrayList
// The void pointer pValue need to be casted to corresponding type to retrieve its value


struct RRList;	//Forward declaration for RRListItem
/*!@struct*/
/*!@brief A single list element type */
typedef struct RRListItem
{
    enum ListItemType ItemType;  	 /*!< The type of the item in this list element */
    union
    {
       int 	     		iValue;  /*!< Integer value */
	   double    		dValue;  /*!< Double value */
	   char*     		sValue;  /*!< String value */
	   struct RRList*  		lValue;  /*!< List value */
	} data;                      /*!< Union */
} *RRListItemHandle;             /*!< Pointer to cRRArrayListItemHandle struct */


/*!@struct*/
/*!@brief A list type, stores int, double, strings and lists */
typedef struct RRList
{
    int                   Count; /*!< The number elements in this list */
    RRListItemHandle    *Items;  /*!< A pointer to a list of items */
}  *RRListHandle;                /*!< Pointer to cRRArrayListHandle struct */

/*!@enum*/
/*!@brief The parameter type scan be string, integer or double */
enum RRParameterType {ptString, ptInteger, ptDouble};

/*!@struct*/
/*!@brief A single parameter type */
typedef struct RRParameter
{
    enum RRParameterType ParaType;  	 /*!< The type of the item held by the parameter */
    union
    {
       int 	     		iValue;  /*!< Integer value */
	   double    		dValue;  /*!< Double value */
	   char*     		sValue;  /*!< String value */
	} data;                      /*!< Union */
    char* 				mHint;
	char* 				mName;
} *RRParameterHandle;             /*!< Pointer to cRRParameter */


#if defined( __cplusplus)
}
}//Namespace
#endif


#endif
