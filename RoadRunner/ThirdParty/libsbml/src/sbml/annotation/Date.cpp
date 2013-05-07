/**
 * @file    Date.cpp
 * @brief   Date I/O
 * @author  Sarah Keating
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
 * the Free Software Foundation.  A copy of the license agreement is
 * provided in the file named "LICENSE.txt" included with this software
 * distribution.  It is also available online at
 * http://sbml.org/software/libsbml/license.html
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */


#include <sbml/annotation/Date.h>
#include <sbml/common/common.h>
#include <sbml/SBase.h>
#include <cstdio>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/*
 * creates a date from the individual fields entered as numbers
 */
Date::Date(unsigned int year, unsigned int month, 
    unsigned int day, unsigned int hour, 
    unsigned int minute, unsigned int second,
    unsigned int sign, unsigned int hoursOffset,
    unsigned int minutesOffset) :
  mHasBeenModified (false)
{
  mYear   = year;
  mMonth  = month;
  mDay    = day;
  mHour   = hour;  
  mMinute = minute;
  mSecond = second;
  
  mSignOffset   = sign;
  mHoursOffset  = hoursOffset;
  mMinutesOffset  = minutesOffset;;
  
  parseDateNumbersToString();
}


/*
 * creates a date from a string
 */
Date::Date (const std::string& date) :
  mHasBeenModified (false)
{ 
  if (&(date) == NULL)
    mDate = "";
  else
    mDate = date; 

  parseDateStringToNumbers();
  parseDateNumbersToString();
}

Date::~Date() {}

/*
 * Copy constructor.
 */
Date::Date(const Date& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mYear   = orig.mYear;
    mMonth  = orig.mMonth;
    mDay    = orig.mDay;
    mHour   = orig.mHour;  
    mMinute = orig.mMinute;
    mSecond = orig.mSecond;
    
    mSignOffset     = orig.mSignOffset;
    mHoursOffset    = orig.mHoursOffset;
    mMinutesOffset  = orig.mMinutesOffset;;

    mDate = orig.mDate;

    mHasBeenModified = orig.mHasBeenModified;
  }
}

/*
 * Assignment operator
 */
Date& Date::operator=(const Date& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    mYear   = rhs.mYear;
    mMonth  = rhs.mMonth;
    mDay    = rhs.mDay;
    mHour   = rhs.mHour;  
    mMinute = rhs.mMinute;
    mSecond = rhs.mSecond;
    
    mSignOffset     = rhs.mSignOffset;
    mHoursOffset    = rhs.mHoursOffset;
    mMinutesOffset  = rhs.mMinutesOffset;;

    mDate = rhs.mDate;

    mHasBeenModified = rhs.mHasBeenModified;
  }

  return *this;
}

/*
 * @return a (deep) copy of this Date.
 */
Date* Date::clone () const
{
  return new Date(*this);
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setYear    (unsigned int year)
{
  if (year <1000 || year > 9999)
  {
    mYear = 2000;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mYear = year;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setMonth   (unsigned int month)
{
  if (month < 1 || month > 12)
  {
    mMonth = 1;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mMonth = month;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setDay     (unsigned int day)
{
  bool validDay = true;
  if (day < 1 || day > 31)
  {
    validDay = false;
  }
  else
  {
    switch (mMonth)
    {
    case 4:
    case 6:
    case 9:
    case 11:
      if (day > 30) validDay = false;
      break;
    case 2:
      if (mYear % 4 == 0)
      {
        if (day > 29) validDay = false;
      }
      else
      {
         if (day > 28) validDay = false;
      }
      break;
    case 1:
    case 3:
    case 5:
    case 7:
    case 8:
    case 10:
    case 12:
    default:
      break;
    }
  }
  
  if (!validDay)
  {
    mDay = 1;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mDay = day;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
} 

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setHour    (unsigned int hour)
{
  if (/*hour < 0 ||*/ hour > 23)
  {
    mHour = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mHour = hour;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setMinute  (unsigned int minute)
{
  if (/*minute < 0 ||*/ minute > 59)
  {
    mMinute = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mMinute = minute;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setSecond  (unsigned int second)
{
  if (/*second < 0 ||*/ second > 59)
  {
    mSecond = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mSecond = second;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setSignOffset    (unsigned int sign)
{
  if (/*sign < 0 ||*/ sign > 1)
  {
    mSignOffset = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mSignOffset = sign;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setHoursOffset    (unsigned int hour)
{
  if (/*hour < 0 ||*/ hour > 12)
  {
    mHoursOffset = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mHoursOffset = hour;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the year checking appropriateness
 */
int 
Date::setMinutesOffset  (unsigned int minute)
{
  if (/*minute < 0 ||*/ minute > 59)
  {
    mMinutesOffset = 0;
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mMinutesOffset = minute;
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * sets the value of the date string checking appropriateness
 */
int 
Date::setDateAsString (const std::string& date)
{
  /* if date is NULL consider this as resetting 
   * the date completely
   */
 
  if (&(date) == NULL)
  {
    mDate = "";
    // revert to default numbers
    // rewrite date string to reflect the defaults
    parseDateStringToNumbers();
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (date.empty())
  {
    mDate = "";
    // revert to default numbers
    // rewrite date string to reflect the defaults
    parseDateStringToNumbers();
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }

  /* Date must be: YYYY-MM-DDThh:mm:ssTZD
   * where TZD is either Z or +/-HH:MM
   */
  mDate = date;

  if (!representsValidDate())
  {
    mDate = "";
    parseDateNumbersToString();
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    parseDateStringToNumbers();
    parseDateNumbersToString();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}



/** @cond doxygen-libsbml-internal */
/*
 * returns the date in numbers as a W3CDTF string
 */
void
Date::parseDateNumbersToString()
{
  char cdate[10];

  if (mMonth < 10)
    sprintf(cdate, "%u-0%u-", mYear, mMonth);
  else
    sprintf(cdate, "%u-%u-", mYear, mMonth);
  mDate = cdate;
  
  if (mDay < 10)
    sprintf(cdate, "0%uT", mDay);
  else
    sprintf(cdate, "%uT", mDay);
  mDate.append(cdate);

  if (mHour < 10)
    sprintf(cdate, "0%u:", mHour);
  else
    sprintf(cdate, "%u:", mHour);
  mDate.append(cdate);
  
  if (mMinute < 10)
    sprintf(cdate, "0%u:", mMinute);
  else
    sprintf(cdate, "%u:", mMinute);
  mDate.append(cdate);
  
  if (mSecond < 10)
    sprintf(cdate, "0%u", mSecond);
  else
    sprintf(cdate, "%u", mSecond);
  mDate.append(cdate);

  if (mHoursOffset == 0 && mMinutesOffset == 0)
  {
    sprintf(cdate, "Z");
    mDate.append(cdate);
  }
  else
  {
    if (mSignOffset == 0)
      sprintf(cdate, "-");
    else
      sprintf(cdate, "+");
    mDate.append(cdate);

    if (mHoursOffset < 10)
      sprintf(cdate, "0%u:", mHoursOffset);
    else
      sprintf(cdate, "%u:", mHoursOffset);
    mDate.append(cdate);
    
    if (mMinutesOffset < 10)
      sprintf(cdate, "0%u", mMinutesOffset);
    else
      sprintf(cdate, "%u", mMinutesOffset);
    mDate.append(cdate);
  }

}
/** @endcond */


/** @cond doxygen-libsbml-internal */
void
Date::parseDateStringToNumbers()
{
  if (mDate.length() == 0)
  {
    mYear   = 2000;
    mMonth  = 1;
    mDay    = 1;
    mHour   = 0;  
    mMinute = 0;
    mSecond = 0;
    
    mSignOffset   = 0;
    mHoursOffset  = 0;
    mMinutesOffset  = 0;
  }
  else
  {
    const char * cdate = mDate.c_str();
    char year[5];
    year[4] = '\0';
    char block[3];
    block[2] = '\0';
    
    year[0] = cdate[0];
    year[1] = cdate[1];
    year[2] = cdate[2];
    year[3] = cdate[3];

    mYear = strtol(year, NULL, 10);

    block[0] = cdate[5];
    block[1] = cdate[6];
    
    mMonth = strtol(block, NULL, 10);

    block[0] = cdate[8];
    block[1] = cdate[9];
    
    mDay = strtol(block, NULL, 10);

    block[0] = cdate[11];
    block[1] = cdate[12];
    
    mHour = strtol(block, NULL, 10);

    block[0] = cdate[14];
    block[1] = cdate[15];
    
    mMinute = strtol(block, NULL, 10);

    block[0] = cdate[17];
    block[1] = cdate[18];
    
    mSecond = strtol(block, NULL, 10);

    if (cdate[19] == '+')
    {
      mSignOffset = 1;
      block[0] = cdate[20];
      block[1] = cdate[21];
      mHoursOffset = strtol(block, NULL, 10);

      block[0] = cdate[23];
      block[1] = cdate[24];
      mMinutesOffset = strtol(block, NULL, 10);
    }
    else if (cdate[19] == '-')
    {
      mSignOffset = 0;
      block[0] = cdate[20];
      block[1] = cdate[21];
      mHoursOffset = strtol(block, NULL, 10);

      block[0] = cdate[23];
      block[1] = cdate[24];
      mMinutesOffset = strtol(block, NULL, 10);
    }
    else
    {
      mSignOffset = 0;
      mHoursOffset = 0;
      mMinutesOffset = 0;
    }
  }
}

bool
Date::representsValidDate()
{
  bool valid = true;
//  parseDateNumbersToString();
  const char * cdate = mDate.c_str();

  if (mDate.length() != 20 && mDate.length() != 25)
  {
    valid = false;
  }
  else if (cdate[4]  != '-' ||
      cdate[7]  != '-' ||
      cdate[10] != 'T' ||
      cdate[13] != ':' ||
      cdate[16] != ':')
  {
    valid = false;
  }
  else if (cdate[19] != 'Z' &&
      cdate[19] != '+' && 
      cdate[19] != '-')
  {
    valid = false;
  }
  else if (cdate[19] != 'Z' &&
           cdate[22] != ':')
  {
    valid = false;
  }


  if (getMonth() > 12 ||
      getDay() > 31   ||
      getHour() > 23  ||
      getMinute() > 59 ||
      getSecond() > 59 ||
      getSignOffset() > 1 ||
      getHoursOffset() > 11 ||
      getMinutesOffset() > 59)
  {
    valid = false;
  }
  else
  {
    switch(getMonth())
    {
    case 4:
    case 6:
    case 9:
    case 11:
      if (getDay() > 30)
        valid = false;
      break;
    case 2:
      if (getYear() % 4 == 0)
      {
        if (getDay() > 29)
          valid = false;
      }
      else
      {
        if (getDay() > 28)
          valid = false;
      }
      break;
    default:
      break;
    }
  }
  
  return valid;
}

bool
Date::hasBeenModified()
{
  return mHasBeenModified;
}

void
Date::resetModifiedFlags()
{
  mHasBeenModified = false;
}



/** @endcond */


/**
 * Creates a date optionally from the individual fields entered as numbers.
 *
 * @param year an unsigned int representing the year.
 * @param month an unsigned int representing the month.
 * @param day an unsigned int representing the day.
 * @param hour an unsigned int representing the hour.
 * @param minute an unsigned int representing the minute.
 * @param second an unsigned int representing the second.
 * @param sign an unsigned int representing the sign of the offset 
 * (0/1 equivalent to +/-). 
 * @param hoursOffset an unsigned int representing the hoursOffset.
 * @param minutesOffset an unsigned int representing the minutesOffset.
 *
 * @return pointer to the newly created Date_t structure.
 */
LIBSBML_EXTERN
Date_t *
Date_createFromValues(unsigned int year, unsigned int month, 
    unsigned int day, unsigned int hour, 
    unsigned int minute, unsigned int second,
    unsigned int sign, unsigned int hoursOffset,
    unsigned int minutesOffset)
{
  return new(nothrow) Date(year, month, day, hour, minute,
    second, sign, hoursOffset, minutesOffset);
}


/**
 * Creates a date from a string.
 *
 * @param date a string representing the date.
 *
 * @return pointer to the newly created Date_t structure.
 *
 * @note the string should be in W3CDTF format 
 * YYYY-MM-DDThh:mm:ssTZD (eg 1997-07-16T19:20:30+01:00)
 * where TZD is the time zone designator.
 */
LIBSBML_EXTERN
Date_t *
Date_createFromString (const char * date)
{
  if (date == NULL ) return NULL;
  return new(nothrow) Date(date);
}


/**
 * Destroys this Date.
 *
 * @param date Date_t structure to be freed.
 */
LIBSBML_EXTERN
void
Date_free(Date_t * date)
{
  delete static_cast<Date*>(date);
}


/**
 * Creates a deep copy of the given Date_t structure
 * 
 * @param date the Date_t structure to be copied
 * 
 * @return a (deep) copy of the given Date_t structure.
 */
LIBSBML_EXTERN
Date_t *
Date_clone (const Date_t* date)
{
  if (date == NULL ) return NULL;
  return static_cast<Date*>( date->clone() );
}


/**
 * Returns the Date as a string.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the date as a string.
 */
LIBSBML_EXTERN
const char *
Date_getDateAsString(Date_t * date)
{
  if (date == NULL) return NULL;
  return date->getDateAsString().c_str();
}


/**
 * Returns the year from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the year from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getYear(Date_t * date)
{
  if (date == NULL) return SBML_INT_MAX;
  return date->getYear();
}


/**
 * Returns the month from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the month from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getMonth(Date_t * date)
{
  if (date == NULL) return SBML_INT_MAX;
  return date->getMonth();
}


/**
 * Returns the day from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the day from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getDay(Date_t * date)
{
  if (date == NULL) return SBML_INT_MAX;
  return date->getDay();
}


/**
 * Returns the hour from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the hour from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getHour(Date_t * date)
{
  if (date == NULL) return SBML_INT_MAX;
  return date->getHour();
}


/**
 * Returns the minute from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the minute from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getMinute(Date_t * date)
{
  if (date == NULL) return SBML_INT_MAX;
  return date->getMinute();
}


/**
 * Returns the seconds from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the seconds from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getSecond(Date_t * date) 
{ 
  if (date == NULL) return SBML_INT_MAX;
  return date->getSecond(); 
} 


/**
 * Returns the sign of the offset from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the sign of the offset from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getSignOffset(Date_t * date) 
{ 
  if (date == NULL) return SBML_INT_MAX;
  return date->getSignOffset(); 
} 


/**
 * Returns the hours of the offset from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the hours of the offset from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getHoursOffset(Date_t * date) 
{ 
  if (date == NULL) return SBML_INT_MAX;
  return date->getHoursOffset(); 
} 


/**
 * Returns the minutes of the offset from this Date.
 *
 * @param date the Date_t structure to be queried
 * 
 * @return the minutes of the offset from this Date.
 */
LIBSBML_EXTERN
unsigned int
Date_getMinutesOffset(Date_t * date) 
{ 
  if (date == NULL) return SBML_INT_MAX;
  return date->getMinutesOffset(); 
} 


/**
 * Sets the value of the year checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the year to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setYear(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setYear(value); 
}


/**
 * Sets the value of the month checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the month to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setMonth(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setMonth(value); 
}


/**
 * Sets the value of the day checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the day to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setDay(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setDay(value); 
}


/**
 * Sets the value of the hour checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the hour to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT;
 */
LIBSBML_EXTERN
int
Date_setHour(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setHour(value); 
}


/**
 * Sets the value of the minute checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the minute to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setMinute(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setMinute(value); 
}


/**
 * Sets the value of the second checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the second to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setSecond(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setSecond(value); 
}


/**
 * Sets the value of the offset sign checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the sign of the 
 * offset to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setSignOffset(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setSignOffset(value); 
}


/**
 * Sets the value of the offset hour checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the hours of the 
 * offset to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setHoursOffset(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setHoursOffset(value); 
}


/**
 * Sets the value of the offset minutes checking appropriateness.
 *  
 * @param date the Date_t structure to be set
 * @param value an unsigned int representing the minutes of the 
 * offset to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setMinutesOffset(Date_t * date, unsigned int value) 
{ 
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return date->setMinutesOffset(value); 
}

/**
 * Sets the value of the date from a string.
 *  
 * @param date the Date_t structure to be set
 * @param str string representing the date to set.  
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Date_setDateAsString(Date_t * date, const char *str)
{
  if (date == NULL) return LIBSBML_INVALID_OBJECT;
  return (str == NULL) ? date->setDateAsString("") :
                          date->setDateAsString(str);
}


LIBSBML_EXTERN
int
Date_representsValidDate(Date_t *date)
{
  if (date == NULL) return (int)false;
  return static_cast<int> (date->representsValidDate());
}


LIBSBML_CPP_NAMESPACE_END

