/*******************************************************************\

              Copyright (C) 2003 Joseph Coffland

    This program is free software; you can redistribute it and/or
     modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
             GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
      Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
                           02111-1307, USA.

            For information regarding this software email
                   jcofflan@users.sourceforge.net

\*******************************************************************/
#include "BasicString.h"

#include "BasicException.h"

//#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <cstdio>

using namespace std;

BasicString::BasicString(const int x) {
  char buf[12];
  sprintf(buf, "%i", x);
  *this = buf;
}

BasicString::BasicString(const unsigned x) {
  char buf[12];
  sprintf(buf, "%i", x);
  *this = buf;
}

BasicString::BasicString(const long x) {
  char buf[12];
  sprintf(buf, "%li", x);
  *this = buf;
}

BasicString::BasicString(const unsigned long x) {
  char buf[12];
  sprintf(buf, "%lu", x);
  *this = buf;
}

BasicString::BasicString(const double x) {
  char buf[16];
  sprintf(buf, "%f", x);
  *this = buf;
}

unsigned char BasicString::parseUByte(const std::string s) {
  unsigned int v = parseUInteger(s);
  ASSERT_OR_THROW(string("Byte value '") + s + "'out of range!", v < 256);

  return (unsigned char)v;
}

char BasicString::parseByte(const std::string s) {
  int v = parseUInteger(s);
  ASSERT_OR_THROW(string("Byte value '") + s + "'out of range!",
		  -128 < v  && v < 128);

  return (char)v;
}

unsigned short BasicString::parseUShort(const std::string s) {
  unsigned int v = parseUInteger(s);
  ASSERT_OR_THROW(string("Short value '") + s + "'out of range!", v < 65536);

  return (unsigned short)v;
}

short BasicString::parseShort(const std::string s) {
  int v = parseUInteger(s);
  ASSERT_OR_THROW(string("Byte value '") + s + "'out of range!",
		  -32768 < v  && v < 32768);

  return (short)v;
}

unsigned int BasicString::parseUInteger(const std::string s) {
  errno = 0;
//  unsigned long v = strtol(s.c_str(), 0, 10);
//  ASSERT_OR_THROW(string("parseUInteger() Invalid unsigned integer '") + s +
//		  "'!", errno == 0 && v >= 0);

  unsigned long v = 0;
  if (s[0] == '#') {
    cout << "WARNING:  BasicString::parseUInteger got hex string=" << s <<endl;
  }
  else {
    v = strtol(s.c_str(), 0, 10);
    ASSERT_OR_THROW(string("parseUInteger() Invalid unsigned integer '") + s +
                  "'!", errno == 0 && v >= 0);
  }

  return (unsigned int)v;
}

int BasicString::parseInteger(const std::string s) {
  errno = 0;
  long v = strtol(s.c_str(), 0, 10);
  ASSERT_OR_THROW(string("parseInteger() Invalid integer '") + s +
		  "'!", errno == 0);

  return (int)v;
}

double BasicString::parseDouble(const std::string s) {
  errno = 0;
  double v = strtod(s.c_str(), 0);
  ASSERT_OR_THROW(string("parseDouble() Invalid double '") + s +
		  "'!", errno == 0);
  return v;
}

bool BasicString::parseBool(const std::string s) {
  string v = toLower(trim(s));
  if (v == "true") return true;
  if (v == "false") return false;
  THROW(string("parseBool() Invalid bool '") + s + "'!");
}

string BasicString::trim(const string s) {
  string::size_type start = s.find_first_not_of(" \t\n\r");
  string::size_type end = s.find_last_not_of(" \t\n\r");

  if (start == string::npos) return "";
  return s.substr(start, (end - start) + 1);
}

string BasicString::toUpper(const string s) {
  string v;
  string::size_type len = s.length();
  v.resize(len, ' ');

  for (string::size_type i = 0; i < len; i++)
    v[i] = toupper(s[i]);

  return v;
}

string BasicString::toLower(const string s) {
  string v;
  string::size_type len = s.length();
  v.resize(len, ' ');

  for (string::size_type i = 0; i < len; i++)
    v[i] = tolower(s[i]);

  return v;
}
