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


#ifndef BASICFILELOCATION_H
#define BASICFILELOCATION_H

#include <string>

/** 
 * This class is mainly used by BasicException, but can be used
 * as a general class for recording a line and column location
 * with in a file.
 */
class BasicFileLocation {
  std::string filename;
  long line;
  long col;
  bool empty;

public:
  /** 
   * Construct a default BasicFileLocation with an empty value.
   */
  BasicFileLocation() : line(-1), col(-1), empty(true) {}

  /** 
   * Copy constructor.
   */
  BasicFileLocation(const BasicFileLocation &x) : 
    filename(x.filename), line(x.line), col(x.col), empty(x.empty) {}

  /** 
   * @param filename The name of the file.
   * @param line The line with that file.
   * @param col The column on that line.
   */
  BasicFileLocation(const std::string filename, const long line, 
		    const long col) : 
    filename(filename), line(line), col(col), empty(false) {}

  virtual ~BasicFileLocation() {}

  const std::string getFilename() const {return filename;}

  /** 
   * @return -1 if no line was set the line number otherwise.
   */  
  const long getLine() const {return line;}

  /** 
   * @return -1 of no column was set the column number otherwise.
   */
  const long getCol() const {return col;}

  /** 
   * @return True of no filename, line, or column have been set.
   */
  bool isEmpty() const {return empty;}

  friend std::ostream &operator<<(std::ostream &stream, 
				  const BasicFileLocation &fl);
};  

/** 
 * Print a file location to a stream.  The format is as follows.
 *
 * filename[:line[:col]]
 *
 * If no line or column has been set then they will not be displayed.
 * 
 * @return A reference to the passed stream.
 */
std::ostream &operator<<(std::ostream &stream, const BasicFileLocation &fl);

#define FILE_LOCATION BasicFileLocation(__FILE__, __LINE__, -1)

#endif
