/*******************************************************************\

              Copyright (C) 2004 Joseph Coffland

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
#include "BasicProcess.h"

#include "BasicException.h"
#include "BasicPipe.h"
#include "BasicString.h"
#include "Zap.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <string>

using namespace std;


BasicProcess::BasicProcess() {
  pid = 0;
  running = false;
  returnCode = 0;
}

BasicProcess::~BasicProcess() {
  if (isRunning()) {
    kill(SIGHUP);
    wait();
  }

  pipes_t::iterator it;
  for (it = pipes.begin(); it != pipes.end(); it++)
    delete *it;
}

void BasicProcess::exec(list<string> &args) {

  BasicSmartPointer<char *, SP_ARRAY> argv = new char *[args.size() + 1];
  list<string>::iterator it;
  unsigned int i = 0;
  for (i = 0, it = args.begin(); it != args.end(); i++, it++)
    argv[i] = (char *)it->c_str();
  argv[i] = 0;
  
  exec(argv.get());
}

void BasicProcess::exec(char *args) {

  char *argv[256];
  int argc = 0;
  parseArgs(args, argc, argv, 256);
  exec(argv);
}

void BasicProcess::exec(char *argv[]) {
  ASSERT_OR_THROW("Process all ready running!", !isRunning());

  pid = fork();

  if (pid == 0) {
    pipes_t::iterator it;
    for (it = pipes.begin(); it != pipes.end(); it++) {
      ChildPipe *pipe = *it;

      if (pipe->direction == TO_CHILD) {
	pipe->pipe.closeIn();
	if (pipe->fd >= 0) pipe->pipe.moveOutFD(pipe->fd);
	
      } else {
	pipe->pipe.closeOut();
	if (pipe->fd >= 0) pipe->pipe.moveInFD(pipe->fd);
      }
    }

    execvp(argv[0], argv);
    
    // Execution failed!
    perror("BasicProcess() executing child process");
    exit(1);

  } else if (pid == -1)
    THROW("Failed to spawn child!");

  running = true;

  pipes_t::iterator it;
  for (it = pipes.begin(); it != pipes.end(); it++)
    if ((*it)->direction == TO_CHILD) (*it)->pipe.closeOut();
    else (*it)->pipe.closeIn();
}

void BasicProcess::parseArgs(char *args, int &argc, char *argv[], int n) {
  if (args) {

    bool inArg = false;
    bool inSQuote = false;
    bool inDQuote = false;
    bool addChar;
    int i = 0;
    
    for (char *ptr = args; *ptr; ptr++) {
      addChar = false;
      
      switch (*ptr) {
      case '\\':
	if (ptr[1] != '\0') {
	  ptr++;
	  switch (*ptr) {
	  case 'n': *ptr = '\n'; break;
	  case 't': *ptr = '\t'; break;
	  case 'r': *ptr = '\r'; break;
	  default: break;
	  }
	  addChar = true;
	} 
	break;
	
      case '\'':
	if (inDQuote) addChar = true;
	else {
	  if (inSQuote) inSQuote = false;
	  else inSQuote = true;
	}
	break;
	
      case '"':
	if (inSQuote) addChar = true;
	else {
	  if (inDQuote) inDQuote = false;
	  else inDQuote = true;
	}
	break;
	
      case '\t':
      case ' ':
      case '\n':
      case '\r':
	if (inArg) {
	  if (inSQuote || inDQuote) addChar = true;
	  else {
	    args[i++] = 0;
	    inArg = false;
	  }
	}
	break;
	
      default: addChar = true; break;
      }
    
      if (addChar) {
	if (!inArg) {
	  ASSERT_OR_THROW("Too many arguments!", argc < n - 1);
	  argv[argc++] = &args[i];
	  inArg = true;
	}
	args[i++] = *ptr;
      }
    }

    if (inArg) args[i] = 0;
  }

  argv[argc] = 0;
}

BasicPipe *BasicProcess::getChildPipe(dir_t dir, int childFD) {
  ASSERT_OR_THROW("Process all ready running!", !running);

  ChildPipe *pipe = new ChildPipe(dir, childFD);
  pipes.push_back(pipe);

  return &pipe->pipe;
}


void BasicProcess::kill(int sig) {
  ASSERT_OR_THROW("Process not running!", running);

  if (::kill(pid, sig) != 0) {
    THROW(string("Failed to kill process ") + BasicString(pid) + ":" + 
	  strerror(errno));
  }
}

int BasicProcess::wait(int options) {
  ASSERT_OR_THROW("Process not running!", running);

  int status = 0;
  int retVal = waitpid(pid, &status, options);

  if (retVal == -1) {
    THROW(string("Failed to wait on process ") + BasicString(pid) + ":" + 
	  strerror(errno));
  }

  if (retVal) {
    if (WIFEXITED(status)) {
      returnCode = WEXITSTATUS(status);
      running = false;
    }
  }

  return status;
}

bool BasicProcess::isRunning() {
  if (running) wait(WNOHANG);
  return running;
}
