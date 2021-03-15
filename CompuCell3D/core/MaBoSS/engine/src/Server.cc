/*
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2011-2020 Institut Curie, 26 rue d'Ulm, Paris, France       #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################

   Module:
     Server.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     May 2018
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <signal.h>

#include "Client.h"
#include "Server.h"
#include "DataStreamer.h"
#include "RPC.h"
#include "Utils.h"
#include "MaBEstEngine.h"
#include "FinalStateSimulationEngine.h"
#include "Function.h"

Server* Server::server;
static const char* RPC_portname;

static void unlink_tempfiles_handler(int sig)
{
  const std::string& pidfile = Server::getInstance()->getPidFile();
  if (pidfile.length() > 0) {
    unlink(pidfile.c_str());
  }
  if (NULL != RPC_portname) {
    unlink(RPC_portname);
  }
  exit(1);
}

int Server::manageRequests()
{
  MaBEstEngine::init();
  if (bind(&RPC_portname) == rpc_Success) {
    if (0 != pidfile.length()) {
      std::ofstream fpidfile(pidfile.c_str());
      if (fpidfile.bad() || fpidfile.fail()) {
	std::cerr << "cannot open pidfile " << pidfile << " for writing\n";
	return 1;
      }
      fpidfile << getpid() << '\n';
      fpidfile.close();
    }
    signal(SIGINT, unlink_tempfiles_handler);
    signal(SIGQUIT, unlink_tempfiles_handler);
    signal(SIGTERM, unlink_tempfiles_handler);
    signal(SIGABRT, unlink_tempfiles_handler);

    signal(SIGCHLD, SIG_IGN);
    
    time_t t = time(NULL);
    char* now = ctime(&t);
    if (!quiet) {
      std::cerr << "\n" << prog << " [listen=" << host << ":" << port << "] Ready at " << now;
    }
    listen();
    return 0;
  }
  return 1;
}

#define ostringstream2str(OSTR) (((std::ostringstream*)(OSTR))->str())

void Server::run(const ClientData& client_data, ServerData& server_data)
{
  static const std::string hst = "==================";
  std::ostream* output_run = NULL;
  std::ostream* output_traj = NULL;
  std::ostream* output_probtraj = NULL;
  std::ostream* output_statdist = NULL;
  std::ostream* output_fp = NULL;

  std::ostringstream ostr;
  struct timeval tv;
  gettimeofday(&tv, 0);
  ostr << "/tmp/MaBoSS-server_" << tv.tv_sec << "_" << tv.tv_usec << "_" << getpid();
  std::string tmp_output = ostr.str();

  try {
    time_t start_time, end_time;
    time(&start_time);
    char* timebuf = ctime(&start_time);
    timebuf[strlen(timebuf)-1] = 0;
    if (!quiet) {
      std::cerr << "\n" << hst << " " << prog << " running simulation at " << timebuf << " " << hst << "\n";
    }

    Node::setOverride((client_data.getFlags() & DataStreamer::OVERRIDE_FLAG) != 0);
    Node::setAugment((client_data.getFlags() & DataStreamer::AUGMENT_FLAG) != 0);
    if (Node::isOverride() && Node::isAugment()) {
      server_data.setStatus(2);
      server_data.setErrorMessage("override and augment are exclusive flags");
      return;
    }

    Network* network = new Network();
    // EV: 2020-12-11: currently mandatory to create a temporary file as parsing can only be done from a FILE*
    std::string network_file = tmp_output + "_network.bnd";
    filePutContents(network_file, client_data.getNetwork());

    network->parse(network_file.c_str(), NULL, true);

    RunConfig* runconfig = new RunConfig();
    const std::string& config_vars = client_data.getConfigVars();
    if (config_vars.length() > 0) {
      std::vector<std::string> runconfig_var_v;
      runconfig_var_v.push_back(config_vars);
      if (setConfigVariables(network, prog, runconfig_var_v)) {
        //return 1;
        // TBD: error
        return;
      }      
    }

    const std::vector<std::string>& configs = client_data.getConfigs();
    for (std::vector<std::string>::const_iterator iter = configs.begin(); iter != configs.end(); ++iter) {
      runconfig->parseExpression(network, iter->c_str());
    }

    const std::vector<std::string>& config_exprs = client_data.getConfigExprs();
    for (std::vector<std::string>::const_iterator iter = config_exprs.begin(); iter != config_exprs.end(); ++iter) {
      runconfig->parseExpression(network, iter->c_str());
    }

    IStateGroup::checkAndComplete(network);

    network->getSymbolTable()->checkSymbols();

    if (client_data.getCommand() == DataStreamer::CHECK_COMMAND) {
      server_data.setStatus(0);
      return;
    }

    if (runconfig->displayTrajectories()) {
      if (runconfig->getThreadCount() > 1) {
	if (!quiet) {
	  std::cerr << '\n' << prog << ": warning: cannot display trajectories in multi-threaded mode\n";
	}
      } else {
	output_traj = new std::ostringstream();
      }
    }

    bool hexfloat = (client_data.getFlags() & DataStreamer::HEXFLOAT_FLAG) != 0;
    bool final_simulation = (client_data.getFlags() & DataStreamer::FINAL_SIMULATION_FLAG) != 0;

    if (final_simulation) {
      std::ostream* output_final = new std::ostringstream();
      FinalStateSimulationEngine engine(network, runconfig);
      engine.run(NULL);
      engine.displayFinal(*output_final, hexfloat);
      server_data.setStatus(0);
      server_data.setFinalProb(ostringstream2str(output_final));
      delete output_final;
    } else {
      // EV: 2020-11-11: instead of using temp files, use std::ostringstream to make Gautier happy
      output_run = new std::ostringstream();
      output_probtraj = new std::ostringstream();
      output_statdist = new std::ostringstream();
      output_fp = new std::ostringstream();

      MaBEstEngine mabest(network, runconfig);
      mabest.run(output_traj);
      mabest.display(*output_probtraj, *output_statdist, *output_fp, hexfloat);
      time(&end_time);

      runconfig->display(network, start_time, end_time, mabest, *output_run);

      server_data.setStatus(0);
      server_data.setStatDist(ostringstream2str(output_statdist));
      server_data.setProbTraj(ostringstream2str(output_probtraj));
      server_data.setRunLog(ostringstream2str(output_run));
      server_data.setFP(ostringstream2str(output_fp));
      if (NULL != output_traj) {
	server_data.setTraj(ostringstream2str(output_traj));
      }

      if (!quiet) {
	std::cerr << "\n" << server_data.getRunLog();
      }
      timebuf = ctime(&end_time);
      timebuf[strlen(timebuf)-1] = 0;
      if (!quiet) {
	std::cerr << hst << " " << prog << " simulation finished at " << timebuf << " " << hst << "\n";;
      }
    }
    delete runconfig;
    delete network;
  } catch(const BNException& e) {
    if (!quiet) {
      std::cerr << "\n" << hst << " " << prog << " simulation error [[\n" << e << "]] " << hst << "\n";
    }
    server_data.setStatus(1);
    server_data.setErrorMessage(e.getMessage());

    return;
  }
  Function::destroyFuncMap();
}

void Server::manageRequest(int fd, const char* request)
{
  ClientData client_data;
  ServerData server_data;
  std::string err_data;
  int status;

  if (verbose) {
    std::cout << "request [" << request << "]\n";
  }

  if ((status = DataStreamer::parseStreamData(client_data, request, err_data)) != 0) {
    server_data.setStatus(status);
    server_data.setErrorMessage(err_data);
    //rpc_writeStringData(fd, err_data.c_str(), err_data.length());
    //return;
  } else {
    if (client_data.getCommand() == DataStreamer::RUN_COMMAND || client_data.getCommand() == DataStreamer::CHECK_COMMAND) {
      run(client_data, server_data);
    } else {
      std::cerr << "unknown command \"" << client_data.getCommand() << "\"\n";
      return;
    }
  }

  std::string data;
  DataStreamer::buildStreamData(data, server_data);
  if (verbose) {
    std::cout << "response [" << data << "]\n";
  }
  rpc_writeStringData(fd, data.c_str(), data.length());
}
