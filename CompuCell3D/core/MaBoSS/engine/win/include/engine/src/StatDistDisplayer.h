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
     StatDistDisplayer.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     Decembre 2020
*/

#ifndef _STATDIST_DISPLAYER_H_
#define _STATDIST_DISPLAYER_H_

#include <iostream>

class StatDistDisplayer {

protected:
  Network* network;
  bool hexfloat;
  size_t max_size;
  size_t statdist_traj_count;

  size_t current_line;
  size_t num;

  StatDistDisplayer(Network* network, bool hexfloat = false) : network(network), hexfloat(hexfloat), current_line(0) { }

public:
  void begin(size_t max_size, size_t statdist_traj_count) {
    this->max_size = max_size;
    this->statdist_traj_count = statdist_traj_count;
    beginDisplay();
  }

  void beginStateProba(size_t num) {
    this->num = num;
    beginStateProbaDisplay();
  }

  virtual void addStateProba(const NetworkState_Impl& state, double proba) = 0;

  void endStateProba() {
    endStateProbaDisplay();
    current_line++;
  }

  void end() {
    endDisplay();
  }

  virtual void beginDisplay() = 0;

  virtual void beginStatDistDisplay() = 0;
  virtual void beginStateProbaDisplay() = 0;
  virtual void endStateProbaDisplay() = 0;
  virtual void endStatDistDisplay() = 0;

  virtual void beginFactoryCluster() { }
  virtual void endFactoryCluster() { }

  virtual void beginCluster(size_t num, size_t size) = 0;
  virtual void endCluster() = 0;

  virtual void beginClusterFactoryStationaryDistribution() = 0;
  virtual void endClusterFactoryStationaryDistribution() = 0;

  virtual void beginClusterStationaryDistribution(size_t num) = 0;
  virtual void endClusterStationaryDistribution() = 0;

  virtual void addProbaVariance(const NetworkState_Impl& state, double proba, double variance) = 0;
  virtual void endDisplay() = 0;

  virtual ~StatDistDisplayer() { }
};

class CSVStatDistDisplayer : public StatDistDisplayer {

  std::ostream& os_statdist;

public:
  CSVStatDistDisplayer(Network* network, std::ostream& os_statdist, bool hexfloat = false) : StatDistDisplayer(network, hexfloat), os_statdist(os_statdist) { }

  virtual void beginDisplay();

  virtual void beginStatDistDisplay();
  virtual void beginStateProbaDisplay();
  virtual void addStateProba(const NetworkState_Impl& state, double proba);
  virtual void endStateProbaDisplay();
  virtual void endStatDistDisplay();

  virtual void beginCluster(size_t num, size_t size);
  virtual void endCluster();

  virtual void beginClusterFactoryStationaryDistribution();
  virtual void endClusterFactoryStationaryDistribution();

  virtual void beginClusterStationaryDistribution(size_t num);
  virtual void endClusterStationaryDistribution();

  virtual void addProbaVariance(const NetworkState_Impl& state, double proba, double variance);

  virtual void endDisplay();
};

class JSONStatDistDisplayer : public StatDistDisplayer {

  std::ostream& os_statdist;
  std::ostream& os_statdist_cluster;
  std::ostream& os_statdist_distrib;
  size_t current_state_proba;
  bool cluster_mode;

public:
  JSONStatDistDisplayer(Network* network, std::ostream& os_statdist, std::ostream& os_statdist_cluster, std::ostream& os_statdist_distrib, bool hexfloat = false) : StatDistDisplayer(network, hexfloat), os_statdist(os_statdist), os_statdist_cluster(os_statdist_cluster), os_statdist_distrib(os_statdist_distrib), current_state_proba(0), cluster_mode(false) { }

  virtual void beginDisplay();

  virtual void beginStatDistDisplay();
  virtual void beginStateProbaDisplay();
  virtual void addStateProba(const NetworkState_Impl& state, double proba);
  virtual void endStateProbaDisplay();
  virtual void endStatDistDisplay();

  virtual void beginFactoryCluster();
  virtual void endFactoryCluster();
  virtual void beginCluster(size_t num, size_t size);
  virtual void endCluster();

  virtual void beginClusterFactoryStationaryDistribution();
  virtual void endClusterFactoryStationaryDistribution();

  virtual void beginClusterStationaryDistribution(size_t num);
  virtual void endClusterStationaryDistribution();

  virtual void addProbaVariance(const NetworkState_Impl& state, double proba, double variance);
  virtual void endDisplay();
};

#endif
