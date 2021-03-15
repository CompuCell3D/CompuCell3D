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
     FixedPointDisplayer.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     Decembre 2020
*/

#ifndef _FIXEDPOINT_DISPLAYER_H_
#define _FIXEDPOINT_DISPLAYER_H_

#include <iostream>
class NetworkState;
class Network;

class FixedPointDisplayer {

public:
  virtual void begin(size_t size) = 0;
  virtual void displayFixedPoint(size_t num, const NetworkState& state, unsigned int val, unsigned int sample_count) = 0;
  virtual void end() = 0;
};

class CSVFixedPointDisplayer : public FixedPointDisplayer {

  std::ostream& os;
  Network* network;
  bool hexfloat;
public:
  CSVFixedPointDisplayer(Network* network, std::ostream& os, bool hexfloat) : os(os), network(network), hexfloat(hexfloat) {}
  virtual void begin(size_t size);
  virtual void displayFixedPoint(size_t num, const NetworkState& state, unsigned int val, unsigned int sample_count);
  virtual void end();
};

class JsonFixedPointDisplayer : public FixedPointDisplayer {

  std::ostream& os;
  Network* network;
  bool hexfloat;
public:
  JsonFixedPointDisplayer(Network* network, std::ostream& os, bool hexfloat) : os(os), network(network), hexfloat(hexfloat) {}
  virtual void begin(size_t size);
  virtual void displayFixedPoint(size_t num, const NetworkState& state, unsigned int val, unsigned int sample_count);
  virtual void end();
};

#endif

