# Makefile for C++ test application.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

CPPFLAGS=-I../include/ -Wall

HEADERS=../include/arithmetic.hpp ../include/autodouble.hpp \
        ../include/quaternion.hpp ../include/types.hpp \
        ../include/constants.hpp ../include/functions.hpp

ALLHEADERS = $(HEADERS) ../include/dataset.hpp ../include/projecteddata.hpp \
             ../include/cost.hpp ../include/optimize.hpp ../include/bfgs.hpp \
             ../include/linalg.hpp

ALLSOURCES = ../src/dataset.cpp ../src/projecteddata.cpp ../src/cost.cpp \
             ../src/arithmetic.cpp ../src/labordecylinder.cpp \
             ../src/functions.cpp ../src/ctypesinterface.cpp \
             ../src/optimize.cpp

all: testing doomercat.so


testing: $(HEADERS) arithmetic.o testing.o labordecylinder.o dataset.o \
         projecteddata.o cost.o optimize.o
	g++ -o testing arithmetic.o testing.o

dataset.o: $(HEADERS) ../include/dataset.hpp ../src/dataset.cpp
	g++ $(CPPFLAGS) -c ../src/dataset.cpp

projecteddata.o: $(HEADERS) ../include/dataset.hpp ../include/projecteddata.hpp\
                 ../src/projecteddata.cpp
	g++ $(CPPFLAGS) -c ../src/projecteddata.cpp

cost.o: $(HEADERS) ../include/cost.hpp ../include/dataset.hpp \
                   ../include/projecteddata.hpp ../src/cost.cpp
	g++ $(CPPFLAGS) -c ../src/cost.cpp

testing.o: $(HEADERS) testing.cpp
	g++ $(CPPFLAGS) -c testing.cpp

arithmetic.o: $(HEADERS) ../src/arithmetic.cpp
	g++ $(CPPFLAGS) -c ../src/arithmetic.cpp

labordecylinder.o: $(HEADERS) ../src/labordecylinder.cpp
	g++ $(CPPFLAGS) -c ../src/labordecylinder.cpp

functions.o: $(HEADERS) ../src/functions.cpp
	g++ $(CPPFLAGS) -c ../src/functions.cpp

optimize.o: $(ALLHEADERS) ../src/optimize.cpp
	g++ $(CPPFLAGS) -c ../src/optimize.cpp

doomercat.so: $(ALLHEADERS) $(ALLSOURCES)
	g++ $(CPPFLAGS) -O3 -g0 -march=native -fPIC $(ALLSOURCES) -shared \
	   -o doomercat.so -Wl,-z,defs
