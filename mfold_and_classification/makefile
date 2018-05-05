CXX = g++ -std=c++11
INCLUDES = ../Matrix/include
CXXFLAGS = -O2 -g -Wall -I$(INCLUDES)
LIBS = $(INCLUDES)/libmatrix.a
EXECUTABLES = driver
HEADERS = mat.hpp validate.hpp

all: $(EXECUTABLES)

.PHONY: clean
clean:
		@rm -f core $(EXECUTABLES) *.o *.txt

#.SUFFIXES: .cpp .o
#.cpp.o:
#		$(CXX) $(CXXFLAGS) -c $*.cpp

validate.o: validate.cpp mat.cpp $(HEADERS)
		$(CXX) $(CXXFLAGS) -c validate.cpp

mat.o: mat.cpp mat.hpp
		$(CXX) $(CXXFLAGS) -c mat.cpp

driver.o: mat.cpp validate.cpp $(HEADERS)
		$(CXX) $(CXXFLAGS) -c driver.cpp

driver: validate.o mat.o driver.o
		@$(CXX) -g -o driver mat.o validate.o driver.o $(LIBS)
