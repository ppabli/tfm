CXX = mpicxx
MPIRUN = mpirun

NP = 4

TARGET = montecarlo
SRCS = malleable.cpp montecarlo.cpp
OBJS = $(SRCS:.cpp=.o)

CXXFLAGS = -O2 -Wall -std=c++17
LIBS = -l pthread

.PHONY: all run clean info re

all: $(TARGET)

$(TARGET): $(OBJS)

	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

malleable.o: malleable.cpp malleable.hpp

	$(CXX) $(CXXFLAGS) -c $< -o $@

montecarlo.o: montecarlo.cpp malleable.hpp

	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)

	$(MPIRUN) -n $(NP) ./$(TARGET)

re: clean all

clean:

	$(RM) $(OBJS) $(TARGET)