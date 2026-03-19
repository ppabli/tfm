CXX = mpicxx
MPIRUN = mpirun

NP = 4

CXXFLAGS = -O2 -Wall -std=c++20 -MMD -MP
LIBS = -l pthread

BUILD_DIR = build

TARGET_MC = $(BUILD_DIR)/montecarlo
SRCS_MC = malleable.cpp montecarlo.cpp
OBJS_MC = $(SRCS_MC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_VEC = $(BUILD_DIR)/vector
SRCS_VEC = malleable.cpp vector.cpp
OBJS_VEC = $(SRCS_VEC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MM = $(BUILD_DIR)/matmul
SRCS_MM = malleable.cpp matmul.cpp
OBJS_MM = $(SRCS_MM:%.cpp=$(BUILD_DIR)/%.o)

TARGET_CONV = $(BUILD_DIR)/convolution
SRCS_CONV = malleable.cpp convolution.cpp
OBJS_CONV = $(SRCS_CONV:%.cpp=$(BUILD_DIR)/%.o)
DEPS = $(OBJS_MC:.o=.d) $(OBJS_VEC:.o=.d) $(OBJS_MM:.o=.d) $(OBJS_CONV:.o=.d)

.PHONY: all run_montecarlo run_vector run_vector_collapse run_matmul run_convolution clean re

all: $(TARGET_MC) $(TARGET_VEC) $(TARGET_MM) $(TARGET_CONV)

$(TARGET_MC): $(OBJS_MC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(TARGET_VEC): $(OBJS_VEC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(TARGET_MM): $(OBJS_MM)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(TARGET_CONV): $(OBJS_CONV)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) -lm

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run_montecarlo: $(TARGET_MC)
	$(MPIRUN) -n $(NP) ./$(TARGET_MC)

run_vector: $(TARGET_VEC)
	$(MPIRUN) -n $(NP) ./$(TARGET_VEC)

run_vector_collapse: $(TARGET_VEC)
	$(MPIRUN) -n $(NP) ./$(TARGET_VEC) collapse

run_matmul: $(TARGET_MM)
	$(MPIRUN) -n $(NP) ./$(TARGET_MM) mm

run_matvec: $(TARGET_MM)
	$(MPIRUN) -n $(NP) ./$(TARGET_MM) mv

run_convolution: $(TARGET_CONV)
	$(MPIRUN) -n $(NP) ./$(TARGET_CONV)

re: clean all

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)