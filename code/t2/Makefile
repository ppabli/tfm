CXX = mpicxx
AR = ar
MPIRUN = mpirun

NP = 4
MAL_DEFAULT_RESIZE_SEQ = 2,4,4,3,1,4

CXXFLAGS = -O2 -Wall -std=c++20 -MMD -MP -Imalleable/include -DMAL_DEFAULT_RESIZE_SEQ=\"$(MAL_DEFAULT_RESIZE_SEQ)\"
LIBS = -lpthread

BUILD_DIR = build

LIB_MAL = $(BUILD_DIR)/libmalleable.a
LIB_SRCS = malleable/src/malleable.cpp
LIB_OBJS = $(LIB_SRCS:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MC = $(BUILD_DIR)/montecarlo
SRCS_MC = examples/montecarlo.cpp
OBJS_MC = $(SRCS_MC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_VEC = $(BUILD_DIR)/vector
SRCS_VEC = examples/vector.cpp
OBJS_VEC = $(SRCS_VEC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MM = $(BUILD_DIR)/matmul
SRCS_MM = examples/matmul.cpp
OBJS_MM = $(SRCS_MM:%.cpp=$(BUILD_DIR)/%.o)

DEPS = $(LIB_OBJS:.o=.d) $(OBJS_MC:.o=.d) $(OBJS_VEC:.o=.d) $(OBJS_MM:.o=.d)

.PHONY: all run_montecarlo run_vector run_vector_collapse run_matmul run_matvec test clean re

all: $(LIB_MAL) $(TARGET_MC) $(TARGET_VEC) $(TARGET_MM)

$(LIB_MAL): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(TARGET_MC): $(LIB_MAL) $(OBJS_MC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MC) $(LIB_MAL) $(LIBS)

$(TARGET_VEC): $(LIB_MAL) $(OBJS_VEC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_VEC) $(LIB_MAL) $(LIBS)

$(TARGET_MM): $(LIB_MAL) $(OBJS_MM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MM) $(LIB_MAL) $(LIBS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run_montecarlo: $(TARGET_MC)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC)

run_vector: $(TARGET_VEC)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC)

run_vector_collapse: $(TARGET_VEC)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) collapse

run_matmul: $(TARGET_MM)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM) mm

run_matvec: $(TARGET_MM)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM) mv

test: all
	@out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) 2>&1`; \
	echo "$$out" | grep -q "\[RESULT\] vector OK" && echo "TEST vector OK" || { echo "TEST vector FAIL"; exit 1; }
	@out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) collapse 2>&1`; \
	echo "$$out" | grep -q "\[RESULT\] vector OK" && echo "TEST vector collapse OK" || { echo "TEST vector collapse FAIL"; exit 1; }
	@out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM) mv 2>&1`; \
	echo "$$out" | grep -q "\[RESULT\] mat-vec OK" && echo "TEST matvec OK" || { echo "TEST matvec FAIL"; exit 1; }
	@out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM) 2>&1`; \
	echo "$$out" | grep -q "\[RESULT\] mat-mat OK" && echo "TEST matmul OK" || { echo "TEST matmul FAIL"; exit 1; }
	@out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC) 2>&1`; \
	echo "$$out" | grep -q "\[RESULT\] montecarlo OK" && echo "TEST montecarlo OK" || { echo "TEST montecarlo FAIL"; exit 1; }
	@echo "TEST ALL OK"

re: clean all

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)