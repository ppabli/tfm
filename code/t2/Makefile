CXX = mpicxx
AR = ar
MPIRUN = mpirun

NP = 4
BENCH_CSV ?= 0

USE_PAPI ?= 0
PAPI_PREFIX ?= $(HOME)/Desktop/papi/src/install

CXXFLAGS = -O2 -march=native -Wall -std=c++20 -MMD -MP -Imalleable/include -DBENCH_CSV=$(BENCH_CSV)
LIBS = -lpthread

ifeq ($(USE_PAPI), 1)

	CXXFLAGS += -DMAL_USE_PAPI -I$(PAPI_PREFIX)/include
	LIBS += -L$(PAPI_PREFIX)/lib -lpapi

endif

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

TARGET_SP = $(BUILD_DIR)/sparse
SRCS_SP = examples/sparse.cpp
OBJS_SP = $(SRCS_SP:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MC_NORMAL = $(BUILD_DIR)/montecarlo_normal
SRCS_MC_NORMAL = examples/montecarlo_normal.cpp
OBJS_MC_NORMAL = $(SRCS_MC_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_VEC_NORMAL = $(BUILD_DIR)/vector_normal
SRCS_VEC_NORMAL = examples/vector_normal.cpp
OBJS_VEC_NORMAL = $(SRCS_VEC_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MM_NORMAL = $(BUILD_DIR)/matmul_normal
SRCS_MM_NORMAL = examples/matmul_normal.cpp
OBJS_MM_NORMAL = $(SRCS_MM_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SP_NORMAL = $(BUILD_DIR)/sparse_normal
SRCS_SP_NORMAL = examples/sparse_normal.cpp
OBJS_SP_NORMAL = $(SRCS_SP_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

DEPS = $(LIB_OBJS:.o=.d) $(OBJS_MC:.o=.d) $(OBJS_VEC:.o=.d) $(OBJS_MM:.o=.d) $(OBJS_SP:.o=.d) \
	$(OBJS_MC_NORMAL:.o=.d) $(OBJS_VEC_NORMAL:.o=.d) $(OBJS_MM_NORMAL:.o=.d) $(OBJS_SP_NORMAL:.o=.d)

.PHONY: all run_montecarlo run_vector run_vector_collapse run_matmul run_matvec run_sparse \
	run_montecarlo_normal run_vector_normal run_vector_normal_collapse run_matmul_normal run_matvec_normal run_sparse_normal \
	test clean re

all: $(LIB_MAL) $(TARGET_MC) $(TARGET_VEC) $(TARGET_MM) $(TARGET_SP) \
	$(TARGET_MC_NORMAL) $(TARGET_VEC_NORMAL) $(TARGET_MM_NORMAL) $(TARGET_SP_NORMAL)

$(LIB_MAL): Makefile $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $(LIB_OBJS)

$(TARGET_MC): $(LIB_MAL) $(OBJS_MC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MC) $(LIB_MAL) $(LIBS)

$(TARGET_VEC): $(LIB_MAL) $(OBJS_VEC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_VEC) $(LIB_MAL) $(LIBS)

$(TARGET_MM): $(LIB_MAL) $(OBJS_MM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MM) $(LIB_MAL) $(LIBS)

$(TARGET_SP): $(LIB_MAL) $(OBJS_SP)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SP) $(LIB_MAL) $(LIBS)

$(TARGET_MC_NORMAL): $(OBJS_MC_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MC_NORMAL) $(LIBS)

$(TARGET_VEC_NORMAL): $(OBJS_VEC_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_VEC_NORMAL) $(LIBS)

$(TARGET_MM_NORMAL): $(OBJS_MM_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MM_NORMAL) $(LIBS)

$(TARGET_SP_NORMAL): $(OBJS_SP_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SP_NORMAL) $(LIBS)

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

run_sparse: $(TARGET_SP)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SP)

run_montecarlo_normal: $(TARGET_MC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC_NORMAL)

run_vector_normal: $(TARGET_VEC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC_NORMAL)

run_vector_normal_collapse: $(TARGET_VEC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC_NORMAL) collapse

run_matmul_normal: $(TARGET_MM_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM_NORMAL) mm

run_matvec_normal: $(TARGET_MM_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM_NORMAL) mv

run_sparse_normal: $(TARGET_SP_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SP_NORMAL)

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
