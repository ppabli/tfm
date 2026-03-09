CC = mpicc
CFLAGS = -Wall -O2 -D TIMING

TARGETS = sessions spawn hybrid
RESULTS_DIR = ../data/results

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	CORES = $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
else
	CORES = $(shell sysctl -n hw.ncpu 2>/dev/null || echo 1)
endif

MAX_PROC = $(CORES)
RUNS = 10

.PHONY: all compile test parse clean config test-sessions test-spawn test-hybrid parse-sessions parse-spawn parse-hybrid

all: compile test parse clean

compile: $(TARGETS)

sessions: sessions.c
	$(CC) $(CFLAGS) -o $@ $<

spawn: spawn.c
	$(CC) $(CFLAGS) -o $@ $<

hybrid: hybrid.c
	$(CC) $(CFLAGS) -o $@ $<

test: compile
	@mkdir -p $(RESULTS_DIR)
	@for i in $(shell seq 1 $(RUNS)); do \
		echo "===== Iteration $$i ====="; \
		echo "Running sessions test with $(MAX_PROC) processes"; \
		mpirun -np $(MAX_PROC) ./sessions > $(RESULTS_DIR)/sessions_$${i}_$(MAX_PROC)p.out 2>&1; \
		echo "Running spawn test"; \
		mpirun -np 1 ./spawn > $(RESULTS_DIR)/spawn_$${i}_1p.out 2>&1; \
		echo "Running hybrid test"; \
		mpirun -np 1 ./hybrid > $(RESULTS_DIR)/hybrid_$${i}_1p.out 2>&1; \
	done

parse: parse-sessions parse-spawn parse-hybrid

parse-sessions:
	@echo "Parsing sessions results"
	@grep -h -E "TESTING|TIMING" $(RESULTS_DIR)/sessions_*.out > $(RESULTS_DIR)/sessions_parsed.csv 2>/dev/null || true

parse-spawn:
	@echo "Parsing spawn results"
	@grep -h -E "TESTING|TIMING" $(RESULTS_DIR)/spawn_*.out > $(RESULTS_DIR)/spawn_parsed.csv 2>/dev/null || true

parse-hybrid:
	@echo "Parsing hybrid results"
	@grep -h -E "TESTING|TIMING" $(RESULTS_DIR)/hybrid_*.out > $(RESULTS_DIR)/hybrid_parsed.csv 2>/dev/null || true

config:
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "TARGETS: $(TARGETS)"
	@echo "CORES: $(CORES)"
	@echo "MAX_PROC: $(MAX_PROC)"
	@echo "RUNS: $(RUNS)"

clean:
	rm -f $(TARGETS) *.o

clean-all:
	rm -f $(TARGETS) *.o
	rm -rf $(RESULTS_DIR)
