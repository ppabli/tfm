#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>

#define MAX_ITERATIONS 100
#define ROOT_PROCESS 0

#define CTRL_CONTINUE 0
#define CTRL_GROW 1
#define CTRL_SHRINK 2
#define CTRL_STOP 3

typedef struct {
	int signal;
	int target_size;
	int iteration;
} ControlMsg;

typedef struct {
	MPI_Session session;
	MPI_Group world_group;
	MPI_Comm universe_comm;
	MPI_Comm active_comm;
	int universe_rank;
	int universe_size;
	int current_rank;
	int current_size;
} MalleableState;

typedef enum {
	METHOD_GROW = 0,
	METHOD_SHRINK = 1,
	METHOD_STOP = 2,
} ResizeMethod;

typedef struct {
	int iteration;
	int target_size;
	ResizeMethod method;
} TestSequence;

typedef struct {
	double resize_time;
	int success;
	int new_size;
	int new_rank;
} ResizeResult;

static TestSequence test_sequence[] = {
	{2, 2, METHOD_GROW},
	{3, 4, METHOD_GROW},
	{5, 4, METHOD_GROW},
	{7, 3, METHOD_SHRINK},
	{8, 1, METHOD_SHRINK},
	{10, 4, METHOD_GROW},
	{12, 0, METHOD_STOP},
};

static void malleable_init(MalleableState *state) {

	state->active_comm = MPI_COMM_NULL;
	state->universe_comm = MPI_COMM_NULL;
	state->world_group = MPI_GROUP_NULL;
	state->current_rank = -1;
	state->current_size = 0;

	MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &state->session);

	#ifdef TIMING

		double t_init_start = MPI_Wtime();

	#endif

	MPI_Group_from_session_pset(state->session, "mpi://WORLD", &state->world_group);

	MPI_Comm_create_from_group(state->world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &state->universe_comm);

	MPI_Comm_rank(state->universe_comm, &state->universe_rank);
	MPI_Comm_size(state->universe_comm, &state->universe_size);

	const int init_size = 1;
	int color = (state->universe_rank < init_size) ? 0 : MPI_UNDEFINED;

	#ifdef TIMING

		double t_comm_start = MPI_Wtime();

	#endif

	MPI_Comm_split(state->universe_comm, color, state->universe_rank, &state->active_comm);

	#ifdef TIMING

		double t_comm_end = MPI_Wtime();

	#endif

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(state->active_comm, &state->current_rank);
		MPI_Comm_size(state->active_comm, &state->current_size);

	}

	#ifdef TIMING

		double t_init_end = MPI_Wtime();

		if (state->universe_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,INIT,total,%.8f\n", t_init_end - t_init_start);
			fprintf(stdout, "TIMING,INIT,comm_create,%.8f\n", t_comm_end - t_comm_start);

		}

	#endif

	if (state->universe_rank == ROOT_PROCESS) {

		fprintf(stdout, "[INIT] universe_size=%d | initial active_size=%d\n", state->universe_size, init_size);

	}

}

static ResizeResult execute_resize(MalleableState *state, int target_size) {

	ResizeResult result = {0};
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;

	if (target_size == state->current_size) {

		result.success = 1;

		return result;

	}

	const int old_size = state->current_size;
	double start_time = MPI_Wtime();

	#ifdef TIMING

		double t_free_start = MPI_Wtime();

	#endif

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&state->active_comm);
		state->active_comm = MPI_COMM_NULL;

	}

	#ifdef TIMING

		double t_free_end = MPI_Wtime();
		double t_create_start = MPI_Wtime();

	#endif

	int color = (state->universe_rank < target_size) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(state->universe_comm, color, state->universe_rank, &state->active_comm);

	#ifdef TIMING

		double t_create_end = MPI_Wtime();

	#endif

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(state->active_comm, &state->current_rank);
		MPI_Comm_size(state->active_comm, &state->current_size);

	} else {

		state->current_rank = -1;
		state->current_size = 0;

	}

	result.resize_time = MPI_Wtime() - start_time;
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;
	result.success = 1;

	#ifdef TIMING

		if (state->universe_rank == ROOT_PROCESS) {

			const char *method = (target_size > old_size) ? "GROW" : "SHRINK";
			fprintf(stdout, "TIMING,RESIZE_%s,total,%.8f\n", method, result.resize_time);
			fprintf(stdout, "TIMING,RESIZE_%s,comm_free,%.8f\n", method, t_free_end - t_free_start);
			fprintf(stdout, "TIMING,RESIZE_%s,comm_create,%.8f\n", method, t_create_end - t_create_start);

		}

	#else

		(void)old_size;

	#endif

	return result;

}

static int reconfigure(MalleableState *state, int iter, int *seq_index, int num_sequences) {

	#ifdef TIMING

		double t_reconf_start = MPI_Wtime();
		double t_bcast_start = 0.0, t_bcast_end = 0.0;

	#endif

	ControlMsg ctrl = {CTRL_CONTINUE, 0, iter};

	if (state->universe_rank == ROOT_PROCESS && *seq_index < num_sequences) {

		TestSequence seq = test_sequence[*seq_index];

		if (iter == seq.iteration) {

			switch (seq.method) {

				case METHOD_GROW:

					if (seq.target_size > state->universe_size) {

						fprintf(stderr, "[ROOT] Warning: grow target %d > universe %d, clamping\n", seq.target_size, state->universe_size);
						seq.target_size = state->universe_size;

					}

					if (seq.target_size > state->current_size) {

						ctrl.signal = CTRL_GROW;
						ctrl.target_size = seq.target_size;

					}

					(*seq_index)++;

					break;

				case METHOD_SHRINK:

					if (seq.target_size < 0) {

						fprintf(stderr, "[ROOT] Warning: shrink target < 0, clamping to 0\n");
						seq.target_size = 0;

					}

					if (seq.target_size < state->current_size) {

						ctrl.signal = CTRL_SHRINK;
						ctrl.target_size = seq.target_size;

					}

					(*seq_index)++;

					break;

				case METHOD_STOP:

					ctrl.signal = CTRL_STOP;

					(*seq_index)++;

					break;

				default:

					fprintf(stderr, "[ROOT] Unknown method, stopping.\n");

					ctrl.signal = CTRL_STOP;

					(*seq_index)++;

					break;

			}

		}

	}

	int signal = ctrl.signal;

	#ifdef TIMING

		t_bcast_start = MPI_Wtime();

	#endif

	MPI_Bcast(&signal, 1, MPI_INT, ROOT_PROCESS, state->universe_comm);

	#ifdef TIMING

		t_bcast_end = MPI_Wtime();

	#endif

	if (signal == CTRL_CONTINUE) {

		#ifdef TIMING

			if (state->universe_rank == ROOT_PROCESS) {

				fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
				fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

			}

		#endif

		return 1;

	}

	if (signal == CTRL_GROW || signal == CTRL_SHRINK) {

		int target = ctrl.target_size;
		MPI_Bcast(&target, 1, MPI_INT, ROOT_PROCESS, state->universe_comm);

		#ifdef TIMING

			t_bcast_end = MPI_Wtime();

		#endif

		ctrl.target_size = target;
		ctrl.signal = signal;

	}

	switch (signal) {

		case CTRL_STOP:

			if (state->universe_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | STOP\n", iter);

			}

			#ifdef TIMING

				if (state->universe_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 0;

		case CTRL_GROW:
		case CTRL_SHRINK: {

			if (state->universe_rank == ROOT_PROCESS) {

				const char *tag = (signal == CTRL_GROW) ? "GROW" : "SHRINK";
				fprintf(stdout, "[RECONFIGURE] iter=%d | %s | %d -> %d\n", iter, tag, state->current_size, ctrl.target_size);

			}

			ResizeResult res = execute_resize(state, ctrl.target_size);

			if (state->universe_rank == ROOT_PROCESS && res.success) {

				fprintf(stdout, "[RECONFIGURE] Done. new_size=%d | time=%.8f s\n", res.new_size, res.resize_time);

			}

			#ifdef TIMING

				if (state->universe_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 1;

		}

		default:

			return 1;

	}

}

static void perform_computation(MalleableState *state, int iter) {

	if (state->active_comm == MPI_COMM_NULL) {

		return;

	}

	#ifdef TIMING

		double t_compute_start = MPI_Wtime();

	#endif

	int local_val = state->current_rank + 1;
	int global_sum = 0;

	MPI_Reduce(&local_val, &global_sum, 1, MPI_INT, MPI_SUM, ROOT_PROCESS, state->active_comm);

	if (state->current_rank == ROOT_PROCESS) {

		assert(global_sum == (state->current_size * (state->current_size + 1)) / 2);

		fprintf(stdout, "[COMPUTE] iter=%d | active=%d | rank_sum=%d\n", iter, state->current_size, global_sum);

		#ifdef TIMING

			fprintf(stdout, "TIMING,COMPUTE,total,%.8f\n", MPI_Wtime() - t_compute_start);

		#endif

	}

}

static void malleable_cleanup(MalleableState *state) {

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&state->active_comm);
		state->active_comm = MPI_COMM_NULL;

	}

	MPI_Comm_free(&state->universe_comm);
	MPI_Group_free(&state->world_group);
	MPI_Session_finalize(&state->session);

}

int main(int argc, char *argv[]) {

	MalleableState state;
	memset(&state, 0, sizeof(state));

	malleable_init(&state);

	#ifdef TIMING

		double t_main_start = MPI_Wtime();

	#endif

	const int num_sequences = (int)(sizeof(test_sequence) / sizeof(test_sequence[0]));
	int seq_index = 0;
	int running = 1;

	for (int iter = 0; iter < MAX_ITERATIONS && running; iter++) {

		running = reconfigure(&state, iter, &seq_index, num_sequences);

		if (running) {

			perform_computation(&state, iter);

		}

	}

	#ifdef TIMING

		if (state.universe_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,MAIN,total,%.8f\n", MPI_Wtime() - t_main_start);

		}

	#endif

	malleable_cleanup(&state);

	return EXIT_SUCCESS;

}
