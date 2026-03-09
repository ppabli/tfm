#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_ITERATIONS 100
#define ROOT_PROCESS 0

typedef enum {
	CTRL_CONTINUE = 0,
	CTRL_GROW = 1,
	CTRL_SHRINK = 2,
	CTRL_STOP = 3
} ControlSignal;

typedef enum {
	PARK_WAKE = 10,
	PARK_EXIT = 11
} ParkSignal;

typedef struct {
	int signal;
	int target_size;
	int iteration;
} ControlMsg;

typedef struct {
	MPI_Comm universe_comm;
	MPI_Comm active_comm;
	MPI_Comm parked_comm;
	int universe_rank;
	int universe_size;
	int current_rank;
	int current_size;
	int pool_size;
	int start_iter;
	bool is_newly_spawned;
	bool is_parked;
	bool need_rank_update;
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
	bool success;
	int new_size;
	int new_rank;
} ResizeResult;

TestSequence test_sequence[] = {
	{2, 2, METHOD_GROW},
	{3, 4, METHOD_GROW},
	{5, 4, METHOD_GROW},
	{7, 3, METHOD_SHRINK},
	{8, 1, METHOD_SHRINK},
	{10, 4, METHOD_GROW},
	{12, 0, METHOD_STOP},
};

void update_comm_info(MalleableState *state) {

	if (!state->need_rank_update) {

		return;

	}

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(state->active_comm, &state->current_rank);
		MPI_Comm_size(state->active_comm, &state->current_size);

	}

	if (state->universe_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(state->universe_comm, &state->universe_rank);
		MPI_Comm_size(state->universe_comm, &state->universe_size);

	}

	state->need_rank_update = false;

}

bool parking_loop(MalleableState *state) {

	if (state->parked_comm == MPI_COMM_NULL) {

		return false;

	}

	int cmd = PARK_EXIT;
	MPI_Bcast(&cmd, 1, MPI_INT, ROOT_PROCESS, state->parked_comm);

	return (cmd == PARK_WAKE);

}

void malleable_init(MalleableState *state) {

	#ifdef TIMING

		double t_init_start = MPI_Wtime();

	#endif

	memset(state, 0, sizeof(MalleableState));
	state->parked_comm = MPI_COMM_NULL;
	state->is_parked = false;
	state->need_rank_update = true;

	MPI_Comm parent;
	MPI_Comm_get_parent(&parent);

	if (parent != MPI_COMM_NULL) {

		#ifdef TIMING

			double t_merge_start = MPI_Wtime();

		#endif

		MPI_Comm merged;
		MPI_Intercomm_merge(parent, 1, &merged);

		MPI_Comm_dup(merged, &state->active_comm);
		MPI_Comm_dup(merged, &state->universe_comm);

		MPI_Comm_free(&merged);
		MPI_Comm_free(&parent);

		#ifdef TIMING

			double t_merge_end = MPI_Wtime();

		#endif


		state->need_rank_update = true;
		update_comm_info(state);

		state->pool_size = state->current_size;
		state->is_newly_spawned = true;

		int spawn_iter = 0;
		MPI_Bcast(&spawn_iter, 1, MPI_INT, ROOT_PROCESS, state->active_comm);
		state->start_iter = spawn_iter;

		#ifdef TIMING

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "TIMING,INIT,spawned_merge,%.8f\n", t_merge_end - t_merge_start);
				fprintf(stdout, "TIMING,INIT,spawned_total,%.8f\n", MPI_Wtime() - t_init_start);

			}

		#endif

	} else {

		#ifdef TIMING

			double t_split_start = MPI_Wtime();

		#endif

		MPI_Comm world_comm;
		MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);

		int world_rank;
		MPI_Comm_rank(world_comm, &world_rank);

		MPI_Comm initial_comm;
		MPI_Comm_split(world_comm, 0, world_rank, &initial_comm);

		#ifdef TIMING

			double t_split_end = MPI_Wtime();

		#endif

		MPI_Comm initial_universe;
		MPI_Comm_dup(world_comm, &initial_universe);
		MPI_Comm_free(&world_comm);

		if (initial_comm == MPI_COMM_NULL) {

			state->active_comm = MPI_COMM_NULL;
			state->universe_comm = initial_universe;
			state->start_iter = 0;
			state->is_newly_spawned = false;
			state->is_parked = true;
			state->current_rank = -1;
			state->current_size = 0;
			state->pool_size = 0;

			return;

		}

		state->active_comm = initial_comm;
		state->universe_comm = initial_universe;
		state->start_iter = 0;
		state->is_newly_spawned = false;

		state->need_rank_update = true;
		update_comm_info(state);
		state->pool_size = state->current_size;

		if (state->current_rank == ROOT_PROCESS) {

			fprintf(stdout, "[INIT] universe_size=%d | initial active_size=%d\n", state->universe_size, state->current_size);

			#ifdef TIMING

				fprintf(stdout, "TIMING,INIT,comm_split,%.8f\n", t_split_end - t_split_start);
				fprintf(stdout, "TIMING,INIT,total,%.8f\n", MPI_Wtime() - t_init_start);

			#endif

		}

	}

}

ResizeResult execute_grow(MalleableState *state, int target_size, int current_iter, char **argv) {

	ResizeResult result = {0};

	result.new_size = state->current_size;
	result.new_rank = state->current_rank;

	if (target_size <= state->current_size) {

		result.success = true;

		return result;

	}

	double start_time = MPI_Wtime();

	int parked_count = state->pool_size - state->current_size;
	int need_spawn = target_size - state->pool_size;

	if (need_spawn < 0) {

		need_spawn = 0;

	}

	if (parked_count > 0 && state->parked_comm != MPI_COMM_NULL) {

		int wake_signal = PARK_WAKE;

		MPI_Bcast(&wake_signal, 1, MPI_INT, ROOT_PROCESS, state->parked_comm);

		int color = (state->universe_rank < target_size && need_spawn == 0) ? 0 : (state->universe_rank < state->pool_size) ? 0 : MPI_UNDEFINED;

		MPI_Comm new_active;
		MPI_Comm_split(state->universe_comm, color, state->universe_rank, &new_active);

		MPI_Comm_free(&state->active_comm);

		if (state->parked_comm != MPI_COMM_NULL) {

			MPI_Comm_free(&state->parked_comm);
			state->parked_comm = MPI_COMM_NULL;

		}

		state->active_comm = new_active;
		state->need_rank_update = true;
		update_comm_info(state);

	}

	if (need_spawn > 0) {

		#ifdef TIMING

			double t_spawn_start = MPI_Wtime();

		#endif

		MPI_Comm intercomm;
		MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, need_spawn, MPI_INFO_NULL, ROOT_PROCESS, state->active_comm, &intercomm, MPI_ERRCODES_IGNORE);

		#ifdef TIMING

			double t_spawn_end = MPI_Wtime();
			double t_merge_start = MPI_Wtime();

		#endif

		MPI_Comm merged;
		MPI_Comm new_active;
		MPI_Comm new_universe;

		MPI_Intercomm_merge(intercomm, 0, &merged);

		MPI_Comm_dup(merged, &new_active);
		MPI_Comm_dup(merged, &new_universe);

		MPI_Comm_free(&merged);
		MPI_Comm_free(&intercomm);

		#ifdef TIMING

			double t_merge_end = MPI_Wtime();

		#endif

		MPI_Comm_free(&state->active_comm);
		MPI_Comm_free(&state->universe_comm);

		state->active_comm = new_active;
		state->universe_comm = new_universe;

		state->need_rank_update = true;
		update_comm_info(state);

		MPI_Bcast(&current_iter, 1, MPI_INT, ROOT_PROCESS, state->active_comm);

		state->pool_size = state->current_size;

		#ifdef TIMING

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "TIMING,RESIZE_GROW,comm_spawn,%.8f\n", t_spawn_end - t_spawn_start);
				fprintf(stdout, "TIMING,RESIZE_GROW,intercomm_merge,%.8f\n", t_merge_end - t_merge_start);

			}

		#endif

	} else {

		state->pool_size = state->current_size;

	}

	result.resize_time = MPI_Wtime() - start_time;
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;
	result.success = true;

	#ifdef TIMING

		if (state->current_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,RESIZE_GROW,total,%.8f\n", result.resize_time);

		}

	#endif

	return result;

}

ResizeResult execute_shrink(MalleableState *state, int target_size) {

	ResizeResult result = {0};
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;

	if (target_size >= state->current_size) {

		result.success = true;
		return result;

	}

	int old_size = state->current_size;
	double start_time = MPI_Wtime();

	int color = (state->current_rank < target_size) ? 0 : 1;

	#ifdef TIMING

		double t_split_start = MPI_Wtime();

	#endif

	MPI_Comm new_active;
	MPI_Comm new_parked;

	int park_color = (color == 1) ? 0 : MPI_UNDEFINED;

	MPI_Comm_split(state->active_comm, color, state->current_rank, &new_active);
	MPI_Comm_split(state->active_comm, park_color, state->current_rank, &new_parked);

	#ifdef TIMING

		double t_split_end = MPI_Wtime();

	#endif

	MPI_Comm old_active = state->active_comm;

	if (color == 0) {

		state->active_comm = new_active;

		if (new_parked != MPI_COMM_NULL) {

			MPI_Comm_free(&new_parked);

		}

		if (state->parked_comm != MPI_COMM_NULL) {

			MPI_Comm_free(&state->parked_comm);

		}

		state->parked_comm = new_parked;
		MPI_Comm_free(&old_active);

		state->need_rank_update = true;
		update_comm_info(state);

		state->pool_size = old_size;
		result.resize_time = MPI_Wtime() - start_time;
		result.new_size = state->current_size;
		result.new_rank = state->current_rank;
		result.success = true;

		#ifdef TIMING

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "TIMING,RESIZE_SHRINK,comm_split,%.8f\n", t_split_end - t_split_start);
				fprintf(stdout, "TIMING,RESIZE_SHRINK,total,%.8f\n", result.resize_time);

			}

		#endif

	} else {

		if (new_active != MPI_COMM_NULL) {

			MPI_Comm_free(&new_active);

		}

		MPI_Comm_free(&old_active);

		if (state->parked_comm != MPI_COMM_NULL) {

			MPI_Comm_free(&state->parked_comm);

		}

		state->active_comm = MPI_COMM_NULL;
		state->parked_comm = new_parked;
		state->current_rank = -1;
		state->current_size = 0;
		state->is_parked = true;

		if (state->universe_comm != MPI_COMM_NULL) {

			MPI_Comm_rank(state->universe_comm, &state->universe_rank);
			MPI_Comm_size(state->universe_comm, &state->universe_size);

		}

		fprintf(stdout, "[SHRINK] Process %d parked | old_size=%d | target_size=%d\n", state->universe_rank, old_size, target_size);

		result.success = true;
		result.new_size = 0;
		result.new_rank = -1;

		result.success = true;
		result.new_size = state->current_size;
		result.new_rank = state->current_rank;

	}

	return result;

}

int reconfigure(MalleableState *state, int iter, int *seq_index, int num_sequences, char **argv) {

	if (state->is_newly_spawned) {

		state->is_newly_spawned = false;

		return 1;

	}

	if (state->is_parked && state->parked_comm != MPI_COMM_NULL) {

		bool woken = parking_loop(state);

		if (woken) {

			state->need_rank_update = true;
			update_comm_info(state);
			state->is_parked = false;

			fprintf(stdout, "[RECONFIGURE] Process %d woken from parking | new_rank=%d | new_size=%d\n", state->universe_rank, state->current_rank, state->current_size);

			return 1;

		} else {

			fprintf(stdout, "[RECONFIGURE] Process %d exiting from parking (received PARK_EXIT)\n", state->universe_rank);

			return 0;

		}

	}

	#ifdef TIMING

		double t_reconf_start = MPI_Wtime();

	#endif

	int needs_action = 0;

	ControlMsg ctrl = {CTRL_CONTINUE, 0, iter};

	if (state->current_rank == ROOT_PROCESS && *seq_index < num_sequences) {

		TestSequence seq = test_sequence[*seq_index];

		if (iter == seq.iteration) {

			needs_action = 1;
			ctrl.iteration = iter;

			switch (seq.method) {

				case METHOD_GROW:

					if (seq.target_size > state->current_size) {

						ctrl.signal = CTRL_GROW;
						ctrl.target_size = seq.target_size;

					} else {

						needs_action = 0;

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

					} else {

						needs_action = 0;

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

	#ifdef TIMING

		double t_bcast_flag_start = MPI_Wtime();

	#endif

	MPI_Bcast(&needs_action, 1, MPI_INT, ROOT_PROCESS, state->active_comm);

	#ifdef TIMING

		double t_bcast_flag_end = MPI_Wtime();

	#endif

	if (!needs_action) {

		#ifdef TIMING

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "TIMING,RECONFIGURE,bcast_flag,%.8f\n", t_bcast_flag_end - t_bcast_flag_start);
				fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

			}

		#endif

		return 1;

	}

	#ifdef TIMING

		double t_bcast_ctrl_start = MPI_Wtime();

	#endif

	MPI_Bcast(&ctrl, sizeof(ControlMsg), MPI_BYTE, ROOT_PROCESS, state->active_comm);

	#ifdef TIMING

		double t_bcast_ctrl_end = MPI_Wtime();

	#endif

	switch (ctrl.signal) {

		case CTRL_STOP:

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | STOP\n", ctrl.iteration);

				if (state->parked_comm != MPI_COMM_NULL) {

					int exit_signal = PARK_EXIT;
					MPI_Bcast(&exit_signal, 1, MPI_INT, ROOT_PROCESS, state->parked_comm);

				}

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast_flag,%.8f\n", t_bcast_flag_end - t_bcast_flag_start);
					fprintf(stdout, "TIMING,RECONFIGURE,bcast_ctrl,%.8f\n", t_bcast_ctrl_end - t_bcast_ctrl_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 0;

		case CTRL_GROW: {

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | GROW | %d -> %d (pool=%d, parked=%d)\n", ctrl.iteration, state->current_size, ctrl.target_size, state->pool_size, state->pool_size - state->current_size);

			}

			ResizeResult res = execute_grow(state, ctrl.target_size, iter, argv);

			if (state->current_rank == ROOT_PROCESS && res.success) {

				fprintf(stdout, "[RECONFIGURE] Done. new_size=%d | time=%.8f s\n", res.new_size, res.resize_time);

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast_flag,%.8f\n", t_bcast_flag_end - t_bcast_flag_start);
					fprintf(stdout, "TIMING,RECONFIGURE,bcast_ctrl,%.8f\n", t_bcast_ctrl_end - t_bcast_ctrl_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 1;
		}

		case CTRL_SHRINK: {

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | SHRINK | %d -> %d\n", ctrl.iteration, state->current_size, ctrl.target_size);

			}

			ResizeResult res = execute_shrink(state, ctrl.target_size);

			if (state->current_rank == ROOT_PROCESS && res.success) {

				fprintf(stdout, "[RECONFIGURE] Done. new_size=%d | pool=%d | time=%.8f s\n", res.new_size, state->pool_size, res.resize_time);

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast_flag,%.8f\n", t_bcast_flag_end - t_bcast_flag_start);
					fprintf(stdout, "TIMING,RECONFIGURE,bcast_ctrl,%.8f\n", t_bcast_ctrl_end - t_bcast_ctrl_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 1;

		}

		case CTRL_CONTINUE:
		default:

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast_flag,%.8f\n", t_bcast_flag_end - t_bcast_flag_start);
					fprintf(stdout, "TIMING,RECONFIGURE,bcast_ctrl,%.8f\n", t_bcast_ctrl_end - t_bcast_ctrl_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 1;

	}

}

void perform_computation(MalleableState *state, int iter) {

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

void malleable_cleanup(MalleableState *state) {

	if (state->active_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&state->active_comm);

	}

	if (state->parked_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&state->parked_comm);

	}

	if (state->universe_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&state->universe_comm);

	}

}

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	MalleableState state;

	#ifdef TIMING

		double t_main_start = MPI_Wtime();

	#endif

	malleable_init(&state);

	int num_sequences = (int)(sizeof(test_sequence) / sizeof(test_sequence[0]));
	int seq_index = 0;
	int running = 1;

	if (state.is_parked && state.parked_comm == MPI_COMM_NULL && state.active_comm == MPI_COMM_NULL) {

		fprintf(stdout, "[MAIN] Process initially parked, waiting for program end\n");

	} else {

		for (int iter = state.start_iter; iter < MAX_ITERATIONS && running; iter++) {

			running = reconfigure(&state, iter, &seq_index, num_sequences, argv);

			if (running) {

				perform_computation(&state, iter);

			}

		}

	}

	#ifdef TIMING

		if (state.current_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,MAIN,total,%.8f\n", MPI_Wtime() - t_main_start);

		}

	#endif

	malleable_cleanup(&state);

	MPI_Finalize();

	return EXIT_SUCCESS;

}