#include <stdlib.h>
#include <stdio.h>
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
	MPI_Comm active_comm;
	int universe_rank;
	int universe_size;
	int current_rank;
	int current_size;
	int start_iter;
	int is_newly_spawned;
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
	int terminated_processes;
	int success;
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


static MPI_Info make_spawn_info(void) {

	MPI_Info info;

	MPI_Info_create(&info);
	MPI_Info_set(info, "mpi_initial_errhandler", "mpi_errors_return");

	return info;

}

void malleable_init(MalleableState *state) {

	#ifdef TIMING

		double t_init_start = MPI_Wtime();

	#endif

	MPI_Comm parent;
	MPI_Comm_get_parent(&parent);

	if (parent != MPI_COMM_NULL) {

		#ifdef TIMING

			double t_merge_start = MPI_Wtime();

		#endif

		MPI_Group remote_group, local_group, union_group;
		MPI_Comm_remote_group(parent, &remote_group);
		MPI_Comm_group(parent, &local_group);
		MPI_Group_union(remote_group, local_group, &union_group);

		MPI_Comm merged_comm;

		MPI_Comm_create_from_group(union_group, "malleable.active", MPI_INFO_NULL, MPI_ERRORS_RETURN, &merged_comm);

		MPI_Group_free(&remote_group);
		MPI_Group_free(&local_group);
		MPI_Group_free(&union_group);
		MPI_Comm_free(&parent);

		state->active_comm = merged_comm;
		MPI_Comm_rank(state->active_comm, &state->current_rank);
		MPI_Comm_size(state->active_comm, &state->current_size);
		state->universe_rank = state->current_rank;
		state->universe_size = state->current_size;

		#ifdef TIMING

			double t_merge_end = MPI_Wtime();

		#endif

		int spawn_iter = 0;
		MPI_Bcast(&spawn_iter, 1, MPI_INT, ROOT_PROCESS, state->active_comm);

		state->start_iter = spawn_iter;
		state->is_newly_spawned = 1;

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

		int world_size, world_rank;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		MPI_Comm initial_comm;
		MPI_Comm_dup(MPI_COMM_WORLD, &initial_comm);

		#ifdef TIMING

			double t_split_end = MPI_Wtime();

		#endif

		if (initial_comm == MPI_COMM_NULL) {

			MPI_Finalize();

			exit(EXIT_SUCCESS);

		}

		state->active_comm = initial_comm;
		state->start_iter = 0;
		state->is_newly_spawned = 0;

		MPI_Comm_rank(state->active_comm, &state->current_rank);
		MPI_Comm_size(state->active_comm, &state->current_size);
		state->universe_rank = state->current_rank;
		state->universe_size = world_size;

		if (state->current_rank == ROOT_PROCESS) {

			fprintf(stdout, "[INIT] universe_size=%d | initial active_size=%d\n", state->universe_size, state->current_size);

			#ifdef TIMING

				fprintf(stdout, "TIMING,INIT,comm_split,%.8f\n", t_split_end - t_split_start);
				fprintf(stdout, "TIMING,INIT,total,%.8f\n", MPI_Wtime() - t_init_start);

			#endif

		}

	}

}

ResizeResult execute_spawn(MalleableState *state, int target_size, int current_iter, char **argv) {

	ResizeResult result = {0};
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;

	if (target_size <= state->current_size) {

		result.success = 1;

		return result;

	}

	double start_time = MPI_Wtime();
	int num_to_spawn = target_size - state->current_size;

	#ifdef TIMING

		double t_spawn_start = MPI_Wtime();

	#endif

	MPI_Info spawn_info = make_spawn_info();

	MPI_Comm intercomm;

	int spawn_result = MPI_Comm_spawn( argv[0], MPI_ARGV_NULL, num_to_spawn, spawn_info, ROOT_PROCESS, state->active_comm, &intercomm, MPI_ERRCODES_IGNORE);

	MPI_Info_free(&spawn_info);

	if (spawn_result != MPI_SUCCESS) {

		fprintf(stderr, "[ERROR] MPI_Comm_spawn failed\n");

		result.success = 0;

		return result;

	}

	#ifdef TIMING

		double t_spawn_end = MPI_Wtime();
		double t_merge_start = MPI_Wtime();

	#endif

	MPI_Group old_group, new_group, union_group;
	MPI_Comm_group(state->active_comm, &old_group);
	MPI_Comm_remote_group(intercomm, &new_group);
	MPI_Group_union(old_group, new_group, &union_group);

	MPI_Comm new_active;
	MPI_Comm_create_from_group(union_group, "malleable.active", MPI_INFO_NULL, MPI_ERRORS_RETURN, &new_active);

	MPI_Group_free(&old_group);
	MPI_Group_free(&new_group);
	MPI_Group_free(&union_group);

	MPI_Comm_free(&intercomm);

	#ifdef TIMING

		double t_merge_end = MPI_Wtime();

	#endif

	MPI_Comm_free(&state->active_comm);
	state->active_comm = new_active;

	MPI_Comm_rank(state->active_comm, &state->current_rank);
	MPI_Comm_size(state->active_comm, &state->current_size);
	state->universe_rank = state->current_rank;
	state->universe_size = state->current_size;

	MPI_Bcast(&current_iter, 1, MPI_INT, ROOT_PROCESS, state->active_comm);

	result.resize_time = MPI_Wtime() - start_time;
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;
	result.success = 1;

	#ifdef TIMING

		if (state->current_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,RESIZE_GROW,total,%.8f\n", result.resize_time);
			fprintf(stdout, "TIMING,RESIZE_GROW,comm_spawn,%.8f\n", t_spawn_end - t_spawn_start);
			fprintf(stdout, "TIMING,RESIZE_GROW,comm_create,%.8f\n", t_merge_end - t_merge_start);

		}

	#endif

	return result;

}

ResizeResult execute_shrink(MalleableState *state, int target_size) {

	ResizeResult result = {0};
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;

	if (target_size >= state->current_size) {

		result.success = 1;
		return result;

	}

	double start_time = MPI_Wtime();
	int num_to_terminate = state->current_size - target_size;

	int will_survive = (state->current_rank < target_size);

	if (!will_survive) {

		MPI_Comm_free(&state->active_comm);

		MPI_Finalize();

		exit(EXIT_SUCCESS);

	}

	#ifdef TIMING

		double t_split_start = MPI_Wtime();

	#endif

	MPI_Group world_group, keep_group;
	MPI_Comm_group(state->active_comm, &world_group);

	int ranges[1][3] = {{ 0, target_size - 1, 1 }};
	MPI_Group_range_incl(world_group, 1, ranges, &keep_group);

	MPI_Comm new_comm;
	MPI_Comm_create_group(state->active_comm, keep_group, 0, &new_comm);

	MPI_Group_free(&world_group);
	MPI_Group_free(&keep_group);

	#ifdef TIMING

		double t_split_end = MPI_Wtime();

	#endif

	MPI_Comm_free(&state->active_comm);
	state->active_comm = new_comm;

	MPI_Comm_rank(state->active_comm, &state->current_rank);
	MPI_Comm_size(state->active_comm, &state->current_size);
	state->universe_rank = state->current_rank;
	state->universe_size = state->current_size;

	result.resize_time = MPI_Wtime() - start_time;
	result.new_size = state->current_size;
	result.new_rank = state->current_rank;
	result.terminated_processes = num_to_terminate;
	result.success = 1;

	#ifdef TIMING

		if (state->current_rank == ROOT_PROCESS) {

			fprintf(stdout, "TIMING,RESIZE_SHRINK,total,%.8f\n", result.resize_time);
			fprintf(stdout, "TIMING,RESIZE_SHRINK,comm_create,%.8f\n", t_split_end - t_split_start);

		}

	#endif

	return result;

}

int reconfigure(MalleableState *state, int iter, int *seq_index, int num_sequences, char **argv) {

	if (state->is_newly_spawned) {

		state->is_newly_spawned = 0;

		return 1;

	}

	#ifdef TIMING

		double t_reconf_start = MPI_Wtime();

	#endif

	ControlMsg ctrl = {CTRL_CONTINUE, 0, iter};

	if (state->current_rank == ROOT_PROCESS && *seq_index < num_sequences) {

		TestSequence seq = test_sequence[*seq_index];

		if (iter == seq.iteration) {

			switch (seq.method) {

				case METHOD_GROW:

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

	#ifdef TIMING

		double t_bcast_start = MPI_Wtime();

	#endif

	MPI_Bcast(&ctrl, sizeof(ControlMsg), MPI_BYTE, ROOT_PROCESS, state->active_comm);

	#ifdef TIMING

		double t_bcast_end = MPI_Wtime();

	#endif

	switch (ctrl.signal) {

		case CTRL_STOP:

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | STOP\n", ctrl.iteration);

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 0;

		case CTRL_GROW: {

			if (state->current_rank == ROOT_PROCESS) {

				fprintf(stdout, "[RECONFIGURE] iter=%d | GROW | %d -> %d\n", ctrl.iteration, state->current_size, ctrl.target_size);

			}

			ResizeResult res = execute_spawn(state, ctrl.target_size, iter, argv);

			if (state->current_rank == ROOT_PROCESS && res.success) {

				fprintf(stdout, "[RECONFIGURE] Done. new_size=%d | time=%.8f s\n", res.new_size, res.resize_time);

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
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

				fprintf(stdout, "[RECONFIGURE] Done. new_size=%d | terminated=%d | time=%.8f s\n", res.new_size, res.terminated_processes, res.resize_time);

			}

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
					fprintf(stdout, "TIMING,RECONFIGURE,total,%.8f\n", MPI_Wtime() - t_reconf_start);

				}

			#endif

			return 1;

		}

		case CTRL_CONTINUE:
		default:

			#ifdef TIMING

				if (state->current_rank == ROOT_PROCESS) {

					fprintf(stdout, "TIMING,RECONFIGURE,bcast,%.8f\n", t_bcast_end - t_bcast_start);
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

	for (int iter = state.start_iter; iter < MAX_ITERATIONS && running; iter++) {

		running = reconfigure(&state, iter, &seq_index, num_sequences, argv);

		if (running) {

			perform_computation(&state, iter);

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