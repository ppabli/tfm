#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include "example_utils.hpp"

#define MAL_N 20
#define COLLAPSE_ROWS 4
#define COLLAPSE_COLS 5

struct BlockRange {
	long start;
	long count;
};

BlockRange block_range(long total, int rank, int size) {

	const long base = total / size;
	const long rem = total % size;
	const long count = base + (rank < rem ? 1 : 0);
	const long start = rank * base + std::min<long>(rank, rem);

	return {start, count};

}

void build_gatherv_layout(long total, int world_size, std::vector<int>& counts, std::vector<int>& displs) {

	counts.resize(world_size);
	displs.resize(world_size);

	for (int r = 0; r < world_size; r++) {

		const BlockRange rr = block_range(total, r, world_size);
		counts[r] = static_cast<int>(rr.count);
		displs[r] = static_cast<int>(rr.start);

	}

}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	double* A = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));
	double* B = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));
	double* C = (world_rank == 0) ? static_cast<double*>(std::malloc(MAL_N * sizeof(double))) : nullptr;

	if (world_rank == 0) {

		for (int k = 0; k < MAL_N; k++) {

			A[k] = static_cast<double>(k + 1);
			B[k] = static_cast<double>(MAL_N - k);

		}

	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Bcast(A, MAL_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, MAL_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	const bool use_collapse = (argc > 1 && std::strcmp(argv[1], "collapse") == 0);
	const BlockRange range = block_range(MAL_N, world_rank, world_size);
	std::vector<double> local_c(static_cast<size_t>(range.count));
	const useconds_t delay_us = example_delay_us(200000);

	for (long local_i = 0; local_i < range.count; local_i++) {

		const long flat = range.start + local_i;
		long idx = flat;

		if (use_collapse && COLLAPSE_ROWS * COLLAPSE_COLS == MAL_N) {

			const long row = flat / COLLAPSE_COLS;
			const long col = flat % COLLAPSE_COLS;
			idx = row * COLLAPSE_COLS + col;

		}

		local_c[static_cast<size_t>(local_i)] = A[idx] + B[idx];
		usleep(delay_us);

	}

	std::vector<int> recv_counts;
	std::vector<int> recv_displs;
	if (world_rank == 0) {

		build_gatherv_layout(MAL_N, world_size, recv_counts, recv_displs);

	}

	MPI_Gatherv(local_c.data(), static_cast<int>(range.count), MPI_DOUBLE, C, (world_rank == 0) ? recv_counts.data() : nullptr, (world_rank == 0) ? recv_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		int errors = 0;
		for (int k = 0; k < MAL_N; k++) {

			if (std::fabs(C[k] - static_cast<double>(MAL_N + 1)) > 1e-9) {

				errors++;
				break;

			}

		}

		if (errors == 0) {

			std::printf("[RESULT] vector OK\n");

		} else {

			std::printf("[RESULT] vector WRONG\n");

		}

		std::printf("[TIME] vector normal mpi mode=%s np=%d seconds=%.6f\n", use_collapse ? "collapse" : "flat", world_size, t1 - t0);
		std::free(C);

	}

	std::free(A);
	std::free(B);

	MPI_Finalize();
	return EXIT_SUCCESS;

}
