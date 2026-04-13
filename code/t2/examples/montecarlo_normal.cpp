#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include "example_utils.hpp"

#define MAL_TOTAL_POINTS 20

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

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	unsigned int seed = static_cast<unsigned int>(world_rank);
	const BlockRange range = block_range(MAL_TOTAL_POINTS, world_rank, world_size);
	const useconds_t delay_us = example_delay_us(200000);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	long local_hits = 0;
	for (long i = 0; i < range.count; i++) {

		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {

			local_hits++;

		}

		usleep(delay_us);

	}

	long hits = 0;
	MPI_Reduce(&local_hits, &hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		const double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(MAL_TOTAL_POINTS);
		std::printf("[RESULT] montecarlo OK total_points=%d  hits=%ld  pi~=%.6f  error=%.2e\n", MAL_TOTAL_POINTS, hits, pi_approx, std::fabs(pi_approx - 3.14159265358979));
		std::printf("[TIME] montecarlo normal mpi np=%d seconds=%.6f\n", world_size, t1 - t0);

	}

	MPI_Finalize();
	return EXIT_SUCCESS;

}
