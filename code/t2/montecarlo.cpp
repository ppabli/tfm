#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <unistd.h>
#include "malleable.hpp"

#define MAL_TOTAL_POINTS 20

int main(int /*argc*/, char* /*argv*/[]) {

	mal_init();

	const int urank = mal_state.universe_rank;
	unsigned int seed = urank;

	long i;
	long limit;
	long hits = 0;

	MalReduce r = mal_for_reduce(MAL_TOTAL_POINTS, i, limit, hits);

	for (; i < limit; ++i) {

		MAL_LOG("ITER", "Iteration %ld / %ld", i, limit);


		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {

			MAL_LOG("HIT", "i=%ld x=%.4f y=%.4f", i, x, y);
			hits++;

		}

		usleep(1000 * 1000 * 2);

		mal_check_reduce(r);

	}

	if (mal_state.active_comm != MPI_COMM_NULL && mal_state.current_rank == 0) {

		double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(MAL_TOTAL_POINTS);

		MAL_LOG("RESULT", "total_points=%d  hits=%ld  pi~=%.6f  error=%.2e", MAL_TOTAL_POINTS, hits, pi_approx, std::fabs(pi_approx - 3.14159265358979));

	}

	mal_finalize();

	return EXIT_SUCCESS;

}

