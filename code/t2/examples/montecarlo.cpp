#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

#define MAL_TOTAL_POINTS 20

int main(int /*argc*/, char* /*argv*/[]) {

	mal_init();

	unsigned int seed = static_cast<unsigned int>(mal_rank());

	long i, limit;
	MalFor f = mal_for(MAL_TOTAL_POINTS, i, limit);
	const useconds_t delay_us = example_delay_us(200000);

	long hits = 0;
	mal_attach_acc(f, hits);

	for (; i < limit; i++) {

		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {

			hits++;

		}

		MAL_LOG(MAL_LOG_INFO, "[ITER] i=%ld hits_so_far=%ld", i, hits);

		usleep(delay_us);

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(MAL_TOTAL_POINTS);

		MAL_LOG(MAL_LOG_INFO, "[RESULT] montecarlo OK total_points=%d  hits=%ld  pi~=%.6f  error=%.2e", MAL_TOTAL_POINTS, hits, pi_approx, std::fabs(pi_approx - 3.14159265358979));

	}

	return EXIT_SUCCESS;

}

