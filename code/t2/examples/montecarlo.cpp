#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

int main(int argc, char* argv[]) {

	mal_init();

	const long total_points = parse_arg_long(argc, argv, "n", 20);
	unsigned int seed = static_cast<unsigned int>(mal_rank());

	long i, limit;
	MalFor f = mal_for(total_points, i, limit);
	const useconds_t delay_us = example_delay_us(200000);
	const double t0 = MPI_Wtime();

	long hits = 0;
	mal_attach_acc(f, hits);

	for (; i < limit; i++) {

		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {

			hits++;

		}

		#if !BENCH_CSV
		MAL_LOG(MAL_LOG_INFO, "[ITER] i=%ld hits_so_far=%ld", i, hits);
		#endif

		usleep(delay_us);

		mal_check_for(f);

	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV
	(void)compute_seconds;
	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

			print_bench_csv("montecarlo", "malleable", "std", mal_size(), total_points, compute_seconds, 0);

		#else

			double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(total_points);

			MAL_LOG(MAL_LOG_INFO, "[RESULT] montecarlo OK total_points=%ld hits=%ld pi~=%.6f error=%.2e", total_points, hits, pi_approx, std::fabs(pi_approx - 3.14159265358979));

		#endif

	}

	return EXIT_SUCCESS;

}

