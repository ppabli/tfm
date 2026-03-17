#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <unistd.h>
#include "malleable.hpp"

#define MAL_N 20

int main(int /*argc*/, char* /*argv*/[]) {

	mal_init();

	double *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));
		B = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));
		C = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));

		for (int k = 0; k < MAL_N; k++) {

			A[k] = static_cast<double>(k + 1);
			B[k] = static_cast<double>(MAL_N - k);

		}

	}

	long i, limit;
	MalFor f = mal_for(MAL_N, i, limit);

	mal_attach_vec(f, (void**)&A, sizeof(double), MAL_N, -1);
	mal_attach_vec(f, (void**)&B, sizeof(double), MAL_N, -1);
	mal_attach_vec(f, (void**)&C, sizeof(double), MAL_N,  0);

	for (; i < limit; i++) {

		C[i] = A[i] + B[i];

		MAL_LOG(MAL_LOG_INFO, "[ITER] C[%ld] = %.1f + %.1f = %.1f", i, A[i], B[i], C[i]);

		usleep(1000 * 1000 * 2);

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (int k = 0; k < MAL_N; k++) {

			if (C[k] != static_cast<double>(MAL_N + 1)) {

				errors++;
				break;

			}

		}

		if (errors == 0) {

			MAL_LOG(MAL_LOG_INFO, "[RESULT] All elements correct");

		} else {

			MAL_LOG(MAL_LOG_ERROR, "[RESULT] Some elements WRONG");

		}

		std::free(C);

	}

	return EXIT_SUCCESS;

}

