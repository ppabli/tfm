#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

#define MAL_N 20

int main(int argc, char* argv[]) {

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

	bool use_collapse = (argc > 1 && std::strcmp(argv[1], "collapse") == 0);

	if (use_collapse) {

		long i, limit_rows, j, limit_cols;

		const long starts[2] = {0, 0};
		const long limits2d[2] = {4, 5};

		long* iters[2] = {&i, &j};
		long* loop_limits[2] = {&limit_rows, &limit_cols};

		MalForND nd = mal_for_nd_begin(iters, loop_limits, starts, limits2d, 2);

		mal_attach_vec(nd, (void**)&A, sizeof(double), MAL_N, -1);
		mal_attach_vec(nd, (void**)&B, sizeof(double), MAL_N, -1);
		mal_attach_vec(nd, (void**)&C, sizeof(double), MAL_N,  0);


		const useconds_t delay_us = example_delay_us(200000);
		for (; i < limit_rows; i++) {

			for (; j < limit_cols; j++) {

				long idx = i * limit_cols + j;
				C[idx] = A[idx] + B[idx];

				MAL_LOG(MAL_LOG_INFO, "[ITER] C[%ld] = %.1f + %.1f = %.1f", idx, A[idx], B[idx], C[idx]);
				usleep(delay_us);

				mal_check_for(nd);

			}

		}

	} else {

		long i, limit;

		MalFor f = mal_for(MAL_N, i, limit);

		mal_attach_vec(f, (void**)&A, sizeof(double), MAL_N, -1);
		mal_attach_vec(f, (void**)&B, sizeof(double), MAL_N, -1);
		mal_attach_vec(f, (void**)&C, sizeof(double), MAL_N,  0);
		const useconds_t delay_us = example_delay_us(200000);

		for (; i < limit; i++) {

			C[i] = A[i] + B[i];

			MAL_LOG(MAL_LOG_INFO, "[ITER] C[%ld] = %.1f + %.1f = %.1f", i, A[i], B[i], C[i]);
			usleep(delay_us);

			mal_check_for(f);

		}

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

			MAL_LOG(MAL_LOG_INFO, "[RESULT] vector OK");

		} else {

			MAL_LOG(MAL_LOG_ERROR, "[RESULT] vector WRONG");

		}

		std::free(C);

	}

	return EXIT_SUCCESS;

}
