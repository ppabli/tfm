#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"

#define MAL_N 1000
#define RADIUS 3

static void make_kernel(double* k, int R) {

	double sigma = R / 2.0, sum = 0;

	for (int t = -R; t <= R; t++) {

		k[t + R] = std::exp(-0.5 * (t / sigma) * (t / sigma));
		sum += k[t + R];

	}

	for (int t = 0; t <= 2 * R; t++) {

		k[t] /= sum;

	}

}

int main(int /*argc*/, char* /*argv*/[]) {

	mal_init();

	double *A = nullptr, *C = nullptr;
	double kernel[2 * RADIUS + 1];

	make_kernel(kernel, RADIUS);

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(MAL_N * sizeof(double)));

		for (int i = 0; i < MAL_N; i++) {

			A[i] = std::sin(2.0 * M_PI * i / MAL_N);

		}

	}

	long i, limit;
	MalFor f = mal_for(MAL_N, i, limit);

	mal_attach_vec(f, (void**)&A, sizeof(double), MAL_N, -1);
	mal_attach_vec(f, (void**)&C, sizeof(double), MAL_N, 0);
	mal_attach_halo(f, (void**)&A, RADIUS);

	for (; i < limit; i++) {

		double s = 0;

		for (int k = -RADIUS; k <= RADIUS; k++) {

			s += kernel[k + RADIUS] * A[i + k];

		}

		C[i] = s;

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (int j = RADIUS; j < MAL_N - RADIUS; j++) {

			double ref = 0;

			for (int k = -RADIUS; k <= RADIUS; k++) {

				ref += kernel[k + RADIUS] * std::sin(2.0 * M_PI * (j + k) / MAL_N);

			}

			if (std::fabs(C[j] - ref) > 1e-12) {

				errors++;
				break;

			}

		}

		if (errors == 0) {

			MAL_LOG(MAL_LOG_INFO, "[RESULT] Convolution correct (N=%d, radius=%d)", MAL_N, RADIUS);

		} else {

			MAL_LOG(MAL_LOG_ERROR, "[RESULT] WRONG result");

		}

		std::free(C);

	}

	return EXIT_SUCCESS;

}
