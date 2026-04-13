#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

#define M 12
#define K 6
#define N 4

void run_matvec() {

	double *A = nullptr, *x = nullptr, *y = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(M * K * sizeof(double)));
		x = static_cast<double*>(std::malloc(K * sizeof(double)));
		y = static_cast<double*>(std::malloc(M * sizeof(double)));

		for (int r = 0; r < M; r++) {

			for (int c = 0; c < K; c++) {

				A[r * K + c] = static_cast<double>(r * K + c + 1);

			}

		}

		for (int c = 0; c < K; c++) {

			x[c] = 1.0;

		}

	}

	long i, lim;
	MalFor f = mal_for(M, i, lim);
	const useconds_t delay_us = example_delay_us(100000);

	mal_attach_mat(f, (void**)&A, sizeof(double), M, K, -1, MAL_ATTACH_PARTITIONED);
	mal_attach_mat(f, (void**)&x, sizeof(double), 1, K, -1, MAL_ATTACH_SHARED_ACTIVE);
	mal_attach_mat(f, (void**)&y, sizeof(double), M, 1, 0, MAL_ATTACH_PARTITIONED);

	for (; i < lim; i++) {

		double acc = 0.0;

		for (long k = 0; k < K; k++) {

			acc += A[i * K + k] * x[k];

		}

		y[i] = acc;

		MAL_LOG(MAL_LOG_INFO, "[MV] y[%ld] = %.1f", i, acc);

		usleep(delay_us);

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (int r = 0; r < M; r++) {

			double expected = 0.0;

			for (int c = 0; c < K; c++) {

				expected += (double)(r * K + c + 1);

			}

			if (std::fabs(y[r] - expected) > 1e-9) {

				errors = 1;

				break;

			}

		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-vec %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		std::free(y);

	}

	std::free(x);

}

void run_matmul() {

	double *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(M * K * sizeof(double)));
		B = static_cast<double*>(std::malloc(K * N * sizeof(double)));
		C = static_cast<double*>(std::malloc(M * N * sizeof(double)));

		for (int r = 0; r < M; r++) {

			for (int c = 0; c < K; c++) {

				A[r * K + c] = static_cast<double>(r + 1);

			}

		}

		for (int r = 0; r < K; r++) {

			for (int c = 0; c < N; c++) {


				B[r * N + c] = 1.0;

			}

		}

	}

	long i, lim;
	MalFor f = mal_for(M, i, lim);
	const useconds_t delay_us = example_delay_us(100000);

	mal_attach_mat(f, (void**)&A, sizeof(double), M, K, -1, MAL_ATTACH_PARTITIONED);
	mal_attach_mat(f, (void**)&B, sizeof(double), K, N, -1, MAL_ATTACH_SHARED_ACTIVE);
	mal_attach_mat(f, (void**)&C, sizeof(double), M, N, 0, MAL_ATTACH_PARTITIONED);

	for (; i < lim; i++) {

		for (long j = 0; j < N; j++) {

			double acc = 0.0;

			for (long k = 0; k < K; k++) {

				acc += A[i * K + k] * B[k * N + j];

			}

			C[i * N + j] = acc;

		}

		MAL_LOG(MAL_LOG_INFO, "[MM] C[%ld, 0..%d] computed, C[%ld,0]=%.1f", i, N-1, i, C[i * N]);

		usleep(delay_us);

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (int r = 0; r < M && errors == 0; r++) {

			for (int c = 0; c < N; c++) {

				if (std::fabs(C[r * N + c] - (double)(r + 1) * K) > 1e-9) {

					errors = 1;

					break;

				}

			}

		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-mat %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		std::free(C);

	}

	std::free(B);

}

int main(int argc, char* argv[]) {

	mal_init();

	bool do_mv = (argc > 1 && std::strcmp(argv[1], "mv") == 0);

	if (do_mv) {

		run_matvec();

	} else {

		run_matmul();

	}

	return EXIT_SUCCESS;

}
