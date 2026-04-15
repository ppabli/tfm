#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

void run_matvec(int M, int K) {

	double *A = nullptr, *x = nullptr, *y = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(static_cast<size_t>(M * K) * sizeof(double)));
		x = static_cast<double*>(std::malloc(static_cast<size_t>(K) * sizeof(double)));
		y = static_cast<double*>(std::malloc(static_cast<size_t>(M) * sizeof(double)));

		for (int r = 0; r < M; r++) {

			for (int c = 0; c < K; c++) {

				A[r * K + c] = static_cast<double>(r * K + c + 1);

			}

		}

		for (int c = 0; c < K; c++) {

			x[c] = 1.0;

		}

	}

	const double t0 = MPI_Wtime();
	long i, lim;
	MalFor f = mal_for(M, i, lim);

	#if !BENCH_CSV

		const useconds_t delay_us = example_delay_us(100000);

	#endif

	mal_attach_mat(f, (void**)&A, sizeof(double), M, K, -1, MAL_ATTACH_PARTITIONED);
	mal_attach_mat(f, (void**)&x, sizeof(double), 1, K, -1, MAL_ATTACH_SHARED_ACTIVE);
	mal_attach_mat(f, (void**)&y, sizeof(double), M, 1, 0, MAL_ATTACH_PARTITIONED);

	for (; i < lim; i++) {

		double acc = 0.0;

		for (long k = 0; k < K; k++) {

			acc += A[i * K + k] * x[k];

		}

		y[i] = acc;

		#if !BENCH_CSV

			MAL_LOG(MAL_LOG_INFO, "[MV] y[%ld] = %.1f", i, acc);
			usleep(delay_us);

		#endif

		mal_check_for(f);

	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV

		(void)compute_seconds;

	#endif

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

		#if BENCH_CSV

			print_bench_csv("matmul", "malleable", "mv", mal_size(), M, compute_seconds, errors);

		#else

			MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-vec %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		#endif

		std::free(y);

	}

	std::free(x);

}

void run_matmul(int M, int K, int N) {

	double *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<double*>(std::malloc(static_cast<size_t>(M * K) * sizeof(double)));
		B = static_cast<double*>(std::malloc(static_cast<size_t>(K * N) * sizeof(double)));
		C = static_cast<double*>(std::malloc(static_cast<size_t>(M * N) * sizeof(double)));

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

	const double t0 = MPI_Wtime();
	long i, lim;
	MalFor f = mal_for(M, i, lim);

	#if !BENCH_CSV

		const useconds_t delay_us = example_delay_us(100000);

	#endif

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

		#if !BENCH_CSV

			MAL_LOG(MAL_LOG_INFO, "[MM] C[%ld, 0..%d] computed, C[%ld,0]=%.1f", i, N-1, i, C[i * N]);
			usleep(delay_us);

		#endif

		mal_check_for(f);

	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV

		(void)compute_seconds;

	#endif

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

		#if BENCH_CSV

			print_bench_csv("matmul", "malleable", "mm", mal_size(), static_cast<long>(M) * static_cast<long>(N), compute_seconds, errors);

		#else

			MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-mat %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		#endif

		std::free(C);

	}

	std::free(B);

}

int main(int argc, char* argv[]) {

	mal_init();

	const int M = static_cast<int>(parse_arg_long(argc, argv, "m", 12));
	const int K = static_cast<int>(parse_arg_long(argc, argv, "k", 6));
	const int N = static_cast<int>(parse_arg_long(argc, argv, "n", 4));

	bool do_mv = (argc > 1 && std::strcmp(argv[1], "mv") == 0);

	if (do_mv) {

		run_matvec(M, K);

	} else {

		run_matmul(M, K, N);

	}

	return EXIT_SUCCESS;

}
