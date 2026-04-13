#pragma once

#include <cstdlib>
#include <unistd.h>

[[maybe_unused]] inline long parse_arg_long(int argc, char** argv, const char* key, long default_val) {

	for (int i = 1; i < argc; i++) {

		const char* arg = argv[i];

		if (arg[0] != '-' || arg[1] != '-') continue;

		const char* p = arg + 2;
		const char* q = key;

		while (*q && *p == *q) { p++; q++; }

		if (*q == '\0' && *p == '=') {

			char* end = nullptr;
			long v = std::strtol(p + 1, &end, 10);

			if (end != p + 1) return v;

		}

	}

	return default_val;

}

[[maybe_unused]] inline useconds_t parse_env_uint(const char* var, useconds_t default_val) {

	const char* env = std::getenv(var);

	if (!env || !*env) return default_val;

	char* end = nullptr;
	long v = std::strtol(env, &end, 10);

	return (end == env || v <= 0) ? default_val : (useconds_t)v;

}

[[maybe_unused]] inline useconds_t example_delay_us(useconds_t default_us) {

	return parse_env_uint("MAL_EXAMPLE_DELAY_US", default_us);

}

[[maybe_unused]] inline useconds_t sparse_delay_scale_percent() {

	return parse_env_uint("MAL_SPARSE_DELAY_SCALE_PERCENT", 100);

}
