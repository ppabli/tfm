#pragma once

#include <cstdlib>
#include <unistd.h>

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
