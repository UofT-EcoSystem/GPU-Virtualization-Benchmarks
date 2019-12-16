#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include "rodinia_common.h"

long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void fatal(const char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}
