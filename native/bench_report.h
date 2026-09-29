#ifndef SQ_BENCH_REPORT_H
#define SQ_BENCH_REPORT_H

#include "bench_types.h"

/* Writes the one-line bench JSON report to stdout. The stage columns come from
   result->columns. */
void sq_bench_print_json(const sq_bench_result *result);

#endif
