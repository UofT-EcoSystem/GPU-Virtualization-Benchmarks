#include <parboil.h>

#include "cuda_runtime.h"
#include "cutoff.h"

#include <functional>

int gpu_compute_cutoff_potential_lattice6overlap(
    struct pb_TimerSet *timers,        /* for measuring execution time */
    Lattice *lattice,
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    int verbose,                        /* print info/debug messages */
    std::function<int(const int, cudaStream_t &)> & kernel,
    std::function<void(void)> & cleanup
    );
