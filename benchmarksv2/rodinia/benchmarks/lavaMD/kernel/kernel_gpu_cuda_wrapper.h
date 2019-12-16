#include "cuda_runtime_api.h"


#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER HEADER
//========================================================================================================================================================================================================200

void kernel_gpu_cuda_wrapper(	par_str parms_cpu,
								dim_str dim_cpu,
								box_str* box_cpu,
								FOUR_VECTOR* rv_cpu,
								fp* qv_cpu,
								FOUR_VECTOR* fv_cpu,
								int uid,
								cudaStream_t & stream);

#ifdef __cplusplus
}
#endif
