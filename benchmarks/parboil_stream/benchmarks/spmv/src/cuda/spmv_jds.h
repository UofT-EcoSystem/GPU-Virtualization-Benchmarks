




 
 
__global__ void spmv_jds(float *dst_vector,
							   const float *d_data,const int *d_index, const int *d_perm,
							   const float *x_vec,const int *d_nzcnt,const int dem);
							   
