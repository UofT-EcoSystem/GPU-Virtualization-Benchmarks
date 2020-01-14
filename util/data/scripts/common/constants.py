import os

DATA_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

max_cta_volta = 32
max_thread_volta = 2048
max_smem = 96*1024
max_register = 64*1024
num_sm_volta = 80

# max ctas according to resource constraints, grid size
app_dict = {
            'cut_sgemm-0': [2, 128],
            'cut_sgemm-1': [2, 512],
            'cut_wmma-0': [4, 128],
            'cut_wmma-1': [4, 1024],
            'parb_sgemm-0': [11, 528],
            'parb_cutcp-0': [16, 121],
            'parb_stencil-0': [16, 1024],
            'parb_lbm-0': [12, 18000],
            'parb_spmv-0': [16, 1147],
            'rod_mummer-0': [8, 5727],
            'rod_heartwall-0': [5, 51],
            'rod_hotspot-0': [8, 1849],
            'rod_hotspot3d-0': [8, 1024],
            'rod_streamcluster-0': [3, 128],
            'rod_pathfinder-0': [8, 463],
            'rod_lavamd-0': [9, 1000],
}

