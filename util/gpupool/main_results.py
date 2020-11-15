import sys
import numpy as np
import argparse
import os
from gpupool.workload import BatchJob, GpuPoolConfig, Job
from gpupool.predict import Allocation, StageOne, StageTwo
from gpupool.workload import Violation


NUM_JOBS = 100


def parse_args():
    parser = argparse.ArgumentParser("Get main paper results from pickle.")

    parser.add_argument('--pkl',
                        help='Path to pickle file')
    parser.add_argument('--stage2_buffer', default=0.1,
                        type=float,
                        help='Amount of buffer to tighten qos check in stage2.')
    # hack
    parser.add_argument('--offset',
                        type=int,
                        default=0,
                        help='ID Offset for job synthesis.')

    results = parser.parse_args()

    return results


def main():
    args = parse_args()
    pkl_name = str(os.path.basename(args.pkl).split('.')[0])
    params = pkl_name.split('-')

    Job.count = args.offset
    batch = BatchJob(rand_seed=int(params[1]), num_jobs=NUM_JOBS)
    batch.load_df_from_pickle(args.pkl)

    # GPUPool
    config = GpuPoolConfig(Allocation[params[2]], StageOne[params[3]],
                           StageTwo[params[4]],
                           at_least_once=(params[5] == "True"),
                           accuracy_mode=False,
                           stage2_buffer=args.stage2_buffer)

    gpupool_matching = batch.max_matching(config, cores=32)
    gp_count, gp_violation, gp_ws, gp_ws_list, gp_isolated = gpupool_matching
    print("GPUPool GPU count:", gp_count)
    print("WS:", gp_ws)
    print(gp_violation.to_string(NUM_JOBS))

    # MIG
    mig_count, mig_ws, ws_list_mig = batch.calculate_gpu_count_mig()
    print("MIG count:", mig_count)
    print("MIG WS", mig_ws)

    # Heuristic
    heuristic_count = NUM_JOBS / 2
    bw_violation, bw_ws_list, bw_gpu_total = \
        batch.calculate_qos_viol_dram_bw(heuristic_count,
                                         cores=32)
    bw_ws = sum(bw_ws_list) / bw_gpu_total

    # heuristic_migrated_count_final = NUM_JOBS
    # ws_list_heuristic_final = []
    # heuristic_count_final = NUM_JOBS
    # heuristic_violation_final = Violation()
    # # Iteratively find the best config
    # while violations_count > 0:
    #     heuristic_count += 2
    #     violation, ws_list, gpu_migrated = \
    #         batch.calculate_qos_viol_dram_bw(heuristic_count,
    #                                          cores=32)
    #     violations_count = violation.count
    #     if gpu_migrated < heuristic_migrated_count_final:
    #         heuristic_migrated_count_final = gpu_migrated
    #         ws_list_heuristic_final = ws_list
    #         heuristic_count_final = heuristic_count
    #         heuristic_violation_final = violation
    #
    #     sys.exit(0)
    #
    print("Heuristic count", bw_gpu_total)
    print("Heuristic WS", bw_ws)

    # Data for sorted ws per GPU
    sorted_gpupool_ws = sorted(gp_ws_list, reverse=True)
    gpupool_xs = range(len(gp_ws_list))

    sorted_mig_ws = sorted(ws_list_mig, reverse=True)
    mig_xs = range(len(ws_list_mig))

    sorted_heuristic_ws = sorted(bw_ws_list, reverse=True)
    heuristic_xs = range(len(bw_ws_list))

    csv = open('best.csv', 'w')

    csv.write("#ws-plot")

    sorted_mig_ws = [str(ws) for ws in sorted_mig_ws]
    sorted_heuristic_ws = [str(ws) for ws in sorted_heuristic_ws]
    sorted_gpupool_ws = [str(ws) for ws in sorted_gpupool_ws]

    csv.write('Heuristic,{}'.format(','.join(sorted_heuristic_ws)))
    csv.write('Coarse,{}'.format(','.join(sorted_mig_ws)))
    csv.write('GPUPool,{}'.format(','.join(sorted_gpupool_ws)))

    csv.write("#gpu-count")
    csv.write("No-Sharing,{}".format(NUM_JOBS))
    csv.write("Coarse,{}".format(mig_count))
    csv.write("Heuristic,{}".format(bw_gpu_total))
    csv.write("GPUPool,{}".format(gp_count))

    csv.write("#ws")
    csv.write("No-Sharing,{}".format(1))
    csv.write("Coarse,{}".format(mig_ws))
    csv.write("Heuristic,{}".format(bw_ws))
    csv.write("GPUPool,{}".format(gp_ws))

    csv.close()

    # Achieved QoS per Job
    job_ids = [job.id for job in batch.list_jobs]
    id_offset = batch.list_jobs[0].id

    def get_job_norm_sld(violation):
        norm_sld = []
        for job_id in job_ids:
            job_qos = batch.list_jobs[job_id - id_offset].qos.value
            if job_id in violation.job_sld:
                norm_sld.append(violation.job_sld[job_id] / job_qos)
            else:
                norm_sld.append(1 / job_qos)

        return norm_sld

    norm_sld_gpupool = get_job_norm_sld(gp_violation)
    norm_sld_mig = [job.sld_mig / job.qos.value for job in batch.list_jobs]
    norm_sld_heuristic = get_job_norm_sld(bw_violation)

    xs = range(len(batch.list_jobs))

    # csv.write('xs, Coarse, Heuristic, GPUPool')
    scatter_data = np.array([np.array(xs),
                             np.array(norm_sld_mig),
                             np.array(norm_sld_heuristic),
                             np.array(norm_sld_gpupool)]).T

    np.savetxt('scatter.csv', scatter_data, delimiter=',')


if __name__ == '__main__':
    main()
