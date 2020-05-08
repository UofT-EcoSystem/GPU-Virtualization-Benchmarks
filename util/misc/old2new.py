import os
import glob

current = os.path.dirname(os.path.realpath(__file__))
logs = glob.glob(os.path.join(current, '*/*/*.log'))

for log in logs:
    print(log)
    with open(log, 'rt') as f:
        data = f.read()
        data = data.replace('gpu_tot_sim_cycle[1]' , 'gpu_tot_sim_cycle[1][0]')
        data = data.replace('gpu_tot_sim_insn[1]' , 'gpu_tot_sim_insn[1][0]')
        data = data.replace('L2_BW_total[1]' , 'L2_BW_total[1][0]')
        data = data.replace('barrier_cycles[1]', 'barrier_cycles[1][0]')
        data = data.replace('inst_empty_cycles[1]', 'inst_empty_cycles[1][0]')
        data = data.replace('branch_cycles[1]', 'branch_cycles[1][0]')
        data = data.replace('stall_scoreboard_cycles[1]', 'stall_scoreboard_cycles[1][0]')
        data = data.replace('stall_sp_cycles[1]', 'stall_sp_cycles[1][0]')
        data = data.replace('stall_dp_cycles[1]', 'stall_dp_cycles[1][0]')
        data = data.replace('stall_int_cycles[1]', 'stall_int_cycles[1][0]')
        data = data.replace('stall_tensor_cycles[1]', 'stall_tensor_cycles[1][0]')
        data = data.replace('stall_sfu_cycles[1]', 'stall_sfu_cycles[1][0]')
        data = data.replace('stall_control_cycles[1]', 'stall_control_cycles[1][0]')
        data = data.replace('stall_mem_cycles[1]', 'stall_mem_cycles[1][0]')
        data = data.replace('not_selected_cycles[1]', 'not_selected_cycles[1][0]')
        data = data.replace('cycles_per_issue[1]', 'cycles_per_issue[1][0]')
        data = data.replace('averagemflatency[1]', 'averagemflatency[1][0]')
        data = data.replace('avg_icnt2mem_latency[1]', 'avg_icnt2mem_latency[1][0]')
        data = data.replace('avg_offchip2mem_latency[1]', 'avg_offchip2mem_latency[1][0]')
        data = data.replace('avg_icnt2mem_per_submem[1]', 'avg_icnt2mem_per_submem[1][0]')
        data = data.replace('avg_mrq_latency[1]', 'avg_mrq_latency[1][0]')
        data = data.replace('avg_icnt2sh_latency[1]', 'avg_icnt2sh_latency[1][0]')

    with open(log, 'wt') as f:
        f.write(data)
