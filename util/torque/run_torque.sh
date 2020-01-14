#!/bin/sh

# start server
sudo pbs_server
sudo pbs_mom
sudo pbs_sched


qsub torque.sim

qstat

sleep 5

ls ~/run
cat ~/run/bar3.txt
