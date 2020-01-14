#!/bin/sh

if [ "$#" -lt 1 ]; then
  echo "Usage install_torque.sh num_cores"
  exit
fi

echo "Installing torque and configure server to have $1 cores."

# torque SERVER setup
sudo apt update && sudo apt -y install torque-server torque-scheduler torque-client

# torque SERVER setup
hostname | sudo tee /var/spool/torque/server_name

echo y | sudo pbs_server -t create


sudo qmgr -c "set server scheduling=true"
sudo qmgr -c "create queue batch queue_type=execution"
sudo qmgr -c "set queue batch started=true"
sudo qmgr -c "set queue batch enabled=true"
sudo qmgr -c "set queue batch resources_default.nodes=1"
sudo qmgr -c "set queue batch resources_default.walltime=3600"
sudo qmgr -c "set server default_queue=batch"

# stop pbs_server, can be used to restart pbs server and reload node config
sudo qterm

echo "`hostname` np=$1" | sudo tee -a /var/spool/torque/server_priv/nodes


# torque CLIENT setup
sudo apt update && sudo apt install -y torque-client torque-mom

echo "pbs_server = `hostname`" | sudo tee /var/spool/torque/mom_priv/config
