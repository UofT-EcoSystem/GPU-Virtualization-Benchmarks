rsync -a --prune-empty-dirs --include '*/' --include="*.log" --exclude '*' run-* me:/media/hdisk/home/serina/gpusim
scp logfiles/* me:/media/hdisk/home/serina/gpusim/logfiles

