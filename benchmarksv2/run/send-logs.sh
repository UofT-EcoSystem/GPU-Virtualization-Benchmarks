if [ "$#" -ne 1 ]; then
  echo "Usage: send-logs.sh <run-dir>"
fi

rsync -avz --prune-empty-dirs --include '*/' --include="*commit*.log" --exclude '*' $1 me:/media/hdisk/home/serina/gpusim

