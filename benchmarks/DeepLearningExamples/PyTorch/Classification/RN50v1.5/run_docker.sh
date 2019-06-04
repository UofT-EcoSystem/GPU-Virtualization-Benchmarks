docker run --privileged --runtime=nvidia -dit --ipc=host -v /home/serinatan/:/mnt -v /scratch/dataset/imagenet/:/dataset --name res gnmt:latest
docker exec -it res /bin/bash

