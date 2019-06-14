docker run --privileged --runtime=nvidia -dit --ipc=host -v /home/serinatan/project:/mnt -v /media/hdisk/home/serina/datasets/tiny-imagenet-200/:/dataset --name torch torch_img:latest
docker exec -it torch /bin/bash

