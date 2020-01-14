docker run --privileged --hostname <hostname> -dit --runtime=nvidia -v <project_path>:/mnt --ipc=host --name gpusim gpusim_serina
docker exec -it gpusim bash
