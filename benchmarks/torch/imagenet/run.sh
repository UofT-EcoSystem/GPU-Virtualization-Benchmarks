#/usr/local/NVIDIA-Nsight-Compute-2019.3/nv-nsight-cu-cli -o resnet50 --target-processes all --section-folder ../../sections --section PertSustainedActive -f --profile-from-start off --csv ./main.py -a resnet50 --pretrained /imagenet -b 32 --gpu 0 --profile 5 --epochs 1
#/usr/local/NVIDIA-Nsight-Compute-2019.3/nv-nsight-cu-cli -o resnet50-inst --target-processes all \ 
#--section-folder ../../sections --section InstCounter -f --profile-from-start off \ 
#--csv /opt/conda/bin/python main.py -a resnet50 --pretrained /dataset -b 32 --gpu 2 \
#--profile 5 --epochs 1
/usr/bin/python main.py -a resnet50 --pretrained /dataset -b 32 --gpu 2 --profile 5 --epochs 1
