python nsight.py --types inst comp mem --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Translation/GNMT/profile/train-2080 --name "GNMT FP16 Training" --drop_threshold 1 --width 25 --height 30 --name_len 0.4

python nsight.py --types inst comp mem --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Translation/GNMT/profile/infer-2080 --name "GNMT FP16 Inference" --drop_threshold 1 --width 25 --height 30 --name_len 0.2

python nsight.py --types inst comp mem --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Classification/RN50v1.5/profile/train-2080 --name "Resnet50 FP32 Training" --drop_threshold 1 --width 25 --height 30 --name_len 0.4

python nsight.py --types inst comp mem --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Classification/RN50v1.5/profile/infer-2080 --name "Resnet50 FP32 Inference" --drop_threshold 1 --width 25 --height 30 --name_len 0.2

python nsight.py --types comptime memtime --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Classification/RN50v1.5/profile/train-2080 --name "Resnet50 FP32 Training" --drop_threshold 1 --width 50 --height 20 --name_len 0.4

python nsight.py --types memtime comptime --indir /home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/DeepLearningExamples/PyTorch/Classification/RN50v1.5/profile/infer-2080 --name "Resnet50 FP32 Inference" --drop_threshold 1 --width 50 --height 20 --name_len 0.2
