import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def inception_tf():
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    dataset_dir = dir_path + '/../ImageClassification-Inception_v3/TensorFlow/dataset/'
    source_dir = dir_path + '/../ImageClassification-Inception_v3/TensorFlow/source/'
    os.system('python ' + source_dir + 'download_and_convert_data.py --dataset_name cifar10 --dataset_dir ' + dataset_dir)


downloader_map = {
    ('inception', 'tf') : inception_tf,
}