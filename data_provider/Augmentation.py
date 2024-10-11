import Augmentor
import os

def augment(classname, samplename, method, type, source_dir, output_path):
    path = f"{output_path}/{classname}/{samplename}_{method[0]}" #_0"
    os.makedirs(path, exist_ok=True)
    source_dir = f"{source_dir}/{classname}/{samplename}"
    p = Augmentor.Pipeline(source_directory=source_dir,output_directory=path)

    # p.skew(probability = 0.5)
    # p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=6)
    # p.random_distortion(probability=0.5, grid_width=20, grid_height=20, magnitude=12)

    # augmented picture
    if type:
        p.random_erasing(probability=1, rectangle_area=method[1])
    else:
    # original picture
        p.random_distortion(probability=1, grid_width=method[0], grid_height=method[1], magnitude=method[2])
    p.process()

if '__name__' == '__main__':
    output_path = 'dataset\\New_Data\\Augmented'
    source_dir = 'dataset\\New_Data\\STFT_train'
    a = 1  # Start (static)
    b = 23  # Total TrainSet Number + 1: for example, 11 classes with 22 samples in each class -> 22 + 1
    method0 = [[10, 10, 0]]  # reserve one set of original dataset
    methods_imp = [[1, 0.5], [2, 0.6], [3, 0.7]]  # Augmented dataset *3

    # IMP Distance Augmentation
    allclassname =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for __, method in enumerate(methods_imp):
        for _, classname in enumerate(allclassname):
            print(classname)
            for i in range(a, b):
                    augment(classname, i, method, True, source_dir, output_path)
    for __, method in enumerate(method0):
        for _, classname in enumerate(allclassname):
            print(classname)
            for i in range(a, b):
                    augment(classname, i, method, False, source_dir, output_path)


