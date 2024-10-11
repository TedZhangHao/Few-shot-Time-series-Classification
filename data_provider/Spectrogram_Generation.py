import os, sys
from tqdm import tqdm
import torchvision.transforms as transforms
from scipy.signal import stft,istft
import matplotlib.pyplot as plt
import numpy as np
from Spectrogram_Dataset_Obtain import CustomDataset

def Series_Show(data, channel_num=9):
    fig, ax = plt.subplots(3, 3, figsize=(10, 4))
    band_name = ['Z11', 'Z12', 'Z13', 'Z21', 'Z22', 'Z23', 'Z31', 'Z32', 'Z33']
    a = 0
    b = 0
    for i in range(channel_num):
        if i > 0 and i % 3 == 0:
            a += 1
            b -= 3
        ax[a][b].plot(data[i])
        ax[a][b].set_xlabel('Time/ms')
        ax[a][b].set_ylabel('Voltage/mv')
        ax[a][b].set_title(band_name[i])
        ax[a][b].grid()
        b += 1
    plt.show()
    plt.figure(figsize=(10, 4))
    for i in range(channel_num):
        plt.plot(data[i], label=band_name[i])
        plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Sample')
    plt.grid()
    plt.show()

def STFT_Show(data, channel_num=9):
    fs = 100
    window = 'hann'
    # frame长度
    n = 62  # 135
    n1 = n
    overlap = 0.91
    f_all = [];
    t_all = [];
    Z_all = [];
    Z1_all = []
    for i in range(channel_num):
        f, t, Z = stft(data[i], fs=fs, window=window, noverlap=n * overlap, nperseg=n,
                       nfft=n, return_onesided=True,
                       boundary='even')  # nperseg = 窗函数长度, nfft = fft长度 boundary = 'even',
        f_all.append(f)
        t_all.append(t)
        Z_all.append(Z)
        Z1 = np.log(np.abs(Z))
        Z1_all.append(Z1)
    cmap = 'gray'
    fig, ax = plt.subplots(3, 3, figsize=(10, 4))
    band_name = ['Z11', 'Z12', 'Z13', 'Z21', 'Z22', 'Z23', 'Z31', 'Z32', 'Z33']
    a = 0
    b = 0
    print(Z.shape)
    for i in range(channel_num):
        if i > 0 and i % 3 == 0:
            a += 1
            b -= 3
        ax[a][b].pcolormesh(t_all[i], f_all[i], Z1_all[i], cmap=cmap)  # t,f 横/纵坐标系 Z颜色
        ax[a][b].set_title('Log STFT Magnitude')
        ax[a][b].set_ylabel('Frequency [Hz]')
        ax[a][b].set_xlabel('Time [sec]')
        b += 1
    plt.show()

def save_image(x, classname, samplename, picname, fs, window, n, path):
    f, t, Z = stft(x, fs=fs, window=window, noverlap=n * 0.91, nperseg=n, boundary='even', nfft=n,
                   return_onesided=True)  # nperseg = Window length, nfft = fft length
    # Amplitude
    Z = np.abs(Z)
    Z = np.log(Z)  # Increase Contrast

    plt.figure()
    plt.pcolormesh(t, f, Z, cmap='gray')
    plt.axis('off')
    # plt.tight_layout()
    if picname is not None:
        path_ = f"{path}\\{classname}\\{samplename}"
        os.makedirs(path_, exist_ok=True)
        plt.savefig(path + '\\' + str(classname) + '\\' + str(samplename) + '\\' + str(
            picname) + '.jpg', bbox_inches='tight', pad_inches=0.0, dpi=400)  # save images
    plt.clf()
    plt.close()

def transform_set(train_root, test_root, resize_size=(32, 35)):

    # define transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize_size),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=0.5,std=0.5),
    ])
    # create available dataset
    trainset1 = CustomDataset(train_root, transform=transform)
    valset1 = CustomDataset(test_root, transform=transform)
    return trainset1, valset1

def data_adjustment(trainset1, valset1, num_train, num_test, channel_num, resize_size, num_class=11):
    trainset = np.zeros((num_train*num_class, channel_num, resize_size[0], resize_size[1]))
    for i in tqdm(range(len(trainset1))):
        trainset[i] = np.array(trainset1[i].squeeze())
    valset = np.zeros((num_test*num_class, channel_num, resize_size[0], resize_size[1]))
    for i in tqdm(range(len(valset1))):
        valset[i] = np.array(valset1[i].squeeze())
    return np.array(trainset), np.array(valset)

if __name__ == '__main__':
    train_data = np.load("dataset\\New_Data\\X_train_series.npy")
    test_data = np.load("dataset\\New_Data\\X_test_series.npy")
    train_label = np.load("dataset\\New_Data\\y_train.npy")
    test_label = np.load("dataset\\New_Data\\y_test.npy")
    # Show Wind Turbine Series Example
    data_example = train_data[0]
    Series_Show(data_example)
    STFT_Show(data_example)

    window = 'hann'
    n1 = 62
    allclassname = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    num_train = 22  # training sample number
    num_test = 6  # testing sample number
    fs = 100  # Sample Frequency
    channel_num = 9
    save_STFT_train_path = 'dataset\\New_Data\\STFT_train'
    save_STFT_test_path = 'dataset\\New_Data\\STFT_test'
    for _, classname in enumerate(allclassname):
        print(classname)
        for i in range(num_train):
            for j in range(channel_num):
                save_image(train_data[i + num_train*int(classname)][j], f"{classname}", f"{i + 1}", f"{j + 1}",
                           fs, window, n=n1, path=save_STFT_train_path)
        for i in range(num_test):
            for j in range(channel_num):
                save_image(test_data[i + num_test*int(classname)][j], f"{classname}", f"{i + 1}", f"{j + 1}",
                           fs, window, n=n1, path=save_STFT_test_path)
    # Transform Spectrogram to TrainSet and ValSet
    resize_size = (32, 35)
    X_train, X_test = transform_set(train_root=save_STFT_train_path, test_root=save_STFT_test_path, resize_size=resize_size)
    np.save("dataset\\New_Data\\11classes_imp\\X_train_imp_graph_stft", X_train)
    np.save("dataset\\New_Data\\11classes_imp\\X_test_imp_graph_stft", X_test)

