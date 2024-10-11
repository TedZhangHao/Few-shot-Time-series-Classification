import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['STSong']
import math

def load_data(path_data, path_distance):
    distance_data = pd.read_csv(path_distance, header=None)
    turbine_data = np.load(path_data)
    turbine_data = turbine_data.swapaxes(1, 2)
    distance_data = np.array(distance_data)
    return turbine_data, distance_data

def xyz_to_d(labels):
    distance = []
    o = np.array([-0.18193, 1.0946, -0.00020664])
    for i in range(len(labels)):
        d = math.sqrt((labels[i, 0] - o[0])**2 + (labels[i, 1] - o[1])**2 + (labels[i, 2] - o[2])**2)
        distance.append(d)
    distance = np.array(distance)
    return distance

def plot_series_data(distance_label1, distance_label2, distance_label3):
    plt.figure()
    plt.plot(distance_label1, label='1')
    plt.plot(distance_label2, label='2')
    plt.plot(distance_label3, label='3')
    plt.legend()
    plt.show()


def define_label(data, distance):
    data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10 = [], [], [], [], [], [], [], [], [], []
    a = 0.15
    b = 0.03
    for i in range(len(data)):
        if distance[i] < a:
            data_1.append(data[i])
        elif a <= distance[i] < a + b:
            data_2.append(data[i])
        elif a + b <= distance[i] < a + 2 * b:
            data_3.append(data[i])
        elif a + 2 * b <= distance[i] < a + 3 * b:
            data_4.append(data[i])
        elif a + 3 * b <= distance[i] < a + 4 * b:
            data_5.append(data[i])
        elif a + 4 * b <= distance[i] < a + 5 * b:
            data_6.append(data[i])
        elif a + 5 * b <= distance[i] < a + 6 * b:
            data_7.append(data[i])
        elif a + 6 * b <= distance[i] < a + 7 * b:
            data_8.append(data[i])
        elif a + 7 * b <= distance[i] < a + 8 * b:
            data_9.append(data[i])
        else:
            data_10.append(data[i])

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_3 = np.array(data_3)
    data_4 = np.array(data_4)
    data_5 = np.array(data_5)
    data_6 = np.array(data_6)
    data_7 = np.array(data_7)
    data_8 = np.array(data_8)
    data_9 = np.array(data_9)
    data_10 = np.array(data_10)
    return data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10

def data_merge_sampling(turbine_data1, distance_label1, turbine_data2, distance_label2, turbine_data3, distance_label3, turbine_data4, num = 28):
    data1, data2, data3, data4, data5, data6, data7, data8, data9, data10 = [], [], [], [], [], [], [], [], [], []
    for turbine, distance in zip([turbine_data1, turbine_data2, turbine_data3], [distance_label1, distance_label2, distance_label3]):
        data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10 = define_label(turbine, distance)
        data1.append(data_1)
        data2.append(data_2)
        data3.append(data_3)
        data4.append(data_4)
        data5.append(data_5)
        data6.append(data_6)
        data7.append(data_7)
        data8.append(data_8)
        data9.append(data_9)
        data10.append(data_10)
    data_list = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
    for i in range(len(data_list)):
        data_list[i] = np.concatenate(data_list[i])
        data_list[i] = select_elements(data_list[i], num)
    data11 = select_elements(turbine_data4, num)
    return data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5],\
           data_list[6], data_list[7], data_list[8], data_list[9], data11

def select_elements(data, num):
    data_ = []
    selected_elements = np.random.choice(data.shape[0], size=num, replace=False)
    for i in selected_elements:
        data_.append(data[i])
    data_ = np.array(data_)
    return data_

def dataset_split(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, train_num = 22, num=28):
    label_refer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_data = [data1[:train_num], data2[:train_num], data3[:train_num], data4[:train_num], data5[:train_num],
                  data6[:train_num], data7[:train_num], data8[:train_num], data9[:train_num], data10[:train_num],
                  data11[:train_num]]
    train_data = np.concatenate(train_data)
    train_label = np.repeat(label_refer, train_num, axis=0)
    test_data = [data1[train_num:], data2[train_num:], data3[train_num:], data4[train_num:], data5[train_num:],
                 data6[train_num:], data7[train_num:], data8[train_num:], data9[train_num:], data10[train_num:],
                 data11[train_num:]]
    test_data = np.concatenate(test_data)
    test_label = np.repeat(label_refer, num - train_num, axis=0)
    print(train_data.shape, test_data.shape, train_label, test_label)
    np.save("dataset\\New_Data\\X_train_series.npy", train_data)
    np.save("dataset\\New_Data\\X_test_series.npy", test_data)
    np.save("dataset\\New_Data\\y_train.npy", train_label)
    np.save("dataset\\New_Data\\y_test.npy", test_label)

if __name__ == '__main__':
    file_path1 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\label_119node1.csv'
    turbine_data1 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\dataset_119node1.npy'
    file_path2 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\label_119node2.csv'
    turbine_data2 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\dataset_119node2.npy'
    file_path3 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\label_119node3.csv'
    turbine_data3 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\dataset_119node3.npy'
    # For No Impact Selection
    turbine_data4 = 'dataset\\Example_WTIL\\Impact_data_resource\\TIMdata\\dataset_1114.npy'
    # Obtain data
    turbine_data1, distance_data1 = load_data(turbine_data1, file_path1)
    turbine_data2, distance_data2 = load_data(turbine_data2, file_path2)
    turbine_data3, distance_data3 = load_data(turbine_data3, file_path3)
    turbine_data4 = np.load(turbine_data4).swapaxes(1, 2)
    # Distance Calibration
    distance_label1 = xyz_to_d(distance_data1)
    distance_label2 = xyz_to_d(distance_data2)
    distance_label3 = xyz_to_d(distance_data3)
    # Plot 3 Sensors Series
    # plot_series_data(distance_label1, distance_label2, distance_label3)
    # Define Class 11 Segmentation, Sampling Few-shot data
    num = 28  # number of each class
    data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11 = data_merge_sampling(turbine_data1, distance_label1, turbine_data2, distance_label2, turbine_data3, distance_label3,
                        turbine_data4, num=num)
    # Split Dataset and Save
    dataset_split(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, train_num=22, num=num)

