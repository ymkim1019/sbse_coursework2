import argparse
import numpy as np
import pandas as pd
import sys
import os
import glob
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

def load_dataset(data_files):
    faulty_functions = []
    normal_functions = []

    for f in data_files:
        df = pd.read_csv(f)
        data = df.values

        for each in data:
            if each[42] == 1:
                faulty_functions.append(each)
            else:
                normal_functions.append(each)

    return np.vstack(faulty_functions), np.vstack(normal_functions)


# def main(args):
def main():
    # windows
    is_windows = sys.platform.startswith('win')
    if is_windows:
        dataset_path = os.getcwd() + '\\fluccs_data\\'
    # linux
    else:
        dataset_path = os.getcwd() + '/fluccs_data/'
    extension = 'csv'
    files = [i for i in glob.glob(dataset_path + '*.{}'.format(extension))]

    faulty_dataset, normal_dataset = load_dataset(files)
    # faulty_dataset, normal_dataset = load_dataset(args.data_files)
    imbalance_ratio = np.ceil(normal_dataset.shape[0] / faulty_dataset.shape[0])
    faulty_dataset = np.tile(faulty_dataset, (int(imbalance_ratio - 1), 1))
    data = np.vstack([faulty_dataset, normal_dataset])
    np.random.shuffle(data)
    x_train = data[:, 1:42]
    y_train = np.reshape(data[:, -1], (data.shape[0],1))
    print(x_train.shape, y_train.shape)

    model = Sequential()
    model.add(Dense(16, input_dim=41, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]

    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2, callbacks=callbacks_list)

    model.save('my_model.h5')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_files', type=str, nargs='*')
    # args = parser.parse_args()
    #
    # main(args)
    #
    main()