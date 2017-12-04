import argparse
import numpy as np
import pandas as pd
from keras.models import load_model

def load_dataset(data_file):
    df = pd.read_csv(data_file)
    return df.values

def main(args):
    model = load_model(args.model_file)

    for f in args.data_files:
        data = load_dataset(f)
        pred_y = model.predict(data[:,1:42], batch_size=32)
        print(np.argmax(pred_y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('data_files', type=str, nargs='*')
    args = parser.parse_args()

    main(args)