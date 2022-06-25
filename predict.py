import argparse
from awpy import DemoParser
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
from matplotlib import pyplot as plt
from PIL import Image
import glob
from utils import graphpos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Position Prediction')
    parser.add_argument("--layout", "-l", help="path to layout image")
    parser.add_argument("--demo", "-d", help="path to demo file")
    parser.add_argument("--side", "-s", help="side to predict positions, values: (ct or t)")
    parser.add_argument("--model", "-m", help="model type, values: (rnn or ltsm)")
    parser.add_argument("--output", "-o", help="output path for predicted images")
    parser.add_argument("--debug", "-g", help="OPTIONAL, whether to print debug info, 0 or 1", required=False)

    args = parser.parse_args()

    layout_path = args.layout
    demo_path = args.demo
    side = args.side
    model = args.model
    output_path = args.output

    assert side == "ct" or side == "t", f"Unexpected side argument passed: {side}"
    assert model == "rnn" or side == "ltsm", f"Unexpected model argument passed: {model}"

    demo_parser = DemoParser(demofile="infexample.dem", parse_rate=128)
    data = demo_parser.parse(return_type="json")
    all_rounds = data["gameRounds"]

    player_pos = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    print("[INFO] Extracting positions")
    for round in all_rounds:
        # temp = np.array([])
        for frame in round["frames"]:
            temp = np.array([])
            for p in frame[side]["players"]:
                temp = np.append(temp, p["x"])
                temp = np.append(temp, p["y"])
            # print(np.expand_dims(temp, axis=0))
            player_pos = np.row_stack((player_pos, np.expand_dims(temp, axis=0)))
            # print(t_player_pos)

    player_pos = np.delete(player_pos, 0, axis=0)

    print("[INFO] Normalizing data")
    MEAN = np.mean(player_pos)
    STD = np.std(player_pos)

    data = (player_pos - MEAN) / STD
    X = data[:-1]
    y = data[1:]

    n_points = len(player_pos)
    train_points = int(2 / 3 * n_points) + 1
    X_train, X_test = X[:train_points], X[train_points:]
    y_train, y_test = y[:train_points], y[train_points:]
    print("[INFO] We have", len(X_train), "training points and", X_test.shape[0], "testing points")

    X_train = X_train.reshape(X_train.shape[0], 1, 10)
    X_test = X_test.reshape(X_test.shape[0], 1, 10)

    if model == "rnn":
        print("[INFO] Using Simple RNN")
        model = Sequential()
        model.add(SimpleRNN(units=10, input_shape=(1, 10)))
        model.add(Dense(10))
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        print("[INFO] Using LSTM")
        model = Sequential()
        model.add(LSTM(10, input_shape=(1, 10)))
        model.add(Dense(10))
        model.compile(loss='mean_squared_error', optimizer='adam')

    print("[INFO] Training model")
    H = model.fit(X_train, y_train, epochs=100, batch_size=16)

    print("[INFO] Plotting loss, saved to loss.png")
    plt.plot(H.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss.png")

    print("[INFO] Predicting...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = y_train_pred.reshape(X_train.shape[0], 10)
    y_test_pred = y_test_pred.reshape(X_test.shape[0], 10)

    print("[INFO] Graphing the predictions")
    for idx, pos in enumerate(y_test_pred):
        graphpos(poses=pos, orig_std=STD, orig_mean=MEAN, debug=0, output_filename=output_path+"/out_" + str(idx) + ".png",
                 layout_path=layout_path)

    frames = []
    imgs = glob.glob(output_path + "/*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    model = args.model
    print("[INFO] Converting the png predictions to a gif file")
    # Save into a GIF file that loops forever
    frames[0].save(f'{model}_{side}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)

