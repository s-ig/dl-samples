import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

TRAINING_DATA_LENGTH=2056
HIDDEN_UNITS_NUM=16
LEARNING_RATE=0.01
MAX_EPOCHS=200

def main():
    training_data=get_sin_data(TRAINING_DATA_LENGTH)
    model=build_model(HIDDEN_UNITS_NUM,LEARNING_RATE)
    print("learning...")
    model.fit(training_data["x"],training_data["y"],epochs=MAX_EPOCHS)
    predicting_x=np.arange(0, 2*np.pi, np.pi/512)
    print("predicting...")
    predicting_y=model.predict(predicting_x)
    plot(predicting_x,predicting_y)


#教師データを生成
def get_sin_data(length):
    x=np.random.rand(length)*2*np.pi
    y=np.sin(x)
    return {"x":x,"y":y}

#推論結果と実際のsin(x)をプロット
def plot(x,y):
    correct_y=np.sin(x)
    plt.figure(figsize=(16,9))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y,label="predicted")
    plt.plot(x,correct_y,label="sin(x)")
    plt.legend()
    plt.show()

#モデルを生成
def build_model(hidden_units_num,lr):
    model = Sequential()
    model.add(Dense(hidden_units_num,input_shape=(1,), activation='tanh'))#中間層
    model.add(Dense(1, activation='linear'))#出力層
    model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr))
    return model


if __name__ == "__main__":
    main()