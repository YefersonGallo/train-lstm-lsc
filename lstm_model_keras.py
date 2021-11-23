from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import numpy as np
import os
from s3fs.core import S3FileSystem

import tensorflow_cloud as tfc

tfc.run(
    entry_point=None,
    distribution_strategy='auto',
    requirements_txt='requirements.txt',
    chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=2),
    worker_count=0)

s3 = S3FileSystem(anon=True)

actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z'])

bucket = 'dataset-lsc'

no_sequences = 2

sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

print(label_map)


sequences, labels = [], []
for action in actions:
    print(action)
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            key = 'MP_Data/{}/{}/{}.npy'.format(action, str(sequence), frame_num)
            res = np.load(s3.open('{}/{}'.format(bucket, key)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.08)

log_dir = os.path.join('Logs_Multiple_dataset')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='poisson', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=10, callbacks=[tb_callback])

model.summary()

res = model.predict(X_test)

print(actions[np.argmax(res[3])])

print(actions[np.argmax(y_test[3])])

model.save('action_new_dataset.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(accuracy_score(ytrue, yhat))
