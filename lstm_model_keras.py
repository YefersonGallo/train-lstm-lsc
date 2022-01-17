from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import numpy as np
import os
import datetime
from s3fs.core import S3FileSystem

import tensorflow_cloud as tfc

# Note: Please change the gcp_bucket to your bucket name.
gcp_bucket = "train-lsc"

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

checkpoint_path = os.path.join("gs://", gcp_bucket, "lsc-train-2", "epochs", "save_at_{epoch}")

tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
    "gs://", gcp_bucket, "logs-2", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

callbacks = [
    # TensorBoard will store logs for each epoch and graph performance for us.
    TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    # ModelCheckpoint will save models after each epoch for retrieval later.
    ModelCheckpoint(checkpoint_path),
    # EarlyStopping will terminate training when val_loss ceases to improve.
    EarlyStopping(monitor="loss", baseline=0.8),
]

actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'ñ', 'u', 'v', 'w', 'z'])

bucket = 'lsc-dataset'

no_sequences = 300

sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    print(action)
    for sequence in range(no_sequences):
        print(sequence)
        window = []
        for frame_num in range(sequence_length):
            key = '{}/{}/{}.npy'.format(action, str(sequence), frame_num)
            res = np.load(s3.open('{}/{}'.format(bucket, key)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.08)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=10000, callbacks=callbacks)

save_path = os.path.join("gs://", gcp_bucket, "lsc-train-2", "model")

if tfc.remote():
    model.save(save_path)
