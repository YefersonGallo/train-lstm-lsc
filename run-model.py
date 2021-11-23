import tensorflow_cloud as tfc

tfc.run(entry_point='lstm_model_keras.py', chief_config=tfc.COMMON_MACHINE_CONFIGS['T4_4X'])