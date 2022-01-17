from s3fs.core import S3FileSystem
import numpy as np

s3 = S3FileSystem(anon=True)

bucket = 'lsc-dataset'
action = "ñ"
sequence = "231"
frame_num = "20"

key = '{}/{}/{}.npy'.format(action, str(sequence), frame_num)
res = np.load(s3.open('{}/{}'.format(bucket, key)))
print(res)