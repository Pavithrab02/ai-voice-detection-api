# import numpy as np

# X = np.load("ml/features/X_signal.npy")
# print(X.shape)
# print(X[:5])

# import numpy as np
# langs = np.load("ml/features/lang_labels.npy")
# y = np.load("ml/features/y_labels.npy")

# for l in np.unique(langs):
#     print(l, np.bincount(y[langs == l]))



#to chk prediction of a file:
from inference.predict import predict

print(predict("ml/data/audio_clean/download.wav"))
