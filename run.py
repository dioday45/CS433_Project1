import numpy as np
import src.helpers as hlp
from src.nn import NN
import src.nn as nn
import pickle

np.random.seed(42)

# defining paths
TEST_DATA_PATH = "data/test.csv"
MODEL0_PATH = "models/model0_3001_ep.pkl"
MODEL1_PATH = "models/model1_1501_ep.pkl"
MODEL2_PATH = "models/model2_1501_ep.pkl"

# specify NNs architectures

nn_architecture0 = [
    {"input_dim": 36, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 2, "activation": "softmax"},
]
nn_architecture1 = [
    {"input_dim": 44, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 2, "activation": "softmax"},
]
nn_architecture2 = [
    {"input_dim": 58, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 2, "activation": "softmax"},
]

# load test data
print("\n-----loading test data-----\n")
test_data_raw0, test_data_raw1, test_data_raw2 = hlp.load_split_data(
    "data/test.csv", one_hot=True
)
test_data0, test_data1, test_data2 = hlp.process_data(
    test_data_raw0, test_data_raw1, test_data_raw2
)
del test_data_raw0
del test_data_raw1
del test_data_raw2

id0, test_data0 = (
    test_data0[:, 0].astype("int32"),
    np.delete(test_data0, [0, 1], axis=1),
)
id1, test_data1 = (
    test_data1[:, 0].astype("int32"),
    np.delete(test_data1, [0, 1], axis=1),
)
id2, test_data2 = (
    test_data2[:, 0].astype("int32"),
    np.delete(test_data2, [0, 1], axis=1),
)

# loading saved weights to each model
print("\n-----loading models-----\n")
model0 = NN(nn_architecture0)
params0 = pickle.load(open(MODEL0_PATH, "rb"))
model0.init_weights()
model0.params = params0

model1 = NN(nn_architecture1)
params1 = pickle.load(open(MODEL1_PATH, "rb"))
model1.init_weights()
model1.params = params1

model2 = NN(nn_architecture2)
params2 = pickle.load(open(MODEL2_PATH, "rb"))
model2.init_weights()
model2.params = params2

# making predictions
print("\n-----making predictions-----\n")
logits_test0 = model0.forward(test_data0)
y_test0_preds = np.argmax(logits_test0, axis=1)

logits_test1 = model1.forward(test_data1)
y_test1_preds = np.argmax(logits_test1, axis=1)

logits_test2 = model2.forward(test_data2)
y_test2_preds = np.argmax(logits_test2, axis=1)

# formatting for AIcrowd
y_test0_preds[y_test0_preds == 0] = -1
y_test1_preds[y_test1_preds == 0] = -1
y_test2_preds[y_test2_preds == 0] = -1

# create csv
print("\n-----saving predictions-----\n")
ids = np.concatenate((id0, id1, id2))
preds = np.concatenate((y_test0_preds, y_test1_preds, y_test2_preds))
hlp.create_csv_submission(ids, preds, "run_script_predictions.csv")
print("-----done-----")
