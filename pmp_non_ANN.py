num_neurons = 512

import tensorflow as tf
import torch
import numpy as np

class PMP(tf.keras.Model):
    def __init__(self, units_layer1=num_neurons, units_layer2=num_neurons, units_layer3=num_neurons, units_layer4=num_neurons, **kwargs):
        super(PMP, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units_layer1, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(units_layer2, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(units_layer3, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(units_layer4, activation='tanh')
        self.dense5 = tf.keras.layers.Dense(3, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)


keras_model = tf.keras.models.load_model(
    "fine_tuned_model.keras",
    custom_objects={"PMP": PMP},
)


input_dim = 6

import torch
import torch.nn as nn
from torch.nn import functional as F

class PMP2(nn.Module):
    def __init__(self, num_neurons):
        super(PMP2, self).__init__()
        self.dense1 = nn.Linear(6, num_neurons)
        self.dense2 = nn.Linear(num_neurons, num_neurons)
        self.dense3 = nn.Linear(num_neurons, num_neurons)
        self.dense4 = nn.Linear(num_neurons, num_neurons)
        self.dense5 = nn.Linear(num_neurons, 3)

    def forward(self, x):
        x = F.sigmoid(self.dense1(x)) #selu
        x = F.tanh(self.dense2(x))
        x = F.tanh(self.dense3(x))
        x = F.tanh(self.dense4(x))
        return self.dense5(x)

pytorch_model = PMP2(num_neurons)


def convert_dense(keras_layer, torch_layer):
    weights = keras_layer.get_weights()
    keras_w, keras_b = weights[0], weights[1]
    torch_layer.weight.data = torch.tensor(keras_w.T, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(keras_b, dtype=torch.float32)


convert_dense(keras_model.layers[0], pytorch_model.dense1)
convert_dense(keras_model.layers[1], pytorch_model.dense2)
convert_dense(keras_model.layers[2], pytorch_model.dense3)
convert_dense(keras_model.layers[3], pytorch_model.dense4)
convert_dense(keras_model.layers[4], pytorch_model.dense5)


# save
torch.save(pytorch_model.state_dict(), "fine_tuned_model.pth")

#verify
q=[90, -90, -90, -100, -100, 0]

u=tf.constant([q])
a = keras_model(u)
x = a.numpy()
print(x[0])

u = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
pytorch_model.eval()
with torch.no_grad():
  a = pytorch_model(u)
x = a.numpy()
print( x[0])
