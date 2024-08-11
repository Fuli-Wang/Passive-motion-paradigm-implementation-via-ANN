import tensorflow as tf
import numpy as np
import random

inputL=6
hiddenL1=48
hiddenL2=55
outputL=3
num_data = 50000


class PMP(tf.keras.Model):
    def __init__(self, units_layer1=48, units_layer2=55, **kwargs):
        super(PMP, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units_layer1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units_layer2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(3, activation='linear')

    def call(self, inputs):
        x1 = self.dense1(inputs)
        x2 = self.dense2(x1)
        output = self.dense3(x2)
        return output

    def get_config(self):
        config = super(PMP, self).get_config()
        config.update({
            'units_layer1': self.dense1.units,
            'units_layer2': self.dense2.units,
            'units_layer3': self.dense3.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_jack(self, Jan):
        h =np.zeros((hiddenL1,1))
        z =np.zeros((hiddenL1,1))
        p =np.zeros((hiddenL2,1))
        hinter =np.zeros((hiddenL1,1))
        p1 =np.zeros((hiddenL2,1))
        pinter =np.zeros((hiddenL2,1))
        Jack =np.zeros((outputL,inputL))
        JacT =[]

        weights1=self.dense1.get_weights()
        w1 = weights1[0].T
        b1 = weights1[1].T

        weights2=self.dense2.get_weights()
        w2 = weights2[0].T
        b2 = weights2[1].T

        weights3=self.dense3.get_weights()
        w3 = weights3[0].T
        b3 = weights3[1].T

        for u in range(hiddenL1):
          sum = 0
          for i in range(inputL):
            sum = sum+(w1[u][i]*Jan[i])
          h[u][0]=sum+b1[u]
          z[u][0] = np.maximum(0, h[u][0])  # ReLU activation    Change it based on the activation
          hinter[u][0] = 1.0 if h[u][0] > 0 else 0.0  # ReLU derivative

        for u in range(hiddenL2):
          sum2 = 0
          for i in range(hiddenL1):
            sum2=sum2+w2[u][i]*z[i][0]
          p[u][0]=sum2+b2[u]
          p1[u][0] = np.maximum(0, p[u][0])  # ReLU activation
          pinter[u][0] = 1.0 if p[u][0] > 0 else 0.0  # ReLU derivative

        for k in range(outputL):
          for n in range(inputL):
            inter1=0
            for a in range(hiddenL2):
                for b in range(hiddenL1):
                    inter1=inter1 +((w3[k][a]*pinter[a][0])*((w2[a][b]*hinter[b][0])*w1[b][n]))
            Jack[k][n]=inter1
        return Jack



def compute_loss(model, x, y):
    with tf.GradientTape() as tape:
        # 计算预测值
        y_pred = model(x)

        residual = tf.norm(y - y_pred, ord='euclidean')
        f_loss = tf.reduce_mean(residual)

    return f_loss


def train(model, x, y, optimizer, epochs):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x, y, qv, xv)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 1000 == 0:
            print('Epoch {} - Loss: {}'.format(epoch, loss))

        if loss<0.0001:
            break

def print_checkpoint(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
          f"value={reader.get_tensor(key)})")


try:
    f = open('IndhRobot.txt') #joint angles
    matrix = f.read().split()
    x = np.zeros((num_data,6))
    a = 0
    for i in range(num_data):
        for j in range(6):
            x[i][j] = matrix[a]
            a = a+1
except:
    print("Oops!  Cannot find the IndhRobot file.")

try:
    f1 = open('OutdhRobot.txt') #end position
    matrix1 = f1.read().split()
    y = np.zeros((num_data,3))
    a = 0
    for i in range(num_data):
        for j in range(3):
            y[i][j] = matrix1[a]
            a = a+1
except:
    print("Oops!  Cannot find the OutdhRobot file.")


model = PMP()

#model = tf.keras.models.load_model(
#    "my_model.keras",
#    custom_objects={"PMP": PMP},
#)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 进行训练
train(model, x, y, optimizer, epochs=20000)

#Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')

#Example
Jan = [-167.4, -86.4,  78.3,  73.8,  45.9, 50.4]
x=tf.constant([Jan])
y_p = model(x)
Jack = model.get_jack(Jan)
