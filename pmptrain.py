import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split

num_neurons = 128

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

    def get_jacobian(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.call(inputs)
        return tape.batch_jacobian(outputs, inputs)

# Function to load data
def load_data(file_path, num_rows, num_cols):
    data = np.zeros((num_rows, num_cols), dtype=np.float32)
    try:
        with open(file_path, 'r') as f:
            matrix = f.read().split()
            for i in range(num_rows):
                for j in range(num_cols):
                    data[i, j] = matrix[i * num_cols + j]
    except FileNotFoundError:
        print(f"Oops! Cannot find the file {file_path}.")
    return data


# Custom callback to print messages every 50 epochs
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, print_every=50):
        super(EpochLogger, self).__init__()
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every == 0:
            print(f"Epoch {epoch + 1}: loss = {logs['loss']}, val_loss = {logs['val_loss']}")

num_data = 500000 #adjust it based on your data size
x = load_data('IndhRobot_ur5.txt', num_data, 6) #each row is a data
y = load_data('OutdhRobot_ur5.txt', num_data, 3)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = PMP()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.85)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_squared_error')


# Define callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=0
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=250,
    verbose=1,
    restore_best_weights=True
)

epoch_logger = EpochLogger(print_every=50)

# Train the model
batch_size = num_neurons
epochs = 2000
# Measure training time
start_time = time.time()

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_callback, early_stopping_callback, epoch_logger],
    verbose=0  # Set to 0 to suppress detailed output
)

# Calculate and print the total training time
end_time = time.time()
training_time_seconds = end_time - start_time
training_time_hours = training_time_seconds / 3600
print(f"Total training time: {training_time_hours:.2f} hours")

# Save training history
with open('losses.txt', 'w') as f:
    for epoch, loss, val_loss in zip(range(1, epochs+1), history.history['loss'], history.history['val_loss']):
        f.write(f"Epoch {epoch}: Loss: {loss}, Val Loss: {val_loss}\n")

#Example
Jan = [-167.4, -86.4,  78.3,  73.8,  45.9, 50.4]
tf_Jan = tf.convert_to_tensor([Jan], dtype=tf.float32)
tf_Jack = model.get_jacobian(tf_Jan)
Jack = (tf_Jack.numpy()).squeeze(axis=0)
print(Jack)
