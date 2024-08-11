import numpy as np
import tensorflow as tf
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

# Load new data
num_data = 500000
new_x = load_data('IndhRobot_ur3.txt', num_data, 6)  # Replace with actual new data file
new_y = load_data('OutdhRobot_ur3.txt', num_data, 3)


# Split new data into training and validation sets
x_train_new, x_val_new, y_train_new, y_val_new = train_test_split(new_x, new_y, test_size=0.2, random_state=42)

# Load the pre-trained model
model = tf.keras.models.load_model('best_model.keras', custom_objects={'PMP': PMP})

# Ensure all layers are trainable initially
for layer in model.layers:
    layer.trainable = True

# Recompile the model with a lower learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error')

# Define the number of layers in the model
max_layers = len(model.layers)

# Define a function to freeze the first N layers
def freeze_layers(model, num_layers):
    for layer in model.layers[:num_layers]:
        layer.trainable = False
    for layer in model.layers[num_layers:]:
        layer.trainable = True
    print(f"Freezing the first {num_layers} layers.")
    # Recompile the model with a lower learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error')

# Initialize variables
best_val_loss = np.Inf
patience = 100
wait = 0
layers_to_freeze = 0
losses = []  # List to store all loss values
val_losses = []  # List to store all validation loss values

# Define common callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='fine_tuned_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

epoch_logger = EpochLogger(print_every=50)

# Fine-tune the model on the new data
batch_size = num_neurons
epochs = 5000
# Measure training time
start_time = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    history = model.fit(
        x_train_new, y_train_new,
        validation_data=(x_val_new, y_val_new),
        epochs=1,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, epoch_logger],
        verbose=0  # Suppress all output
    )

    current_loss = history.history['loss'][0]
    current_val_loss = history.history['val_loss'][0]
    losses.append(current_loss)
    val_losses.append(current_val_loss)

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience and layers_to_freeze < max_layers:
            wait = 0
            layers_to_freeze += 1
            freeze_layers(model, layers_to_freeze)

    # Check if all layers were freezed
    if layers_to_freeze >= max_layers:
        print("All layers are freezed, stopping training.")
        break

# Calculate and print the total training time
end_time = time.time()
training_time_seconds = end_time - start_time
training_time_hours = training_time_seconds / 3600
print(f"Total training time: {training_time_hours:.2f} hours")
# Save the final fine-tuned model
model.save('last_tuned_model_ur3.keras')

# Save training history
with open('fine_tuning_losses.txt', 'w') as f:
    for epoch, loss, val_loss in zip(range(1, len(losses) + 1), losses, val_losses):
        f.write(f"Epoch {epoch}: Loss: {loss}, Val Loss: {val_loss}\n")
