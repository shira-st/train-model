# Save and restore training of a keras model

Train a keras model
- The result is the same as `model.fit()`.
- The training process can be saved and restored.
- The same result is obtained from wherever restarting the training process.

## Example

### Preparation
```python
import numpy as np
import tensorflow as tf
from train_model import train_model

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def create_sample_dataset():
    x = np.random.random((100, 10))
    y = np.random.randint(0, 2, (100, 1))
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((x, y)).shuffle(100).batch(10)
    return dataset


def create_sample_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
    return model
```

### Training by `model.fit()`
```python
seed = 0
set_seed(seed)
dataset = create_sample_dataset()
model = create_sample_model()

epochs = 5
history = model.fit(dataset, epochs=epochs)
print(history.history)
```
```
Epoch 1/5
10/10 [==============================] - 0s 776us/step - loss: 2.1481 - accuracy: 0.5100
Epoch 2/5
10/10 [==============================] - 0s 776us/step - loss: 1.9838 - accuracy: 0.4900
Epoch 3/5
10/10 [==============================] - 0s 720us/step - loss: 1.8324 - accuracy: 0.4900
Epoch 4/5
10/10 [==============================] - 0s 776us/step - loss: 1.8124 - accuracy: 0.4900
Epoch 5/5
10/10 [==============================] - 0s 1ms/step - loss: 1.8042 - accuracy: 0.5000
{'loss': [2.1480836868286133, 1.9837868213653564, 1.8324090242385864, 1.812350869178772, 1.8041961193084717], 'accuracy': [0.5099999904632568, 0.49000000953674316, 0.49000000953674316, 0.49000000953674316, 0.5]}
```

### Training by `train_model()`
```python
seed = 0
set_seed(seed)
dataset = create_sample_dataset()
model = create_sample_model()

epochs = 5
step_period = 3
epoch_period = 2
history = train_model(dataset, model, epochs, step_period=step_period, epoch_period=epoch_period)
print(history.history)
```
```
Epoch 1/5
10/10 [==============================] - 0s 30ms/step - loss: 2.1481 - accuracy: 0.5100
Epoch 2/5
10/10 [==============================] - 0s 10ms/step - loss: 1.9838 - accuracy: 0.4900
Epoch 3/5
10/10 [==============================] - 0s 12ms/step - loss: 1.8324 - accuracy: 0.4900
Epoch 4/5
10/10 [==============================] - 0s 16ms/step - loss: 1.8124 - accuracy: 0.4900
Epoch 5/5
10/10 [==============================] - 0s 13ms/step - loss: 1.8042 - accuracy: 0.5000
{'loss': [2.1480836868286133, 1.9837868213653564, 1.8324090242385864, 1.812350869178772, 1.8041961193084717], 'accuracy': [0.5099999904632568, 0.49000000953674316, 0.49000000953674316, 0.49000000953674316, 0.5]}
```

### Restart the training process
```python
seed = 0
set_seed(seed)
dataset = create_sample_dataset()
model = create_sample_model()

epochs = 5
initial_step = 3
initial_epoch = 2
history = train_model(dataset, model, epochs, initial_step=initial_step, initial_epoch=initial_epoch)
print(history.history)
```
```
Epoch 3/5
10/10 [==============================] - 0s 21ms/step - loss: 1.8324 - accuracy: 0.4900
Epoch 4/5
10/10 [==============================] - 0s 537us/step - loss: 1.8124 - accuracy: 0.4900
Epoch 5/5
10/10 [==============================] - 0s 798us/step - loss: 1.8042 - accuracy: 0.5000
{'loss': [2.1480836868286133, 1.9837868213653564, 1.8324090242385864, 1.812350869178772, 1.8041961193084717], 'accuracy': [0.5099999904632568, 0.49000000953674316, 0.49000000953674316, 0.49000000953674316, 0.5]}
```