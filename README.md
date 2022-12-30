# Train a keras model

You can train a keras model and
- save and restore the training process.
- get the same result as `model.fit()`.
- get the same result wherever restarting the training process.

## Usage

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

### Note

#### Callbacks
You can use customize callbacks and save their states. \
Please specify the names and the variables to save as follows.

```python
callbacks = MyCallback()
callback_variables = {"MyCallback": ["variable1", "variable2"]}

train_model(..., callbacks=callbacks, callback_variables=callback_variables, ...)
```

The variables `epoch` and `history` of the default callback `History` are saved without specification.
