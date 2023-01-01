# Train a keras model

You can train a keras model and
- save and restore the training process.
- get the same result as `model.fit()`.
- get the same result wherever restarting the training process.

## Usage
Train a model while creating checkpoints every epoch and every two steps.

```python
from train import ModelTrainer


ckpt_dir = "checkpoint"

trainer = ModelTrainer(model, ckpt_dir, epoch_period=1, step_period=2)
history = trainer.fit(data, epochs=2)
```

Restart training from the second step of the first epoch.

```python
from train import ModelTrainer


ckpt_dir = "checkpoint"

trainer = ModelTrainer(model, ckpt_dir, initial_epoch=1, initial_step=2)
history = trainer.fit(data, epochs=2)
```

## Note

### Callbacks
You can use customized callbacks and save their states. \
Please specify the names and the attributes to save as follows.

```python
from train import ModelTrainer


callbacks = MyCallback()
callback_attributes = {"MyCallback": ["attribute1", "attribute2"]}

trainer = ModelTrainer(..., callback_attributes=callback_attributes)
history = trainer.fit(data, epochs=2, callbacks=[callbacks])
```

The attributes `epoch` and `history` of the default callback `History` are saved without specification.

### Transfer learning
Models which contain pretrained models may not give the same results as `model.fit()`.
I have observed that some of the pretrained models provided by `tf.keras.applications` do not give the same results, 
and that some of the models provided in TensorFlow Hub give the same results. 
So, when using a pretrained model, compare it with `model.fit()` to be sure.