import os
import joblib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


class CheckpointManager():
    def __init__(self, ckpt_dir):
        self.ckpt_path = os.path.join(ckpt_dir, "ckpt_{epoch:04d}_{step:04d}", "{name}")

    # save a dataset
    def save_iterator(self, iterator, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="iterator")
        ckpt = tf.train.Checkpoint(iterator)
        ckpt.write(path)
        return

    # restore a dataset
    def restore_iterator(self, iterator, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="iterator")
        ckpt = tf.train.Checkpoint(iterator)
        ckpt.read(path).assert_consumed()
        return

    # save model weights
    def save_model(self, model, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="model")
        model.save_weights(path)
        return

    # restore model weights
    def restore_model(self, model, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="model")
        model.load_weights(path)
        return

    # save metrics
    def save_metrics(self, model, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="metrics")
        ckpt = tf.train.Checkpoint()
        ckpt.metrics = model.metrics
        ckpt.save(path)
        return

    # restore metrics
    def restore_metrics(self, model, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="metrics-1")
        ckpt = tf.train.Checkpoint()

        # create initialized metrics
        restored = []
        for metric in model.metrics:
            name = metric.name
            if name == "loss":
                name = "Mean"
            name = "".join([w.capitalize() for w in  name.split("_")])
            restored.append(tf.keras.metrics.get(name))

        # restore metrics
        ckpt.metrics = restored
        ckpt.restore(path)

        # merge states and update logs
        logs = {}
        for m1, m2 in zip(model.metrics, restored):
            m1.merge_state([m2])
            logs[m1.name] = m1.result()
        return logs

    # save a history
    def save_history(self, callbacks, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="history.pkl")

        # get a history object from callbacks
        history = None
        for callback in callbacks.callbacks:
            if callback.__class__.__name__ == "History":
                history = callback
                break
        joblib.dump({"epoch": history.epoch, "history": history.history}, path)
        return

    # restore a history
    def restore_history(self, callbacks, epoch, step):
        path = self.ckpt_path.format(epoch=epoch, step=step, name="history.pkl")
        restored = joblib.load(path)

        for callback in callbacks.callbacks:
            if callback.__class__.__name__ == "History":
                callback.epoch = restored["epoch"]
                callback.history = restored["history"]
                break
        return
    

def train_model(data, model, epochs, validation_data=None, initial_epoch=0, initial_step=0, epoch_period=0, step_period=0, ckpt_dir="checkpoint"):
    cm = CheckpointManager(ckpt_dir)

    iterator = iter(data)

    # restore the dataset and the model weights before creating callbacks
    if initial_epoch != 0 or initial_step != 0:
        cm.restore_iterator(iterator, initial_epoch, initial_step)
        cm.restore_model(model, initial_epoch, initial_step)

    steps = len(data)
    callbacks = tf.keras.callbacks.CallbackList(add_history=True, add_progbar=True, model=model, epochs=epochs, steps=steps, verbose=True)

    callbacks.on_train_begin()
    train_fn = model.make_train_function()

    # restore the hisotry after calling callbacks.on_train_begin()
    if initial_epoch != 0 or initial_step != 0:
        cm.restore_history(callbacks, initial_epoch, initial_step)

    logs = None
    for epoch in range(initial_epoch, epochs):

        # initialize states 
        if initial_step == 0:
            iterator = iter(data)
            model.reset_metrics()
        
        callbacks.on_epoch_begin(epoch)

        for step in range(initial_step, steps):
            callbacks.on_train_batch_begin(step)
            logs = train_fn(iterator)

            # restore the metrics and update the logs after calling the first train_fn()
            if initial_step != 0 and step == initial_step:
                logs = cm.restore_metrics(model, initial_epoch, initial_step)

            callbacks.on_train_batch_end(step + 1, logs)
            
            # save information every "epochs_period" epochs
            if step_period != 0 and (step + 1) % step_period == 0:
                cm.save_iterator(iterator, epoch, step + 1)
                cm.save_model(model, epoch, step + 1)
                cm.save_metrics(model, epoch, step + 1)
                cm.save_history(callbacks, epoch, step + 1)

        # reset after the first epoch
        if initial_step != 0:
            initial_step = 0

        # validation step
        if validation_data:
            val_logs = model.evaluate(validation_data, callbacks=callbacks, return_dict=True)
            val_logs = {"val_" + name: val for name, val in val_logs.items()}
            logs.update(val_logs)

        callbacks.on_epoch_end(epoch, logs)

        # save information every "epochs_period" epochs
        if epoch_period != 0 and (epoch + 1) % epoch_period == 0:
            cm.save_iterator(iterator, epoch + 1, 0)
            cm.save_model(model, epoch + 1, 0)
            cm.save_metrics(model, epoch + 1, 0)
            cm.save_history(callbacks, epoch + 1, 0)

    callbacks.on_train_end(logs=logs)
    return model.history
