import os
import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


class CheckpointManager():
    def __init__(self, ckpt_dir, callback_attributes):
        self.ckpt_dir = ckpt_dir
        self.callback_attributes = callback_attributes

    def _get_ckpt_path(self, epoch, step, name):
        ckpt_format = os.path.join(self.ckpt_dir, "ckpt_{:04d}_{:04d}/{}")
        return ckpt_format.format(epoch, step, name)

    # save dataset
    def save_iterator(self, epoch, step, iterator):
        ckpt_path = self._get_ckpt_path(epoch, step, "iterator")
        ckpt = tf.train.Checkpoint(iterator)
        ckpt.write(ckpt_path)
        return

    # restore dataset
    def restore_iterator(self, epoch, step, iterator):
        ckpt_path = self._get_ckpt_path(epoch, step, "iterator")
        ckpt = tf.train.Checkpoint(iterator)
        ckpt.read(ckpt_path).assert_consumed()
        return

    # save model weights
    def save_model(self, epoch, step, model):
        ckpt_path = self._get_ckpt_path(epoch, step, "model")
        model.save_weights(ckpt_path)
        return

    # restore model weights
    def restore_model(self, epoch, step, model):
        ckpt_path = self._get_ckpt_path(epoch, step, "model")
        model.load_weights(ckpt_path)
        return

    # save metrics
    def save_metrics(self, epoch, step, model):
        ckpt_path = self._get_ckpt_path(epoch, step, "metrics")
        ckpt = tf.train.Checkpoint()
        ckpt.metrics = model.metrics
        ckpt.save(ckpt_path)
        return

    # restore metrics
    def restore_metrics(self, epoch, step, model):
        ckpt_path = self._get_ckpt_path(epoch, step, "metrics-1")
        ckpt = tf.train.Checkpoint()

        # create initialized metrics
        restored = []
        for metric in model.metrics:
            name = metric.name
            if name == "loss":
                name = "Mean"
            name = "".join([w.capitalize() for w in name.split("_")])
            restored.append(tf.keras.metrics.get(name))

        # restore metrics
        ckpt.metrics = restored
        ckpt.restore(ckpt_path)

        # merge states and update logs
        logs = {}
        for m1, m2 in zip(model.metrics, restored):
            m1.merge_state([m2])
            logs[m1.name] = m1.result()
        return logs

    # save callbacks
    def save_callbacks(self, epoch, step, callbacks):
        ckpt_path = self._get_ckpt_path(epoch, step, "callbacks.pkl")
        data = {}
        for callback in callbacks.callbacks:
            name = callback.__class__.__name__
            if name in self.callback_attributes.keys():
                data[name] = {}
                for a in self.callback_attributes[name]:
                    data[name][a] = getattr(callback, a)
        joblib.dump(data, ckpt_path)
        return

    # restore callbacks
    def restore_callbacks(self, epoch, step, callbacks):
        ckpt_path = self._get_ckpt_path(epoch, step, "callbacks.pkl")
        restored = joblib.load(ckpt_path)
        for callback in callbacks.callbacks:
            name = callback.__class__.__name__
            if name in restored.keys():
                for k, v in restored[name].items():
                    setattr(callback, k, v)
        return


class ModelTrainer():
    """
    model
    - model to train.

    callback_attributes
    - Dictionary which shows attributes of callbacks to be saved and restored.
    - Each key is a class name of a callback.
    - Each value is a list of names of attributes.

    initial_epoch
    - Epoch where to start training.

    initial_step
    - Step where to start training.

    epoch_period
    - Epoch interval to create checkpoints.

    step_period
    - Step interval to create checkpoints.
    """

    def __init__(
        self,
        model,
        ckpt_dir,
        callback_attributes={},
        initial_epoch=0,
        initial_step=0,
        epoch_period=0,
        step_period=0
    ):
        self.model = model
        callback_attributes["History"] = ["epoch", "history"]
        self.ckpt_manager = CheckpointManager(ckpt_dir, callback_attributes)
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.epoch_period = epoch_period
        self.step_period = step_period

    def fit(
        self,
        data,
        epochs,
        callbacks=None,
        validation_data=None
    ):
        iterator = iter(data)

        # restore dataset and model weights before creating CallbackList
        if (self.initial_epoch != 0) or (self.initial_step != 0):
            self.ckpt_manager.restore_iterator(
                self.initial_epoch, self.initial_step, iterator
            )
            self.ckpt_manager.restore_model(
                self.initial_epoch, self.initial_step, self.model
            )

        steps = len(data)
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=True,
            model=self.model,
            epochs=epochs,
            steps=steps,
            verbose=True
        )

        callbacks.on_train_begin()
        train_fn = self.model.make_train_function()

        # restore callbacks after calling on_train_begin
        if (self.initial_epoch != 0) or (self.initial_step != 0):
            self.ckpt_manager.restore_callbacks(
                self.initial_epoch, self.initial_step, callbacks
            )

        logs = None
        for epoch in range(self.initial_epoch, epochs):

            # initialize states
            if self.initial_step == 0:
                iterator = iter(data)
                self.model.reset_metrics()

            callbacks.on_epoch_begin(epoch)

            for step in range(self.initial_step, steps):
                callbacks.on_train_batch_begin(step)
                logs = train_fn(iterator)

                # restore metrics after calling the first train_fn
                if (step != 0) and (step == self.initial_step):
                    logs = self.ckpt_manager.restore_metrics(
                        self.initial_epoch, self.initial_step, self.model
                    )

                callbacks.on_train_batch_end(step, logs)

                # save states every step_period steps
                if (
                    (self.step_period != 0)
                    and ((step + 1) % self.step_period == 0)
                ):
                    self.ckpt_manager.save_iterator(
                        epoch, step + 1, iterator
                    )
                    self.ckpt_manager.save_model(
                        epoch, step + 1, self.model
                    )
                    self.ckpt_manager.save_metrics(
                        epoch, step + 1, self.model
                    )
                    self.ckpt_manager.save_callbacks(
                        epoch, step + 1, callbacks
                    )

            # reset initial_step after the first epoch
            if self.initial_step != 0:
                self.initial_step = 0

            # validation step
            if validation_data:
                val_logs = self.model.evaluate(
                    validation_data, callbacks=callbacks, return_dict=True
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                logs.update(val_logs)

            callbacks.on_epoch_end(epoch, logs)

            # save states every epoch_period epochs
            if (
                (self.epoch_period != 0)
                and ((epoch + 1) % self.epoch_period == 0)
            ):
                self.ckpt_manager.save_iterator(
                    epoch + 1, 0, iterator
                )
                self.ckpt_manager.save_model(
                    epoch + 1, 0, self.model
                )
                self.ckpt_manager.save_metrics(
                    epoch + 1, 0, self.model
                )
                self.ckpt_manager.save_callbacks(
                    epoch + 1, 0, callbacks
                )

        callbacks.on_train_end(logs=logs)
        return self.model.history
