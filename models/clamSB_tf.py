"""
Model CLAM SB
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input


class Attn_Net(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.dense1 = Dense(D, activation="tanh")
        self.dropout = Dropout(0.25) if dropout else None
        self.dense2 = Dense(n_classes)

    def call(self, x):
        x = self.dense1(x)
        if self.dropout:
            x = self.dropout(x)
        return self.dense2(x), x


class Attn_Net_Gated(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.dense1_a = Dense(D, activation="tanh")
        self.dense1_b = Dense(D, activation="sigmoid")
        self.dropout = Dropout(0.25) if dropout else None
        self.dense2 = Dense(n_classes)

    def call(self, x):
        a = self.dense1_a(x)
        b = self.dense1_b(x)
        A = tf.multiply(a, b)
        if self.dropout:
            A = self.dropout(A)
        return self.dense2(A), x


class CLAM_SB(tf.keras.Model):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        instance_eval=True,
        attention_only=False,
        return_features=False,
        bag_loss_weight=None,
        inst_loss_weight=None,
    ):
        super(CLAM_SB, self).__init__()
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.instance_eval = instance_eval
        self.attention_only = attention_only
        self.return_features = return_features
        self.bag_loss_weight = bag_loss_weight
        self.inst_loss_weight = inst_loss_weight

        size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = size_dict[size_arg]
        self.feature_extractor = ResNet50(
            weights="imagenet", include_top=False, pooling="avg"
        )
        self.dense = Dense(size[1], activation=tf.nn.relu)
        self.attention_net = (
            Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
            if gate
            else Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        )
        self.classifiers = Dense(2)
        self.instance_classifiers = Dense(2)

    @staticmethod
    def create_positive_targets(length):
        return tf.ones((length,), dtype=tf.int64)

    @staticmethod
    def create_negative_targets(length):
        return tf.zeros((length,), dtype=tf.int64)

    def inst_eval(self, A, h):

        # Get top & bottom k_sample attention scores
        top_p_ids = tf.math.top_k(A, k=self.k_sample)[-1][-1]
        top_p = tf.gather(h, top_p_ids)
        top_n_ids = tf.math.top_k(-A, k=self.k_sample)[-1][-1]
        top_n = tf.gather(h, top_n_ids)

        # Assign pseudo-labels
        p_targets = self.create_positive_targets(self.k_sample)
        n_targets = self.create_negative_targets(self.k_sample)
        all_targets = tf.concat([p_targets, n_targets], axis=0)
        all_instances = tf.concat([top_p, top_n], axis=0)

        # Get predictions for all instances
        logits = self.instance_classifiers(all_instances)
        all_preds = tf.math.top_k(logits, k=1)[-1]
        instance_loss = self.instance_loss_fn(all_targets, logits)

        return instance_loss, all_preds, all_targets

    def call(self, h, training=False):

        # h = self.feature_extractor(preprocess_input(x))
        h = self.dense(h)
        A, h = self.attention_net(h)
        A = tf.transpose(A)
        if self.attention_only:
            return A
        A_raw = A
        A = tf.nn.softmax(A, axis=1)

        if self.instance_eval:
            instance_loss, preds, targets = self.inst_eval(A, h)
            results_dict = {
                "inst_labels": targets,
                "inst_preds": preds,
                "instance_loss": instance_loss,
            }
        else:
            results_dict = {}

        M = tf.matmul(A, h)
        logits = self.classifiers(M)
        Y_hat = tf.math.top_k(logits, k=1)[-1]
        Y_prob = tf.nn.softmax(logits, axis=1)

        if self.return_features:
            results_dict.update({"features": M})
        self.return_dict = results_dict

        if training:
            return Y_prob, instance_loss
        else:
            return tf.argmax(Y_prob, axis=-1)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, instance_loss = self(x, training=True)
            bag_loss = self.compute_loss(y=y, y_pred=y_pred)
            loss = (
                self.bag_loss_weight * bag_loss + self.inst_loss_weight * instance_loss
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def model(args, params):
    model = CLAM_SB(
        k_sample=params["k_sample"],
        dropout=params["dropout"],
        bag_loss_weight=params["loss_weights"]["bag"],
        inst_loss_weight=params["loss_weights"]["instance"],
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=float(params["learning_rate"])),
        metrics=["accuracy"],
    )

    # model.build(input_shape=(args["batch_size"], *args["patch_dims"]))
    model.build(input_shape=(88, 1024))

    return model


def model_callbacks(args, params):
    return []
