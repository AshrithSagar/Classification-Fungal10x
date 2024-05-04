"""
Model CLAM SB
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
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
        instance_eval=True,
        attention_only=False,
    ):
        super(CLAM_SB, self).__init__()
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_eval = instance_eval
        self.attention_only = attention_only

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
        self.bag_classifier = Dense(2, name="bag")
        self.instance_classifier = Dense(2, name="instance")

    @staticmethod
    def create_positive_targets(length):
        return tf.ones((length,), dtype=tf.int64)

    @staticmethod
    def create_negative_targets(length):
        return tf.zeros((length,), dtype=tf.int64)

    def inst_eval(self, A, h):

        # Get top & bottom k_sample attention scores
        topk_p = tf.math.top_k(A.reshape(-1), k=self.k_sample)
        topk_n = tf.math.top_k(-A.reshape(-1), k=self.k_sample)
        top_p = tf.gather(h, topk_p.indices, axis=1)
        top_n = tf.gather(h, topk_n.indices, axis=1)
        all_instances = tf.concat([top_p, top_n], axis=0)

        logits = self.instance_classifier(all_instances)
        logits = logits.reshape((2 * self.k_sample, 2))
        y_pred_instance = tf.math.top_k(logits, k=1)[-1]
        return logits

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
            logits_instance = self.inst_eval(A, h)

        A = A.reshape((1, 88))
        h = h.reshape((88, 512))
        features_bag = tf.matmul(A, h)

        logits_bag = self.bag_classifier(features_bag)
        Y_hat_bag = tf.math.top_k(logits_bag, k=1)[-1]
        Y_prob_bag = tf.nn.softmax(logits_bag, axis=1)

        if training:
            return logits_bag, logits_instance
        else:
            return tf.argmax(Y_prob_bag, axis=-1)

    def train_step(self, data):
        X_bag_instances, y_true_bag = data
        with tf.GradientTape() as tape:
            y_pred_bag, y_pred_instance = self(X_bag_instances, training=True)
            loss = self.loss.call(y_true_bag, y_pred_bag, y_pred_instance)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y_true_bag, y_pred_bag)
        if self.instance_eval:
            self.compiled_metrics.update_state(y_true_bag, y_pred_instance)

        return {m.name: m.result() for m in self.metrics}


class CLAMLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        bag_loss_fn=BinaryCrossentropy(from_logits=True),
        instance_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        bag_weight=0.6,
        instance_weight=0.4,
        k_sample=8,
    ):
        super().__init__()
        self.bag_loss_fn = bag_loss_fn
        self.instance_loss_fn = instance_loss_fn
        self.bag_weight = bag_weight
        self.instance_weight = instance_weight
        self.k_sample = k_sample

    @staticmethod
    def create_positive_targets(length):
        return tf.ones((length,), dtype=tf.int64)

    @staticmethod
    def create_negative_targets(length):
        return tf.zeros((length,), dtype=tf.int64)

    def call(self, y_true_bag, y_pred_bag, y_pred_instance):
        # Assign pseudo-labels to instances
        p_targets = tf.cond(
            tf.equal(tf.reduce_mean(y_true_bag), 1),
            lambda: self.create_positive_targets(self.k_sample),
            lambda: self.create_negative_targets(self.k_sample),
        )
        n_targets = self.create_negative_targets(self.k_sample)
        y_pseudo_instance = tf.concat([p_targets, n_targets], axis=0)

        bag_loss = self.bag_loss_fn(y_true_bag, y_pred_bag)
        instance_loss = self.instance_loss_fn(y_pseudo_instance, y_pred_instance)

        combined_loss = (
            self.bag_weight * bag_loss + self.instance_weight * instance_loss
        )
        return combined_loss


def model(args, params):

    model = CLAM_SB(
        k_sample=params["k_sample"],
        dropout=params["dropout"],
    )
    metrics = ["accuracy"]
    optimizer = Adam(learning_rate=float(params["learning_rate"]))
    loss = CLAMLoss(
        bag_loss_fn=BinaryCrossentropy(),
        instance_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        bag_weight=params["loss_weights"]["bag"],
        instance_weight=params["loss_weights"]["instance"],
        k_sample=params["k_sample"],
    )

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    # model.build(input_shape=(args["batch_size"], *args["patch_dims"]))
    model.build(input_shape=(None, 88, 1024))

    return model


def model_callbacks(args, params):
    return []
