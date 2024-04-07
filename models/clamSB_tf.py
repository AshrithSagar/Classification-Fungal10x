"""
Model CLAM SB
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class Attn_Net(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [Dense(D, activation="tanh")]

        if dropout:
            self.module.append(Dropout(0.25))

        self.module.append(Dense(n_classes))
        self.module = Sequential(self.module)

    def call(self, x):
        return self.module(x), x


class Attn_Net_Gated(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [tf.keras.layers.Dense(D, activation="tanh")]
        self.attention_b = [tf.keras.layers.Dense(D, activation="sigmoid")]

        if dropout:
            self.attention_a.append(Dropout(0.25))
            self.attention_b.append(Dropout(0.25))

        self.attention_a = Sequential(self.attention_a)
        self.attention_b = Sequential(self.attention_b)
        self.attention_c = Dense(n_classes)

    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = tf.multiply(a, b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(tf.keras.Model):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        subtyping=False,
    ):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [Dense(size[1], activation=tf.nn.relu)]
        if dropout:
            fc.append(Dropout(0.25))
        self.attention_net = (
            Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
            if gate
            else Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        )
        fc.append(self.attention_net)
        self.attention_net = Sequential(fc)
        self.classifiers = Dense(n_classes)
        instance_classifiers = [Dense(2) for _ in range(n_classes)]
        self.instance_classifiers = Sequential(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length):
        return tf.ones((length,), dtype=tf.int64)

    @staticmethod
    def create_negative_targets(length):
        return tf.zeros((length,), dtype=tf.int64)

    def inst_eval(self, A, h, classifier):

        top_p_ids = tf.math.top_k(A, k=self.k_sample)[-1][-1]
        top_p = tf.gather(h, top_p_ids)

        top_n_ids = tf.math.top_k(-A, k=self.k_sample)[-1][-1]
        top_n = tf.gather(h, top_n_ids)

        p_targets = self.create_positive_targets(self.k_sample)
        n_targets = self.create_negative_targets(self.k_sample)

        all_targets = tf.concat([p_targets, n_targets], axis=0)
        all_instances = tf.concat([top_p, top_n], axis=0)

        logits = classifier(all_instances)
        all_preds = tf.math.top_k(logits, k=1)[-1].squeeze()

        instance_loss = self.instance_loss_fn(all_targets, logits)

        return instance_loss, all_preds, all_targets

    def call(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        training=False,
    ):
        A, h = self.attention_net(h)
        A = tf.transpose(A)
        if attention_only:
            return A
        A_raw = A
        A = tf.nn.softmax(A, axis=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = tf.one_hot(label, depth=self.n_classes)
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[:, i]
                classifier = self.instance_classifiers[i]
                if tf.reduce_any(inst_label == 1):
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.numpy())
                    all_targets.extend(targets.numpy())
                    total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = tf.matmul(A, h, transpose_a=True)
        logits = self.classifiers(M)
        Y_hat = tf.math.top_k(logits, k=1)[-1]
        Y_prob = tf.nn.softmax(logits, axis=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})

        # return logits, Y_prob, Y_hat, A_raw, results_dict

        if training:
            return Y_prob
        else:
            return tf.argmax(Y_prob, axis=-1)


def model(args, params):
    model = CLAM_SB(
        k_sample=params["k_sample"],
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=float(params["learning_rate"])),
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, 256, 256, 3))

    return model


def model_callbacks(args, params):
    return []
