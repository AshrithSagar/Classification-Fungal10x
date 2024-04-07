"""
Model CLAM SB
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.optimizers import Adam


class Attn_Net(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            Dense(D, activation=tf.nn.tanh),
        ]

        if dropout:
            self.module.append(Dropout(0.25))

        self.module.append(Dense(n_classes))

        self.module = Sequential(self.module)

    def call(self, x):
        return self.module(x), x


class Attn_Net_Gated(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            Dense(D, activation=tf.nn.tanh),
        ]

        self.attention_b = [
            Dense(D, activation=tf.nn.sigmoid),
        ]
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
        instance_loss_fn=CategoricalCrossentropy(),
        subtyping=False,
    ):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [
            Dense(size[1], activation=tf.nn.relu),
        ]
        if dropout:
            fc.append(Dropout(0.25))
        if gate:
            self.attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            self.attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        fc.append(self.attention_net)
        self.attention_net = Sequential(fc)
        self.classifiers = Dense(n_classes)
        self.instance_classifier = Dense(2)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length):
        return tf.fill((length,), 1)

    @staticmethod
    def create_negative_targets(length):
        return tf.fill((length,), 0)

    def inst_eval(
        self,
        A,
        h,
        classifier,
        label,
        bool_annot,
        training,
    ):
        if len(A.shape) == 1:
            A = tf.reshape(A, (1, -1))

        if training:
            bool_annot = bool_annot.numpy()
        else:
            bool_annot = True

        # Get instance
        top_p_ids = tf.math.top_k(A, k=self.k_sample)[1]
        top_n_ids = tf.math.top_k(-A, k=self.k_sample)[1]
        top_p = tf.gather(h, top_p_ids, axis=0)
        top_n = tf.gather(h, top_n_ids, axis=0)
        all_instances = tf.concat([top_p, top_n], axis=0)

        logits = classifier(all_instances)
        logits = tf.reshape(logits, (2 * self.k_sample, 2))
        all_preds = tf.math.top_k(logits, k=1)[1].squeeze(1)

        # Get target labels
        if label:
            p_targets = self.create_positive_targets(self.k_sample)  # 1's
        else:
            p_targets = self.create_negative_targets(self.k_sample)  # 0's
        n_targets = self.create_negative_targets(self.k_sample)  # 0's
        all_targets = tf.concat([p_targets, n_targets], axis=0)

        instance_loss = self.instance_loss_fn(logits, all_targets)

        return instance_loss, all_preds, all_targets

    def call(
        self,
        h,
        bool_annot=None,
        patch_annot=None,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        training=True,
    ):
        A, h = self.attention_net(h)  # NxK
        A = tf.transpose(A)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = tf.nn.softmax(A, axis=1)  # softmax over N

        if instance_eval:
            classifier = self.instance_classifier
            instance_loss, preds, targets, loss_pos_L1, loss_neg_L1 = self.inst_eval(
                A,
                h,
                classifier,
                label,
                bool_annot,
                patch_annot,
                semi_supervised,
                alpha_weight,
                weight_alpha,
                training,
            )

        if semi_supervised and bool_annot:
            A_attention_preds = tf.reshape(A, [77])
            attention_labels_loss = self.attention_labels_loss_fn(
                A_attention_preds, tf.cast(targets, tf.float32)
            )
        else:
            attention_labels_loss = None

        M = tf.matmul(A, h)
        logits = self.classifiers(M)
        Y_hat = tf.math.top_k(logits, k=1)[1]
        Y_prob = tf.nn.softmax(logits)

        if instance_eval:
            results_dict = {
                "instance_loss": instance_loss,
                "loss_pos_L1": loss_pos_L1,
                "loss_neg_L1": loss_neg_L1,
                "inst_labels": np.array(targets.numpy()),
                "inst_preds": np.array(preds.numpy()),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict, attention_labels_loss


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
