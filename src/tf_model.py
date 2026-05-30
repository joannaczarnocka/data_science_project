"""TensorFlow/Keras neural network — sklearn-compatible wrapper."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class TFNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Siec neuronowa Keras z interfejsem sklearn.

    Architektura: Input -> Dense(relu, L2) -> BN -> Dropout (x N warstw) -> sigmoid
    Balansowanie: class_weight (bez SMOTE).
    Early stopping: monitoruje val_loss z restore_best_weights=True.
    """

    def __init__(
        self,
        hidden_units=(64, 32, 16),
        dropout_rate=0.3,
        learning_rate=1e-3,
        epochs=200,
        batch_size=32,
        patience=20,
        l2_reg=0.01,
        val_fraction=0.15,
        random_state=42,
    ):
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.l2_reg = l2_reg
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ── budowa modelu ──────────────────────────────────────────────────────────

    def _build_model(self, n_features: int):
        import tensorflow as tf  # lazy import — TF nie zawsze dostepny

        tf.random.set_seed(self.random_state)
        reg = tf.keras.regularizers.l2(self.l2_reg)

        layers = [tf.keras.layers.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers += [
                tf.keras.layers.Dense(units, activation="relu",
                                      kernel_regularizer=reg),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate,
                                        seed=self.random_state),
            ]
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc",
                                           curve="PR")],  # precision-recall AUC
        )
        return model

    # ── sklearn interface ──────────────────────────────────────────────────────

    def fit(self, X, y):
        import tensorflow as tf

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        n_neg = float((y_arr == 0).sum())
        n_pos = float((y_arr == 1).sum())
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1.0)}

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_arr.shape[1]
        self.model_ = self._build_model(X_arr.shape[1])

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_arr,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=y_arr,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        self.history_ = self.model_.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )
        self.n_epochs_trained_ = len(self.history_.history["loss"])
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        proba = self.model_.predict(X_arr, verbose=0).ravel()
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
