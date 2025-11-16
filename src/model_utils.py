import tensorflow as tf


def build_model(img_size=224, lr=1e-3):
    """
    Build an EfficientNetB0-based binary classifier.

    Returns (model, base) where:
      - model is the compiled Keras model ready to train
      - base is the EfficientNet backbone (for later fine-tuning)
    """
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input((img_size, img_size, 3))
    z = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base(z, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)

    loss = "binary_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss,
        metrics=metrics,
    )

    return model, base


def prepare_for_stage2(base, model, lr2, unfreeze_from_fraction=0.60):
    ##Stage 2: unfreeze top ~40% of the EfficientNet base (except BatchNorms) and recompile the model with a smaller learning rate.
    unfreeze_from = int(unfreeze_from_fraction * len(base.layers))

    for layer in base.layers[unfreeze_from:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    loss = "binary_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr2),
        loss=loss,
        metrics=metrics,
    )

    return model
