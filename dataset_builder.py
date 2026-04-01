import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def augment_spectrogram(spectrogram, label):
    noise       = tf.random.normal(shape=tf.shape(spectrogram), mean=0.0, stddev=0.02)
    spectrogram = spectrogram + noise
    t       = tf.shape(spectrogram)[0]
    t_mask  = tf.random.uniform((), 0, t // 5, dtype=tf.int32)
    t_start = tf.random.uniform((), 0, t - t_mask, dtype=tf.int32)
    t_vec   = tf.cast((tf.range(t) < t_start) | (tf.range(t) >= t_start + t_mask), tf.float32)
    spectrogram = spectrogram * tf.reshape(t_vec, [-1, 1, 1])
    f       = tf.shape(spectrogram)[1]
    f_mask  = tf.random.uniform((), 0, f // 5, dtype=tf.int32)
    f_start = tf.random.uniform((), 0, f - f_mask, dtype=tf.int32)
    f_vec   = tf.cast((tf.range(f) < f_start) | (tf.range(f) >= f_start + f_mask), tf.float32)
    spectrogram = spectrogram * tf.reshape(f_vec, [1, -1, 1])
    spectrogram = spectrogram * tf.random.uniform((), 0.75, 1.25)
    shift       = tf.random.uniform((), -10, 10, dtype=tf.int32)
    spectrogram = tf.roll(spectrogram, shift, axis=0)
    drop_channel = tf.random.uniform((), 0, 3, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(
        tf.ones([3], dtype=tf.float32),
        [[drop_channel]],
        [tf.cast(tf.random.uniform(()) > 0.15, tf.float32)]
    )
    spectrogram = spectrogram * tf.reshape(mask, [1, 1, 3])
    return spectrogram, label


def normalise(spectrogram, label):
    mean, variance = tf.nn.moments(spectrogram, axes=[0, 1, 2])
    spectrogram    = (spectrogram - mean) / (tf.sqrt(variance) + 1e-7)
    return spectrogram, label


def apply_mixup(train_ds, alpha=0.2, shuffle_buffer=500):
    ds1 = train_ds.repeat()
    ds2 = train_ds.repeat().shuffle(shuffle_buffer)

    def mixup_fn(batch1, batch2):
        x1, y1 = batch1
        x2, y2 = batch2
        lam   = tf.random.uniform((), alpha, 1.0 - alpha)
        x_mix = lam * x1 + (1.0 - lam) * x2
        y_mix = lam * y1 + (1.0 - lam) * y2
        return x_mix, y_mix

    return tf.data.Dataset.zip((ds1, ds2)).map(
        mixup_fn, num_parallel_calls=tf.data.AUTOTUNE
    )


def build_dataset(data_path, batch_size=32, validation_split=0.2, shuffle_buffer=500):

    file_paths = []
    labels     = []

    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_path, cls)
        if not os.path.exists(cls_dir):
            print(f"WARNING: class folder not found — {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith('.npy'):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    file_paths = np.array(file_paths)
    labels     = np.array(labels)

    print(f"\nTotal samples found: {len(file_paths)}")
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {np.sum(labels == idx)}")

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        file_paths, labels,
        test_size=validation_split,
        stratify=labels,
        random_state=42
    )

    print(f"\nTrain samples      : {len(tr_paths)}")
    print(f"Validation samples : {len(val_paths)}")

    def generator(paths, labs):
        for path, lab in zip(paths, labs):
            features = np.load(path).astype(np.float32)
            yield features, lab

    train_ds = tf.data.Dataset.from_generator(
        lambda: generator(tr_paths, tr_labels),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(),            dtype=tf.int32)
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: generator(val_paths, val_labels),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(),            dtype=tf.int32)
        )
    )

    train_ds = train_ds.map(
        lambda x, y: (x, tf.one_hot(y, depth=len(CLASS_NAMES))),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (x, tf.one_hot(y, depth=len(CLASS_NAMES))),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.map(augment_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(normalise,           num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(normalise,             num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (train_ds
                .shuffle(buffer_size=shuffle_buffer)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))

    train_ds = apply_mixup(train_ds, alpha=0.2, shuffle_buffer=shuffle_buffer)

    return train_ds, val_ds