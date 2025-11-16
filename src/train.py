import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img_dir", required=True)
    p.add_argument("--test_img_dir", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--epochs_stage1", type=int, default=5)
    p.add_argument("--epochs_stage2", type=int, default=20)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TRAIN_IMG_DIR = args.train_img_dir
    TEST_IMG_DIR  = args.test_img_dir
    TRAIN_CSV     = args.train_csv
    EPOCHS_STAGE1 = args.epochs_stage1
    EPOCHS_STAGE2 = args.epochs_stage2

#Baseline Config
FAST = False
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 20
SUBSET_PER_CLASS = None

IMG_SIZE = 224
BATCH    = 32
LR1, LR2 = 1e-3, 1e-4

#Imports & setup
import os, random, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from data_utils import to_stem
from data_utils import (
    scan_isic_dir,
    to_stem,
    decode_rgb,
    decode_mask,
    crop_to_mask_with_margin,
)

np.random.seed(42); random.seed(42); tf.random.set_seed(42)
AUTOTUNE = tf.data.AUTOTUNE
print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU'))

try:
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
except Exception as e:
    print("Mixed precision not enabled:", e)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif"}

#Separate photos & masks
def scan_isic_dir(dirpath):
    """Return dicts {stem: path} for photos and masks. Masks end with '_Segmentation' stem."""
    images, masks = {}, {}
    for fname in os.listdir(dirpath):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS: 
            continue
        stem = os.path.splitext(fname)[0]
        full = os.path.join(dirpath, fname)
        if stem.endswith("_Segmentation"):
            images_stem = stem[:-13]
            masks[images_stem] = full
        else:
            images[stem] = full
    return images, masks

train_images, train_masks = scan_isic_dir(TRAIN_IMG_DIR)
test_images,  test_masks  = scan_isic_dir(TEST_IMG_DIR)
print(f"TRAIN: {len(train_images)} photos, {len(train_masks)} masks")
print(f"TEST : {len(test_images)} photos, {len(test_masks)} masks (test masks optional)")

#Align rows to photo paths by stem
df = pd.read_csv(TRAIN_CSV, header=None, names=["image","label"])

df["stem"] = df["image"].apply(to_stem)
df["img_path"]  = df["stem"].map(train_images)
df["mask_path"] = df["stem"].map(train_masks)
df = df.dropna(subset=["img_path"]).reset_index(drop=True)
print("Rows mapped to existing photos:", len(df))

if SUBSET_PER_CLASS is not None:
    parts = []
    for c in df["label"].unique():
        part = df[df["label"].astype(str)==c].sample(min(SUBSET_PER_CLASS, (df["label"]==c).sum()), random_state=42)
        parts.append(part)
    df = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("Subset size:", len(df))

#Encode labels
labels_raw = df["label"].astype(str).values
classes, y_idx = np.unique(labels_raw, return_inverse=True)
num_classes = len(classes)
class_to_index = {c:i for i,c in enumerate(classes)}
index_to_class = {i:c for c,i in class_to_index.items()}
print("Classes:", classes.tolist())

#Split and class weights
y_idx = np.array([class_to_index[str(x)] for x in df["label"].tolist()])
train_df, val_df, y_train_idx, y_val_idx = train_test_split(
    df[["img_path","mask_path","label","stem"]], y_idx, test_size=0.15, stratify=y_idx, random_state=42
)

cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train_idx)
class_weights = {int(i): float(w) for i,w in enumerate(cw)}
print("Train:",len(train_df), " Val:",len(val_df), " Class weights:", class_weights)

#Mask-aware preprocessing
def decode_rgb(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  #uint8
    img = tf.image.convert_image_dtype(img, tf.float32)  #[0,1]
    return img

def decode_mask(path):
    m = tf.io.read_file(path)
    m = tf.image.decode_image(m, channels=1, expand_animations=False)  #uint8
    m = tf.image.convert_image_dtype(m, tf.float32)                    #[0,1]
    return m

def crop_to_mask_with_margin(img, mask, has_mask, margin=0.10):
    """Crop image to mask bbox (+margin). If mask missing/empty → return img."""
    def _crop():
        m = tf.squeeze(mask, -1)                                  #(H,W)
        coords = tf.cast(tf.where(m > 0.05), tf.int32)

        def _no_pixels():
            return img

        def _do_crop():
            ys = coords[:, 0]; xs = coords[:, 1]
            y1 = tf.reduce_min(ys); y2 = tf.reduce_max(ys)
            x1 = tf.reduce_min(xs); x2 = tf.reduce_max(xs)
            h = tf.shape(img, out_type=tf.int32)[0]
            w = tf.shape(img, out_type=tf.int32)[1]
            hbb = y2 - y1 + 1; wbb = x2 - x1 + 1
            y_pad = tf.cast(tf.round(tf.cast(hbb, tf.float32) * margin), tf.int32)
            x_pad = tf.cast(tf.round(tf.cast(wbb, tf.float32) * margin), tf.int32)
            y1m = tf.clip_by_value(y1 - y_pad, 0, h - 1); y2m = tf.clip_by_value(y2 + y_pad, 0, h - 1)
            x1m = tf.clip_by_value(x1 - x_pad, 0, w - 1); x2m = tf.clip_by_value(x2 + x_pad, 0, w - 1)
            return tf.image.crop_to_bounding_box(img, y1m, x1m, y2m - y1m + 1, x2m - x1m + 1)

        return tf.cond(tf.equal(tf.size(coords), 0), _no_pixels, _do_crop)
    return tf.cond(has_mask, _crop, lambda: img)

#Augment & resize
augment = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
])
resize_only = tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE)

def preprocess_row(img_path, mask_path, label, training=True, zero_background=True):
    """Decode RGB, optional mask crop/zero, resize/augment, make label tensor."""
    img = decode_rgb(img_path)                            #float32 [0,1], (H,W,3)
    has_mask = tf.greater(tf.strings.length(mask_path), 0)

    def _load_mask():  #returns float32 [0,1], (H,W,1)
        return decode_mask(mask_path)
    mask = tf.cond(has_mask, _load_mask, lambda: tf.zeros([1, 1, 1], tf.float32))

    #Crop around mask bbox (+10% margin)
    img = crop_to_mask_with_margin(img, mask, has_mask, margin=0.10)

    #Background zeroing
    if zero_background:
        def _apply_zero():
            m_res = tf.image.resize(mask, tf.shape(img)[0:2], method="nearest")
            return img * m_res
        img = tf.cond(has_mask, _apply_zero, lambda: img)

    #Resize
    if training:
        img = augment(img)
    else:
        img = resize_only(img)

    #Label tensor
    if num_classes == 2:
        y = tf.cast(label, tf.int32)
    else:
        y = tf.one_hot(label, num_classes)

    return img, y

def make_ds(frame, training=True, zero_background=True):
    paths  = frame["img_path"].tolist()
    masks  = [m if isinstance(m, str) else "" for m in frame["mask_path"].tolist()]
    labels = [class_to_index[str(x)] for x in frame["label"].tolist()]
    ds = tf.data.Dataset.from_tensor_slices((paths, masks, labels))
    ds = ds.map(lambda p, m, y: preprocess_row(p, m, y, training=training, zero_background=zero_background),
                num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    opt = tf.data.Options(); opt.experimental_deterministic = False
    ds = ds.with_options(opt)
    return ds

train_ds = make_ds(train_df, training=True,  zero_background=True)
val_ds   = make_ds(val_df,   training=False, zero_background=True).cache()

bx, by = next(iter(train_ds.take(1)))
plt.figure(figsize=(10,3))
for i in range(min(6, bx.shape[0])):
    plt.subplot(2,3,i+1); plt.imshow(np.clip(bx[i].numpy(),0,1)); plt.axis("off")
plt.suptitle("Augmented, mask-cropped samples"); plt.show()

#Test loader
def make_test_ds(paths, masks):
    ds = tf.data.Dataset.from_tensor_slices((paths, masks))
    ds = ds.map(lambda p, m: preprocess_row(p, m, 0, training=False, zero_background=True)[0],
                num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

#Model (EfficientNetB0, two-stage fine-tune)
base = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
z = tf.keras.layers.Rescaling(255.0)(inputs)
x = base(z, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
loss = "binary_crossentropy"
metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(LR1), loss=loss, metrics=metrics)
model.summary()

#Stage 1 (AUC-based)
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "lesion_best_stage1.weights.h5", monitor="val_auc", mode="max",
    save_best_only=True, save_weights_only=True, verbose=1
)
early = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=10, restore_best_weights=True, verbose=1
)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
csv = tf.keras.callbacks.CSVLogger("lesion_training_log_stage1.csv", append=False)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=[ckpt, early, rlrop, csv],
    verbose=1
)

#Stage 2: unfreeze top ~40% (skip BatchNorms)
unfreeze_from = int(0.60 * len(base.layers))
for layer in base.layers[unfreeze_from:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(LR2), loss=loss, metrics=metrics)

ckpt2 = tf.keras.callbacks.ModelCheckpoint(
    "lesion_best_stage2.weights.h5", monitor="val_auc", mode="max",
    save_best_only=True, save_weights_only=True, verbose=1
)
csv2 = tf.keras.callbacks.CSVLogger("lesion_training_log_stage2.csv", append=False)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[ckpt2, early, rlrop, csv2],
    verbose=1
)

#Curves & val metrics
def plot_curves(h1, h2, key="val_auc", title=None):
    series = (h1.history.get(key, []) or []) + (h2.history.get(key, []) or [])
    plt.figure(figsize=(12,4)); plt.plot(series); 
    plt.title(title or key); plt.xlabel("Epoch"); plt.ylabel(key); plt.show()

plot_curves(history1, history2, "val_auc", "Validation AUC (both stages)")
plot_curves(history1, history2, "val_loss", "Val Loss (both stages)")

val_probs = model.predict(val_ds, verbose=0).ravel()
y_true = np.array(y_val_idx)[:len(val_probs)]

print(f"Val AUC : {roc_auc_score(y_true, val_probs):.3f}")

#Threshold tuning
taus = np.linspace(0.05, 0.95, 181)
f1s  = [f1_score(y_true, (val_probs>=t).astype(int)) for t in taus]
bacc = [balanced_accuracy_score(y_true, (val_probs>=t).astype(int)) for t in taus]
t_f1   = float(taus[int(np.argmax(f1s))])
t_bacc = float(taus[int(np.argmax(bacc))])
print(f"Best τ for F1   : {t_f1:.2f}  (F1={max(f1s):.3f})")
print(f"Best τ for BAcc : {t_bacc:.2f}  (BAcc={max(bacc):.3f})")

t_star = t_f1
y_pred = (val_probs >= t_star).astype(int)
print(f"Val Acc @τ*={t_star:.2f}: {accuracy_score(y_true, y_pred):.3f} | F1: {f1_score(y_true, y_pred):.3f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm, cmap="Blues"); plt.title("Validation confusion"); plt.xlabel("Pred"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks([0,1], classes); plt.yticks([0,1], classes); plt.tight_layout(); plt.show()

#Predict TEST
test_stems_sorted = sorted(test_images.keys())
test_paths_sorted = [test_images[s] for s in test_stems_sorted]
test_mask_paths   = [test_masks.get(s, "") for s in test_stems_sorted]

test_ds = make_test_ds(test_paths_sorted, test_mask_paths)
probs = model.predict(test_ds, verbose=1).ravel()
pred_idx = (probs >= t_star).astype(int)
pred_labels = [index_to_class[i] for i in pred_idx]

sub = pd.DataFrame({0: test_stems_sorted, 1: pred_labels})
sub.to_csv("SkinLesionTest_Predictions.csv", index=False, header=False)
print(sub.head(), "\nSaved -> SkinLesionTest_Predictions.csv")

check = pd.read_csv("SkinLesionTest_Predictions.csv", header=None)
assert check.shape[0] == len(test_images) and check.shape[1] == 2
assert set(check[1].unique()) <= set(classes)
print("✅ Submission shape OK:", check.shape)


# In[3]:


#Predict TEST
test_stems_sorted = sorted(test_images.keys())
test_paths_sorted = [test_images[s] for s in test_stems_sorted]
test_mask_paths   = [test_masks.get(s, "") for s in test_stems_sorted]

test_ds = make_test_ds(test_paths_sorted, test_mask_paths)
probs = model.predict(test_ds, verbose=1).ravel()
pred_idx = (probs >= t_star).astype(int)
pred_labels = [index_to_class[i] for i in pred_idx]

sub = pd.DataFrame({0: test_stems_sorted, 1: pred_labels})
sub.to_csv("SkinLesionTest_Predictions.csv", index=False, header=False)
print(sub.head(), "\nSaved -> SkinLesionTest_Predictions.csv")

check = pd.read_csv("SkinLesionTest_Predictions.csv", header=None)
assert check.shape[0] == len(test_images) and check.shape[1] == 2
assert set(check[1].unique()) <= set(classes)
print("✅ Submission shape OK:", check.shape)


# In[ ]:




