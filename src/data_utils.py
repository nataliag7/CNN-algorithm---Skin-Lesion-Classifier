import os
import tensorflow as tf

#Image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def scan_isic_dir(dirpath):
    """
    Return dicts {stem: path} for photos and masks in an ISIC-style folder.
    Masks end with '_Segmentation' in the filename.
    """
    images, masks = {}, {}
    for fname in os.listdir(dirpath):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue

        stem = os.path.splitext(fname)[0]
        full = os.path.join(dirpath, fname)

        if stem.endswith("_Segmentation"):
            image_stem = stem[:-13]
            masks[image_stem] = full
        else:
            images[stem] = full

    return images, masks


def to_stem(s):
    """
    Convert an image filename (or path) to its base stem.
    Also strips '_Segmentation' from mask filenames.
    """
    s = os.path.basename(str(s))
    stem = os.path.splitext(s)[0]
    if stem.endswith("_Segmentation"):
        stem = stem[:-13]
    return stem


def decode_rgb(path):
    """Read an RGB image from disk and return float32 tensor in [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def decode_mask(path):
    """Read a 1-channel mask from disk and return float32 tensor in [0,1]."""
    m = tf.io.read_file(path)
    m = tf.image.decode_image(m, channels=1, expand_animations=False)
    m = tf.image.convert_image_dtype(m, tf.float32)
    return m


def crop_to_mask_with_margin(img, mask, has_mask, margin=0.10):
    """
    Crop image to the bounding box of the mask plus a margin (fraction of bbox size).
    If there is no mask or it's empty, just return the original image.
    """

    def _crop():
        #mask: (H, W, 1) -> (H, W)
        m = tf.squeeze(mask, -1)

        #coords: [N, 2] (y, x) where mask > threshold
        coords = tf.where(m > 0.05)

        def _no_pixels():
            #No foreground pixels -> return original image
            return img

        def _do_crop():
            ys = coords[:, 0]
            xs = coords[:, 1]

            y1 = tf.reduce_min(ys)
            y2 = tf.reduce_max(ys)
            x1 = tf.reduce_min(xs)
            x2 = tf.reduce_max(xs)

            h = tf.shape(img)[0]
            w = tf.shape(img)[1]

            hbb = y2 - y1 + 1
            wbb = x2 - x1 + 1

            y_pad = tf.cast(tf.round(tf.cast(hbb, tf.float32) * margin), tf.int32)
            x_pad = tf.cast(tf.round(tf.cast(wbb, tf.float32) * margin), tf.int32)

            y1m = tf.clip_by_value(y1 - y_pad, 0, h - 1)
            y2m = tf.clip_by_value(y2 + y_pad, 0, h - 1)
            x1m = tf.clip_by_value(x1 - x_pad, 0, w - 1)
            x2m = tf.clip_by_value(x2 + x_pad, 0, w - 1)

            return tf.image.crop_to_bounding_box(
                img,
                y1m,
                x1m,
                y2m - y1m + 1,
                x2m - x1m + 1,
            )

        num_pixels = tf.shape(coords)[0]
        return tf.cond(tf.equal(num_pixels, 0), _no_pixels, _do_crop)

    #If there is a mask for this image, crop around it; otherwise just return img
    return tf.cond(has_mask, _crop, lambda: img)
