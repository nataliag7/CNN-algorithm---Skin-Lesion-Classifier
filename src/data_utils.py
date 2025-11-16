import os

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
