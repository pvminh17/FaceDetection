# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:04:21 2021

@author: minh
"""


import os
import tarfile
import shutil
import hashlib
import glob
import random
import pickle
from datetime import datetime
from typing import *

from numba import jit
import requests
from joblib import Parallel, delayed

from PIL import Image, ImageOps
import numpy as np
from sklearn.metrics import *

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette('muted')

original_image_path = "images/hongkong.jpg"
original_image = Image.open(original_image_path)

target_size = (384, 288)
thumbnail_image = original_image.copy()
thumbnail_image.thumbnail(target_size, Image.ANTIALIAS)


def to_float_array(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32) / 255.

def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values * 255.))

original = to_float_array(thumbnail_image)

def gamma(values: np.ndarray, coeff: float=2.2) -> np.ndarray:
    return values**(1./coeff)

def gleam(values: np.ndarray) -> np.ndarray:
    return np.sum(gamma(values), axis=2) / values.shape[2]


grayscale = gleam(original)

# =============================================================================
# Image._show(to_image(grayscale), )
# Image._show(thumbnail_image, )
# =============================================================================

WINDOW_SIZE = 15
EINSTEIN_POS = (73, 207)
EINSTEIN = grayscale[EINSTEIN_POS[0]:EINSTEIN_POS[0]+WINDOW_SIZE, EINSTEIN_POS[1]:EINSTEIN_POS[1]+WINDOW_SIZE]

def to_integral(img: np.ndarray) -> np.ndarray:
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]

integral = to_integral(grayscale)

class Box:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.coords_x = [x, x + width, x,          x + width]
        self.coords_y = [y, y,         y + height, y + height]
        self.coeffs   = [1, -1,        -1,         1]

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))


dataset_path = 'dataset'

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

def download_file(url: str, path: str):
    print('Downloading file ...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    print('Download completed.')

def md5(path: str, chunk_size: int=65536) -> str:
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def untar(file_path: str, dest_path: str):
    print('Extracting file.')
    with tarfile.open(file_path, 'r:gz') as f:
        f.extractall(dest_path)
    print('Extraction completed.')

faces_url = 'https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=1'
faces_md5 = 'ab853c17ca6630c191457ff1fb16c1a4'

faces_archive = os.path.join(dataset_path, 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz')
faces_dir = os.path.join(dataset_path, 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned')

if not os.path.exists(faces_archive) or md5(faces_archive) != faces_md5:
    download_file(faces_url, faces_archive)

if not os.path.exists(faces_dir):
    untar(faces_archive, dataset_path)

face_image_files = glob.glob(os.path.join(faces_dir, '**', '*.png'), recursive=True)


def open_face(path: str, resize: bool=True) -> Image.Image:
    CROP_TOP = 50

    img = Image.open(path)
    img = to_image(gamma(to_float_array(img)[CROP_TOP:, :]))
    min_size = np.min(img.size)
    img = ImageOps.fit(img, (min_size, min_size), Image.ANTIALIAS)
    if resize:
        img = img.resize((WINDOW_SIZE, WINDOW_SIZE), Image.ANTIALIAS)
    return img.convert('L')

def merge_images(files: Iterable[str], open_fun: Callable, resize: bool=False) -> Image.Image:
    images = [open_fun(f, resize) for f in files]
    sizes = [img.size for img in images]
    collage_width = np.sum([size[0] for size in sizes])
    collage_height = np.max([size[1] for size in sizes])

    result = Image.new('L', (collage_width, collage_height))
    x_offset = 0
    for img, size in zip(images, sizes):
        result.paste(im=img, box=(x_offset, 0))
        x_offset += size[0]
    return result

random.seed(1000)
random_face_files = random.sample(face_image_files, 5)
merge_images(random_face_files, open_face)

backgrounds_url = 'http://dags.stanford.edu/data/iccv09Data.tar.gz'
backgrounds_md5 = 'f469cf0ab459d94990edcf756694f4d5'

backgrounds_archive = os.path.join(dataset_path, 'iccv09Data.tar.gz')
backgrounds_dir = os.path.join(dataset_path, 'iccv09Data')

if not os.path.exists(backgrounds_archive) :
    download_file(backgrounds_url, backgrounds_archive)

if not os.path.exists(backgrounds_dir):
    untar(backgrounds_archive, dataset_path)

background_image_files = glob.glob(os.path.join(backgrounds_dir, '**', '*.jpg'), recursive=True)


def random_crop(img: Image.Image) -> Image.Image:
    max_allowed_size = np.min(img.size)
    size = random.randint(WINDOW_SIZE, max_allowed_size)
    max_width = img.size[0] - size - 1
    max_height = img.size[1] - size - 1
    left = 0 if (max_width <= 1)  else random.randint(0, max_width)
    top  = 0 if (max_height <= 1) else random.randint(0, max_height)
    return img.crop((left,top,left+size,top+size))

def open_background(path: str, resize: bool=True) -> Image.Image:
    img = Image.open(path)
    img = to_image(gleam(to_float_array(img)))
    img = random_crop(img)
    if resize:
        img = img.resize((WINDOW_SIZE, WINDOW_SIZE), Image.ANTIALIAS)
    return img.convert('L')

# =============================================================================
# sample_image =  np.array([
#     [5, 2, 3, 4, 1], 
#     [1, 5, 4, 2, 3],
#     [2, 2, 1, 3, 4],
#     [3, 5, 6, 4, 5],
#     [4, 1, 3, 2, 6]])
# 
# 
# sample_integral = to_integral(sample_image)
# sample_image= np.pad(sample_image, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]
# print(sample_integral)
# print(sample_image)
# print(sample_integral[sample_image >3])
# =============================================================================



class Feature:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        self.coords_x = [x,      x + hw,     x,          x + hw,
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + height, y + height,
                         y,      y,          y + height, y + height]
        self.coeffs   = [1,     -1,         -1,          1,
                         -1,     1,          1,         -1]

class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hh = height // 2
        self.coords_x = [x,      x + width,  x,          x + width,
                         x,      x + width,  x,          x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs   = [-1,     1,          1,         -1,
                         1,     -1,         -1,          1]
# =============================================================================
#     
# expected = - Box(0, 1, 4, 2)(sample_integral) + Box(0, 3, 4, 2)(sample_integral)
# actual = Feature2v(0, 1, 4, 4)(sample_integral)
# assert expected == actual, f'{expected} == {actual}'
# =============================================================================

class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        self.coords_x = [x,        x + tw,    x,          x + tw,
                         x + tw,   x + 2*tw,  x + tw,     x + 2*tw,
                         x + 2*tw, x + width, x + 2*tw,   x + width]
        self.coords_y = [y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height]
        self.coeffs   = [-1,       1,         1,         -1,
                          1,      -1,        -1,          1,
                         -1,       1,         1,         -1]

# =============================================================================
# expected = - Box(0, 0, 1, 2)(sample_integral) + Box(1, 0, 1, 2)(sample_integral) - Box(2, 0, 1, 2)(sample_integral)
# actual = Feature3h(0, 0, 3, 2)(sample_integral)
# assert expected == actual, f'{expected} == {actual}'
# =============================================================================

class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 3
        self.coords_x = [x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width]
        self.coords_y = [y,        y,          y + th,     y + th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + 2*th, y + 2*th,   y + height, y + height]
        self.coeffs   = [-1,        1,         1,         -1,
                          1,       -1,        -1,          1,
                         -1,        1,         1,         -1]

# =============================================================================
# expected = - Box(0, 0, 2, 1)(sample_integral) + Box(0, 1, 2, 1)(sample_integral) - Box(0, 2, 2, 1)(sample_integral)
# actual = Feature3v(0, 0, 2, 3)(sample_integral)
# assert expected == actual, f'{expected} == {actual}'
# =============================================================================

class Feature4(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.coords_x = [x,      x + hw,     x,          x + hw,     # upper row
                         x + hw, x + width,  x + hw,     x + width,
                         x,      x + hw,     x,          x + hw,     # lower row
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,     # upper row
                         y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height, # lower row
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs   = [1,     -1,         -1,          1,          # upper row
                         -1,     1,          1,         -1,
                         -1,     1,          1,         -1,          # lower row
                          1,    -1,         -1,          1]



Size = NamedTuple('Size', [('height', int), ('width', int)])
Location = NamedTuple('Location', [('top', int), ('left', int)])

def possible_position(size: int, window_size: int = WINDOW_SIZE) -> Iterable[int]:
    return range(0, window_size - size + 1)

def possible_locations(base_shape: Size, window_size: int = WINDOW_SIZE) -> Iterable[Location]:
    return (Location(left=x, top=y)
            for x in possible_position(base_shape.width, window_size)
            for y in possible_position(base_shape.height, window_size))

def possible_shapes(base_shape: Size, window_size: int = WINDOW_SIZE) -> Iterable[Size]:
    base_height = base_shape.height
    base_width = base_shape.width
    return (Size(height=height, width=width)
            for width in range(base_width, window_size + 1, base_width)
            for height in range(base_height, window_size + 1, base_height))



feature2h = list(Feature2h(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=1, width=2), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature2v = list(Feature2v(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=2, width=1), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature3h = list(Feature3h(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=1, width=3), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature3v = list(Feature3v(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=3, width=1), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature4  = list(Feature4(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=2, width=2), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

features = feature2h + feature2v + feature3h + feature3v + feature4

# =============================================================================
# print(f'Number of feature2h features: {len(feature2h)}')
# print(f'Number of feature2v features: {len(feature2v)}')
# print(f'Number of feature3h features: {len(feature3h)}')
# print(f'Number of feature3v features: {len(feature3v)}')
# print(f'Number of feature4 features:  {len(feature4)}')
# print(f'Total number of features:     {len(features)}')
# =============================================================================

def sample_data(p: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    xs.extend([to_float_array(open_face(f)) for f in random.sample(face_image_files, p)])
    xs.extend([to_float_array(open_background(f)) for f in np.random.choice(background_image_files, n, replace=True)])

    ys = np.hstack([np.ones((p,)), np.zeros((n,))])
    return np.array(xs), ys




# =============================================================================
# left, width = 2, 10
# top, height = 3, 4
# 
# EINSTEIN_MASKED = EINSTEIN.copy()
# EINSTEIN_MASKED[top:top+height//2, left:left+width] = 0
# EINSTEIN_MASKED[top+height//2:top+height, left:left+width] = 1
# 
# f2v = Feature2v(x=left, y=top, width=width, height=height)
# 
# random.seed(1000)
# np.random.seed(1000)
# xs, ys = sample_data_normalized(50, 50)
# xs = np.array([to_integral(x) for x in xs])
# zs = np.array([f2v(x) for x in xs])
# =============================================================================

# =============================================================================
# a = sns.distplot(zs[ys > .5], rug=True, hist=False, color='g', kde_kws={'shade': True});
# sns.distplot(zs[ys < .5], rug=True, hist=False, color='r', kde_kws={'shade': True}, ax=a);
# 
# plt.xlabel('Feature value $z = f(x)$')
# plt.ylabel('Value probability')
# plt.tight_layout()
# 
# 
# average_precision = average_precision_score(ys, zs)
# precision, recall, thresholds = precision_recall_curve(ys, zs)
# 
# plt.figure(figsize=(18, 5))
# plt.step(recall, precision, color='b', alpha=0.2, where='post');
# plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
# 
# for i, (p, r, t) in enumerate(zip(precision, recall, thresholds)):
#     if i % 2 == 0:
#         plt.annotate(f'{t:.2f}', xy=(r, p))
# 
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# plt.tight_layout()
# =============================================================================


PredictionStats = NamedTuple('PredictionStats', [('tn', int), ('fp', int), ('fn', int), ('tp', int)])

def prediction_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, PredictionStats]:
    c = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c.ravel()
    return c, PredictionStats(tn=tn, fp=fp, fn=fn, tp=tp)

# =============================================================================
# theta = 0.05 #thresholds duoc chon
# c, s = prediction_stats(ys, zs >= theta)
# 
# print(f'Precision {s.tp/(s.tp+s.fp):.3}, recall {s.tp/(s.tp+s.fn):.3}.')
# 
# sns.heatmap(c, cmap='YlGnBu', annot=True, square=True, 
#             xticklabels=['Predicted negative', 'Predicted positive'], 
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'Confusion matrix for $\Theta={theta}$');
# 
# p = np.argsort(zs)
# zs = zs[p]
# ys = ys[p]
# 
# s_minuses, s_pluses = [], []
# s_minus, s_plus = 0., 0.
# t_minus, t_plus = 0., 0.
# 
# for z, y in zip(zs, ys):
#     if y == 0:
#         s_minus += 1
#         t_minus += 1
#     else:
#         s_plus += 1
#         t_plus += 1
#     print(f'{s_minus} __ {s_plus}')
#     s_minuses.append(s_minus)
#     s_pluses.append(s_plus)
#     
# errors_1, errors_2 = [], []
# 
# min_e = float('inf')
# min_idx = 0
# polarity = 0
# for i, (s_m, s_p) in enumerate(zip(s_minuses, s_pluses)):
#     error_1 = s_p + (t_minus - s_m)
#     error_2 = s_m + (t_plus - s_p)
#     errors_1.append(error_1)
#     errors_2.append(error_2)
#     #print(f'errornow== {error_1} -- {min_idx}')
#     if error_1 < min_e:
#         min_e = error_1
#         min_idx = i
#         polarity = -1
#         #print(f'error1 {min_e} -- {min_idx}')
#     elif error_2 < min_e:
#         min_e = error_2
#         min_idx = i
#         polarity = 1
# =============================================================================
        #print(f'error2 {min_e} -- {min_idx}')

#print(f'Minimal error: {min_e:.2} at index {min_idx} with threshold {zs[min_idx]:.2}. Classifier polarity is {polarity}.')

# =============================================================================
# theta = zs[min_idx]
# c, s = prediction_stats(ys, zs >= theta)
# 
# sns.heatmap(c, cmap='YlGnBu', annot=True, square=True, 
#             xticklabels=['Predicted negative', 'Predicted positive'], 
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'Confusion matrix for $\Theta={theta:.2}$');
# 
# print(f'Precision {s.tp/(s.tp+s.fp):.2}, recall {s.tp/(s.tp+s.fn):.2}, false positive rate {s.fp/(s.fp+s.tn)}.')
# =============================================================================

ThresholdPolarity = NamedTuple('ThresholdPolarity', [('threshold', float), ('polarity', float)])

ClassifierResult = NamedTuple('ClassifierResult', [('threshold', float), ('polarity', int),
                                                   ('classification_error', float),
                                                   ('classifier', Callable[[np.ndarray], float])])

WeakClassifier = NamedTuple('WeakClassifier', [('threshold', float), ('polarity', int),
                                               ('alpha', float),
                                               ('classifier', Callable[[np.ndarray], float])])


@jit
def weak_classifier(x: np.ndarray, f: Feature, polarity: float, theta: float) -> float:
    # return 1. if (polarity * f(x)) < (polarity * theta) else 0.
    return (np.sign((polarity * theta) - (polarity * f(x))) + 1) // 2

@jit
def run_weak_classifier(x: np.ndarray, c: WeakClassifier) -> float:
    return weak_classifier(x=x, f=c.classifier, polarity=c.polarity, theta=c.threshold)


@jit
def strong_classifier(x: np.ndarray, weak_classifiers: List[WeakClassifier]) -> int:
    sum_hypotheses = 0.
    sum_alphas = 0.
    for c in weak_classifiers:
        sum_hypotheses += c.alpha * run_weak_classifier(x, c)
        sum_alphas += c.alpha
    return 1 if (sum_hypotheses >= .5*sum_alphas) else 0

def normalize_weights(w: np.ndarray) -> np.ndarray:
    return w / w.sum()


@jit
def build_running_sums(ys: np.ndarray, ws: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.
    s_minuses, s_pluses = [], []

    for y, w in zip(ys, ws):
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)
    return t_minus, t_plus, s_minuses, s_pluses


@jit
def find_best_threshold(zs: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float], s_pluses: List[float]) -> ThresholdPolarity:
    min_e = float('inf')
    min_z, polarity = 0, 0
    for z, s_m, s_p in zip(zs, s_minuses, s_pluses):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        if error_1 < min_e:
            min_e = error_1
            min_z = z
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            min_z = z
            polarity = 1
    return ThresholdPolarity(threshold=min_z, polarity=polarity)


def determine_threshold_polarity(ys: np.ndarray, ws: np.ndarray, zs: np.ndarray) -> ThresholdPolarity:
    # Sort according to score
    p = np.argsort(zs)
    zs, ys, ws = zs[p], ys[p], ws[p]

    # Determine the best threshold: build running sums
    t_minus, t_plus, s_minuses, s_pluses = build_running_sums(ys, ws)

    # Determine the best threshold: select optimal threshold.
    return find_best_threshold(zs, t_minus, t_plus, s_minuses, s_pluses)

def apply_feature(f: Feature, xis: np.ndarray, ys: np.ndarray, ws: np.ndarray, parallel: Optional[Parallel] = None) -> ClassifierResult:
    if parallel is None:
        parallel = Parallel(n_jobs=-1, backend='threading')

    # Determine all feature values
    zs = np.array(parallel(delayed(f)(x) for x in xis))

    # Determine the best threshold
    result = determine_threshold_polarity(ys, ws, zs)

    # Determine the classification error
    classification_error = 0.
    for x, y, w in zip(xis, ys, ws):
        h = weak_classifier(x, f, result.polarity, result.threshold)
        classification_error += w * np.abs(h - y)

    return ClassifierResult(threshold=result.threshold, polarity=result.polarity,
                            classification_error=classification_error, classifier=f)

STATUS_EVERY     = 1000
KEEP_PROBABILITY = 1./50.
list_weak_classifiers = 'weak_classifiers'


def build_weak_classifiers(prefix: str, num_features: int, xis: np.ndarray, ys: np.ndarray, features: List[Feature], ws: Optional[np.ndarray] = None) -> Tuple[List[WeakClassifier], List[float]]:
    if ws is None:
        m = len(ys[ys < .5])  # number of negative example
        l = len(ys[ys > .5])  # number of positive examples

        # Initialize the weights
        ws = np.zeros_like(ys)
        ws[ys < .5] = 1./(2.*m)
        ws[ys > .5] = 1./(2.*l)

    # Keep track of the history of the example weights.
    w_history = [ws]

    total_start_time = datetime.now()
    with Parallel(n_jobs=-1, backend='threading') as parallel:
        weak_classifiers = []  # type: List[WeakClassifier]
        for t in range(num_features):
            print(f'Building weak classifier {t+1}/{num_features} ...')
            start_time = datetime.now()

            # Normalize the weights
            ws = normalize_weights(ws)

            status_counter = STATUS_EVERY

            # Select best weak classifier for this round
            best = ClassifierResult(polarity=0, threshold=0, classification_error=float('inf'), classifier=None)
            for i, f in enumerate(features):
                status_counter -= 1
                improved = False

                # Python runs singlethreaded. To speed things up,
                # we're only anticipating every other feature, give or take.
                if KEEP_PROBABILITY < 1.:
                    skip_probability = np.random.random()
                    if skip_probability > KEEP_PROBABILITY:
                        continue

                result = apply_feature(f, xis, ys, ws, parallel)
                if result.classification_error < best.classification_error:
                    improved = True
                    best = result

                # Print status every couple of iterations.
                if improved or status_counter == 0:
                    current_time = datetime.now()
                    duration = current_time - start_time
                    total_duration = current_time - total_start_time
                    status_counter = STATUS_EVERY
                    if improved:
                        print(f't={t+1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s in this stage) {i+1}/{len(features)} {100*i/len(features):.2f}% evaluated. Classification error improved to {best.classification_error:.5f} using {str(best.classifier)} ...')
                    else:
                        print(f't={t+1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s in this stage) {i+1}/{len(features)} {100*i/len(features):.2f}% evaluated.')

            # After the best classifier was found, determine alpha
            beta = best.classification_error / (1 - best.classification_error)
            alpha = np.log(1. / beta)

            # Build the weak classifier
            classifier = WeakClassifier(threshold=best.threshold, polarity=best.polarity, classifier=best.classifier, alpha=alpha)

            # Update the weights for misclassified examples
            for i, (x, y) in enumerate(zip(xis, ys)):
                h = run_weak_classifier(x, classifier)
                e = np.abs(h - y)
                ws[i] = ws[i] * np.power(beta, 1-e)

            # Register this weak classifier           
            weak_classifiers.append(classifier)
            w_history.append(ws)
            path =os.path.join(list_weak_classifiers,f'{prefix}-weak-learner-{t+1}-of-{num_features}.pickle')
            pickle.dump(classifier, open(path, 'wb'))

    print(f'Done building {num_features} weak classifiers.')
    return weak_classifiers, w_history



seed = 243
random.seed(seed)
np.random.seed(seed)

image_samples, _ = sample_data(1000, 1000)

sample_mean = image_samples.mean()
sample_std = image_samples.std()
del image_samples

#print(f'Sample mean: {sample_mean}, standard deviation: {sample_std}')

def normalize(im: np.ndarray, mean: float = sample_mean, std: float = sample_std) -> np.ndarray:
    return (im - mean) / std

def sample_data_normalized(p: int, n: int, mean: float = sample_mean, std: float = sample_std) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = sample_data(p, n)
    xs = normalize(xs, mean, std)
    return xs, ys

# =============================================================================
# 
# random.seed(123)
# np.random.seed(123)
# test_xs, test_ys = sample_data_normalized(1000, 1000)
# test_xis = np.array([to_integral(x) for x in test_xs])
# assert test_xis.shape[1:3] == (WINDOW_SIZE+1, WINDOW_SIZE+1), test_xis.shape
# weak_classifiers, w_history = build_weak_classifiers('1st', 2, test_xis, test_ys, features)
# 
# 
# random.seed(234)
# np.random.seed(234)
# test_xs, test_ys = sample_data_normalized(1000, 1000)
# test_xis = np.array([to_integral(x) for x in test_xs])
# assert test_xis.shape[1:3] == (WINDOW_SIZE+1, WINDOW_SIZE+1), test_xis.shape
# weak_classifiers, w_history = build_weak_classifiers('2nd', 10, test_xis, test_ys, features)
# 
# 
# random.seed(345)
# np.random.seed(345)
# test_xs, test_ys = sample_data_normalized(1000, 1000)
# test_xis = np.array([to_integral(x) for x in test_xs])
# assert test_xis.shape[1:3] == (WINDOW_SIZE+1, WINDOW_SIZE+1), test_xis.shape
# weak_classifiers, w_history = build_weak_classifiers('3rd', 10, test_xis, test_ys, features)
# 
# 
# random.seed(345)
# np.random.seed(345)
# test_xs, test_ys = sample_data_normalized(1000, 1000)
# test_xis = np.array([to_integral(x) for x in test_xs])
# assert test_xis.shape[1:3] == (WINDOW_SIZE+1, WINDOW_SIZE+1), test_xis.shape
# weak_classifiers, w_history = build_weak_classifiers('4th', 25, test_xis, test_ys, features)
# =============================================================================


def cascade (x: np.ndarray ,weak_classifiers: List[List[WeakClassifier]])->int:
    probably_face = strong_classifier(x, weak_classifiers[0])
    if probably_face > .5:
        probably_face = strong_classifier(x, weak_classifiers[1])
        if probably_face > .5:
            probably_face = strong_classifier(x, weak_classifiers[2])
            if probably_face > .5:
                probably_face = strong_classifier(x, weak_classifiers[3])
                if probably_face > .5:
                    probably_face = strong_classifier(x, weak_classifiers[4])
                    if probably_face > .5:
                      return 1
    return 0






seed = 34534
random.seed(seed)
np.random.seed(seed)
test_xs, test_ys = sample_data_normalized(1000, 1000)
test_xis = np.array([to_integral(x) for x in test_xs])



weak_classifiers = [];
name_classifier = ['1st','2nd','3rd','4th','5th']
num_classifier = [2,10,10,25,30]
weak_classifiers = []  # type: List[WeakClassifier]
for j, (name, num) in enumerate(zip(name_classifier,num_classifier)):
    weak_classifiers.append([])
    for i in range(num):
        path = os.path.join(list_weak_classifiers,f'{name}-weak-learner-{i+1}-of-{num}.pickle')
        if not os.path.exists(path):
            build_weak_classifiers(name, num, test_xis, test_ys, features)
        f = open(path, 'rb')
        weak_classifiers[j].append(pickle.load(f));



# =============================================================================
seed = 7777
random.seed(seed)
np.random.seed(seed)
test_xs, test_ys = sample_data_normalized(1000, 1000)
test_xis = np.array([to_integral(x) for x in test_xs])

ys_strong = np.array([cascade(x, weak_classifiers) for x in test_xis])
c, s = prediction_stats(test_ys, ys_strong)

sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
            xticklabels=['Predicted negative', 'Predicted positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Cascade 5 stages');
# =============================================================================


rows, cols = integral.shape[0:2]
print(integral.shape[0:2])
print(rows)
print(cols)
HALF_WINDOW = WINDOW_SIZE // 2

face_positions_1 = []
face_positions_2 = []
face_positions_3 = []
face_positions_4 = []
face_positions_5 = []

normalized_integral = to_integral(normalize(grayscale))

for row in range(HALF_WINDOW + 1, rows - HALF_WINDOW):
    for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
        window = normalized_integral[row-HALF_WINDOW-1:row+HALF_WINDOW+1, col-HALF_WINDOW-1:col+HALF_WINDOW+1]

        # First cascade stage
        probably_face = strong_classifier(window, weak_classifiers[0])
        if probably_face < .5:
            continue
        face_positions_1.append((row, col))

        # Second cascade stage
        probably_face = strong_classifier(window, weak_classifiers[1])
        if probably_face < .5:
            continue
        face_positions_2.append((row, col))

        # Third cascade stage
        probably_face = strong_classifier(window, weak_classifiers[2])
        if probably_face < .5:
            continue
        face_positions_3.append((row, col))

        # 4th cascade stage
        probably_face = strong_classifier(window, weak_classifiers[3])
        if probably_face < .5:
            continue
        face_positions_4.append((row, col))

        probably_face = strong_classifier(window, weak_classifiers[4])
        if probably_face < .5:
            continue
        face_positions_5.append((row, col))

def render_candidates(image: Image.Image, candidates: List[Tuple[int, int]]) -> Image.Image:
    canvas = to_float_array(image.copy())
    for row, col in candidates:
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col-HALF_WINDOW-1, :] = [1., 0., 0.]
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col+HALF_WINDOW-1, :] = [1., 0., 0.]
        canvas[row-HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 0., 0.]
        canvas[row+HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 0., 0.]
    return to_image(canvas)

Image._show(render_candidates(thumbnail_image,face_positions_1), )
Image._show(render_candidates(thumbnail_image,face_positions_2), )
Image._show(render_candidates(thumbnail_image,face_positions_3), )
Image._show(render_candidates(thumbnail_image,face_positions_4), )
Image._show(render_candidates(thumbnail_image,face_positions_5), )

























