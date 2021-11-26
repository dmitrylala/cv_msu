import numpy as np
from scipy.ndimage import convolve

from typing import Tuple


def get_bayer_masks(n_rows: int, n_cols: int) -> np.ndarray:
    """
    Получение масок трех цветовых каналов по шаблону байера
    :param n_rows: высота масок
    :param n_cols: ширина масок

    :return: маски указанных размеров
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"Bad masks sizes: {n_rows}, {n_cols}")

    red = np.array([[i % 2 == 1 and j % 2 == 0 for i in range(n_cols)] for j in range(n_rows)])
    green = np.array([[i % 2 == j % 2 for i in range(n_cols)] for j in range(n_rows)])
    blue = np.array([[i % 2 == 0 and j % 2 == 1 for i in range(n_cols)] for j in range(n_rows)])
    return np.stack([red, green, blue], axis=2)


def get_colored_img(raw_img: np.ndarray) -> np.ndarray:
    """
    Получение из одноканального изображения трехканального с неизвестными значениями, используя маски байера
    :param raw_img: исходное изображение в градациях серого

    :return: rgb-представление изображения, полученное с помощью масок
    """
    bayer_masks = get_bayer_masks(*raw_img.shape)
    color_components = [raw_img * bayer_masks[:, :, k] for k in range(3)]
    return np.stack(color_components, axis=2)


def bilinear_interpolation(colored_img: np.ndarray) -> np.ndarray:
    """
    Нахождение неизвестных значений с помощью билинейной интерполяции (граничные пиксели игнорируются)
    :param colored_img: цветное изображение с неизвестными значениями, согласно байеровскому шаблону

    :return: восстановленное изображение
    """
    n_rows, n_cols = colored_img.shape[:2]
    bayer_masks = get_bayer_masks(n_rows, n_cols)
    new_img = np.array(colored_img, dtype=np.float32)

    for k in range(3):
        color_mask = bayer_masks[:, :, k]
        color_pixels = colored_img[:, :, k]
        for j in range(1, n_rows - 1):
            for i in range(1, n_cols - 1):
                if not bayer_masks[j, i, k]:
                    new_img[j, i, k] = color_pixels[j - 1:j + 2, i - 1:i + 2][
                        color_mask[j - 1:j + 2, i - 1:i + 2]].mean()

    return new_img.astype(np.uint8)


def get_improved_masks(n_rows: int, n_cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Получение масок трех цветовых каналов по предложенному в статье шаблону
    :param n_rows: высота масок
    :param n_cols: ширина масок

    :return: маски указанных размеров
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"Bad masks sizes: {n_rows}, {n_cols}")

    masks = get_bayer_masks(n_rows, n_cols)
    mask_r = masks[:, :, 0]
    mask_g1 = np.array([[i % 2 == j % 2 == 1 for i in range(n_cols)] for j in range(n_rows)])
    mask_g2 = np.array([[i % 2 == j % 2 == 0 for i in range(n_cols)] for j in range(n_rows)])
    mask_b = masks[:, :, 2]
    return mask_r, mask_g1, mask_g2, mask_b


def get_improved_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Получение весов пикселей в масках по предложенному в статье шаблону

    :return: веса для масок
    """
    weight0 = np.array(
        [
            [0, 0, 1 / 2, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 1 / 2, 0, 0]
        ], dtype=np.float64
    ) / 8

    weight1 = weight0.transpose()

    weight2 = np.array(
        [
            [0, 0, -3 / 2, 0, 0],
            [0, 2, 0, 2, 0],
            [-3 / 2, 0, 6, 0, -3 / 2],
            [0, 2, 0, 2, 0],
            [0, 0, -3 / 2, 0, 0]
        ], dtype=np.float64
    ) / 8

    weight3 = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ], dtype=np.float64
    ) / 8

    return np.stack([weight0, weight1, weight2, weight3], axis=2)


def improved_interpolation(raw_img: np.ndarray) -> np.ndarray:
    """
    Реализация улучшенной линейной интерполяции согласно предложенному в статье алгоритму
    :param raw_img: исходное изображение в градациях серого

    :return: трехканальное изображение - результат интерполяции
    """
    raw_img = raw_img.astype(np.float64) / 255
    new_img = np.zeros(raw_img.shape + (3,), dtype=np.float64)

    mask_r, mask_g1, mask_g2, mask_b = get_improved_masks(*raw_img.shape)
    weights = get_improved_weights()

    new_img[:, :, 0] = convolve(raw_img, weights[:, :, 0]) * mask_g2 + convolve(raw_img, weights[:, :, 1]) * mask_g1 + \
        convolve(raw_img, weights[:, :, 2]) * mask_b + raw_img * mask_r
    new_img[:, :, 1] = convolve(raw_img, weights[:, :, 3]) * (mask_r + mask_b) + raw_img * (mask_g1 + mask_g2)
    new_img[:, :, 2] = convolve(raw_img, weights[:, :, 2]) * mask_r + convolve(raw_img, weights[:, :, 0]) * \
        mask_g1 + convolve(raw_img, weights[:, :, 1]) * mask_g2 + raw_img * mask_b

    return (255 * new_img).astype(np.int32).clip(0, 255).astype(np.uint8)


def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Вычисление метрики PSNR
    :param img_pred: результат работы алгоритма интеполяции
    :param img_gt: эталонное изображение

    :return: значение метрики
    """
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse = ((img_pred - img_gt) ** 2).mean()
    if mse == 0:
        raise ValueError

    return 10 / np.log(10) * np.log(np.max(img_gt ** 2) / mse)
