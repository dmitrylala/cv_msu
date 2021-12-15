import numpy as np
from typing import Tuple


def convert_from_rgb(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def grad_norm(img: np.ndarray) -> np.ndarray:
    """
    Вычисление нормы градиента изображения с помощью аппроксимаций первого и второго порядка

    :param img: исходное изображение
    :return: норма градиента
    """
    height, width = img.shape
    der_x = np.zeros_like(img, dtype='float64')
    for y in range(height):
        der_x[y, 0] = img[y, 1] - img[y, 0]
        for x in range(1, width - 1):
            der_x[y, x] = img[y, x + 1] - img[y, x - 1]
        der_x[y, width - 1] = img[y, width - 1] - img[y, width - 2]

    der_y = np.zeros_like(img, dtype='float64')
    for x in range(width):
        der_y[0, x] = img[1, x] - img[0, x]
        for y in range(1, height - 1):
            der_y[y, x] = img[y + 1, x] - img[y - 1, x]
        der_y[height - 1, x] = img[height - 1, x] - img[height - 2, x]

    return np.sqrt(der_x ** 2 + der_y ** 2)


def calculate_seam_matrix(energy: np.ndarray, axis: int) -> np.ndarray:
    """
    Нахождение вспомогательной матрицы "швов"

    :param energy: норма градиента изображения
    :param axis: 0 - для вычисления швов по горизонтали, 1 - по вертикали
    :return: матрица, по которой будет искаться шов
    """
    seam_matrix = np.zeros_like(energy)
    height, width = energy.shape

    # horizontal
    if axis == 0:
        seam_matrix[0, :] = energy[0, :]
        for j in range(1, height):
            seam_matrix[j, 0] = min(seam_matrix[j - 1, : 2]) + energy[j, 0]
            for i in range(1, width - 1):
                seam_matrix[j, i] = min(seam_matrix[j - 1, i - 1: i + 2]) + energy[j, i]
            seam_matrix[j, width - 1] = min(seam_matrix[j - 1, width - 2:]) + energy[j, width - 1]
    # vertical
    elif axis == 1:
        seam_matrix[:, 0] = energy[:, 0]
        for i in range(1, width):
            seam_matrix[0, i] = min(seam_matrix[: 2, i - 1]) + energy[0, i]
            for j in range(1, height - 1):
                seam_matrix[j, i] = min(seam_matrix[j - 1: j + 2, i - 1]) + energy[j, i]
            seam_matrix[height - 1, i] = min(seam_matrix[height - 2:, i - 1]) + energy[height - 1, i]

    return seam_matrix


def find_seam(seam_matrix: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Нахождение шва по вспомогательной матрице

    :param seam_matrix: матрица "швов"
    :param axis: 0 - для нахождения шва по горизонтали, 1 - по вертикали
    :return: маска со швом, индексы строк/столбцов в маске, где находится шов
    """
    seam_mask = np.zeros_like(seam_matrix, dtype='bool')
    height, width = seam_matrix.shape
    seam_indexes = np.zeros(seam_matrix.shape[axis], dtype='int32')

    # horizontal
    if axis == 0:
        index_min = np.argmin(seam_matrix[height - 1, :])
        seam_mask[height - 1, index_min] = True
        seam_indexes[height - 1] = index_min
        for j in range(height - 2, -1, -1):
            index_min = np.argmin(seam_matrix[j, max(index_min - 1, 0): min(index_min + 2, width)]) + \
                        max(index_min - 1, 0)
            seam_mask[j, index_min] = True
            seam_indexes[j] = index_min
    # vertical
    elif axis == 1:
        index_min = np.argmin(seam_matrix[:, width - 1])
        seam_mask[index_min, width - 1] = True
        seam_indexes[width - 1] = index_min
        for i in range(width - 2, -1, -1):
            index_min = np.argmin(seam_matrix[max(index_min - 1, 0): min(index_min + 2, height), i]) + \
                        max(index_min - 1, 0)
            seam_mask[index_min, i] = True
            seam_indexes[i] = index_min

    return seam_mask.astype('uint8'), seam_indexes


def seam_erase(img_rgb: np.ndarray,
               mask: np.ndarray,
               seam_indexes: np.ndarray,
               axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Удаление шва (сжатие изображения)

    :param img_rgb: исходное изображение
    :param mask: маска изображения
    :param seam_indexes: индексы строк/столбцов шва
    :param axis: 0 - для удаления шва по горизонтали, 1 - по вертикали
    :return: новое изображение и новая маска
    """
    height, width = img_rgb.shape[:2]
    shape = (height - int(axis == 1), width - int(axis == 0))
    new_img = np.zeros((shape[0], shape[1], 3), dtype='uint8')
    new_mask = (None, np.zeros(shape, dtype='int8'))[mask is not None]

    # horizontal
    if axis == 0:
        for j in range(height):
            new_img[j, : seam_indexes[j], :] = img_rgb[j, : seam_indexes[j], :]
            new_img[j, seam_indexes[j]:, :] = img_rgb[j, seam_indexes[j] + 1:, :]

            if mask is not None:
                new_mask[j, : seam_indexes[j]] = mask[j, : seam_indexes[j]]
                new_mask[j, seam_indexes[j]:] = mask[j, seam_indexes[j] + 1:]
    # vertical
    elif axis == 1:
        for i in range(width):
            new_img[: seam_indexes[i], i, :] = img_rgb[: seam_indexes[i], i, :]
            new_img[seam_indexes[i]:, i, :] = img_rgb[seam_indexes[i] + 1:, i, :]

            if mask is not None:
                new_mask[: seam_indexes[i], i] = mask[: seam_indexes[i], i]
                new_mask[seam_indexes[i]:, i] = mask[seam_indexes[i] + 1:, i]
    return new_img, new_mask


def seam_add(img_rgb: np.ndarray,
             mask: np.ndarray,
             seam_indexes: np.ndarray,
             axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Добавление шва (расширение изображения)

    :param img_rgb: исходное изображение
    :param mask: маска изображения
    :param seam_indexes: индексы строк/столбцов шва
    :param axis: 0 - для добавления шва по горизонтали, 1 - по вертикали
    :return: новое изображение и новая маска
    """
    height, width = img_rgb.shape[:2]
    shape = (height + int(axis == 1), width + int(axis == 0))
    new_img = np.zeros((shape[0], shape[1], 3), dtype='uint8')
    new_mask = (None, np.zeros(shape, dtype='int8'))[mask is not None]

    # horizontal
    if axis == 0:
        for j in range(height):
            new_img[j, : seam_indexes[j] + 1, :] = img_rgb[j, : seam_indexes[j] + 1, :]
            new_img[j, seam_indexes[j] + 1, :] = (img_rgb[j, seam_indexes[j], :] +
                                                  img_rgb[j, min(seam_indexes[j] + 2, width - 1), :]) // 2
            new_img[j, seam_indexes[j] + 2:, :] = img_rgb[j, seam_indexes[j] + 1:, :]

            if mask is not None:
                new_mask[j, : seam_indexes[j] + 1] = mask[j, : seam_indexes[j] + 1]
                new_mask[j, seam_indexes[j] + 1] = (mask[j, seam_indexes[j]] + mask[
                    j, min(seam_indexes[j] + 2, width - 1)]) // 2
                new_mask[j, seam_indexes[j] + 2:] = mask[j, seam_indexes[j] + 1:]
                new_mask[j, seam_indexes[j]] += 1
    # vertical
    elif axis == 1:
        for i in range(width):
            new_img[: seam_indexes[i] + 1, i, :] = img_rgb[: seam_indexes[i] + 1, i, :]
            new_img[seam_indexes[i] + 1, i, :] = (img_rgb[seam_indexes[i], i, :] +
                                                  img_rgb[min(seam_indexes[i] + 2, height - 1), i, :]) // 2
            new_img[seam_indexes[i] + 2:, i, :] = img_rgb[seam_indexes[i] + 1:, i, :]

            if mask is not None:
                new_mask[: seam_indexes[i] + 1, i] = mask[: seam_indexes[i] + 1, i]
                new_mask[seam_indexes[i] + 1, i] = (mask[seam_indexes[i], i] + mask[
                    min(seam_indexes[i] + 2, height - 1), i]) // 2
                new_mask[seam_indexes[i] + 2:, i] = mask[seam_indexes[i] + 1:, i]
                new_mask[seam_indexes[i], i] += 1

    return new_img, new_mask


def seam_carve(img_rgb: np.ndarray,
               mode: str,
               mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполнение алгоритма контекстно-зависимого масштабирования изображений

    :param img_rgb: изображение, к которому необходимо применить масштабирование
    :param mode: режим работы (horizontal shrink, vertical shrink, horizontal expand, vertical expand)
    :param mask: маска изображения - состоит из элементов {-1, 0, +1}
    -1 означает пиксели, подлежащие удалению, +1 — сохранению; 0 означает, что энергию пикселей менять не надо
    :return: новое изображение и маска, а также маска найденного/созданного шва
    """
    if mask is not None:
        mask = np.array(mask, dtype='int8')
    img = convert_from_rgb(img_rgb)
    height, width = img.shape
    axis = int('vertical' in mode)
    action = int('shrink' in mode)

    # finding norm of gradient in every pixel
    energy = grad_norm(img)

    # adding mask
    if mask is not None:
        mask = np.array(mask, dtype='float64')
        energy += mask * 256 * height * width

    # calculating matrix to define where is seam
    seam_matrix = calculate_seam_matrix(energy, axis=axis)

    # creating seam mask
    seam_mask, seam_indexes = find_seam(seam_matrix, axis=axis)

    seam_actions = {0: seam_add, 1: seam_erase}

    # erasing or adding seam from/to image and mask
    new_img, new_mask = seam_actions[action](img_rgb, mask, seam_indexes, axis=axis)

    return new_img, new_mask, seam_mask
