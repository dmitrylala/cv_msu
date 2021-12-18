import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio

from typing import Tuple, List


def pca_compression(matrix: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Сжатие изображения с помощью PCA

    :param matrix: двумерная матрица (одна цветовая компонента картинки)
    :param p: количество компонент
    :return: собственные векторы, проекция матрицы на новое пр-во и вектор средних по строкам
    """

    # центрирование и сохранение средних
    matrix = matrix.astype(np.float64)
    means = np.mean(matrix, axis=1)
    for j in range(matrix.shape[0]):
        matrix[j, :] -= means[j]

    # нахождение ковариационной матрицы, собственных векторов и собственных значений
    cov = np.cov(matrix)
    eig_val, eig_vec = np.linalg.eig(cov)

    # отбрасывание собственных векторов после p-го
    eig_vec = eig_vec[:, :p]

    # проекция на новое пространство
    projection = np.matmul(eig_vec.T, matrix)

    return eig_vec, projection, means


def pca_decompression(compressed: np.ndarray) -> np.ndarray:
    """
    Разжатие изображения, сжатого с помощью PCA

    :param compressed: список кортежей из собственных векторов, проекций для каждой цветовой компоненты и средних
    :return: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # умножение собственных векторов на проекции и прибавление средних значений по строкам исходной матрицы
        vectors = comp[0].copy()
        projection = comp[1].copy()
        means = compressed[0][2].copy()
        prod = vectors @ projection
        for j in range(prod.shape[0]):
            prod[j, :] += means[j]
        result_img.append(prod)
    result_img = np.clip(np.array(result_img, dtype=np.int64), 0, 255).astype(np.uint8)
    result = np.stack(result_img, axis=2)
    return result


def pca_visualize():
    """
    Визуализация работы сжатия с помощью PCA при различных значениях компонент
    """
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j], p)))
        decompressed = pca_decompression(compressed)

        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img: np.ndarray) -> np.ndarray:
    """
    Переход из цветового пространства RGB в пространство YCbCr

    :param img: RGB изображение
    :return: YCbCr изображение
    """
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ], dtype=np.float64)
    ycbcr = img.dot(matrix.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(img: np.ndarray) -> np.ndarray:
    """
    Переход из цветового пространства YCbCr в пространство RGB

    :param img: YCbCr изображение
    :return: RGB изображение
    """
    matrix = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.77, 0]
    ])
    rgb = img.astype(np.float64)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(matrix.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def get_gauss_1():
    """
    Визуализация размытия цветовых компонент в YCbCr пространстве
    """
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr = rgb2ycbcr(rgb_img)
    sigma = 10.0
    ycbcr[:, :, 1] = gaussian_filter(ycbcr[:, :, 1], sigma)
    ycbcr[:, :, 2] = gaussian_filter(ycbcr[:, :, 2], sigma)
    rgb_img = ycbcr2rgb(ycbcr)
    plt.imshow(rgb_img)

    plt.savefig("gauss_1.png")


def get_gauss_2():
    """
    Визуализация размытия яркостной компоненты в YCbCr пространстве
    """
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr = rgb2ycbcr(rgb_img)
    sigma = 10.0
    ycbcr[:, :, 0] = gaussian_filter(ycbcr[:, :, 0], sigma)
    rgb_img = ycbcr2rgb(ycbcr)
    plt.imshow(rgb_img)

    plt.savefig("gauss_2.png")


def downsampling(component: np.ndarray) -> np.ndarray:
    """
    Уменьшение цветовых компонент в 2 раза

    :param component: цветовая компонента размера [A, B, 1]
    :return: цветовая компонента размера [A // 2, B // 2, 1]
    """
    return gaussian_filter(component, 10.0)[::2, ::2]


def dct(block: np.ndarray) -> np.ndarray:
    """
    Дискретное косинусное преобразование

    :param block: блок размера 8x8
    :return: блок размера 8x8 после ДКП
    """
    def alpha(u):
        return 1 if u != 0 else 2 ** (-1 / 2)

    new_block = np.zeros(block.shape, dtype=np.float64)
    for j in range(new_block.shape[0]):
        for i in range(new_block.shape[1]):
            new_block[j, i] = 1 / 4 * alpha(j) * alpha(i)
            cur_sum = 0
            for k in range(block.shape[0]):
                for m in range(block.shape[1]):
                    cur_sum += block[k, m] * np.cos((2 * k + 1) * j * np.pi / 16) * np.cos((2 * m + 1) * i * np.pi / 16)
            new_block[j, i] *= cur_sum
    return new_block


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block: np.ndarray, quantization_matrix: np.ndarray) -> np.ndarray:
    """
    Квантование

    :param block: блок размера 8x8 после применения ДКП
    :param quantization_matrix: матрица квантования
    :return: блок размера 8x8 после квантования
    """
    return np.round(block / quantization_matrix)


def get_scale_factor(q: int) -> float:
    """
    Получение scale factor для генерации матриц квантования

    :param q: Quality Factor для jpeg-сжатия
    :return: scale factor
    """
    if 1 <= q < 50:
        return 5000.0 / q
    elif 50 <= q <= 99:
        return 200 - 2 * q
    else:
        return 1


def own_quantization_matrix(default_quantization_matrix: np.ndarray, q: int) -> np.ndarray:
    """
    Генерация матрицы квантования по Quality Factor

    :param default_quantization_matrix: "стандартная" матрица квантования
    :param q: Quality Factor
    :return: новая матрица квантования
    """
    assert 1 <= q <= 100
    scale_factor = get_scale_factor(q)
    new_quantization_matrix = np.floor((default_quantization_matrix * scale_factor + 50) / 100)
    new_quantization_matrix[new_quantization_matrix == 0] = 1
    return new_quantization_matrix


def zigzag(block: np.ndarray) -> np.ndarray:
    """
    Зигзаг-сканирование

    :param block: блок размера 8x8
    :return: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    indexes = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ])
    flattened = block.flatten()
    new_block = np.zeros(len(indexes))
    for i in range(len(indexes)):
        new_block[indexes[i]] = flattened[i]
    return new_block


def compression(zigzag_list: List) -> np.ndarray:
    """
    Сжатие последовательности после зигзаг-сканирования

    :param zigzag_list: список после зигзаг-сканирования
    :return: сжатый с помощью RLE список
    """
    result = []
    count_zeros = 0
    for num in zigzag_list:
        if num != 0:
            if count_zeros != 0:
                result.append(count_zeros)
            result.append(num)
            count_zeros = 0
        else:
            if count_zeros == 0:
                result.append(0)
            count_zeros += 1
    if count_zeros != 0:
        result.append(count_zeros)
    return result


def jpeg_compression(img: np.ndarray, quantization_matrices: List[np.ndarray]) -> List[List[np.ndarray]]:
    """
    JPEG-сжатие

    :param img: цветная картинка
    :param quantization_matrices: список из 2-ух матриц квантования
    :return: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переход из RGB в YCbCr
    ycbcr = rgb2ycbcr(img).astype(np.float64)
    height, width = ycbcr.shape[:2]

    # Уменьшение цветовых компонент
    downsampled_color = np.zeros((height // 2, width // 2, 2), dtype=np.float64)
    for k in range(2):
        downsampled_color[:, :, k] = downsampling(ycbcr[:, :, k + 1])

    # Деление компонент на блоки 8x8 и перевод всех элементов блоков из диапазона [0, 255] в [-128, 127]
    blocks = [[], [], []]
    block_size = 8
    for j in range(height // 8):
        for i in range(width // 8):
            height_up, height_down = j * block_size, (j + 1) * block_size
            width_left, width_right = i * block_size, (i + 1) * block_size
            blocks[0].append(ycbcr[height_up:height_down, width_left:width_right, 0] - 128)
    for j in range(height // 16):
        for i in range(width // 16):
            height_up, height_down = j * block_size, (j + 1) * block_size
            width_left, width_right = i * block_size, (i + 1) * block_size
            for k in range(2):
                blocks[k + 1].append(downsampled_color[height_up:height_down, width_left:width_right, k] - 128)

    # Применение ДКП, квантования, зигзаг-сканирования и сжатия
    for m in range(3):
        cur_blocks = blocks[m]
        matrix = quantization_matrices[0] if m == 0 else quantization_matrices[1]
        for k in range(len(cur_blocks)):
            cur_blocks[k] = compression(zigzag(quantization(dct(cur_blocks[k]), matrix)))

    return blocks


def inverse_compression(compressed_list: List) -> List:
    """
    Разжатие последовательности

    :param compressed_list: сжатый с помощью RLE список
    :return: разжатый список
    """
    result = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] != 0:
            result.append(compressed_list[i])
        else:
            count_zeros = compressed_list[i + 1]
            result += [0] * count_zeros
            i += 1
        i += 1
    return result


def inverse_zigzag(elems: List) -> np.ndarray:
    """
    Обратное зигзаг-сканирование

    :param elems: список элементов
    :return: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в
                зигзаг-сканировании
    """
    indexes = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ])
    output = np.zeros(len(indexes))
    for i in range(len(indexes)):
        output[i] = elems[indexes[i]]
    output = output.reshape((8, 8))
    return output


def inverse_quantization(block: np.ndarray, quantization_matrix: np.ndarray) -> np.ndarray:
    """
    Обратное квантование

    :param block: блок размера 8x8 после применения обратного зигзаг-сканирования
    :param quantization_matrix: матрица квантования
    :return: блок размера 8x8 после квантования; округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block: np.ndarray) -> np.ndarray:
    """
    Обратное дискретное косинусное преобразование

    :param block: блок размера 8x8
    :return: блок размера 8x8 после обратного ДКП
    """
    def alpha(u):
        return 1 if u != 0 else 2 ** (-1 / 2)

    new_block = np.zeros(block.shape, dtype=np.float64)
    for j in range(new_block.shape[0]):
        for i in range(new_block.shape[1]):
            cur_sum = 0
            for k in range(block.shape[0]):
                for m in range(block.shape[1]):
                    cur_sum += 1 / 4 * alpha(k) * alpha(m) * block[k, m] * np.cos(
                        (2 * j + 1) * k * np.pi / 16) * np.cos((2 * i + 1) * m * np.pi / 16)
            new_block[j, i] = cur_sum
    return np.round(new_block)


def upsampling(component: np.ndarray) -> np.ndarray:
    """
    Увеличение цветовых компонент в 2 раза

    :param component: цветовая компонента размера [A, B, 1]
    :return: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    height, width = component.shape[:2]
    result = np.zeros((2 * height, 2 * width))
    for j in range(height):
        for i in range(width):
            result[2 * j, 2 * i] = component[j, i]
            result[2 * j, 2 * i + 1] = result[2 * j, 2 * i]
        result[2 * j + 1, :] = result[2 * j, :]
    return result


def jpeg_decompression(result: List[List[np.ndarray]],
                       result_shape: Tuple[int],
                       quantization_matrices: List[np.ndarray]
                       ) -> np.ndarray:
    """
    Разжатие изображения

    :param result: список сжатых данных
    :param result_shape: размер ответа
    :param quantization_matrices: список из 2-ух матриц квантования
    :return: разжатое изображение
    """
    # Применение обратного сжатия, обратного зигзаг-сканирования, обратного квантования и обратного ДКП
    for m in range(3):
        cur_blocks = result[m]
        matrix = quantization_matrices[0] if m == 0 else quantization_matrices[1]
        for k in range(len(cur_blocks)):
            cur_blocks[k] = inverse_dct(
                inverse_quantization(inverse_zigzag(inverse_compression(cur_blocks[k])), matrix))

    # Перевод блоков из диапазона [-128, 127] назад в [0, 255] и объединение их в компоненты
    y_comp = np.zeros(result_shape[:2], dtype=np.float64)
    height, width = result_shape[:2]
    color_components = np.zeros((height // 2, width // 2, 2), dtype=np.float64)
    block_size = 8

    num = 0
    for j in range(height // 8):
        for i in range(width // 8):
            height_up, height_down = j * block_size, (j + 1) * block_size
            width_left, width_right = i * block_size, (i + 1) * block_size
            y_comp[height_up:height_down, width_left:width_right] = result[0][num] + 128
            num += 1
    for k in range(2):
        cur_blocks = result[k + 1]
        num = 0
        for j in range(height // 16):
            for i in range(width // 16):
                height_up, height_down = j * block_size, (j + 1) * block_size
                width_left, width_right = i * block_size, (i + 1) * block_size
                color_components[height_up:height_down, width_left:width_right, k] = cur_blocks[num] + 128
                num += 1

    # Увеличение цветовых компонент и их объединение в изображение
    upsampled = np.zeros((height, width, 2), dtype=np.float64)
    for k in range(2):
        upsampled[:, :, k] = upsampling(color_components[:, :, k])
    ycbcr = np.stack([y_comp, upsampled[:, :, 0], upsampled[:, :, 1]], axis=2)

    # Переход из YCbCr в RGB
    rgb = ycbcr2rgb(ycbcr)

    return rgb


def jpeg_visualize():
    """
    Визуализация JPEG-сжатия с различными значениями Quality Factor
    """
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        matrices = [own_quantization_matrix(y_quantization_matrix, p),
                    own_quantization_matrix(color_quantization_matrix, p)]
        result = jpeg_compression(img, matrices)
        decompressed = jpeg_decompression(result, img.shape, matrices)

        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img: np.ndarray, c_type: str, param: int = 1) -> Tuple[np.ndarray, int]:
    """
    Pipeline для PCA и JPEG

    :param img: исходное изображение
    :param c_type: название метода - 'pca', 'jpeg'
    :param param: кол-во компонент в случае PCA, и Quality Factor для JPEG
    :return: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    compressed = None
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrices = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrices)
        img = jpeg_decompression(compressed, img.shape, matrices)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path: str, c_type: str, param_list: List[int]) -> plt.figure:
    """
    Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков

    :param img_path: пусть до изображения
    :param c_type: тип сжатия
    :param param_list: список параметров - кол-во компонент в случае PCA, и Quality Factor для JPEG
    :return: график с метриками
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    """
    Подсчет метрик для PCA
    """
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    """
    Подсчет метрик для JPEG
    """
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == '__main__':
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
