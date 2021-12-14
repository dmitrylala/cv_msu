import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2

from typing import Tuple


def align(img: np.ndarray,
          g_coord: Tuple[int, int]
          ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Выполнение совмещения с помощью преобразования Фурье

    :param img: Исходное изображение
    :param g_coord: Точка зеленого канала, для которой должны быть найдены
    соответствующие ей точки красного и синего каналов
    :return: Совмещенное изображение и найденные точки красного и синего каналов
    """

    # dividing image on channels
    channels = []
    height = img.shape[0] // 3
    for k in range(3):
        channels.append(img[k * height:(k + 1) * height, :])

    # 5% crop on each side
    for k in range(3):
        cur_height, cur_width = channels[k].shape
        cut_h, cut_w = int(1/20 * cur_height), int(1/20 * cur_width)
        channels[k] = channels[k][cut_h:19 * cut_h, cut_w:19 * cut_w]

    # aligning red and green (green = red + shift)
    green_ft = fft2(channels[1])
    red_ft = np.conj(fft2(channels[0]))
    result = ifft2(green_ft * red_ft)
    shift_red = np.array(np.unravel_index(np.argmax(result), result.shape))
    red_shifted = np.roll(np.roll(channels[0], shift_red[1], axis=1), shift_red[0], axis=0)

    # aligning blue and green (green = blue - shift, blue = green + shift)
    blue_ft = fft2(channels[2])
    green_ft = np.conj(fft2(channels[1]))
    result = ifft2(blue_ft * green_ft)
    shift_blue = -np.array(np.unravel_index(np.argmax(result), result.shape))
    blue_shifted = np.roll(np.roll(channels[2], shift_blue[1], axis=1), shift_blue[0], axis=0)

    # result image
    aligned = np.zeros((red_shifted.shape[0], red_shifted.shape[1], 3), dtype=np.uint8)
    aligned[:, :, 0] = blue_shifted
    aligned[:, :, 1] = channels[1]
    aligned[:, :, 2] = red_shifted

    cur_height, cur_width = channels[0].shape
    height_diff = height - cur_height
    g_row, g_col = g_coord
    b_row = g_row - shift_blue[0] + height_diff
    b_col = (g_col - shift_blue[1]) % cur_width
    r_row = ((g_row - shift_red[0] - cur_height) % cur_height - height_diff) % cur_height
    r_col = (g_col - shift_red[1]) % cur_width

    return aligned, (r_row, r_col), (b_row, b_col)


if __name__ == '__main__':
    src = np.array(Image.open('img.png'), dtype=np.uint8)
    aligned_img, _, _ = align(src, (0, 0))
    Image.fromarray(aligned_img).save('result.png', format='png')
