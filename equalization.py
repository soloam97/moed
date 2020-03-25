from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# Функция нормировки
def normalization(mass, N=255):
    '''
    Функция нормировки
    :param mass: список значений оттенков серости пикселей
    :param N: количество элементов
    :return: норма
    '''
    norm = []
    min_mass = min(mass)
    max_mass = max(mass)
    for elem in mass:
        norm.append(((elem - min_mass) / (max_mass - min_mass)) * N)
    return norm


# Функция для чтения изображения и перевод в оттенки серого
def read_jpg_gray(file):
    image = Image.open(file).convert('L')
    return image


# Построение гистограммы изображения
def image_histogram(image):
    pixels = image.load()
    w, h = image.size[0], image.size[1]

    i = 0
    pixels_1d = []
    for col in range(w):
        for row in range(h):
            pixels_1d.append(pixels[col, row])

    # нормализация данных
    S = col * row
    pixels_1d = normalization(pixels_1d)
    for i in range(S):
        pixels_1d[i] = int(pixels_1d[i])

    # создаем список гистограммы
    image_hist_y = [0 for i in range(255)]
    image_hist_x = [i for i in range(255)]

    index_hist = 0
    for i in image_hist_x:
        for pix in pixels_1d:
            if pix == i:
                image_hist_y[index_hist] += 1
        index_hist += 1

    return image_hist_x, image_hist_y, pixels_1d


# Создание картинки по считываемым данным по одномерному массиву пикселей
def drawing_image_new(pixels, w, h):
    '''
    Функция рисует картинку в оттенках серого по вхожному одномерному списку пикселей
    :param pixels: одномерный список оттенков серого каждого пикселя
    :param w: ширина создаваемой картинки
    :param h: высота создаваемой картинки
    :return: image_new - картинка в оттенках серого
    '''
    image_new = Image.new('L', (w, h))  # создаем пустую картинку в оттенках серого с шириной w и высотой h
    draw = ImageDraw.Draw(image_new)  # Запускаем инструмент для рисования

    # нормализуем значения оттенков серого
    S = 255
    # image_new_norm = normalization(pixels, S)
    image_new_norm = pixels
    # заполняем значения пикселей новой картинки оттенками серого входного списка
    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(image_new_norm[i]))
            i += 1

    return image_new


def task_equalization():
    # Считываем  файл
    image = read_jpg_gray('HollywoodLC.jpg')
    w, h = image.size[0], image.size[1]
    pix = image.load()
    # Строим гистограмму
    x, y, pixels_1d = image_histogram(image)

    # Выводим гистограмму изображения
    plt.plot(x, y)
    plt.savefig('1')
    plt.show()

    # Производим интегрирование
    cdf = [0]
    for i in range(1, len(y)):
        cdf.append(cdf[i - 1] + ((y[i - 1] + y[i]) * 0.5))
    max_cdf = max(cdf)
    for i in range(255):
        cdf[i] /= max_cdf

    # Выводим кривую для корррекции после интегрирования
    plt.plot(x, cdf)
    plt.savefig('2')
    plt.show()

    new_image = drawing_image_new(pixels_1d, w, h)
    new_image.show()

    for i in range(w * h):
        for j in range(255):
            if pixels_1d[i] == j:
                pixels_1d[i] = cdf[j] * 255

    new_image = drawing_image_new(pixels_1d, w, h)
    new_image.save("HollywoodLC2.jpg", "JPEG")  # сохранить изображение
    new_image.show()


if __name__ == '__main__':
    task_equalization()
