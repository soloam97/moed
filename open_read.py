import struct
from pickle import dump, load


# Чтение массива из файла .dat
def open_reader(file):
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 4):
            format = '4s'
            elem = struct.unpack(format, data[i:i + 4])
            figures.append(elem[0])
        return figures


def open_write(file, data):
    with open(file, 'wb') as binary_file:
        figures = []

        # data = binary_file.read()
        for i in range(0, len(data), 1):
            elem = struct.pack('s', data[i])
            figures.append((elem[0]))
        return figures



def main():
    y = open_reader("C:/Users/soloa/PycharmProjects/application/K1N_1701.bin")
    # y = open_reader("123.bin")
    # print(y)
    # y[0] = b'\xbb'
    print(y)
    print(type(y[0]))
    a = hex(120)
    print(a)
    file = open('123.bin', 'wb')

    file.write(bytes(a, 'UTF-8'))
    file.close()







if __name__ == '__main__':
    main()