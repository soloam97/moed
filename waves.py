import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

def main():

    fs = 44100  # Частота дискретизации
    seconds = 3  # Продолжительность записи
    print('Начать')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Дождитесь окончания записи
    write('output.wav', fs, myrecording)  # Сохранить как WAV файл


if __name__ == '__main__':
    main()