from scipy.io import wavfile
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

def better_fit(vector, scalar):

    min_diff = np.inf
    ans = None

    if scalar < 0:
        scalar *= -1

    for elem in vector:
        diff = (elem - scalar) ** 2

        if diff < min_diff:
            ans = elem
            min_diff = diff

    return ans


records = ['string_1.wav', 'string_2.wav', 'string_3.wav']

#    E1: 329.63 Hz
#    B2: 246.94 Hz
#    G3: 196.00 Hz
#    D4: 146.83 Hz
#    A5: 110.00 Hz
#    E6: 82.41 Hz

strings = {"E1":329.63, "B2":246.94, "G3":196.00, "D4":146.83, "A5":110.00, "E6":82.41}

for record in records: 
    samplerate, data = wavfile.read(record)

    N = len(data)
    T = 1.0 / samplerate
    yf = fft(data)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)

    size = len(xf)

    start = int(9*size/20)
    end = int(11*size/20) 

    xf = xf[start:end]
    yplot = yplot[start:end]

    yplot = 1.0/N * np.abs(yplot)
    plt.plot(xf, yplot)

    peaks, _ = find_peaks(yplot, height=20)
    peak_values = [yplot[peak] for peak in peaks]

    sorted_peak_values = [[x, y] for y, x in sorted(zip(peak_values, peaks), key=lambda pair: pair[0], reverse=True)]

    texts = []
    for i in range(len(sorted_peak_values)):
        peak_position = sorted_peak_values[i][0]

        if xf[peak_position] > 0:
            texts.append(plt.text(xf[peak_position], yplot[peak_position], "{:.2f}".format(xf[peak_position]), fontsize=8))

    plt.grid()
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.savefig(record[:-4] + ".png")
    plt.close()

    max_peak_position = sorted_peak_values[0][0]
    max_peak_frequency = xf[max_peak_position]

    freq = better_fit(list(strings.values()), max_peak_frequency)

    for key in strings.keys():
        if freq == strings[key]:
            print(record, key, strings[key])




