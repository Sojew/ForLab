import mne
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.signal as sp
import pandas as pd
import seaborn as sns


def load_file(file_name, channels=False):
    os.chdir(r"") # EDF data file
    data = mne.io.read_raw_edf(file_name, preload=True)
    channels = data.ch_names
    print(channels)
    dataraw = data.get_data()
    events = np.genfromtxt(file_name[:-4] + ".ann", dtype=int, delimiter=",", skip_header=1)
    if channels == True:
        return dataraw, events, channels
    else:
        return dataraw, events

channels = load_file("data.edf", channels=True)[2]


def FFT(data, window=500, freqs=False):
    spectre = abs(np.fft.fft(data).real) ** 2
    freq = np.fft.fftfreq(window, d=.004)
    if freqs == True:
        return freq, spectre
    else:
        return spectre

data, events = load_file("data.edf")
print(data.shape)


def Epoched_data(data, window):
    e_data = data[:, events[0, 0]:events[0, 0] + window]
    for i in events[1:, 0]:
        e_data = np.append(e_data, data[:, i:i + window], axis=1)
    return e_data[:-4,:].reshape(-1, 19, 500)

print(Epoched_data(load_file("data.edf")[0], 500))

freqs = FFT(Epoched_data(load_file("data.edf")[0], 500), freqs=True)[0][:250]
freqs1 = np.array(list(freqs) * 118).reshape(-1, 1)

ev = list(events[:, 2]) * 250
ev = np.array(ev).reshape(250, -1)
ev = ev.T.reshape(-1, 1)

result = np.append(ev, freqs1, axis=1)

def Feat_Spectre():
    ex_t = np.zeros((1, 19))
    e_data = Epoched_data(load_file("data.edf")[0], 500)
    for j in range(e_data.shape[0]):
        temp = np.zeros((1, 250))
    for i in range(e_data.shape[1]):
        temp = np.append(temp, [FFT(e_data[j, i, :])[:250]], axis=0)
    temp = np.rot90(np.flip(temp[1:20,:], axis = 1), k = 1)
    ex_t = np.append(ex_t, temp, axis=0)
    return ex_t[1:,:]

spectre = Feat_Spectre()
db = pd.DataFrame(spectre)
db.insert(19, "marks", ev, True)
db.insert(0, "freqs", freqs1, True)
print(db)
