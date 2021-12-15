import EdfAnnSpectre

channels = load_file("data.edf", channels=True)[2]

data, events = load_file("data.edf")
print(data.shape)

print(Epoched_data(load_file("data.edf")[0], 500))

freqs = FFT(Epoched_data(load_file("data.edf")[0], 500), freqs=True)[0][:250]
properFreqs = np.array(list(freqs) * 118).reshape(-1, 1)

eventList = list(events[:, 2]) * 250
eventList = np.array(eventList).reshape(250, -1)
eventList = eventList.T.reshape(-1, 1)

result = np.append(eventList, freqs1, axis=1)

spectre = Feat_Spectre()

database = pd.DataFrame(spectre)
database.insert(19, "marks", ev, True)
database.insert(0, "freqs", freqs1, True)
print(database)
