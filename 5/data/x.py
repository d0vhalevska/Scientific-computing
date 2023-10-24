fdata_even = shuffle[::2]
fdata_odd = shuffle[1::2]
terms = np.exp(-2j * np.pi * np.arange(n) / n)
fdata = np.concatenate([fdata_even + terms[:int(n / 2)] * fdata_odd,
                        fdata_even + terms[int(n / 2):] * fdata_odd])