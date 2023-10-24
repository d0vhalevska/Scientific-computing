if n <= 2:
    return dft_matrix(len(data))
else:
    fdata_even = fft(fdata[::2])
    fdata_odd = fft(fdata[1::2])
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    np.concatenate([fdata_even + w[:int(n / 2)] * fdata_odd,
                    fdata_even + w[int(n / 2):] * fdata_odd])

# TODO: normalize fft signal
fdata = np.dot(fdata, 1 / np.sqrt(n))




______________________-

cut = int(np.log(n) / np.log(2))
for i in range(cut):
    m = 2 ** (i - 1)
    w = np.exp(-2 * np.pi * 1j / m)
    for j in range(2 ** cut - i):

