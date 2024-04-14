import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import wave

#open the .wav file and extract sampling frequency, samples and do the DTFT:
def DTFT(wav_file):
    #read .wav file, print sampling rate and # of samples:
    sampling_rate, samples = wavfile.read(wav_file)
    print("Sampling frequency:", sampling_rate)
    print("Number of Samples:", len(samples))
    print("samples = ", samples)

    #Calculate the Discrete time Fourier Transform of input signal
    inputfft = np.fft.fft(samples)
    #Magnitude of input signal:
    Xomega = np.abs(inputfft)
    #frequency axis for plotting:
    faxis = np.fft.fftfreq(len(samples), d=1/sampling_rate)
    return faxis, Xomega

#Function to plot signal
def plotfrequency(horizontal_axis, vertical_axis, plot_label):
    plt.figure(figsize = (22, 14))
    plt.plot(horizontal_axis, np.abs(vertical_axis), label= plot_label)
    plt.title(plot_label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude X(omega)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.autoscale()

######################################### [ MAIN ] ######################################################


#calculate the DTFT of .wav file and plotting it:
input_faxis, input_Xomega = DTFT(r"C:\Users\ducvi\PycharmProjects\S&S 1.94\.venv\thuong em.wav")
plotfrequency(input_faxis, input_Xomega, ".wav file")


#20th Order Butterworth Filter (300Hz) constants:
H1_fo = 44100                       #Resonant frequency
H1_fc = 300                         #Cur off frequency
H1_order = 20                       #order filter
#nyquist frequency and cut off angular frequency:
nyquist_f = 0.5 * H1_fo             #to avoid aliasing
H1_wc = 2 * np.pi * H1_fc           #cut off angular frequency
#Butterworth filter equation:
H1_freqaxis = np.fft.fftfreq(len(input_Xomega), d=1 / H1_fo)    #Frequency axis for H1_omega
H1_freqaxis_nonzero = H1_freqaxis[H1_freqaxis != 0]             #Exclude zero frequencies
# Calculate frequency response of Butterworth filter
H1_omega_squared = 1 / (1 + ((H1_wc) / (H1_freqaxis_nonzero))**(2 * H1_order))
H1_omega = np.sqrt(H1_omega_squared)
# Graph the filter
plotfrequency(H1_freqaxis_nonzero, H1_omega, "Frequency response of filter 1")


#21th Order Butterworth Filter (2000Hz) constants:
H2_fo = 44100                       #Resonant frequency
H2_fc = 2000                         #Cur off frequency
H2_order = 21                       #order filter
#nyquist frequency and cut off angular frequency:
nyquist_f = 0.5 * H2_fo             #to avoid aliasing
H2_wc = 2 * np.pi * H2_fc           #cut off angular frequency
#Butterworth filter equation:
H2_freqaxis = np.fft.fftfreq(len(input_Xomega), d=1 / H2_fo)    #Frequency axis for H1_omega
H2_freqaxis_nonzero = H2_freqaxis[H2_freqaxis != 0]             #Exclude zero frequencies
# Calculate frequency response of Butterworth filter
H2_omega_squared = 1 / (1 + ((H2_wc) / (H2_freqaxis_nonzero))**(2 * H2_order))
H2_omega_low = np.sqrt(H2_omega_squared)
H2_omega_high = 1 - H2_omega_low
# Graph the filter
plotfrequency(H2_freqaxis_nonzero, H2_omega_high, "Frequency response of filter 2")

#plotting the frequency response of two filters put in cascade:
plotfrequency(H2_freqaxis_nonzero, H1_omega + H2_omega_high, "Frequency of combined filters")

#Output function is convolution of System and input
h1_n = np.fft.ifft(H1_omega)
h2_n = np.fft.ifft(H2_omega_high)
xn_sampling, x_n = wavfile.read(r"C:\Users\ducvi\PycharmProjects\S&S 1.94\.venv\thuong em.wav")

convolution_result = np.real(h1_n + h2_n) / 2
convolution_resultflat = convolution_result.flatten()
y_n = np.convolve(x_n, convolution_resultflat, mode = 'full')
print("convolution data:", y_n)
print("data length:", len(y_n))

#output it into a .wav file
with wave.open('output.wav', 'w') as wf:
    output = np.int16(y_n * 32767)
    wf.setnchannels(1)  # mono audio
    wf.setsampwidth(2)  # 2 bytes (16 bit)
    wf.setframerate(44100)
    wf.writeframes(output.tobytes())

