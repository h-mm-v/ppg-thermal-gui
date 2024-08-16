import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import numpy as np
import scipy.signal as sp
from scipy.fft import fft, fftfreq
import scipy.interpolate as spi
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SAMPLE_RATE = 60

######################################################### - Analysis - ###########################################################

def analyze(signal, window_size, monitor_window):
    '''
    takes in a complete PPG signal in a numpy file
    takes in an integer to represent the size of each window in seconds
    takes in a customtkinter/tkinter window to connect the new window to
    returns the filtered ppg graph and its derivatives
    returns the hr and rr estimation for the entire signal
    '''
    #Create arrays to store data from each window
    global ppg_analysis_window
    
    total_datapoints = len(signal)
    try:
        samples_in_window = window_size * SAMPLE_RATE
        total_windows = int(total_datapoints / samples_in_window)
    except:
        samples_in_window = len(signal)
        window_size = samples_in_window / 60
        total_windows = 1

    ppg_analysis_window = ctk.CTkToplevel(master=monitor_window)
    ppg_analysis_window.grid_rowconfigure([0,1], weight=1)
    ppg_analysis_window.grid_columnconfigure([0,1,2], weight=1)
    ppg_analysis_window.geometry('1350x800')

    hr_array = np.zeros(total_windows)
    rr_array = np.zeros(total_windows)
    rr_weight_array = np.zeros(total_windows)
    show_next_window = True
    window_index = 0
    #importing data from numpy file
    while(show_next_window):
        plt.close("all")

        y = signal[int(window_index * samples_in_window):samples_in_window + int(window_index * samples_in_window)]
        raw_y, y, dy, ddy = pre_processing(y, 0.5, 5)

        #Calculating indexes of peak and avg heartrate
        sys_peaks_indexes, dia_peaks_indexes, pulse_onsets, dicrotic_notches = experimental_peak_detection(y = y, dy = dy)

        #Creating PPG waveform
        time_vector = np.arange(0, window_size, window_size/samples_in_window)
        avg_hr1, hr_confidence, power_spectrum, time_freq_vector = freq_spectrum(raw_y, 0.5, 5)

        fig, (ax1) = plt.subplots(1,1, figsize=(6,4))
        fig.tight_layout(pad=2)
        ax1.plot(time_freq_vector, power_spectrum)
        ax1.set(title = "Heartrate Power Spectrum")
        global hr_power_spectrum
        hr_power_spectrum = FigureCanvasTkAgg(fig,
                                            master = ppg_analysis_window)
        hr_power_spectrum.draw()

        top_envelope = create_envelopes(sys_peaks_indexes, raw_y)
        avg_hr2, resp_rate, weight = get_heartrate_biomarkers(sys_peaks_indexes, top_envelope, pulse_onsets, y)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = True, sharey = False, figsize=(6,4))
        fig.tight_layout(pad=2)
        ax1.plot(time_vector, raw_y, color = "red", markevery = sys_peaks_indexes, marker = 'x', markeredgecolor = 'k', markerfacecolor = 'k')
        ax1.set(title = "Raw ppg")
        ax2.plot(time_vector, y, color = "red", markevery = sys_peaks_indexes, marker = 'x', markeredgecolor = 'k', markerfacecolor = 'k')
        ax1.plot(time_vector[:sys_peaks_indexes[-1]], top_envelope, color = "orange")
        ax2.plot(time_vector, y, color = "None", markevery = dia_peaks_indexes, marker = 'x', markeredgecolor = 'b', markerfacecolor = 'b')
        ax2.plot(time_vector, y, color = "None", markevery = pulse_onsets, marker = 'x', markeredgecolor = 'g', markerfacecolor = 'g')
        ax2.plot()
        ax2.set(title = "Filtered ppg")
        ax3.plot(time_vector, dy, color = "blue")
        ax3.set(title = "Filtered ppg\'")
        ax4.plot(time_vector, ddy, color = "green")
        ax4.set(xticks = np.arange(window_size + 1), xlabel = "Seconds", title = "Filtered ppg\'\'")

        waveforms = FigureCanvasTkAgg(fig,
                                    master = ppg_analysis_window)
        waveforms.draw()

        hr_label = ctk.CTkLabel(master = ppg_analysis_window,
                           text = "Window #" + str(window_index + 1) + "/" + str(total_windows)
                           + "\n\nAverage HR  (Freq): " + str(round(avg_hr1, 2)) + " beats/min @ " + str(round(hr_confidence, 2)) + " confidence"
                           + "\nAverage HR (Pulses): " + str(round(avg_hr2, 2)) + " beats/min"
                           + "\nEstimated RR: " + str(round(resp_rate, 2)) + " breaths/min",
                           font = ("Times New Roman", 16))
        
        print("Window " + str(window_index + 1))
        print("Average Heartrate (Freq): " + str(round(avg_hr1, 2)) + " beats / min @ " + str(hr_confidence) + " confidence")
        print("Average Heartrate (Pulses): " + str(round(avg_hr2, 2)) + " beats / min")
        print("Estimated Respiratory Rate: " + str(round(resp_rate, 2)) + " breaths / min")
        print()
        hr_array[window_index] = (avg_hr1 + avg_hr2) / 2
        rr_array[window_index] = resp_rate
        rr_weight_array[window_index] = weight

        hr_label.grid(row=0,column=0, padx=5, ipady = 10, sticky=ctk.S)
        resp_rate_labels.grid(row=1,column=0, padx=5, ipady = 10, sticky=ctk.N)

        waveforms.get_tk_widget().grid(row=0,column=1,padx=5,pady=5)
        hr_power_spectrum.get_tk_widget().grid(row=1,column=1, padx=5, pady=5)
        bio_markers.get_tk_widget().grid(row=0,column=2, padx=5, pady=5)
        power_spectrums.get_tk_widget().grid(row=1,column=2, padx=5, pady=5)

        window_index += 1
        
        if (window_index < total_windows):
            response = CTkMessagebox(master=ppg_analysis_window,
                            title="Show Next Window Prompt",
                            message="Continue to next window?",
                            option_1="Yes",
                            option_2="No")
            if (response.get() == "No"):
                show_next_window = False
        else:
            show_next_window = False

    hr_array = np.trim_zeros(hr_array)
    rr_array = np.trim_zeros(rr_array)
    rr_weight_array = np.trim_zeros(rr_weight_array)
    hr_estimation = round(np.mean(hr_array), 2)
    try:
        rr_estimation = round(np.average(rr_array, weights=rr_weight_array), 2)
    except:
        rr_estimation = round(np.mean(rr_array), 2)

    CTkMessagebox(master=ppg_analysis_window,
                  title="Results",
                  message="Overall Estimations - \nHR: " + str(hr_estimation) + " beats / min\nRR: " + str(rr_estimation) + " breaths / min").get()
    print("Overall Estimations - HR: " + str(hr_estimation) + " beats / min || RR: " + str(rr_estimation) + " breaths / min")

    hr_label.destroy()
    waveforms.get_tk_widget().destroy()
    power_spectrums.get_tk_widget().destroy()
    bio_markers.get_tk_widget().destroy()
    resp_rate_labels.destroy()
    hr_power_spectrum.get_tk_widget().destroy()
    ppg_analysis_window.destroy()

    return fig, hr_estimation, rr_estimation

########################################################################## - Peak Detection - #######################################################################################################

def experimental_peak_detection(y, dy):
    '''
    takes in a signal window and its derivative
    returns systolic peaks, diastolic peaks, pulse onsets, and dicrotic notches
    '''
    peak_enhanced_y = peak_enhance(y,  0, 1024, steps = 2)
    plt.show()
    max_index = np.where(np.gradient(np.sign(dy)) < 0)[0]
    max_index = np.delete(max_index, np.where(y[max_index] < np.percentile(y, 30))[0])

    max_slices = np.empty(len(max_index), dtype=np.ndarray)
    
    last_slice = 0
    slice_counter = 0

    for i in range(len(max_index)):
        if (max_index[i] - max_index[i-1] >= 15):
            max_slices[slice_counter] = max_index[last_slice:i]
            last_slice = i
            slice_counter += 1
    max_slices[slice_counter] = max_index[last_slice:]
    max_slices = max_slices[:slice_counter + 1] #Trim excess space in array

    peaks = np.zeros(len(max_slices), dtype = np.int64)
    for i in range(len(max_slices)):
        peaks[i] = np.argmax(peak_enhanced_y[max_slices[i][0]:max_slices[i][-1] + 1]) + max_slices[i][0]

    sys_peaks = np.delete(peaks, np.where(peak_enhanced_y[peaks] < np.percentile(peak_enhanced_y, 60))[0])
    sys_peaks = np.delete(sys_peaks, np.where(y[sys_peaks] < np.percentile(y, 40)))
    dia_peaks = np.zeros(len(sys_peaks), dtype = np.int64)
    pulse_onsets = np.zeros(len(sys_peaks), dtype = np.int64)
    dic_notches = np.zeros(len(sys_peaks), dtype = np.int64)

    for i in range(len(sys_peaks) - 1):
        pulse_onsets[i] = np.argmin(y[sys_peaks[i]:sys_peaks[i+1]]) + sys_peaks[i]
    pulse_onsets = np.roll(pulse_onsets, 1)
    pulse_onsets[0] = np.argmin(y[:sys_peaks[0]])

    for i in range(len(sys_peaks) - 1):
        dic_notches[i] = np.argmin(y[sys_peaks[i]:sys_peaks[i]+(sys_peaks[i]-pulse_onsets[i])+2]) + sys_peaks[i]

    for i in range(len(dic_notches) - 1):
            try:
                dia_peaks[i] = np.argmax(y[dic_notches[i]+1:pulse_onsets[i+1]]) + dic_notches[i]
            except:
                break

    sys_peaks[-1] = np.argmax(y[pulse_onsets[-1]:]) + pulse_onsets[-1]

    pulse_onsets = np.trim_zeros(pulse_onsets)
    dia_peaks = np.trim_zeros(dia_peaks)
    dic_notches = np.trim_zeros(dic_notches)

    return sys_peaks, dia_peaks, pulse_onsets, dic_notches



############################################## - Gets heartrate and heartrate variability - #####################################

def get_heartrate_biomarkers(sys_peaks, top_envelope, pulse_onsets, y):
    '''
    takes in systolic peaks, pulse onsets, envelope, and signal
    returns average hr, rr estimate array, and rr estimate weight array
    '''
    shifted_peaks = np.roll(sys_peaks, 1)
    shifted_peaks[0] = 0
    samples_between_peaks = sys_peaks - shifted_peaks
    avg_hr = (SAMPLE_RATE / np.mean(samples_between_peaks)) * 60
    shifted_peaks = np.roll(samples_between_peaks, 1)
    shifted_peaks[0] = 0
    hrv = samples_between_peaks - shifted_peaks
    hrv = hrv[2:] / 60
    
    resp_rate, weight = get_resp_rate(top_envelope, sys_peaks, pulse_onsets, hrv, y)
    return avg_hr, resp_rate, weight

def get_resp_rate(top_envelope, sys_peaks, pulse_onsets, hrv, y):
    '''
    takes in envelope, systolic peaks, pulse onsets, heart rate variability, and signal
    returns 3 estimates for rr based on fft analysis of hrv, pulse amplitude, and signal intensity
    returns weights for each rr estimate based on goodness metric
    '''
    resampled_hrv = sp.resample(hrv, len(y))

    peak_to_peak = y[sys_peaks[:len(pulse_onsets)]] - y[pulse_onsets]
    resampled_peak_to_peak = sp.resample(peak_to_peak, len(y))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False, figsize=(6,4))
    fig.tight_layout(pad=2)
    #ax1.plot(hrv)
    ax1.plot(resampled_hrv)
    ax1.set(title = 'Heart Rate Variability')
    #ax2.plot(peakToPeak)
    ax2.plot(resampled_peak_to_peak)
    ax2.set(title = 'Systolic Peak Amplitudes')
    ax3.plot(top_envelope)
    ax3.set(title = 'Top Envelope')

    global bio_markers
    bio_markers = FigureCanvasTkAgg(fig,
                                   master = ppg_analysis_window)
    bio_markers.draw()
    
    resp_rate1, resp_rate1_confidence, power_spectrum1, time_freq_vector1 = freq_spectrum(resampled_hrv, 0.067, 0.6)
    resp_rate2, resp_rate2_confidence, power_spectrum2, time_freq_vector2 = freq_spectrum(resampled_peak_to_peak, 0.2, 0.6)
    resp_rate3, resp_rate3_confidence, power_spectrum3, time_freq_vector3 = freq_spectrum(top_envelope, 0.2, 0.6)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False, figsize=(6,4))
    fig.tight_layout(pad=2)
    ax1.plot(time_freq_vector1, power_spectrum1)
    ax1.set(title = "Heartrate Variability Power Spectrum")
    ax2.plot(time_freq_vector2, power_spectrum2)
    ax2.set(title = "Peak To Peak Power Spectrum")
    ax3.plot(time_freq_vector3, power_spectrum3)
    ax3.set(title = "Top Envelope Power Spectrum")

    global power_spectrums
    power_spectrums = FigureCanvasTkAgg(fig,
                                       master = ppg_analysis_window)
    power_spectrums.draw()

    global resp_rate_labels
    resp_rate_labels = ctk.CTkLabel(master=ppg_analysis_window,
                              text= "HRV RR Estimation: " + str(round(resp_rate1, 2)) + " @ " + str(round(resp_rate1_confidence, 2)) + " confidence"
                              + "\nAmplitude RR Estimation: " + str(round(resp_rate2, 2)) + " @ " + str(round(resp_rate2_confidence, 2)) + " confidence"
                              + "\nEnvelope RR Estimation: " + str(round(resp_rate3, 2)) + " @ " + str(round(resp_rate3_confidence, 2)) + " confidence",
                              font= ("Times New Roman", 16))
    

    #print("Resp Rates: " + str(resp_rate1) + ", " + str(resp_rate2) + ", " + str(resp_rate3))
    #print("Confidence Values: " + str(resp_rate1_confidence) + ", " + str(resp_rate2_confidence) + ", " + str(resp_rate3_confidence))
    weights = [resp_rate1_confidence, resp_rate2_confidence, resp_rate3_confidence]
    try:
        avg_resp_rate = np.average([resp_rate1, resp_rate2, resp_rate3], weights=weights)
    except:
        avg_resp_rate = np.mean([resp_rate1, resp_rate2, resp_rate3])

    return avg_resp_rate, np.sum(weights)


############################### - Uses butterworth bandpass to smooth out signal and calculate derivs - ################################

def pre_processing(y, l_crit_freq, h_crit_freq):
    '''
    takes in signal and critical frequencies for bandpass filter
    passes butterworth bandpass filter forwards and backwards through inputted signal
    '''
    raw_y = y
    b, a = sp.butter(4, [l_crit_freq, h_crit_freq], 'bandpass', fs = 60)
    filtered_y = sp.filtfilt(b, a, y, padtype=None)
    dy = np.gradient(filtered_y)
    ddy = np.gradient(dy)
    return raw_y, filtered_y, dy, ddy

def peak_enhance(y, l_bound, u_bound, steps):
    '''
    takes in signal and parameters for algorithm
    takes in number of times for algorithm to run
    returns peak-enhanced signal
    '''
    for i in range(steps):
        y = np.power(y, 2)
        range_of_bound = u_bound - l_bound
        data_in_range = np.max(y) - np.min(y)
        y = range_of_bound * ((y - (np.min(y)))/data_in_range)+l_bound
    return y

##################################### - Creates Freq Spectrum - #########################################################

def freq_spectrum(y, low_freq_bound, high_freq_bound):
    '''
    takes in signal and window bounds
    returns fft and calculates dominant frequency and goodness metric
    '''
    #Generate fourier transform and bounds
    y = y - np.mean(y)
    yft = np.abs(fft(y, 6000))
    time_freq_vector = fftfreq(6000, 1/SAMPLE_RATE)
    low_freq_bound = np.argwhere(time_freq_vector >= low_freq_bound)[0][0]
    high_freq_bound = np.argwhere(time_freq_vector >= high_freq_bound)[0][0]
    freq_bound_range = high_freq_bound - low_freq_bound

    #Cut fourier transform to window bounds
    yft = yft[low_freq_bound:high_freq_bound]
    time_freq_vector = time_freq_vector[low_freq_bound:high_freq_bound]

    #Square each fft value and calculate dominant freq.
    power_spectrum = yft ** 2
    dom_freq_index = np.argmax(yft)

    #Calculate Confidence Values
    freq_power = np.sum(power_spectrum[dom_freq_index - int(freq_bound_range * .05):dom_freq_index + int(freq_bound_range * .05)])
    noise_power = np.sum(power_spectrum)
    confidence_value = freq_power / noise_power

    return time_freq_vector[dom_freq_index] * SAMPLE_RATE, confidence_value, power_spectrum, time_freq_vector

##################################################### - Envelope - ###########################################################

def create_envelopes(sys_peaks, y):
    '''
    takes in systolic peaks and signal
    returns envelope using cubic spline interpolation
    '''
    for i in range(len(sys_peaks)):
        try:
            sys_peaks[i] = np.argmax(y[sys_peaks[i] - 5:sys_peaks[i] + 5]) + sys_peaks[i] - 5
        except:
            break

    sys_peaks = sys_peaks[[i + np.argmax(y[sys_peaks[i:i + 1]]) for i in range(0, len(sys_peaks), 1)]]

    cubic_top_envelope = spi.CubicSpline(sys_peaks, y[sys_peaks])

    top_envelope = np.zeros(len(y))

    for k in range(0, len(y)):
        top_envelope[k] = cubic_top_envelope(k)

    top_envelope = top_envelope[:sys_peaks[-1]]
    return top_envelope
