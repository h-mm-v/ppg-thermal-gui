import ppg_analysis
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import cv2
import dlib
from PIL import Image
import imageio
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pathlib
import scipy.signal as sp
from scipy.fft import fft, fftfreq
import serial
import tkinter as tk
import threading
from tkinter import ttk
import os

#Class Variables
SAMPLE_RATE = 60

############################## - Import data from pulse ox - ####################################################################
def import_ppg_data_live():
    '''
    Recieves PPG data from a serial port and saves it into a subject folder
    '''
    try:
        num_seconds = int(sample_duration_entry.get())
    except:
        error_message_box()
        return
    #open serial port
    try:
            ser = serial.Serial(port=serial_port_entry.get(),
                                baudrate=115200,
                                bytesize=serial.EIGHTBITS,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                timeout=1,
                                xonxoff=1)
    except serial.SerialException as e:
        error_message_box()
        print(f"Serial exception: {e}")
        return

    waveform_data_list = []
    packages = []

    #Number of packets to read
    num_bytes = num_seconds * SAMPLE_RATE * 9
    ser.write(b'\x7d\x81\xa1\x80\x80\x80\x80\x80\x80')

    num_bytes_recorded = tk.IntVar(master=monitor_window, value=0)
    progress_label = ctk.CTkLabel(master=monitor_window,
                                text="Recording (" + str(num_bytes_recorded.get()) + "/" + str(num_bytes) + ")")
    progress_label.grid(row=2, column=0)
    recording_progress = ttk.Progressbar(master= monitor_window,
                                        orient=tk.HORIZONTAL,
                                        maximum = num_bytes,
                                        variable=num_bytes_recorded,
                                        length=250)
    recording_progress.grid(row=3,column=0)

    live_waveform_window = ctk.CTkToplevel(master=monitor_window)
    #create figure
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    fig.tight_layout(pad=2)
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], "r-")
    canvas = FigureCanvasTkAgg(fig, master=live_waveform_window)
    canvas.get_tk_widget().grid(row=0, column=0)
    ax.set_xlim(0, num_bytes)
    ax.set_ylim(0, 128)

    live_hr = ctk.StringVar(value="")
    live_hr_label = ctk.CTkLabel(master=live_waveform_window,
                           text="Avg. HR: " + str(live_hr.get()),
                           font=("Times New Roman", 16))
    live_hr_label.grid(row=1,column=0)

    def init_plot():
        '''
        Returns a matplotlib line to represent the pulse oximeter waveform
        '''
        line.set_data([], [])
        return line,

    def update_plot(frame):
        '''
        Updates line with new data read from serial port
        '''
        if ser.is_open:
            try:
                data = ser.read(9)
                if data:
                    packages.append(data)
            except serial.SerialException as e:
                print(f"Serial exception during update: {e}")
        num_bytes_recorded.set(num_bytes_recorded.get() + 9)
        progress_label.configure(text="Recording (" + str(num_bytes_recorded.get()) + "/" +
                                str(num_bytes) + ")")
        flattened_packets = np.frombuffer(b''.join(packages), dtype=np.uint8)
        third_byte_indices = (flattened_packets & 0x80 == 0).nonzero()[0] + 3
        try:
            waveform_data_list.append(flattened_packets[third_byte_indices[-1]] & 0x7F)
        except:
            return
        # for idx in third_byte_indices:
        #     if idx < len(flattened_packets):
        #         waveform_data_list.append(flattened_packets[idx] & 0x7F)

        line.set_data(range(len(waveform_data_list)), waveform_data_list)
        if (len(waveform_data_list) < 10 * SAMPLE_RATE):
            ax.set_xlim(0, max(len(waveform_data_list), 1))
        else:
            ax.set_xlim(len(waveform_data_list) - 10 * SAMPLE_RATE, len(waveform_data_list))
        #canvas.draw()

        #if(len(waveform_data_list) > 10 * SAMPLE_RATE):
        live_hr.set(round(ppg_analysis.freq_spectrum(waveform_data_list, .5, 5)[0], 2))
        live_hr_label.configure(text="Avg. HR: " + str(live_hr.get()))



        monitor_window.update()
        return line,
    try:
        ani = animation.FuncAnimation(fig=fig, func=update_plot, init_func=init_plot, blit=True, interval=1, cache_frame_data=False)
    except:
        error_message_box()
        return
    

    def finalize_recording():
        '''
        Destroys display window and creates final estimations
        '''
        ser.close()
        progress_label.destroy()
        recording_progress.destroy()

        np.save(subject_folder_path + "/" + subject_name + "_" + str(sample_index + 1) + "_ppg.npy", waveform_data_list)

        CTkMessagebox(master=live_waveform_window,
                      title="Recording Complete",
                      message="Recording Complete" + 
                      "\nAvg. HR: " + str(live_hr.get())).get()
        ani.event_source.stop()
        live_waveform_window.destroy()

        ppg_analysis.analyze(signal=np.array(waveform_data_list))

        

    def start_recording():
        '''
        Starts and finishes the recording process
        '''
        ani._start()
        monitor_window.after(num_seconds * 1000, finalize_recording)  # Convert seconds to milliseconds

    def run_recording():
        '''
        Starts a thread to run the recording process in
        '''
        threading.Thread(target=start_recording).start()

    run_recording()
    #ani.save("ppg_recording.mp4", writer="ffmpeg", fps=60)

def get_subject_path():
    '''
    Takes subject parameters from the GUI and creates a folder and path to save data in
    '''
    subject_name = subject_name_entry.get().lower()
    
    subject_folder_path = str(pathlib.Path().resolve()) + "/" + subject_name
    if not os.path.exists(subject_folder_path):
        os.makedirs(subject_folder_path)
    try:
        if ppg_check.get() and thermal_check.get():
            sample_index = int(len(os.listdir(subject_folder_path)) / 3)
        elif (thermal_check.get()):
            sample_index = int(len(os.listdir(subject_folder_path)) / 2)
        elif (ppg_check.get()):
            sample_index = int(len(os.listdir(subject_folder_path)))
    except:
        sample_index = 0
    
    return subject_name, subject_folder_path, sample_index

############################################################### - Thermal Camera - ##########################################################

def record_thermal_camera():
    '''
    Records thermal data to the subject folder in a .avi and .npy frames file
    '''
    camera_sample_rate = float(thermal_sample_rate_entry.get())

    frame_total = int(sample_duration_entry.get()) * camera_sample_rate

    cap_vid = cv2.VideoCapture(int(usb_port_entry.get()),cv2.CAP_DSHOW)
    # this should get the raw data
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')  
    fourcc = cv2.VideoWriter.fourcc('Y','1','6',' ')
    ret = cap_vid.set(cv2.CAP_PROP_FOURCC,fourcc)
    ret = cap_vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    #ret = cap_vid.set(cv2.CAP_PROP_FOURCC, fourcc)
    print(ret)
    frames = []
    timestamps = []
    # use datetime.now for timestamp?
    while(len(frames) <= frame_total):
        ret, frame = cap_vid.read()      # frame is still in 8 bit
        # print(ret)
        if ret==True:                   # use capture() function?
            frame = cv2.flip(frame,1)  
            #cv2.normalize(frame, frame, 0, 65535, cv2.NORM_MINMAX)
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB )
            # write the flipped frame  
            frame_time = cap_vid.get(cv2.CAP_PROP_POS_MSEC)
            frames.append(frame)
            # timestamps.append(str(datetime.now()))
            timestamps.append(frame_time)
        disp_frame = (((frame - frame.min())/(frame.max() - frame.min()))*255).astype(np.uint8)
        cv2.imshow('frame',disp_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        print(frame)
        # monitor_window.update()
    frames = np.array(frames)
    # looking at the last frame --> works for lepton
    print(frame.dtype, frames.dtype)
    # print(timestamps[-1])
    cap_vid.release()
    cv2.destroyAllWindows()
    save_frames = (frames - frames.min())/(frames.max() - frames.min())
    save_frames = (save_frames * 255).astype(np.uint8)
    imageio.mimwrite(subject_folder_path + "/" + subject_name + "_" + str(sample_index + 1) + "_thermalvid.avi", save_frames, fps=60)
    np.save(subject_folder_path + "/" + subject_name + "_" + str(sample_index + 1) + "_frames.npy",frames)

    return

def analyze_thermal():
    '''
    Estimates respiratory rate from nose region of the thermal video
    '''
    subject_name = subject_name_entry.get()
    sample_number = sample_number_entry.get()

    visibleframes = np.load(str(pathlib.Path().resolve()) + "/" + subject_name.lower() + "/" + subject_name.lower() + "_" + sample_number + "thermalvid.npy")
    framearrays = np.load(str(pathlib.Path().resolve()) + "/" + subject_name.lower() + "/" + subject_name.lower() + "_" + sample_number + "_frames.npy")

    def gaussfilter(frame):
        '''
        returns blurred frame relative to its size
        '''
        framenumrows = frame.shape[0]

        windowsize = int(0.025*framenumrows)
        if windowsize % 2 == 0 :
            windowsize += 1

        gauss = cv2.GaussianBlur(frame,(windowsize, windowsize),0)
        return gauss


    def framesetup(testframe,scaleval):
        '''
        resizes the given frame
        '''

        highcontrast = imutils.resize(testframe, width=500)

        gauss = gaussfilter(highcontrast)

        return gauss, highcontrast

    def get_coords(detector, image, upsample, model):
        '''
        takes in file paths for detector and model
        takes in frame to analyze and number for upsample
        returns coordinates for face bounding box
        returns coordinates for 4 face landmarks around the nose
        '''
        args = {
            "detector": detector,
            "images" : image,
            'upsample' : upsample,
            'model' : model
        }

        # load the face detector (HOG-SVM)
        detector = dlib.simple_object_detector(args["detector"])

        # load the facial landmarks predictor
        predictor = dlib.shape_predictor(args["model"])

        # copy the image
        image_copy = image.copy()

        # detect faces in the image 
        rects = detector(image, upsample_num_times=args["upsample"])	


        for rect in rects:
            # convert the dlib rectangle into an OpenCV bounding box and
            # draw a bounding box surrounding the face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # predict the location of facial landmark coordinates, 
            # then convert the prediction to an easily parsable NumPy array
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates from our dlib shape
            # predictor model draw them on the image
            # add coordinates of relevant face landmarks to x and y lists
            x_coords = []
            y_coords = []
            i = 1
            for (sx, sy) in shape:
                cv2.circle(image_copy, (sx, sy), 2, (0, 0, 255), -1)
                if i == 28 or i == 51 or i == 49 or i == 50:
                    x_coords.append(sx)
                    y_coords.append(sy)
                i += 1

        return (x,y,w,h), x_coords, y_coords



    def frameanalyze(testframe, firstframe, scaleval):
        '''
        Takes in a frame, the first frame, and a scaling value
        displays frame with face and nose bounding box
        returns a high contrast frame and the coordinates for the nose box
        '''
        # set up the testframe and first frame
        gauss, highcontrast = framesetup(testframe, scaleval)
        firstgauss, firsthighcon = framesetup(firstframe, scaleval)

        # initialize detector and model file paths and upsample value
        detector = 'thermal-facial-landmarks-detection\\models\\dlib_face_detector.svm'
        model = "thermal-facial-landmarks-detection\\models\\dlib_landmark_predictor.dat"
        upsample = 1
        originalsize = highcontrast.shape
        print(originalsize)

        # get face bounding box and landmark coordinates
        rects, x_coords, y_coords = get_coords(detector, highcontrast, upsample, model)
        (x,y,w,h) = rects
        print(f'x-coords: {x_coords}')
        print(f'y-coords: {y_coords}')
        # show image with face bounding box
        image = cv2.rectangle(highcontrast, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Image", image)

        facebox = rects
        print(facebox)
        # initialize nosebox with face landmark coordinates
        nosebox = [np.min(x_coords), np.max(x_coords), np.min(y_coords), y_coords[2]]

        x, y, w, h = facebox

        data = Image.fromarray(testframe)
        data.save('testpic.png')
        frame = cv2.imread('testpic.png')

        # display frame with face and nose bounding box
        highcontrast = cv2.rectangle(highcontrast, (x, y), ((x+w), (y+h)), (255, 0, 0), 2)
        highcontrast = cv2.rectangle(highcontrast, (nosebox[0],nosebox[2]), (nosebox[1],nosebox[3]), (255, 0, 0), 2)
        cv2.imshow('frame', highcontrast)


        return highcontrast, nosebox


    newframes = []
    noserows = []
    rois = []
    scaleval = 1

    # get and analyze the first frame for face and nose finding
    firstframe = visibleframes[0][:,:,0]
    anchorframe = visibleframes[1][:,:,0]
    fhighcon, firstnosebox = frameanalyze(anchorframe, firstframe, scaleval)

    # initialize the coordinates for the box around the nose
    x = int(firstnosebox[0])
    y = int(firstnosebox[2])
    bx = int(firstnosebox[1])
    by = int(firstnosebox[3])
    print(x,bx,y,by)

    _, firstlow = framesetup(framearrays[1][:,:], scaleval)
    _, anchorframe = framesetup(anchorframe, scaleval)
    noseboxh = by - y

    # set the size of the vertical bins
    binsize = 8
    # set the number of bins in each row
    binsperrow = 2
    # calculate the number of vertical bins
    numbins = noseboxh // binsize
    print('numbins ', numbins)

    binneddata = []

    # isolate the nose from the low contrast frame
    boxsegment = firstlow[y:by,x:bx]
    plt.imshow(boxsegment)
    plt.show()
    noseboxseg = boxsegment

    # set up all the frames in the videos
    scaledframearrays = []
    for frame in framearrays:    
        scaledframearrays.append(framesetup(frame[:,:], scaleval)[1])
    scaledframearrays = np.asarray(scaledframearrays)

    # split the nose box into bins
    fracacross = int((bx-x)/binsperrow)
    heatgraph = []
    for f in range(len(scaledframearrays)):
        frame = scaledframearrays[f]
        binavg = []
        framebins = []
        # going through the bins in one frame
        for i in range(numbins):       # each row of bins
            row = []
            for j in range(binsperrow):         # bins in each row
                if (j != binsperrow - 1):
                    boxsegment = frame[y+(i*binsize):y+((i+1)*binsize),(x+j*fracacross):(x+(j+1)*fracacross)]
                else:
                    boxsegment = frame[y+(i*binsize):y+((i+1)*binsize),(x+j*fracacross):bx]
                respsig = np.mean(boxsegment)
                binavg.append(respsig)
                row.append(respsig)
            #noseboxseg = cv2.rectangle(noseboxseg,(0,(i*10)),(-1,(i+1)*10),(0,0,0),2)
            framebins.append(row)
        # all bins avgs in frame f
        binneddata.append(binavg)       # all bins together
        heatgraph.append(framebins)     # bins separated by row
    heatgraph = np.asarray(heatgraph)

    binneddata = np.asarray(binneddata)

    plt.plot(binneddata[:,:])
    plt.show()


    def calcfft(respiratory_rate_data):
        '''
        performs the fft of the function
        finds the freq with the peak in the fft --> primary freq of periodicity
        calculates the goodness metric using the integrals
        '''
        # perform fft
        # filter the signal
        respiratory_rate_data = sp.ndimage.gaussian_filter(respiratory_rate_data,15)
        # take the derivative of the signal
        respiratory_rate_data = np.diff(respiratory_rate_data)
        # calculate fft
        yft = fft(respiratory_rate_data)  # returns complex numbers for mag and phase
        N = len(yft)
        SAMPLE_RATE = 60
        tsteps = 1 / SAMPLE_RATE
        # x axis time (seconds)
        t = np.linspace(0, (N-1)*tsteps, N)     
        yft_mag = np.abs(yft)  /N            # only care about the mag
        yft_mag = yft_mag**2
        # x axis freq (Hz)
        f = fftfreq(N, 1/SAMPLE_RATE)   
        f = f[0:int(N/2)]
        # multiply signal by two except for first value
        yft_mag_plot =2 * yft_mag[0:int(N/2)]
        yft_mag_plot[0] = yft_mag_plot[0]/2

        fftpeak = np.argmax(yft_mag_plot)   # peak of fft --> gives INDEX NOT FREQ

        hzpeak = f[fftpeak] # get freq of peak
        # return fft and frequency arrays
        # return the index and freq of the frequency peak
        return yft_mag_plot, f, hzpeak,fftpeak


    def goodnessmetric(respiratory_rate_data):
        '''
        Takes in respiratory data
        returns periodicity goodness metric
        '''
        # calculate the data fft
        yft_mag_plot, f, hzpeak, fftpeak = calcfft(respiratory_rate_data) 

        # these are in hz --> not the indexes
        # resonable respiration frequency range (0.15 - 60 breaths/min)
        B1 = 0.025
        B2 = 1
        # small range around the frequency spectrum peak
        b = 0.05 * (B2 - B1)    # taking 5% of entire range for the peak range
        # indices for B1 and B2
        B1 = np.argwhere(f < 0.025)[-1][0]
        B2 = np.argwhere(f > 1)[0][0]

        # frequency peak range limits
        if (fftpeak == 0) or (hzpeak-b <= 0):
            lowerlim = 0
        else:
            lowerlim = np.argwhere(f < hzpeak-b)[-1][0]
        upperlim = np.argwhere(f > hzpeak+b)[1][0]

        """ tsteps = 1 / SAMPLE_RATE
        t = np.linspace(0, (N-1)*tsteps, N)     # x axis time (seconds?)
        # plot the time function
        plt.subplot(2,1,1)
        plt.plot(t, respiratory_rate_data)
        plt.title("Time domain")
        plt.xlabel("Time (sec)")
        #plt.show()
        # show fft
        plt.subplot(2,1,2)
        plt.plot(f, yft_mag_plot,'.-')
        plt.plot(f[lowerlim],yft_mag_plot[lowerlim],'.')
        plt.plot(f[upperlim],yft_mag_plot[upperlim],'.')
        plt.title("Frequency Domain")
        plt.xlabel('Hz')
        #plt.xlim(0.1,1)
        plt.show() """

        # components for goodness equation
        # slice around frequency peak
        yft_mag_plot_green = yft_mag_plot[lowerlim:upperlim]
        # slice containing reasonable frequencies
        yft_mag_plot_red = yft_mag_plot[B1:B2]

        # goodness metric equation
        # numerator integration
        green = np.sum(yft_mag_plot_green)
        # denominator integration
        red = np.sum(yft_mag_plot_red) - green

        goodness = green / red

        return goodness, hzpeak

    print(binneddata.shape)

    # binneddata --> one column = one bin's average over all the frames

    """ respiratory_rate_data = heatgraph
    respiratory_rate_data = scipy.ndimage.gaussian_filter(respiratory_rate_data,15)
    goodness, hzpeak = np.apply_over_axes(goodnessmetric,respiratory_rate_data,[3])
    print(goodness.shape, hzpeak.shape) """

    #binneddata = np.resize(binneddata, (binneddata.shape[-1],binneddata.shape[0]))
    #print(binneddata.shape)

    goodnessvals = []
    hzpeaks = []
    weightsignals = []

    # calculate the goodness values for each bin
    goodnessvals = np.apply_along_axis(goodnessmetric,0,binneddata)

    # separate goodness values and peak values
    hzpeaks = goodnessvals[1,:]
    goodnessvals = goodnessvals[0,:]

    # show heat graph of goodness values if 2+ bins per row
    if (binsperrow > 1):
        heatgraph = np.resize(goodnessvals,(numbins,binsperrow))
        plt.imshow(heatgraph)
        plt.colorbar()
        plt.show()

    # multiply each bin's signal by their goodness value
    weightsignals = np.multiply(goodnessvals,binneddata[:,:])

    # take the average of the weighted signals
    avgweightsig = np.mean(weightsignals, axis=1)
    plt.plot(avgweightsig)
    plt.show()

    # calculate the fft and frequency peak of the weighted signal
    weightedfft, weightedf, weighthz, weightftp = calcfft(avgweightsig)

    # print weighted respiration rate
    print(f'The weighted respiratory rate is {weighthz*60} bpm')

    # find most relevant bin and save its signal to numpy file
    respindex = np.argmax(goodnessvals)

    respiratory_rate_data = binneddata[:,respindex]
    np.save('respiratory_rate_data', respiratory_rate_data)
    respiratory_rate_data = sp.ndimage.gaussian_filter(respiratory_rate_data,15)

    # calculate and print respiration rate
    resprate = hzpeaks[respindex] * 60
    print("The respiratory rate is ", resprate, " bpm")


########################################################### - GUI - ######################################################################

def error_message_box():
    '''
    Generates a general error pop-up
    '''
    CTkMessagebox(master=monitor_window,
                  title="Error",
                  message="Invalid Input.")
    
def begin_analysis():
    '''
    Runs analysis scripts depending on GUI parameters
    '''
    if thermal_check.get():
        thermal_analysis_thread = threading.Thread(analyze_thermal)
        thermal_analysis_thread.start()
        #thermal_analysis_thread.join()
    if ppg_check.get():
        monitor_window.withdraw()
        subject_name = subject_folder_entry.get()
        sample_number = sample_number_entry.get()
        signal = np.load(str(pathlib.Path().resolve()) + "/" + subject_name.lower() + "/" + subject_name.lower() + "_" + sample_number + "_ppg.npy", None, allow_pickle=True)
        try:
            ppg_analysis.analyze(signal = signal, window_size=int(window_length_entry.get()), monitor_window = monitor_window)
        except:
            error_message_box()
        #ppgAnalysisThread = threading.Thread(target=analyze_ppg)
        #ppgAnalysisThread.start()
        #ppgAnalysisThread.join()
        monitor_window.deiconify()

def begin_recording():
    '''
    Runs recording scripts based on GUI parameters
    '''
    global subject_name
    global subject_folder_path
    global sample_index

    subject_name, subject_folder_path, sample_index = get_subject_path()

    if ppg_check.get():
        ppg_recording_thread = threading.Thread(target=import_ppg_data_live)
        ppg_recording_thread.start()
        #ppg_recording_thread.join()
    if thermal_check.get():
        thermal_recording_thread = threading.Thread(target=record_thermal_camera)
        thermal_recording_thread.start()
        #thermalRecordingThread.join()
    

ctk.set_appearance_mode("system")

monitor_window = ctk.CTk()
monitor_window.title("PPG Monitor")
monitor_window.geometry('1000x500')
live_data_running = False

control_panel_label = ctk.CTkLabel(master=monitor_window,
                                 text="Control Panel",
                                 font=("Times New Roman", 16))

button_frame = ctk.CTkFrame(master=monitor_window,
                           height=150)

analysis_button = ctk.CTkButton(master = button_frame,
                             command = begin_analysis,
                             height = 50,
                             width = 250,
                             text = "Begin Analysis",
                             font = ("Times New Roman", 16))

record_data_button = ctk.CTkButton(master = button_frame,
                             command = begin_recording,
                             height = 50,
                             width = 250,
                             text = "Begin Recording",
                             font = ("Times New Roman", 16))

hardware_selection_frame = ctk.CTkFrame(master = button_frame,
                                      height = 100,
                                      width = 250)
hardware_selection_frame.grid_propagate(False)

global ppg_check
ppg_check = ctk.CTkSwitch(master=hardware_selection_frame,
                         text="Pulse Oximeter",
                         font=("Times New Roman", 16))

global thermal_check
thermal_check = ctk.CTkSwitch(master=hardware_selection_frame,
                             text="Thermal Camera",
                             font=("Times New Roman", 16))

analysis_params_label = ctk.CTkLabel(master=monitor_window,
                                   text="Analysis Parameters:",
                                   font=("Times New Roman", 16),
                                   height=100,
                                   width=250,
                                   anchor=ctk.S)

analysis_params_frame = ctk.CTkFrame(master=monitor_window,
                                   height = 400,
                                   width = 250)

recording_params_label = ctk.CTkLabel(master=monitor_window,
                                   text="Recording Parameters:",
                                   font=("Times New Roman", 16),
                                   height=100,
                                   width=250,
                                   anchor=ctk.S)

recording_params_frame = ctk.CTkFrame(master=monitor_window,
                                    height=400,
                                    width=250)
recording_params_frame.grid_rowconfigure([0,1,2,3], weight=1)

monitor_window.grid_rowconfigure([1], weight=1)
monitor_window.grid_columnconfigure([0,1,2], weight=1)

control_panel_label.grid(row=0, column=0, sticky = ctk.S)
button_frame.grid(row=1,column=0, sticky = ctk.N)
button_frame.grid_rowconfigure([0,1], weight=1)
analysis_button.grid(row=0, column=0, pady=5, padx=5)
record_data_button.grid(row=1, column=0, pady=5, padx=5)

hardware_selection_frame.grid(row=2,column=0, sticky = ctk.N, pady=5, padx=5)
hardware_selection_frame.rowconfigure([0,1], weight=1)
hardware_selection_frame.columnconfigure([0], weight=1)
ppg_check.grid(row=0,column=0)
thermal_check.grid(row=1,column=0)

recording_params_label.grid(row=0, column=1, pady=10)
recording_params_frame.grid(row = 1, column=1, rowspan=2, sticky = ctk.N)

serial_port_entry_label = ctk.CTkLabel(master=recording_params_frame,
                                    text="Serial Port (Pulse Oximeter): ",
                                    font=("Times New Roman", 16))
global serial_port_entry
serial_port_entry = ctk.CTkEntry(master=recording_params_frame,
                               font=("Times New Roman", 16))
usb_port_entry_label = ctk.CTkLabel(master=recording_params_frame,
                                    text="USB Port (Thermal Camera): ",
                                    font=("Times New Roman", 16))
global usb_port_entry
usb_port_entry = ctk.CTkEntry(master=recording_params_frame)
subject_name_entry_label = ctk.CTkLabel(master=recording_params_frame,
                                     text="Subject Name: ",
                                     font=("Times New Roman", 16))

global thermal_sample_rate_entry
thermal_sample_rate_entry = ctk.CTkEntry(master=recording_params_frame)
thermal_sample_rate_entry_label = ctk.CTkLabel(master=recording_params_frame,
                                               text="Camera Sample Rate (hz): ",
                                               font=("Times New Roman", 16))

global subject_name_entry
subject_name_entry = ctk.CTkEntry(master=recording_params_frame,
                                font=("Times New Roman", 16))
sample_duration_entry_label = ctk.CTkLabel(master=recording_params_frame,
                                        text="Sample Duration (Seconds): ",
                                        font=("Times New Roman", 16))
global sample_duration_entry
sample_duration_entry = ctk.CTkEntry(master=recording_params_frame,
                                   font=("Times New Roman", 16))

serial_port_entry_label.grid(row=0,column=0,pady=15,padx=5)
serial_port_entry.grid(row=0,column=1,pady=15,padx=5)
usb_port_entry_label.grid(row=1,column=0,pady=15,padx=5)
usb_port_entry.grid(row=1,column=1,pady=15,padx=5)
thermal_sample_rate_entry_label.grid(row=2, column=0, pady=15, padx=5)
thermal_sample_rate_entry.grid(row = 2, column = 1, pady=15, padx=5)
subject_name_entry_label.grid(row=3,column=0,pady=15,padx=5)
subject_name_entry.grid(row=3,column=1,pady=15,padx=5)
sample_duration_entry_label.grid(row=4,column=0,pady=15,padx=5)
sample_duration_entry.grid(row=4,column=1,pady=15,padx=5)

analysis_params_label.grid(row=0, column=2, pady=10)
analysis_params_frame.grid(row=1, column=2, rowspan=2, sticky = ctk.N)

subject_folder_entry_label = ctk.CTkLabel(master=analysis_params_frame,
                                       text="Subject Folder: ",
                                       font=("Times New Roman", 16))
global subject_folder_entry
subject_folder_entry = ctk.CTkEntry(master=analysis_params_frame,
                                  font=("Times New Roman", 16))
sample_number_entry_label = ctk.CTkLabel(master=analysis_params_frame,
                                       text="Sample #: ",
                                       font=("Times New Roman", 16))
global sample_number_entry
sample_number_entry = ctk.CTkEntry(master=analysis_params_frame,
                                  font=("Times New Roman", 16))
window_length_entry_label = ctk.CTkLabel(master=analysis_params_frame,
                                       text="Window Length (Seconds): ",
                                       font=("Times New Roman", 16))
global window_length_entry
window_length_entry = ctk.CTkEntry(master=analysis_params_frame,
                                  font=("Times New Roman", 16))

subject_folder_entry_label.grid(row=0,column=0,pady=25,padx=5)
subject_folder_entry.grid(row=0,column=1,pady=25,padx=5)
sample_number_entry_label.grid(row=1,column=0,pady=25,padx=5)
sample_number_entry.grid(row=1,column=1,pady=25,padx=5)
window_length_entry_label.grid(row=2,column=0,pady=25,padx=5)
window_length_entry.grid(row=2,column=1,pady=25,padx=5)

monitor_window.mainloop()
