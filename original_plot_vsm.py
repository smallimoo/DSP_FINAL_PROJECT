#!/usr/bin/python
# *************************************************************************
#  Project   : Part of a mmWave-based single 1Tx1Rx sensor Vital Sign 
#              Monitoring demonstration
#
#                The file data is in binary signed format, Q(16-bit) I(16-bit)
#                Parameter 1: pBinaryFileReadPtr
#                Parameter 2: distance_resolution
#                Parameter 3: detection_rate
#                Parameter 4: fft_length (# IQ data pairs to read for each FFT)
# * MODULE IMPORTS ************************************************************
import platform
# import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import sys

# *****************************************************************************/
def PlotVitalSignData(pBinaryFileReadPtr, distance_resolution, detection_rate, fft_length):
    # Read all IQ data from a binary file (-1 means all) as 16-bit signed integers
    ##########iq_samples_16 = np.fromfile('vital_sign_data_DC1.bin', dtype='i2', count = -1)
    iq_samples_16 = np.fromfile(pBinaryFileReadPtr, dtype='i2', count=-1)
    # int8, int16, int32, int64  使用字符串列代替 = 'i1', 'i2', 'i4', 'i8'
    # # of 16-bit values = 128 (#bins) * 2 (I&Q) * 2048 (# detections) = 524288

    # Change data type of IQ sample array to signed 32-bit to support arithmetic range required later in script
    # e.g. squaring array values
    iq_samples = iq_samples_16.astype('i4')

    #######print(str(len(iq_samples)))

    #######print(str(iq_samples[0]) + ", " + str(iq_samples[1]) + ", " + \
    #######str(iq_samples[2]) + ", " + str(iq_samples[3]) )

    #######print(str(iq_samples[524283]) + ", " + str(iq_samples[524284]) + ", " + str(iq_samples[524285]) + ", " + \
    #######str(iq_samples[524286]) + ", " + str(iq_samples[524287]) )

    real = iq_samples[0:524288:2]
    imag = iq_samples[1:524288:2]
    # Check
    #######print(str(len(real)))
    #######print(str(real[0]) + ", " + str(real[1]))
    #######print(str(len(imag)))
    #######print(str(imag[0]) + ", " + str(imag[1]))
    #######print(str(real[262142]) + ", " + str(real[262143]))
    #######print(str(imag[262142]) + ", " + str(imag[262143]))

    real_length = len(real)
    ##########fft_length = 128
    num_detections = int(real_length / fft_length)  # 2048

    #######print("Shape of real[]")
    #######print(real.shape)
    real_modified = real.reshape(num_detections, fft_length)
    # reshape 改變矩陣的形狀
    # real_modified = real.reshape(fft_length, num_detections)
    # real_modified = real.reshape(fft_length, -1)
    #
    # displays as rows & columns
    #######print("Shape of real_modified[num_detections,fft_length]")
    #######print(real_modified.shape)
    #
    #######print("Shape of imag[]")
    #######print(imag.shape)
    imag_modified = imag.reshape(num_detections, fft_length)
    # imag_modified = imag.reshape(fft_length, num_detections)
    # imag_modified = imag.reshape(fft_length, -1)
    #
    # displays as rows & columns
    #######print("Shape of imag_modified[num_detections, fft_length]")
    #######print(imag_modified.shape)
    #
    fft_length = 128
    fft_num_to_start = 1
    # [0] = num_detections
    # [1] = fft_length
    num_ffts = real_modified.shape[0]
    # check
    print("\nNumber of Detections = " + str(num_ffts) + "\n")
    #

    print("\nPLEASE WAIT, PROCESSING...\n")

    ##############################
    # Top-level figure definition
    ##############################
    fig = plt.figure(num=None, figsize=(17, 12), dpi=80, facecolor='w', edgecolor='k')
    toptitle_str = "Vital Sign Monitoring Data"
    plt.suptitle(toptitle_str, fontsize=20)
    ##############################

    ##############################
    # FIGURE 1
    ##############################
    ax1 = plt.subplot(2, 2, 1)
    title_str1 = "Distance Vs Magnitude (" + str(fft_length) + "-point FFT, detections " + str(
        fft_num_to_start) + " to " + str(num_ffts) + " overlay)"
    ax1.set_title(title_str1)
    ax1.set_xlabel('Distance (cm)')
    ax1.set_ylabel('Magnitude (20log10(sqrt(I^2+Q^2))')
    ax1.grid()
    ###############################
    #######print("distance_resolution = " + str(distance_resolution))
    #######print(str(len(real_modified[0,:])))
    #
    # Create an array of values 0 to 127 (1st row of length 'fft_length')
    x = np.arange(0, len(real_modified[0, :]))
    # Scale the values by the distance resolution for each bin
    scaled_x = x * distance_resolution
    #######print("x[] ...")
    #######print(x)
    #######print("scaled_x[] ...")
    #######print(scaled_x)
    #
    # checking array handling
    # Display 1st 2 rows of FFT data
    #######print("1st row of FFT data real_modified[0,:]")
    #######print(real_modified[0,:])
    #######print("2nd row of FFT data real_modified[0,:]")
    #######print(real_modified[1,:])
    # Display last 2 rows of FFT data
    #######print("Last but 1 row of FFT data real_modified[0,:]")
    #######print(real_modified[2046,:])
    #######print("Last row of FFT data real_modified[0,:]")
    #######print(real_modified[2047,:])
    #
    # Display ALL rows of FFT data
    # print(real_modified[...,:])
    #######print("All rows of FFT data real_modified[0:2048,:]")
    #######print(real_modified[0:2048,:])
    #
    # Calculate real^2 and imag^2
    real_modified_squared = np.square(real_modified[0:2048, :])
    imag_modified_squared = np.square(imag_modified[0:2048, :])
    #######print("real_modified_squared[0:2048,:]")
    #######print(real_modified_squared[0:2048,:])
    #######print("imag_modified_squared[0:2048,:]")
    #######print(imag_modified_squared[0:2048,:])
    #
    # Calculate sum of squares
    sum_real_squared_imag_squared = real_modified_squared[0:2048, :] + imag_modified_squared[0:2048, :]
    #######print("sum_real_squared_imag_squared[0:2048,:]")
    #######print(sum_real_squared_imag_squared[0:2048,:])
    #
    # Calculate sqrt of sum of squares
    sqrt_sum_real_squared_imag_squared = np.sqrt(sum_real_squared_imag_squared)
    #######print("sqrt_sum_real_squared_imag_squared[0:2048,:]")
    #######print(sqrt_sum_real_squared_imag_squared[0:2048,:])
    #
    # Calculate 20*log10 of sqrt of sum of squares
    np.seterr(divide='ignore')
    twentylog10_sqrt_sum_real_squared_imag_squared = 20 * np.log10(sqrt_sum_real_squared_imag_squared[0:2048, :])
    np.seterr(divide='warn')
    #######print("twentylog10_sqrt_sum_real_squared_imag_squared[0:2048,:]")
    #######print(twentylog10_sqrt_sum_real_squared_imag_squared)
    #
    # Setup an array of colors to use when plotting
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_ffts)))
    # Plot the data
    # plt.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[0,:], color='C0')
    # plt.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[1,:], color='C1')
    # plt.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[2,:], color='C2')
    # plt.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[3,:], color='C3')
    ymax = 0
    for i in range(num_ffts):
        c = next(color)
        #############################################################
        # ax1.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[i,:], color=c, linewidth=0.1)
        #############################################################
        plt.plot(scaled_x, twentylog10_sqrt_sum_real_squared_imag_squared[i, :], color=c, linewidth=0.1)
        #############################################################
        if max(twentylog10_sqrt_sum_real_squared_imag_squared[i, :]) > ymax:
            # ymax = max(twentylog10_sqrt_sum_real_squared_imag_squared[i, :])
            binmax = np.nanargmax(twentylog10_sqrt_sum_real_squared_imag_squared[i, :], axis=None)
            ymax = twentylog10_sqrt_sum_real_squared_imag_squared[i, binmax]
    # print('ymax = ')
    # print(ymax)
    plt.plot(binmax * distance_resolution, ymax, marker='*', ms=10)
    plt.text(binmax * distance_resolution + 3, ymax, round(binmax * distance_resolution, 2), rotation=0,
             bbox=dict(facecolor='cyan', alpha=0.5))
    # Draw the plot (non-blocking)
    #############################################################
    # plt.draw()
    #############################################################

    #############################################################
    # FIGURE 2
    #############################################################
    ax2 = plt.subplot(2, 2, 2)
    title_str2 = "Sample Number Vs Magnitude (" + str(fft_length) + "-point FFT, detections " + str(
        fft_num_to_start) + " to " + str(num_ffts) + " overlayed)"
    ax2.set_title(title_str2)
    ax2.set_xlabel('Sample Number (0-127)')
    ax2.set_ylabel('Magnitude (20log10(sqrt(I^2+Q^2))')
    ax2.grid()

    # Create an array of values 0 to 127 (1st row of length 'fft_length'
    xx = np.arange(0, len(real_modified[0, :]))
    #######print("xx[], an array of values 0 thru 127")
    #######print(xx)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_ffts)))
    for i in range(num_ffts):  # num_ffts = detection = 2048
        c = next(color)
        plt.plot(xx, twentylog10_sqrt_sum_real_squared_imag_squared[i, :], color=c, linewidth=0.1)
        ##########################################################################################
        if max(twentylog10_sqrt_sum_real_squared_imag_squared[i, :]) > ymax:
            binmax = np.nanargmax(twentylog10_sqrt_sum_real_squared_imag_squared[i, :], axis=None)
            ymax = twentylog10_sqrt_sum_real_squared_imag_squared[i, binmax]
    # print('ymax = ')
    # print(ymax)
    plt.plot(binmax, ymax, marker='*', ms=10)
    plt.text(binmax + 2, ymax, binmax, rotation=0,
             bbox=dict(facecolor='cyan', alpha=0.5))

    #################################################################################
    # Calculate the std deviation for each FFT bin (there are 'fft_length' of them) 
    #################################################################################
    dtype = [('var', '<i4'), ('idx', '<u4')]
    var = np.zeros(fft_length, dtype=dtype)
    #
    # Calculate the variance for each bin across all FFTs (or detections)
    # Ignore bin 0 (DC). Variance for bin 0 recorded as 0
    for i in range(fft_length):
        if i > 0:
            if i < 129:
                var[i] = (np.var(sqrt_sum_real_squared_imag_squared[:, i]), i)  # assign 'var' and 'idx' values
    #
    # Sort var[] in descending order, sort by 'var, not by 'idx'
    var_ordered = np.sort(var, order='var')[::-1]
    print("var = " + str(var))
    print("var_ordered [0:3] = " + str(var_ordered[0:4]))

    #################################################################################
    # Calculate the average for each FFT bin (there are 'fft_length' of them) 
    #################################################################################

    #########################################################################################
    # Find the 4 bins with the largest variance and average them into a single vector
    # This is what we will use to plot the time-domain signal in Figure 3
    #########################################################################################
    #
    # Create an ordered variance vector containing only the index values (not the variance values)
    # This is used as an index (representing bin) in the FOR loop to index the required [:,bin] vector
    #
    var_ordered_idx = var_ordered['idx']
    #
    # Re-order the 1st 4 elements as desired
    #######print("var_ordered_idx[0] to [3] = " + str(var_ordered_idx[0:4]))
    var_reordered_idx = np.copy(var_ordered_idx)
    # 0 = 36
    # 1 = 37
    # 2 = 42
    # 3 = 38
    var_reordered_idx[0] = var_ordered_idx[0]
    var_reordered_idx[1] = var_ordered_idx[1]
    var_reordered_idx[2] = var_ordered_idx[2]
    var_reordered_idx[3] = var_ordered_idx[3]
    print("var_reordered_idx[0] to [3] = " + str(var_reordered_idx[0:4]))
    #
    #### Create a vector in which to store the average of 4 x [:,bin] vectors accross
    #### all FFTs (i.e. for 'num_detections')
    ###sum_vector = np.zeros(num_detections)
    #
    # Create an average time-domain signal based on FFT data from the 4 bins with the highest variance
    # Sum the 4 column vectors with the largest variance into a single averaged array 'sum_vector[]'
    #
    num_averages = 4
    sum_iq = np.zeros(num_detections, dtype=complex)
    #######print("shape sum_iq: " + str(sum_iq.shape))
    #######print("dtype sum_iq: " + str(sum_iq.dtype))
    #######print("size sum_iq: " + str(len(sum_iq)))
    num_iq_averages = num_averages
    sum_bin_number = 0
    for i in range(num_iq_averages):
        sum_iq = np.add(sum_iq, np.vectorize(complex)(real_modified[:, var_reordered_idx[i]],
                                                      imag_modified[:, var_reordered_idx[i]]))
        print("var_reordered_idx = " + str(var_reordered_idx[i]))
    # calculate average iq vector
    sum_iq = sum_iq / num_iq_averages
    iq = sum_iq

    # Calculate average bin number
    # Round to the average bin number to the nearest integer
    average_bin_number = int(np.round(sum_bin_number / num_iq_averages))
    print("Average bin number = " + str(average_bin_number) + "\n")
    #######print("length of iq[] = " + str(len(iq)))
    #######print("iq[] = " + str(iq))

    #############################################################
    print("\nPLEASE WAIT, PROCESSING...\n")
    #############################################################
    # FIGURE 3 - Plots 'sqrt_sum_real_squared_imag_squared[:,bin]' Vs 't[]'
    #            for each bin being considered. This is a time domain signal
    #            obtained by slicing across N x FFTs for each bin, where N
    #            is 'num_detections'
    #############################################################
    # Sample rate is hard-wired for now
    ##########detection_rate = 40.0545
    #
    Fs = detection_rate;  # from parameter sample rate
    T = 1 / Fs;  # sampling period
    L = len(iq);  # length of complex vector
    #######print("Size of complex iq time vector = " + str(len(iq)))
    t = T * np.arange(0, L);  # Time vector
    #######print("len(t) = " + str(len(t)))
    #
    #############################################################
    ax3 = plt.subplot(2, 2, 3)
    # title_str3 = "Time versus Doppler (1 set of " + str(L) + "-points, bin " + str(average_bin_number_to_display) + ")"
    title_str3 = "data[:,bin]' Vs 't[] (4 sets of " + str(L) + "-points, bins " + str(var_reordered_idx[0]) + ", " + \
                 str(var_reordered_idx[1]) + ", " + \
                 str(var_reordered_idx[2]) + ", " + \
                 str(var_reordered_idx[3]) + ")"
    ax3.set_title(title_str3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Magnitude (sqrt(I^2+Q^2))')
    ax3.grid()

    # plt.plot(t, sum_vector, color='r', linewidth=0.5)
    num_bins_to_plot = num_averages
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_bins_to_plot)))
    for i in range(num_bins_to_plot):
        c = next(color)
        # time_domain_vector = np.add(sum_vector, sqrt_sum_real_squared_imag_squared[:,var_reordered_idx[i]])
        time_domain_vector = sqrt_sum_real_squared_imag_squared[:, var_reordered_idx[i]]
        plt.plot(t, time_domain_vector, color=c, linewidth=1.0, label=var_reordered_idx[i])

    # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.legend(loc='upper left', shadow=True, fontsize='medium')

    #############################################################
    # FIGURE 4 - Plots a figure based on the FFT of the vector
    #            'np.vectorize(complex)(real_modified[:,bin], imag_modified[:,bin])'
    #            averaged over the bins being considered
    #############################################################
    ax4 = plt.subplot(2, 2, 4)
    # title_str4 = "Single-sided " + str(num_ffts) + "-point 2nd FFT, bin " + str(average_bin_number_to_display) + ")"
    title_str4 = "Single-sided " + str(num_ffts) + "-point 2nd FFT, bins " + str(var_reordered_idx[0]) + ", " + \
                 str(var_reordered_idx[1]) + ", " + \
                 str(var_reordered_idx[2]) + ", " + \
                 str(var_reordered_idx[3])
    ax4.set_title(title_str4)
    ax4.set_xlabel('f (Hz)')
    ax4.set_ylabel('|P1|(f)')
    ax4.grid()
    # Exclude DC
    ax4.set_xlim(0.1, 3.5)

    #################################################
    # Average the FFT data for the bins being considered
    #################################################
    # Create an empty matrix for summing
    sum_fft = np.zeros(num_detections)
    #######print("Length of sum_fft is " + str(len(sum_fft)))
    num_fft_averages = num_averages
    for i in range(num_fft_averages):
        sum_fft = np.add(sum_fft, np.fft.fft(
            np.vectorize(complex)(real_modified[:, var_reordered_idx[i]], imag_modified[:, var_reordered_idx[i]])))
    # calculate average iq vector
    sum_fft = sum_fft / num_fft_averages  # num_fft_average = 4
    Y = sum_fft
    # print('Y = ')
    # print(str(Y))
    # print(len(Y))
    # Scale (unity gain)
    P3 = abs(Y / L);  # L = len(iq) L=2048
    # print('L = ')
    # print(L)
    # print('\n P3 = ')
    # print(str(P3))
    # print(len(P3))
    # Remove reflection (Nyquist/2 to DC)
    L_over_2 = int(L / 2)
    # print("L/2 = "+ str(L_over_2))
    P2 = P3[0:L_over_2]
    # print(str(P2))
    # print("len(P2) = " + str(len(P2)))
    # Remove DC & compensate amplitude (x2)
    P1 = 2 * P2[1:L_over_2]
    #######print("Length of P1 is " + str(len(P1)))
    #######print(str(P1))
    #
    # DC has been removed so start frequency vector from 1, not 0
    # Reduce length of frequency vector because DC was removed
    # 'P1' is same length as 'f'
    f = Fs * np.arange(1, ((L_over_2))) / L
    #######print("Length of f is " + str(len(f)))
    #######print("f = " + str(f))
    #
    # Configure y-axis scale based on max value in P1[0.1, 3.5]
    # Find the index into f[] that corresponds to a value of 0.1 (see 'ax4.set_xlim(0.1, 3.5)' above)
    P1_index = round(0.1 * (L / Fs)) - 1
    P1_max = P1[P1_index:-1].max()
    plt.ylim(0, int(P1_max * 1.2))
    #######print("P1_index = " + str(P1_index))
    #######print("P1[P1_index:-1].max()= " + str(P1[P1_index:-1].max()))
    #######print(str(P1[0]))
    #######print(str(P1[1]))
    #######print(str(P1[2]))
    #######print(str(P1[3]))
    #######print(str(P1[4]))
    #
    # plot(f,P1);
    plt.plot(f, P1, color='b', linewidth=2.0)

    ### CHECK
    # for t in range(200):
    #    print("f,P1[" + str(t) + "] = " + str(f[t]) + ", " + str(P1[t]))

    ##################################################################################
    #
    # ADD RESPIRATION RATE MARKERS (RR) TO FIGURE 4
    # Look for first max after 1st 'bin_offset' bins (ignore large values around DC)
    # Assume this max is the RR
    ##################################################################################

    min_frequency_for_max_rr_search_Hz = 0.25
    # min_frequency_for_max_rr_search_Hz = 0.1
    #######print("min_frequency_for_max_search_Hz = " + str(min_frequency_for_max_search_Hz))
    min_bin_for_max_rr_search = int(
        np.round(min_frequency_for_max_rr_search_Hz * L / Fs))  # Fs=detection_rate = 40.0545
    #######print("min_bin_for_max_search = " + str(min_bin_for_max_search))

    # bin_num * Fs/L = frequency(Hz), therefore bin_num = frequency(Hz) * L/Fs
    # Don't search any higher than 0.5 Hz for RR
    max_frequency_for_max_search_Hz = 0.5
    #######print("max_frequency_for_max_search_Hz = " + str(max_frequency_for_max_search_Hz))
    max_bin_for_max_search = int(np.round(max_frequency_for_max_search_Hz * L / Fs))
    #######print("max_bin_for_max_search = " + str(max_bin_for_max_search))
    #
    # [val_rr, idx_rr] = max(P1((1+bin_offset):max_bin_for_max_search))
    # temp_rr contains the part of P1 that we wish to search
    temp_rr = np.array(P1[min_bin_for_max_rr_search:max_bin_for_max_search])
    idx_rr = np.argmax(temp_rr)
    ###val_rr = P1[idx_rr+bin_offset]  # same as temp_rr[idx_rr]
    val_rr = P1[idx_rr + min_bin_for_max_rr_search]  # same as temp_rr[idx_rr]
    #######print("idx_rr = " + str(idx_rr))
    #######print("val_rr = " + str(val_rr))
    #######print("f = " + str(f[idx_rr+bin_offset]))

    rr_freq_points = np.zeros(4)

    rr_freq_points[0] = 1 * f[idx_rr + min_bin_for_max_rr_search]
    rr_freq_points[1] = 2 * f[idx_rr + min_bin_for_max_rr_search]
    rr_freq_points[2] = 3 * f[idx_rr + min_bin_for_max_rr_search]
    rr_freq_points[3] = 4 * f[idx_rr + min_bin_for_max_rr_search]
    print("f1 = " + str(1 * f[idx_rr + min_bin_for_max_rr_search]))
    print("f1' = " + str(f[(1 * (idx_rr + min_bin_for_max_rr_search + 1)) - 1]))
    print("f2 = " + str(2 * f[idx_rr + min_bin_for_max_rr_search]))
    print("f2' = " + str(f[(2 * (idx_rr + min_bin_for_max_rr_search + 1)) - 1]))
    print("f3 = " + str(3 * f[idx_rr + min_bin_for_max_rr_search]))
    print("f3' = " + str(f[(3 * (idx_rr + min_bin_for_max_rr_search + 1)) - 1]))
    print("f4 = " + str(4 * f[idx_rr + min_bin_for_max_rr_search]))
    print("f4' = " + str(f[(4 * (idx_rr + min_bin_for_max_rr_search + 1)) - 1]))
    #
    rr_amp_points = np.zeros(4)
    rr_amp_points[0] = P1[(1 * (idx_rr + min_bin_for_max_rr_search + 1)) - 1]
    markerline, stemlines, baseline = plt.stem(rr_freq_points, rr_amp_points, '-', use_line_collection=True)
    # plt.setp(baseline)
    plt.setp(baseline, linestyle="-", color='black', linewidth=6)

    rr_raise_y_axis = 3.0
    plt.text(rr_freq_points[0], rr_amp_points[0], 'RR=' + str(round(rr_freq_points[0],2)), rotation=0,
             bbox=dict(facecolor='red', alpha=0.5))
    ##################################################################################
    # ADD HEART RATE MARKERS (HR, HR-RR, HR+RR) ON THE FIGURE
    # bin_num * Fs/L = frequency(Hz), therefore bin_num = frequency(Hz) * L/Fs
    # Search from higher than 1.0Hz to 1.5Hz for RR
    ##################################################################################
    min_frequency_for_max_search_Hz = 0.95
    # _frequency_for_max_search_Hz = 0.8
    #######print("min_frequency_for_max_search_Hz = " + str(min_frequency_for_max_search_Hz))
    min_bin_for_max_search = int(np.round(min_frequency_for_max_search_Hz * L / Fs))
    #######print("min_bin_for_max_search = " + str(min_bin_for_max_search))
    max_frequency_for_max_search_Hz = 1.3
    # max_frequency_for_max_search_Hz = 2
    max_bin_for_max_search = int(np.round(max_frequency_for_max_search_Hz * L / Fs))
    #######print("max_bin_for_max_search = " + str(max_bin_for_max_search))
    #
    #######[val_hr, idx_hr] = max(P1(1+min_bin_for_max_search:max_bin_for_max_search));
    # temp_hr contains the part of P1 that we wish to search
    temp_hr = np.array(P1[min_bin_for_max_search:max_bin_for_max_search])
    # Subtract 1 here because we are searching an array where [0] contains a non-zero frequency
    idx_hr = np.argmax(temp_hr)
    val_hr = P1[idx_hr + min_bin_for_max_search]  # same as temp_rr[idx_rr]
    #######print("idx_hr = " + str(idx_hr))
    #######print("val_hr = " + str(val_hr))
    #######print("f = " + str(f[idx_hr+bin_offset]))
    hr_freq_points = np.zeros(3)
    hr_freq_points[0] = f[idx_hr + min_bin_for_max_search]
    #########################################hr_freq_points[1] = f[idx_hr+min_bin_for_max_search]+f[idx_rr+bin_offset]
    #########################################hr_freq_points[2] = f[idx_hr+min_bin_for_max_search]-f[idx_rr+bin_offset]
    #hr_freq_points[1] = f[idx_hr + min_bin_for_max_search] + f[idx_rr + min_bin_for_max_rr_search]
    #hr_freq_points[2] = f[idx_hr + min_bin_for_max_search] - f[idx_rr + min_bin_for_max_rr_search]
    #
    hr_amp_points = np.zeros(3)
    hr_amp_points[0] = P1[idx_hr + min_bin_for_max_search]
    markerline, stemlines, baseline = plt.stem(hr_freq_points, hr_amp_points, ':', use_line_collection=True)
    # plt.setp(baseline)
    plt.setp(baseline, linestyle=":", color='grey', linewidth=6)

    # Add text
    hr_raise_y_axis = 1
    plt.text(hr_freq_points[0], hr_amp_points[0] + hr_raise_y_axis, 'HR = ' + str(round(hr_freq_points[0], 2)), rotation=0,
             bbox=dict(facecolor='cyan', alpha=0.5))
    #########################################################################################

    print("\nPROCESSING COMPLETE !!!\n")
    print("\nCREATING GRAPHS, PLEASE WAIT ...\n")
    print("\nCLOSE GRAPH WINDOW TO EXIT SCRIPT\n")

    #############################################################
    # DRAW ALL FIGURES
    #############################################################
    # DRAW the graph(s)
    # Draw the plot(s) to the screen
    # plt.show(block=False)
    plt.show()
    print("\nCLOSING GRAPHS ...\n")
