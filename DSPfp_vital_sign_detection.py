#!/usr/bin/python
# ********************************************************
#  Project : DSP_final project
#
#  DSPfp_vital_sign_detection.py - Vita sign data collection
#                                  for a single sensor
# ********************************************************

# * MODULE IMPORT*****
import serial
import serial.tools.list_ports
import binascii
import functools
import numpy as np
import time
import math
import collections
from cmath import sqrt
import sys
from datetime import datetime

import mds_msg_lib as mds
import user_lib as user
import original_plot_vsm as kao_plot
#import plot_distance as kao_plot

# *********************
AUTOMOTIVE = 0x00
CONSUMER = 0x01
platform_type = CONSUMER
# ****************
# do_fft=1:single set of FFT data
# do_fft=0:multiple sets of FFT data
do_fft = 0

if do_fft == 1:
    number_of_detections = 1
else:
    number_of_detections = 2048

# **************** #
# SYSTEM START UP  #
# **************** #

# Report Python version
user.vReportPythonVersion_MTK()

# ------------------------------
# Get the command line argument
[serial_port, sensor_id] = user.sProcessCommandLine_MTK()
print(f'sensor_id: {sensor_id}')
print(f'serial_port: {serial_port}')

# ------------------------------
# Open the serial port
if platform_type == CONSUMER:
    uart = user.sOpenSerialPort_MTK(serial_port, 460800)
else:
    uart = user.sOpenSerialPort_MTK(serial_port, 921600)

# -------------------------------
# Ping the sensor to wake it up
user.vWakeupSensor_MTK(sensor_id, uart)

# --------------------------------
# Set the number of chirp tp be 2 - default 32
required_number_of_chirps = 2
user.vSetNumberOfChirps_MTK(sensor_id, uart, required_number_of_chirps)

# ---------------------------------
# Get the current number of chirp
current_number_of_chirps = user.ucGetNumberOfChirps_MTK(sensor_id, uart)
print("number_of_chirps = {:d}".format(current_number_of_chirps))

# ----------------------------------
# Get range and velocity resolution
[rRangeResolution, rVelocityResolution] = user.sGetResolutionParameters_MTK(sensor_id, uart)
print("FINAL Range Resolution(cm): " + str(rRangeResolution))
print("FINAL Velocity Resolution(km/hr): " + str(rVelocityResolution))

# ---------------------------------------------------------
# Set number of velocities per object to report in the detection message response
number_of_velocities_per_object_to_report = 1
user.vSetVelocitiesPerObjectParameter_MTK(sensor_id, uart, number_of_velocities_per_object_to_report)

# ---------------------------------------------------------
# Get number of velocities per object to report in the detection message response
velocities_per_object = user.ucGetVelocitiesPerObjectParameter_MTK(sensor_id, uart)
print("\tFINAL Number of velocities per object to report in detection response = {:d}".format(velocities_per_object))

# **************** #
#  CREATE MESSAGE  #
# **************** #

# -----------------------------------------------------------
# Create the detection command for the sensor specified
ucDetectionMsg = mds.cRadarDetectionMessage(sensor_id, mds.DEBUG_FULL)
ucDetectionMsg.vSetDebugPrintModeOff()

# Create the debug read command
debug_data_index = 0
ucDebugRead = mds.cDebugRead(debug_data_index, sensor_id, mds.DEBUG_FULL)

# Create the debug data read command
ucDebugDataRead = mds.cDebugDataRead(sensor_id, mds.DEBUG_FULL)

# Create the FFT read command
ucFftReadMessage = mds.cFftRead(sensor_id, mds.DEBUG_FULL)

# Create the FFT data read command
ucFftDataReadMessage = mds.cFftDataRead(sensor_id, mds.DEBUG_FULL)

# ****************** #
#  OPEN FFT/IQ FILE  #
# ****************** #

if do_fft == 0:
    filename = "vital_sign_data.bin"
    pInHandle = user.pOpenNewBinaryFile_MTK(filename)
if do_fft == 1:
    filename = "fft_data.bin"
    pFftInHandle = user.pOpenNewBinaryFile_MTK(filename)

# ********************************************** #
#  SETUP FOR ALL IN ONE DETECTION (do_fft == 0)  #
# ********************************************** #
if do_fft == 0:
    # Set all in one detection message
    all_in_one_fft_length = 128
    allInOneMessage = mds.cSetAllInOneDetection(all_in_one_fft_length, sensor_id, mds.DEBUG_FULL)
    # Send the parameter setting command to the sensor specified
    [isSendSensorMsgOkay, ucParameterSettingRsp] = allInOneMessage.sMdsSendSensorMsg(uart)
    allInOneMessage.vSetDebugPrintModeOff()

    # ----------------------------------------------
    # Do one additional detection
    # Throw away the data from the first detection
    # It appears incorrect (different to all subsequent detection data)
    ucDetectionMsg.vSetDebugPrintModeFull()
    [isSendSensorMsgOkay, ucDetectionRsp] = ucDetectionMsg.sMdsSendSensorMsg(uart)
    #
    # Following a detection, since all-in-one detection has been set, read the all-in-one data
    # Throw the all-in-one data away (don't write it to a binary log file)
    allInOneMessage.vSetDebugPrintModeOff()
    [u2BytesRead, ucMsg] = allInOneMessage.sGetAllInOneData(uart)
    #
    # Parse the detection response
    ucDetectionMsg.vSetDebugPrintModeOff()
    [ucNumberOfObjects, rDistance, rVelocity, ucConfidence, rSignalPwr] = ucDetectionMsg.sMdsParseDetectionRsp(
        ucDetectionRsp, rVelocityResolution)

# ****************** #
#      START UP      #
# ****************** #
if do_fft == 0:
    print("")
    print("Starting countdown")
    for p in range(5):
        print("Countdown {:d}".format(5 - p))
        time.sleep(1)
    print("\nGO !!!")

# ******************************************* #
#      SETUP & START DETECTION INTERVAL       #
# ******************************************* #
if do_fft == 0:
    enable_detection_interval_monitoring = 1
else:
    enable_detection_interval_monitoring = 0
#  # end if #  #
# ----------------------------------------------
if enable_detection_interval_monitoring == 1:
    rDetectionIntervalValues_fp = np.array([0] * number_of_detections, np.dtype('float32'))
    ts_detection_time = 0
    ts_detection_interval_ms_max = 0
    ts_detection_interval_ms_min = 1000

    ts1 = time.time()
# #end if # #
# ----------------------------------------------

# ***************************** #
#      MAIN DETECTION LOOP      #
# ***************************** #
for j in range(number_of_detections):
    if (j % 128) == 0:
        print("\nDetection count = {:d}".format(j) + " / {:d}".format(number_of_detections))
    # endif #
    # ------------------------------------------------
    # Measure the detection interval
    if enable_detection_interval_monitoring == 1:
        ts_detection_time_previous = ts_detection_time
        ts_detection_time = time.time()
        # The 1st detection interval to measure is the one between the '0th' and '1st' detection
        if j > 0:
            ts_detection_interval_ms = 1000 * (ts_detection_time - ts_detection_time_previous)
            rDetectionIntervalValues_fp[j] = ts_detection_interval_ms
            if ts_detection_interval_ms > ts_detection_interval_ms_max:
                ts_detection_interval_ms_max = ts_detection_interval_ms
            if ts_detection_interval_ms < ts_detection_interval_ms_min:
                ts_detection_interval_ms_min = ts_detection_interval_ms
            # print("{:3.2f}".format(rDetectionIntervalValues_fp[j]))
        # end if #
    # end if #

    # --------------------------------------------------
    # Detection
    # Select debug o/p mode for detection
    # op mode(Operational Mode) refers to a class located within the source code
    ucDetectionMsg.vSetDebugPrintModeOff()
    # Send the detection to the sensor
    [isSendSensorMsgOkay, ucDetectionRsp] = ucDetectionMsg.sMdsSendSensorMsg(uart)
    # Parse the detection response
    [ucNumberOfObjects, rDistance, rVelocity, ucConfidence, rSignalPwr] = ucDetectionMsg.sMdsParseDetectionRsp(
        ucDetectionRsp, rVelocityResolution)
    # -----------------------------------------------------
    if do_fft == 0:
        # Get all-in-one data from current detection
        allInOneMessage.vSetDebugPrintModeOff()
        [u2BytesRead, ucMsg] = allInOneMessage.sGetAllInOneData(uart)
        allInOneMessage.vSetDebugPrintModeOff()
        # ------------------------------------------------------
        # Formate the all-in-one IQ FFT bin data from the current detection
        allInOneMessage.vSetDebugPrintModeOff()
        [u4WordCount, u4DecryptedValues] = allInOneMessage.sPackAllInOneData(ucMsg)
        allInOneMessage.vSetDebugPrintModeOff()
        # ------------------------------------------------------
        # Write message bytes to file
        u4DecryptedValues.tofile(pInHandle)
    # end if #
# end for #

# Reset the 'all-in-one' detection length to 0 to resume normal sensor behavior
# following a detection. Without this, reading out FFT data following a detection
# will not work as expected because the sensor will be expecting the user to
# read IQ data bytes
# 將"all-in-one"檢測長度重置為0，以恢復檢測後雷達正常的動作。
# 如果不這樣做，在檢測後讀出fft數據將不能像預期的那樣工作，因為雷達將期望讀取到的是IQ數據字節
allInOneMessage.vSetDebugPrintModeOff()
allInOneMessage = mds.cSetAllInOneDetection(0, sensor_id, mds.DEBUG_FULL)
# Send the parameter setting command to the sensor specified
[isSendSensorMsgOkay, ucParameterSettingRsp] = allInOneMessage.sMdsSendSensorMsg(uart)

# ********************************************* #
#      COMPLETE DETECTION INTERVAL CHECKING     #
# ********************************************* #
if enable_detection_interval_monitoring == 1:
    # ---------------------------------------------------------
    # Capture time after last detection has been processed
    ts2 = time.time()
    #
    # ---------------------------------------------------------
    # Display timestamps
    print("\nSTARTING Timestamp = {:f}".format(ts1))
    print("ENDING Timestamp = {:f}".format(ts2))
    duration = ts2 - ts1
    print("\tDuration = {:f} Seconds".format(duration))
# end if #

# *********************************** #
#      REPORT 2ND FFT SAMPLE RATE     #
# *********************************** #
# --------------------------------------------------------------
# Calculate 2nd FFT resolution
if do_fft == 0:
    print("\nSTART: Calculate 2nd FFT resolution")
    number_of_ffts = number_of_detections  # 2048
    print("\tnumber of FFTs = {:d},".format(number_of_ffts))
    sample_period = duration/number_of_ffts
    print("\t2nd FFT sample period = {:f}".format(sample_period))
    sample_rate = 1/sample_period
    print("\t2nd FFT sample rate = {:f}".format(sample_rate))
# end if #

# ************************************* #
#      REPORT RESOLUTION PARAMETERS     #
# ************************************* #
# -------------------------------------------------------
# Get range and velocity resolution
[rRangeResolution, rVelocityResolution] = user.sGetResolutionParameters_MTK(sensor_id, uart)
# print("FINAL Range Resolution(cm): " + str(RangeResolution))
# print("FINAL Velocity Resolution(km/hr): " + str(VelocityResolution))

# ******************************************************** #
#      REPORT DETECTION INTERVAL MONITORING STATISTICS     #
# ******************************************************** #
# Calculate detection interval statistics
if (enable_detection_interval_monitoring == 1):
    print("\nACTION: Calculate detection interval statistics")
    #
    # Display max/min detection intervals
    print("\tMax detection interval (mS) = {:3.2f}".format(ts_detection_interval_ms_max))
    print("\tMin detection interval (mS) = {:3.2f}".format(ts_detection_interval_ms_min))
    #
    # Skip the first value (it is 0, by design)
    sum_fp = 0
    avg_fp = 0.0
    for i in range(1, (number_of_detections - 1), 1):
        sum_fp += rDetectionIntervalValues_fp[i]
        avg_fp = sum_fp / number_of_detections
    print("\tAverage detection interval (mS)= {:.2f}".format(avg_fp))
    #
    # ---------------------------------------------------------
    # Determine how many detection intervals have errors of +- 'tolerated_detection_interval_error_ms' mS
    bad_detection_interval_cnt = 0
    tolerated_detection_interval_error_ms = 5
    for i in range(1, (number_of_detections - 1), 1):
        if ((rDetectionIntervalValues_fp[i] > (avg_fp + tolerated_detection_interval_error_ms)) | \
                (rDetectionIntervalValues_fp[i] < (avg_fp - tolerated_detection_interval_error_ms))):
            bad_detection_interval_cnt += 1
    print("\tNumber of detection intervals outside of +-{:d} mS".format(tolerated_detection_interval_error_ms) + \
          " = {:d}".format(bad_detection_interval_cnt) + \
          " / {:d}".format(number_of_detections - 1))
    percentage_bad_detection_intervals = 100 * np.float(bad_detection_interval_cnt) / np.float(number_of_detections - 1)
    print("\t% of detection intervals outside of +-{:d} mS".format(tolerated_detection_interval_error_ms) + \
          " = {:.2f} %".format(percentage_bad_detection_intervals))
    #
    # ---------------------------------------------------------
    # Write detection interval values to a binary file for post-processing
    #
    debug_interval_filename = "vital_sign_interval_monitoring.bin"
    pMonitorInHandle = user.pOpenNewBinaryFile_MTK(debug_interval_filename)
    #
    print("ACTION: Write detection interval values to a binary file")
    rDetectionIntervalValues_fp.tofile(pMonitorInHandle)
    #
    user.vCloseBinaryFile_MTK(pMonitorInHandle, debug_interval_filename)
## end if ##

# **************************** #
#     CLOSE FFT/IQ LOG FILE    #
# **************************** #
# -------------------------------------------------
# Close the open log file
if do_fft == 0:
    user.vCloseBinaryFile_MTK(pInHandle, filename)
if do_fft == 1:
    user.vCloseBinaryFile_MTK(pFftInHandle, filename)
# -------------------------------------------------
# PLot the VTS data (Respiration & Heartrate)
# Read data from the file just created
if do_fft == 0:
    pReadHandle = user.pOpenExistingBinaryFile_MTK(filename)
    # The distance resolution of the system ->'distance_resolution'
    # The 2nd FFT sample rate ->'sample_rate'
    # The length of IQ data being read from the mmWave sensor on "each detection" is defined in 'fft_length'
    print("rRangeResolution = " + str(rRangeResolution))
    print("sample_rate = " + str(sample_rate))
    print("all_in_one_fft_length = " + str(all_in_one_fft_length))
    kao_plot.PlotVitalSignData(pReadHandle, rRangeResolution, sample_rate, all_in_one_fft_length)
    user.vCloseBinaryFile_MTK(pReadHandle, filename)
# end if #

# *********************** #
#     SHUTDOWN SYSTEM     #
# *********************** #
time.sleep(1)
uart.close()
print("\n*** SYSTEM CLOSED ***")


