# Three-way Bin Sorting Attachment
Using the VGG-16 convolution neural network, transfer-trained, and a Raspberry Pi 3B to recognize and sort cafeteria items with combinations of disposables, compostables, and recyclable items. 

## Contents
- [Setting up the Raspberry Pi](#0)
- [Training and Tuning the Model](#1)
- [Core Sorting Logic](#2)
- [Data Backup](#3)

## <span id="0">0. Setting up the Raspberry Pi

*Tested on Ubuntu 22.04.3 LTS, with all necessary python libraries and dependencies installed. See ```requirements.txt``` for all required python packages.

If starting up in a location with previously unconnected WiFi:
1. If a monitor is available, first connect the RPi to the monitor via HDMI, then power on the RPi by connecting it to a power source via either MicroUSB or the circular power port. Using a mouse and keyboard connected to the RPi via USB, enter the WiFi credentials as for any typical desktop setup.

2. If a monitor is unavailable, power on the travel router and connect it to the WiFi by:
    a. Connecting your personal computer to the router.
    b. Log in using (Beverly needs to update this section)
    c. Connect the router to the WiFi. (Need to figure out how to this, giiven that our school wifi is now credential-based...)

If starting up in a location with previously connected WiFi:
1. Ensure your personal computer and the RPi are connected to the same WiFi.
2. Open your bash shell (WSL for Windows, Terminal for Macbooks) on your personal computer.
3. Enter: ```ssh user@raspberrpi.local```
4. Enter password
   
## 1. <span id="1">Training and Tuning the Model
[to be continued]
