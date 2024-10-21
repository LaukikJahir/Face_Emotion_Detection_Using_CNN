# Real-time Emotion Detection on Raspberry Pi 5 with Webcam

This project implements a real-time emotion detection system using facial recognition techniques, deployed on a Raspberry Pi 5 with a Logitech C310 webcam and a connected monitor. It detects emotions like anger, happiness, sadness, surprise, etc., in real-time, utilizing deep learning models with OpenCV and Keras.

## Table of Contents
- [Aim](#aim)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Working](#working)
- [Results](#results)
- [License](#license)

## Aim
The goal is to detect and display emotions in real-time from facial expressions captured by a webcam, specifically designed for Raspberry Pi 5.

## Prerequisites
1. **Raspberry Pi 5** with Raspberry Pi OS installed.
2. **Any Usb Webcam** webcam connected to Raspberry Pi.
3. A monitor connected to Raspberry Pi for GUI display.
4. Python 3 installed with the following libraries:
    - `opencv-python`
    - `keras`
    - `tensorflow`
    - `pillow`
   
   Install the libraries via pip:
   ```bash
   pip install opencv-python keras tensorflow pillow
