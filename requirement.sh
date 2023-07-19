#!/bin/bash
# Install Tracker Dependencies
echo
echo "Installing dependencies."
echo

read -p "Install dependencies and openCV ? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo apt-get install -y python3-pip python3-edgetpuvision
    sudo apt install python3-opencv
    sudo apt install python3-skimage
    sudo apt install python3-dev
    python3 -m pip install -r requirements_tracker.txt
fi
echo