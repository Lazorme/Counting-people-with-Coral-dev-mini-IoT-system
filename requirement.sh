#!/bin/bash
#Install dependencies

echo "Installing all libraries."
echo

sudo apt-get install -y python3-pip python3-edgetpuvision

sudo apt install python3-opencv #Most important

sudo apt install python3-skimage
sudo apt-get install -y python3-dev
python3 -m pip install -r requirements_tracker.txt

echo
echo "Installation completed."
