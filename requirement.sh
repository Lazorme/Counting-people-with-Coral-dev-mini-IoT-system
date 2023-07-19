#!/bin/bash
#Install dependencies

echo "Installing all libraries."
echo

sudo apt-get install -y python3-pip python3-edgetpuvision
sudo pip3 install opencv-python
sudo pip3 install scikit-image
sudo apt-get install -y python3-dev
python3 -m pip install -r requirements_tracker.txt

echo
echo "Installation completed."
