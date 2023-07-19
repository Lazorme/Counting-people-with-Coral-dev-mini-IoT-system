echo
echo "Installing all library."
echo

sudo apt-get install -y python3-pip python3-edgetpuvision
sudo pip3 install opencv_python
sudo pip3 install python3-skimage
sudo pip3 install python3-dev
python3 -m pip install -r requirements_tracker.txt

echo
