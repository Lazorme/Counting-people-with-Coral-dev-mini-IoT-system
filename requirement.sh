echo
echo "Installing all library."
echo

sudo apt install opencv_python
sudo apt install python3-skimage
sudo apt install python3-dev
python3 -m pip install -r requirements_tracker.txt

echo