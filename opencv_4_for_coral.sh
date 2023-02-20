#bin/bash
sudo apt-get install build-essential cmake unzip pkg-config 
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev 
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev 
sudo apt-get install libxvidcore-dev libx264-dev 
sudo apt-get install libgtk-3-dev 
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-dev 
sudo apt-get update 

git clone https://github.com/pjalusic/opencv4.1.1-for-google-coral.git 
cp opencv4.1.1-for-google-coral/cv2.so /usr/local/lib/python3.7/dist-packages/cv2.so 
sudo cp -r opencv4.1.1-for-google-coral/libraries/. /usr/local/lib 