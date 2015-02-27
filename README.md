cuda-image-recognition
======================

CUDA-aware image recognition application.

This is an application which will become (I hope) my master thesis.

Installation requirements (tested on Ubuntu Linux):
CUDA nvcc etc, CC >= 3.0

OpenCV 2.4.9 - build from sources (GPU module from Ubuntu repository has only method stubs)

ffmpeg - build from sources
./configure --enable-gpl --enable-libass --enable-libfdk-aac --enable-libopus --enable-libfaac --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libtheora --enable-libvorbis --enable-libx264 --enable-libxvid --enable-nonfree --enable-postproc --enable-version3 --enable-x11grab --enable-libvpx --enable-shared

leptonica

tesseract

boost

GTest (disable by switching off tests in CMake)
