#!/bin/bash

cd /media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Glaciers_NeurIPS

source SatelliteImageExtraction/bin/activate 

python ./dataAPI.py

python ./alignment.py

python ./createPatches.py
