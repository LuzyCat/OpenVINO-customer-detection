ECHO OFF
title Run Openvino
cd %vino22dir%
call setupvars.bat
cd C:\Users\keti_nuc1\Documents\age-gender
start age_gender_detection_head.exe -i 0 -u h -min_size 20000
PAUSE