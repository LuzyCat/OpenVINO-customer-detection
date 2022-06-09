ECHO OFF
title Run Openvino
cd %vino22dir%
call setupvars.bat
cd C:\KETI_MJK\Project\22_Retail\OpenVINO-customer-detection
start interactive_face_detection_demo.exe -i 0 -u h -min_size 20000 -reid_th 0.8
PAUSE