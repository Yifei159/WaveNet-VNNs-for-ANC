# WaveNet-VNNs-for-ANC
The unofficial implementation of WaveNet-VNNs for Active Noise Control (ANC).
The orignial thesis can be found in https://arxiv.org/abs/2504.04450.


rtinfer.py is a real-time infer script that use a noisy audio clip as the virtual ambient sound, and then output the audio of the error microphone in real time. By inputting "m", you can switch on/off ANC, and input "q" to exit.
