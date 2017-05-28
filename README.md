# occupancy-cv
An experiment in computer vision, occupancy-cv collects photos while the computer is being used and while it is not.
Then, those images are used to train a system to recognize whether the computer is being used based on an image.

#### How to use:
1. run collection-vlc.py to gather samples
2. run convert.py to prepare the samples
3. run train.py to train and run the neural network

#### Software required:
- VLC
- python3 (3.2+)
- python-xlib
- tensorflow
- Pillow

#### Software recommended:
- devilspie (to auto-minimize VLC)

#### Progress:
- [x] Photo capture
- [x] Training
- [ ] Save training data
- [ ] Smarter image capture (delta-based, maybe motion?)
- [ ] Support capture methods besides VLC
- [ ] Support Windows
