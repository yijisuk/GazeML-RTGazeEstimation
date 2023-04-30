<<<<<<< HEAD
# GazeEstimationDemos
Implementations and modifications of existing Gaze Estimation models.
=======
# GazeML
A deep learning framework based on Tensorflow for the training of high performance gaze estimation.

*Please note that though this framework may work on various platforms, it has only been tested on an Ubuntu 16.04 system.*

*All implementations are re-implementations of published algorithms and thus provided models should not be considered as reference.*

This framework currently integrates the following models:

## ELG

Eye region Landmarks based Gaze Estimation.

> Seonwook Park, Xucong Zhang, Andreas Bulling, and Otmar Hilliges. "Learning to find eye region landmarks for remote gaze estimation in unconstrained settings." In Proceedings of the 2018 ACM Symposium on Eye Tracking Research & Applications, p. 21. ACM, 2018.

- Project page: https://ait.ethz.ch/projects/2018/landmarks-gaze/
- Video: https://youtu.be/cLUHKYfZN5s

## DPG

Deep Pictorial Gaze Estimation

> Seonwook Park, Adrian Spurr, and Otmar Hilliges. "Deep Pictorial Gaze Estimation". In European Conference on Computer Vision. 2018

- Project page: https://ait.ethz.ch/projects/2018/pictorial-gaze

*To download the MPIIGaze training data, please run `bash get_mpiigaze_hdf.bash`*

*Note: This reimplementation differs from the original proposed implementation and reaches 4.63 degrees in the within-MPIIGaze setting. The changes were made to attain comparable performance and results in a leaner model.*

## Running the demo
To run the webcam demo, perform the following:
```
    cd src
    python3 elg_demo.py
```

To see available options, please run `python3 elg_demo.py --help` instead.

## Evaluation
Click the video thumbnails to access the respective demo videos.

The variance in experiment environments is due to the use of cameras with different hardware settings and resolution. The videos are recorded for demonstration purposes; not for performance evaluation/comparison.

The following videos are outputs of live-processed webcam recordings.
Identical eye motions are applied: 
<br>camera center gaze -> 3x left rotation -> 3x right rotation -> camera center gaze -> 2x left rotation -> 2x right rotation -> camera center gaze -> 4x (upper gaze -> lower gaze -> left gaze -> right gaze) -> camera center gaze
### 720p resolution with eyeglasses
<a href="https://www.youtube-nocookie.com/embed/Czl4l3SVKD0" target="_blank">
 <img src="http://img.youtube.com/vi/Czl4l3SVKD0/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

### 720p resolution without eyeglasses
<a href="https://www.youtube-nocookie.com/embed/LB27aihemPo" target="_blank">
 <img src="http://img.youtube.com/vi/LB27aihemPo/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

### 1080p resolution with eyeglasses
<a href="https://www.youtube-nocookie.com/embed/-LGwGGuzqmM" target="_blank">
 <img src="http://img.youtube.com/vi/-LGwGGuzqmM/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

### 1080p resolution without eyeglasses
<a href="https://www.youtube-nocookie.com/embed/O-6Rw6u9ECA" target="_blank">
 <img src="http://img.youtube.com/vi/O-6Rw6u9ECA/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

### 2160p resolution with eyeglasses
<a href="https://www.youtube-nocookie.com/embed/VQqj62NDQ_s" target="_blank">
 <img src="http://img.youtube.com/vi/VQqj62NDQ_s/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

### 2160p resolution without eyeglasses
<a href="https://www.youtube-nocookie.com/embed/eKwxwzY3aVQ" target="_blank">
 <img src="http://img.youtube.com/vi/eKwxwzY3aVQ/maxresdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

During the live-processing of the recording, periodic delays occur, returning a static image with no further updates. The frequency of delays gets lower as video resolution decreases, but still, delays exist in a significant manner.

Will make further modifications on the code, aiming to minimize the delays while live-processing the video.

>>>>>>> fbb8249 (initial commit)
