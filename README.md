## Openface-API
By [Platanus](http://platan.us)

Openface-api is a web API that compares faces on 2 given pictures, and gives back their distance (measure of unsimilarity).  This is useful to check if both pictures belong to the same person.  It was developed for KSEC project, but can serve general purposes.

# Implementation
Openface-api is a Python [Flask](http://flask.pocoo.org/) application, with a face-comparing script based on [Openface](https://cmusatyalab.github.io/openface/) face recognition library.

# Installation
In order to get started, the following must be done:

1. Install Openface https://cmusatyalab.github.io/openface/setup/
2. Install and setup Flask (probably with `sudo pip install Flask`).  Remember to set up FLASK_APP env var
3. Download and copy to root folder the following files: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (extract before copying) and [nn4.small2.v1.t7](http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7) (just copy)

After everything is setup, you should be able to run flask 

# Usage

A route is provided to check if the server is up:  HTTP GET `/status/`

The API route is `/compare/`.  Expects HTTP POST with the params `img1` and `img2` as the urls of both pictures with faces to compare.  Returns a json object with the computed distance: `{distance: 0.1752}`

Then you can define your own distance threshold in order to evaluate if both pictures belong to the same person.  Reasonable values are around 0.5.  So any value higher than that means both faces are not from the same person.

Additionaly, it is possible to run the distance comparer locally in the server, executing in the terminal `python local_exe.py <img1> <img2>