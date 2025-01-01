#!/bin/bash

# I had setup a recording sink with PulseAudio called 'record-n-play' that I am using for 
# recording audio output for reference audio samples.

parec -d record-n-play.monitor --rate=22050 --file-format=wav $(echo $1).wav