# **Behavioral Cloning**

## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIAArch.png "NVIDIA CNN"
[image2]: ./examples/SampleData.png "Sample Data"
[image3]: ./examples/NoDropout.png "No Dropout Layer"
[image4]: ./examples/Dropout.png "Dropout Layer"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

---
### Required Files & Quality of Code

My project includes the following files required to run the simulator in autonomous mode:
* _model.py_ contains the script to create and train the model
* _drive.py_ for driving the car in autonomous mode
* _model.h5_ stores a trained convolution neural network
* _writeup_report.md_ (this file) summarizing the results
* _video.mp4_ capturing the autonomous run

There are additional notebooks in the git repo that were used for development:
* _BehavioralCloning.ipynb_ was the test environment that created model.py
* _DrivingData.ipynb_ helped to produce image captures


The car can be driven autonomously using the Udacity provided simulator by executing:
```sh
docker run -it --rm -p 4567:4567 -v $PWD:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

To efficiently use memory, the model.py file use a Python generator for processing and retrieving the training data.  It also contains code for training and saving the convolution neural network.  This file, along with BehavioralCloning.ipynb, shows the pipeline I used for training and validating the model.

### Model Architecture

My model follows the design of the CNN architecture documented by NVIDIA in [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
![image1]

The NVIDIA engineers have empirically demonstrated, in simulation and in New Jersey driving conditions, that this design adequately controls the steering angle of the vehicle.  Their goal of designing a network that uses a minimal amount of training data (72 hours of image capture) to learn road features using steering alone directly applies to this project.  Therefore, I used this base architecture (three 5x5 filters, two 3x3 filters, and four fully connected layers) with some modifications:
* a Keras lamdba layer normalizes the input data.
```
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
```
* a cropping layer reduces the input to a pertinent roadway frame.  This helps eliminate noise introduced by the sky or side of the road features.
* ELU activations on the layers provide nonlinearity.  Per the theory, ELU activations push the mean activation closer to zero for faster learning and noise robustness.  In addition, the default activation for Dense layers is linear.  Replacing this default with ELU helped with over-fitting [1].
```
model.add(Dense(xxx))
model.add(Activation('elu'))
```
* a dropout layer helps reduce over-fitting.  Training experiments helped confirm the effectiveness of the dropout layer:

| No Dropout Layer | Dropout Layer |
|---------|---------|
| ![image3] | ![image4] |

While training, I discovered that the ELU activation layers were more effective at correcting over-fitting than dropout layers.

To avoid manually tuning the learning rate, an adam optimizer was used.

##### ~~Initial Architecture~~
Prior to using the NVIDIA design, I attempted to use a trained GoogLeNet to make use of the knowledge gained in the Transfer Learning section.  This approach was abandoned due to the Keras version provided in the docker file.

### Training Data and Strategy

The data samples provided for this project were augmented using the Udacity provided simulator.  The `train_test_split` function split the sample data to reserve validation samples, which helps ensure that the model does not memorize the data or over-fit.  These samples were fed into a Python generator.  For each line of sample data, the generator saved six images; center, center flipped, left, left flipped, right, and right flipped.
![image2]

Six angles were also saved, using a correction of +/-0.15 for the left and right camera images.  This data expansion strategy helped maximum the image collection per run.

Training data was collected iteratively.  When the autonomous vehicle encountered problems, the simulation was stopped and new data was collected.  The network was then re-trained and tested again.  The following training table provides the type of data collection that was executed to add to the data samples provided by the project [2]:

| | Style | Purpose |
|--|-------|---------|
| 1 | Complete lap in reverse | Correct right-side bias |
| 2 | Smooth curve shorts | Correct driving on lanes in curves |
| 3 | Bridge entrance recovery shorts | Correct driving off the bridge rather than entering it |
| 4 | Off-road dirt recovery shorts | Correct driving off of road and into dirt, open spaces when lines disappear |
| 5 | Curve recovery shorts | Additional corrections for keeping in-lane on curves |


[1]: Newer versions of Keras include activations as inputs for Dense layers, which makes for a cleaner, easier to read code.
[2]: Video games are not my strong suit.  Therefore I needed to use the provided data set to ensure my training set had enough center-of-the-lane images captured.  I had more than enough "recovery" images.