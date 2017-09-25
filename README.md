# E2E_autopilot
An End-to-end model for self driving car with Comma.ai's data set

### Dataset
The Comma.ai [dataset](https://github.com/commaai/research) consists of 7.25 hours of largely highway driving.
Some oversampling of the original dataset is required to address the imbalance data issue as the number of turning frames 
are significantly smaller than the number of straight steering frames. 

### The model
The model is based on [Nvidia's design](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for 
an end to end deep learning model.
<br/>
However, the model in this repository is slightly different from that of Nvidia's. I stacked multiple convolutional
and pooling layers on top of each others before applying the fully connected layers 
with dropouts and batch normalization.  
 
### Steering angle prediction
![alt text](https://media.giphy.com/media/EUAYMDe25YJ8I/giphy.gif "demo gif")

### Setup and retrain the model
1. You will need python 3.6
2. Download the training data from Comma.ai and store to /data folder
3. Run command: pip install -r requirements.txt
4. Run command: python src/network.py

### To use the pre-trained model 
The data set is recorded at 20 Hz with a camera mounted on the windshield of an Acura ILX 2016.
If you want to use the pre-trained model keep in mind that the camera need to be set up with the same settings.