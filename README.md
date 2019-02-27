# Human Activity Recognition
![](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/readme_images/presentation/ar.png)
___

## Introduction
Activity recognition systems deployed in smart homes are characterized by their ability to detect Activities of Daily Living (ADL) in order to improve assistance. Such solutions have been adopted by smart homes in practice and have delivered promising results for improving the quality of care services for elderly people and responsive assistance in emergency situations.

We propose a novel framework for activity Learning able to extract features from multiple types of sensors located in a sensor-rich environment and make a prediction based on their output.

![](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/readme_images/presentation/home.png?s=10)

 Both CNN and LSTM architectures will be used in a multi-input manner to enhance the classification accuracy making use of all the data sources. 
 
![](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/readme_images/presentation/framework.png)

In addition, an average ensemble of the same model, but with different window sizes is used to overcome the choosing size problem and to learn to discern activities of different complexity.

![](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/readme_images/presentation/model-7-new.png)

In the thesis we explain the phases of data processing and discuss the framework in detail, exploring the unprecedented use of multi-input Neural Networks in AR and the temporal ensemble, motivating the choices of design and evaluating its ability to generalize in different scenarios.

![](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/readme_images/presentation/frame.png)


We evaluated our framework on the UCAmI Cup dataset, which consists in the recognition of 24 activities of daily living and contains data collected from four data sources: binary sensors, an intelligent floor, proximity and acceleration sensors. Our results show that our framework outperforms competing machine and deep learning techniques by 13%.

We demonstrate that the framework can be applied to homogeneous sensor modalities, but can also fuse multimodal sensors to improve performance. We characterise key architectural hyperparameters’ influence on performance to provide insights about their optimisation.

More informations are available in [presentation format](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/slidesThesis.pdf) and in the [thesis](https://github.com/SqrtPapere/ActivityRecognition_DeepLearning/blob/master/thesis.pdf).

