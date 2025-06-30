# Calorie Calculator

 Calorie Calculator is a project that takes an image of a food item and gives the amount of calories in that food item for one serving size.

![image](https://github.com/user-attachments/assets/ce6b6fd2-055d-439a-9a88-ad7621009353)


## The Algorithm

The dataset used for the project can be viewed [here](https://www.kaggle.com/datasets/kmader/food41). Note: some of the lesser-known/less relevant food classes were removed from the final dataset used for training the model to make it smaller and quicker to train. 

The model(resnet18) was trained using the dataset, and classifies a food item as one of the 84 food items from the used dataset and gives a confidence rating accordingly.

In order to print the result of the reconized food item and the calories per serving on top of the output image, code snippets from the imagenet.py file were used.

## Running this project

You can run the trained model through project.py.
1. Make sure all the files(image you will be detecting, project.py, foodtocal.csv, and resnet18.onnx) are in the same directory
2. Make sure you have the jetson-inference library downloaded
3. Run the following command:
   `python3 project.py [name of input image] [name of output image(optional, will output as output.jpg if not specified)]`
Example:
   `python3 project.py pizza_image.jpg output.jpg`
The project will give the output image and will also print the output in the terminal(recognized image and also the calories per serving for that food item).


[View a video explanation here](video link)
