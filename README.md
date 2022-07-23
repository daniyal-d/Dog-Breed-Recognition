# Dog-Breed-Recognition
Web app that uses deep learning to detect the breed on a dog from an image. Has 81.12% accuracy on the validation set. View the notebook by opening `Dog_Breed_Recognition.ipynb` and the validation test by opening `Validation Test.ipynb`

Website link: https://daniyal-d-dog-breed-recognition-main-w86ypc.streamlitapp.com/

This project used Kaggle's Dog Breed Recognition dataset (https://www.kaggle.com/competitions/dog-breed-identification/data) and MobileNet V2 (https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5) to train and test the data. The training folder contained over 10,000 images, which the final website model was trained on. In order to evaluate the dataset, a new model was instantiated, trained on 80% of the training data and then tested on the remaining 20% (validation set). This model had 81.12% accuracy on the validation set, but because the website model was trained on more images, it may be slightly higher.

# How the website works
Simply upload a photo of a dog to the website, and it will output its 10 most confident predictions. The website will also output a downloadable bar graph of its predictions.

[demonstration.webm](https://user-images.githubusercontent.com/31736868/180587221-9134f6da-d63b-4328-9c7c-aca8dd7998f5.webm)


# How the model was trained
This project uses transfer learning and the Adam optimization algorithm to create the model. The Kaggle dataset contains a training folder with over 10,000 images as well as a CSV file with their breeds. The model was given the photos and labels to train off of. The model gives confidence levels of what it believes a dog's breed could be. For example, the model may be 80% confident that a dog is a German Shepherd, 15% confident that its an Inuit Dog, and 5% confident it's a Malamute. These predicted values can also be used to calculate log loss (https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy). In order to evaluate the model, the images were split into training and validation sets. A new, untrained model was given all of the images in the training set (alongisde their labels), and then attempted to predict the labels of the validation set. After training through 7 epochs, the model had approximately 81% accuracy and a loss of .6775 on the validation data. The model used on the website was trained on all images (therefi=ore it was never tested on a validation set). It is important to note that the model was only trained on tested on using the dog breeds in the Kaggle dataset. Because of this, it cannot accurately predict a dog's breed if its breed was not a part of the dataset. For example, this model will not accurately predict a Shiba Inu because the training data did not include any images of them. For a list of all 120 dog breeds, please visit https://www.kaggle.com/competitions/dog-breed-identification/data
