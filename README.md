This is a deep learning test problem provided as part of the admission process for the UTD AI Lab summer internship program. (Update: I got selected!). I was provided with an image dataset (vehicle_classification.zip) that contains 8 different classes of
vehicles:
Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van.
The goal is to train a CNN that can classify these different types of vehicles.
This can be divided into several steps:
1. Load the data.
2. Divide the dataset into a training set and a testing set with a ratio of 8:2.
3. Build a CNN model that takes an image (or a batch of images) as input and output the class that
the input image(s) are belonging to.
4. Train the model and keep track of the loss and accuracy.
5. Print the final accuracy.
6. (Optional bonus task) Plot the loss and accuracy curve during the training.

Approach: 
1. I first tried a simple textbook CNN model, just to set a benchmark.  On the first try I got 79.6% test accuracy on the test dataset; runtime was about 5 minutes on my computer. However, overfitting seemed to be a problem due to the consistently high difference between training loss and test loss. To reduce overfitting, I added drop_out rate and argumentation via rotation and flip. As a result, the overfitting issue got much better, but the training time took much longer due to the increase of image variants. The test accuracy improved to 81.24%; I didn't fix the random seed, so a certain amount of improvement may have been caused by randomness in the training.

<img width="753" height="786" alt="image" src="https://github.com/user-attachments/assets/48993050-af7c-46bd-9f27-73f0a92c2c4b" />


2. Then, I tried a different model structure containing more layers. The run time significantly increased to 45 minutes; test accuracy improved to 84.9%.

<img width="616" height="1405" alt="image" src="https://github.com/user-attachments/assets/4dd572f8-2913-4f4a-9b7d-620316efc654" />


3. Then, I tried another model structure which was simpler than #2 but more sophisticated than #1. The run time was about 20 minutes; test accuracy dropped to 82.58% with an observed increase in overfitting.

<img width="1015" height="1375" alt="image" src="https://github.com/user-attachments/assets/4b986705-a20b-4a38-9c30-9b0df6caaa6d" />



Based on the test results above, I chose the second model as my submission code. Here is the final output:

<img width="1405" height="941" alt="image" src="https://github.com/user-attachments/assets/bafe7af3-add9-4837-823c-04beb331ad0a" />

