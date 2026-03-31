import tensorflow as tf
import matplotlib.pyplot as plt


#check for available gpus on the system
tf.config.list_physical_devices('GPU')

#training dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

#train images are like questions from the flashcards and train labels are like the answer
#download and load the dataset - returns two tuples - first contains training data (images and correct labels to teach the model) 
#                                                   - second contains validation/test data (images and labels to evaluate the model)
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

#question number or image number(to study with) 
#set a variable to select the 169th image from the dataset(0indexed)
data_number = 168

#create  a new blank window for plotting 
plt.figure()

#display data as an image - train_images[data_number] 2D array of pixel values to display - cmap tells matplotliv to map the pixel values to a grayscale color map
plt.imshow(train_images[data_number], cmap="gray")

#add a vertical scale next to the image to show what pixel intensity corresponds to which colors
plt.colorbar()

#disbale grid lines for cleaner look
plt.grid(False)

plt.show()

#need to clasify the loaded image in one of the 10 categories

#label          description

#0              T-shirt/top
#1              Trouser
#2              Pullover
#3              Dress
#4              Coat
#5              Sandal
#6              Shirt
#7              Sneaker
#8              Bag
#9              Ankle Boot

classification_result = train_labels[data_number]   #retrieve the int label that represents the clothing category for the specific training image

print(f"Image belongs to label: {classification_result}")

raw_pixels_val = valid_images[data_number]         #retrieve the raw 2D array of pixel values  for the 169th image  

print(f"Raw pixels value: {raw_pixels_val}")


#the question number to quizz with 
question_number = 6174

plt.figure()
plt.imshow(valid_images[question_number], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

valid_labels[question_number]  #output the int label for the specific image 


#Defining the Neural Network

number_of_classes = train_labels.max() + 1       #find the highest label value and add 1 to det the total nr of unique clothing categories(10)

number_of_classes

#initialize a linear stack of neural network layers 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),         #flatten the 2D image matrix into a 1D array, 28 high By 28 wide become 784 pixels
    tf.keras.layers.Dense(int(number_of_classes))         #create a fully connected layer - the parameter number_of_classes sets the number of output neurons to 10
])

model.summary()   #print a table showing the models layers, output shapes, and the nr of trainable parameters

#parameter calculation and model arhitecture plot

image_height = 28
image_width = 28

number_of_weights = int(image_height * image_width * int(number_of_classes))     #manually calculate the connection between the input layer(784 pixels) and the output layer(10 neurons), in total 7840 weights ignoring biases 
number_of_weights

tf.keras.utils.plot_model(model, show_shapes = True)        #generate a flow chart image of the models arhitecture


#compiling and training the model

#configure the learning process before training - optimizer adam is the algorithm used to update network weights based on the data it sees and the loss function
#                                               - loss function to measure how accurate is the model during training (from_logits = True tells the function that the models output values are raw, unscaled numbers not ptobabilities as theres no softmax activation on the final dense layer)          
#                                               - metrics parameter instructs the model to track the percentage of correctly classified images during training                           
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'])

#training loop - traing images and train label are the data from which the model is learning 
#              - epochs the number of times a model will cycle through the entire training dataset
#              - verbose is telling tensorflow to print a progress bar and metrics for each epoch
#              - validation data is data used to evaluate the models performance at the end of each epoch to ensure the model isnt memorizing the training data 
history = model.fit(
    train_images,
    train_labels,
    epochs = 5,
    verbose = True,
    validation_data = (valid_images, valid_labels)
)

model.predict(train_images[0:10]) #ask the trained model to generate predictions for the first 10 images in the training dataset

new_question_number = 6174

plt.figure()
plt.imshow(train_images[new_question_number], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(train_images[new_question_number:new_question_number+1]).flatten())
plt.xticks(range(10))
plt.show()

print("Correct answer: ", train_labels[new_question_number])