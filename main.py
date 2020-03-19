import pickle
import json
import random
import tensorflow
import tflearn
import numpy
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

try:  # the code isn't going to run everything in the except if it runs successfully and all the data is there
    with open("data.pickle", "rb") as f:
        # code tries to load in and open the pickle file and if it cant do that then it runs the code beneath
        words, labels, training, output = pickle.load(f)
except:
    words = []  # list of words
    labels = []  # list of the labels
    docs_x = []  # list of all of the different patterns
    docs_y = [
    ]  # corrosponding enteries between x and y then the tag for those words (which makes up the pattern)"""

    for intent in data["intents"]:    # this will loop through all the dictionaries"""
        for pattern in intent["patterns"]:  # these patterns will be using for stemming"""
            wrds = nltk.word_tokenize(
                pattern)  # this will return a list with all the different words that are now tokenized"""
            words.extend(
                wrds)  # Put all the tokenized words into the words list"""
            docs_x.append(
                wrds)  # for each pattern another element will be put in docs y that defines what intent (tag) it belongs to"""
            docs_y.append(
                intent["tag"])  # Each entry into docs_x corrosponds with an entry into docs_y -
            # entry into docs_x will be the pattern then the intent (tag) will be in docs_y - this is important for classifying each pattern which is vital component of training the model"""

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]  # (can also be written as *if w != "?"*) stem all of the words in the words list, remove duplicate elements as to figure out the vocab size of the model - how many words the model has seen already"""
    words = sorted(list(set(words)))  # set takes all of the words and makes sure there are no dupe elements, list is going to convert back into a list as set is its own data type - sort will sort the words"""

    labels = sorted(labels)  # sort the labels to make them look nice"""

    """input data is a big list of word entries (bag of words) that the neural network can understand that represents words which says if a word is present or not -
	(bag of words which is a series of numbers in a list, this list is the length of the amount of words present so 100 words means each encoding would have 100 enteries (different numbers can represent frequency of word present), each position in this list will represent if a word exists or not)"""

    training = []  # one hot encoded - one hot as in if the word is there or not determined by a 1 or 0"""
    output = [
    ]  # Another list of 0's and 1's representing words(one hot encoded)"""

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w)for w in doc]

        for w in words:
            if w in wrds:  # if the word exists then (line below) append"""
                bag.append(1)
            else:  # But if the word isn't there then mark it as a 0"""
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(
            docs_y[x])] = 1  # Look through the labels list, find where the tag is in the list then set the value to 1 in the output row"""

        training.append(
            bag)  # both training and output are a list of words that are represented as numbers through one hot encoding"""
        output.append(output_row)

    training = numpy.array(
        training)  # turning the data into numpy arrays (both lines)"""
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        # this writes all the variables into a pickle file so it can be saved for use in the model
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

# /1 From here"""

net = tflearn.input_data(shape=[None, len(
    training[0])])  # Start with input (layer) data which is the length of the training data """
net = tflearn.fully_connected(
    net, 8)  # First hidden layer with 8 neurons fully connected"""
net = tflearn.fully_connected(net, 8)  # Second hidden layer"""
net = tflearn.fully_connected(net, len(
    output[0]), activation="softmax")  # Output layer with neurons representing each of the classes - softmax activation function gives a probabilities to each neuron in the output layer (this layer) so it can guess which tag is appropriate by giving prediction values for every tag"""
net = tflearn.regression(net)

model = tflearn.DNN(net)  # DNN is a type of neural netowork"""
# /2 To hear is the complete AI model"""

try:
    model.load("model.tflearn")
except:  # now if a model already exists then the model will not be re-trained
    model.fit(training, output, n_epoch=1000, batch_size=8,
              show_metric=True)  # the fit function means we are going to be passing the model the training data, number of epochs is the amount of times the model is going to see the data"""
    model.save("model.tflearn")

# making predictions


def bag_of_words(s, words):
    # sets up a list of 0's according to the number of words
    bag = [0 for _ in range(len(words))]

    # this will create a list of tokenised words
    s_words = nltk.word_tokenize(s)
    # this will stem all of the words
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    # takes the bag of words then converts it into a numpy array
    return numpy.array(bag)


def chat():
	print("Start talking with the bot! (type quit to stop)")
	while True:
		inp = input("You:")
		if inp.lower() == "quit":
			break

# if the user hasn't typed quit then turn the users input into a bag of words, feed it to the model and get the models response
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:  # error threshold set at 70%
			for tg in data["intents"]:
				if tg['tag'] == tag:  # open the json file and find the appropriate response to the approptiate intent by finding the right tag to the users input
					responses = tg['responses']

		print(random.choice(responses))
    else:
        print("I didn't quite understand that, please try again")


chat()
