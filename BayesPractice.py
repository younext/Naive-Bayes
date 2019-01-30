
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

#gets the dataset from sklearn
data = fetch_20newsgroups()

#prints category names to be used for classification
#data.target_names

#pick some categories from data.target_names
categories = ['rec.sport.baseball','rec.sport.hockey','sci.space','sci.med',
              'talk.religion.misc','soc.religion.christian','alt.atheism']

#these datasets already have subsets for training and testing
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#prints one of the emails in the training set
#print(train.data[2])

#pipeline attaches the features to the multinomial bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#using the pipeline, apply model to to training data, then predict labels for test data
model.fit(train.data, train.target)
labels = model.predict(test.data)

#can use a confusion matrix to evaluate the performance
confMat = confusion_matrix(test.target, labels)
sns.heatmap(confMat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('actual label')
plt.ylabel('predicted label');
plt.show()

#the more "confused" the model is the more times it will predict a different label from the actual label
#this model can tell the difference between baseball and hockey pretty well,
#but gets confused between atheism and christian talk
#this is an area of confusion that we may expect so not too bad!

#using this pipeline you can input your own strings using the predict() method

def predict_category(str, train=train, model=model):
    predicted = model.predict([str])
    return train.target_names[predicted[0]]

#if you run this file in console you can just call the predict_category() method, you don't have to print unless you run the whole script
print(predict_category('i really love sports with balls'))
print(predict_category('surgery is cool'))

#remember the model will be much better at predicting if you train it with good info
#so inputting a string that is relevant to a topic that is not included
#in the training data will have inaccurate results