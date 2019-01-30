import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!      
        # You need to incorporate self.smooth_param in likelihood calculation  
        ###########################


        # YOUR CODE HERE

        n_doc_per_class = np.zeros(n_classes)
        n_words_doc_class0 = np.zeros(n_words)
        n_words_doc_class1 = np.zeros(n_words)

        for i in range(n_docs):
            if y[i] == 0:   # if doc is class 0
                n_doc_per_class[0] += 1   # add 1 to class 0 count

                for j in range(n_words):
                    n_words_doc_class0[j] += x[i][j]    # array of class 0 words

            else:   # else doc is class 1
                n_doc_per_class[1] += 1   # add 1 to class 1 count

                for j in range(n_words):
                    n_words_doc_class1[j] += x[i][j]    # array of class 1 words


        # prior calc
        prior[0] = n_doc_per_class[0]/n_docs  # prob of doc being class 0 = (number of class 0 docs/all docs)
        prior[1] = n_doc_per_class[1]/n_docs  # same calc but class 1

        for i in range(n_words):
            # need likelihood for both doc classes
            likelihood[i][0] = (n_words_doc_class0[i] + self.smooth_param)/\
                               (n_words_doc_class0.sum() + self.smooth_param * n_words)
            likelihood[i][1] = (n_words_doc_class1[i] + self.smooth_param) / \
                               (n_words_doc_class1.sum() + self.smooth_param * n_words)



        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params


