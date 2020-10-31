#!/usr/bin/env python
# coding: utf-8

# # Arxiv Paper Classification
# ### Matteo Esposito

# In[5]:


import numpy as np
import pandas as pd
import random, unicodedata, re, sys
pd.options.mode.chained_assignment = None


# ## Preliminary

# In[2]:


def split_training_set(df, ratio=0.7):
    """Splits training set into train and validation at a ratio of 70/30"""
    df.sample(frac=1)
    train = df[:int(ratio*df.shape[0])]
    validation = df[int(ratio*df.shape[0]):]
    return train, validation


# In[4]:


# Data Cleanup
def clean(df, t):
    df[t] = df[t].apply(lambda x : re.sub('[0-9]', '', x))
    df[t] = df[t].apply(lambda x : re.sub('\[[^]]*\]', '', x))
    df[t] = df[t].apply(lambda x : re.sub("<$\.*?>", '', x))
    df[t] = df[t].apply(lambda x : re.sub("\n", " ", x))
    df[t] = df[t].apply(lambda x : x.lower())
    df[t] = df[t].apply(lambda x : x.strip())
    df[t] = df[t].apply(lambda x : re.sub("[^A-Za-z0-9]+", " ", x))
    
    # remove stop words
    stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", 
                 "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", 
                 "before", "being", "below", "between", "both", "but", "by", "can", "couldn", 
                 "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", 
                 "doing", "don", "don't", "down", "during", "each", "few", "for", "from", 
                 "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", 
                 "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", 
                 "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", 
                 "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", 
                 "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", 
                 "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", 
                 "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", 
                 "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", 
                 "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", 
                 "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", 
                 "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", 
                 "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", 
                 "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", 
                 "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", 
                 "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", 
                 "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", 
                 "what's", "when's", "where's", "who's", "why's", "would"]
    df[t] = df[t].apply(lambda text: ["".join(w) for w in text.split(" ") if w not in stopwords])
    
    # Convert list back to string
    newabs = [' '.join(map(str, abstract_list)) for abstract_list in df[t]]
    df[t] = newabs
    return df


# ## Q4 - Naive Bayes Baselines

# ### Random Classifier

# In[11]:


def random_classification(out, train, test):
    random_submission = pd.DataFrame({"Id": [], "Category": []})
    random_submission['Id'] = test['Id']
    random_submission['Category'] = np.random.choice(train['Category'].unique(), test.shape[0])
    random_submission.to_csv(out, index=False)
    print(f"Random classification done and output to {out}")


# ### Naive Bayes Classifier using Bag of Words

# In[25]:


class DataFormatter():
    def __init__(self, train, test, validation):
        self.train = train
        self.test = test
        self.validation = validation
        self.vocab = []
        
    def get_vocab(self):
        for wordstring in self.train:
            for word in wordstring.split(" "):
                if word not in self.vocab:
                    self.vocab.append(word)
        
    def create_bow(self):
        returnarrs = []
        for i, df in enumerate([self.train, self.test, self.validation]):
            X_bow = []
            for i in range(df.shape[0]):
                bow = {}
                abstract = df.iloc[i].split()

                # Initialize
                for word in self.vocab:
                    bow[word] = 0
                    
                # Populate with 1 in case where the word is in the provided test case.
                for word in abstract:
                    if word in bow.keys():
                        bow[word] = 1
                        
                X_bow.append(bow)
                
            for i in range(df.shape[0]):
                binary_abstract = X_bow[i]
                X_bow[i] = list(binary_abstract.values()) # Some sort of a conversion
                
            returnarrs.append(np.array(X_bow))
        
        return returnarrs


# In[26]:


class BernoulliNB():
    def fit(self, X, y, alpha=1):
        classes = np.unique(y)
        self.n_classes = len(classes)
        self.n_features = X.shape[1]
        self.likelihoods = np.zeros((self.n_classes, self.n_features))
        for cat_idx, cat in enumerate(classes):
            examples_in_class = X[y==cat].shape[0]
            self.likelihoods[cat_idx] = (X[y==cat].sum(axis=0) + alpha*np.ones(self.n_features)) / (examples_in_class + alpha*self.n_classes)
            
    def get_posteriors(self, test_case):
        ones = np.ones((self.n_classes, self.n_features))
        return np.prod(self.likelihoods * test_case + (ones - self.likelihoods) * (ones - test_case), axis=1)
    
    def predict(self, test_case):
        return np.argmax(self.get_posteriors(test_case))
    
    def predict_matrix(self, X):
        return np.apply_along_axis(self.predict, 1, X)
    
    def error_rate(self, y_pred, y_true):
        er = round(1 - sum(y_pred == y_true)/y_pred.shape[0], 2)
        return er


# In[12]:


def main():
    # Get command line args
    ALGO = sys.argv[1].lower()
    TRAIN = sys.argv[2]
    TEST = sys.argv[3]
    OUTPATH = sys.argv[4]

    print(f"Algo: {ALGO}\nTrain: {TRAIN}\nTest: {TEST}\nOutpath: {OUTPATH}\n")
    
    train_full = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    train, validation = split_training_set(train_full)

    print(">> Data read and split")
    
    train = clean(train, 'Abstract')
    validation = clean(validation, 'Abstract')
    test = clean(test, 'Abstract')
    
    print(">> Data processed")
    
    # Proceed with classification
    if ALGO == "random":
        print(">> Running Random Classifier")
        random_classification(train=train, test=test, out=OUTPATH)
    elif ALGO == "nbbernoulli":
        print(">> Running Bernoulli Naive Bayes Classifier")
        # Format data into acceptable format (binary)
        d = DataFormatter(train=train['Abstract'], test=test['Abstract'], validation=validation['Abstract'])
        d.get_vocab()
        table_output = d.create_bow()
        train_abstracts, test_abstracts, validation_abstracts = (table_output[0], table_output[1], table_output[2])

        # Create classifier and get predictions
        b = BernoulliNB()

        # Get predictions and map integer output to string categories.
        cat_mapping = {
            0:'astro-ph',
            1:'astro-ph.CO',
            2:'astro-ph.GA',
            3:'astro-ph.SR',
            4:'cond-mat.mes-hall',
            5:'cond-mat.mtrl-sci',
            6:'cs.LG',
            7:'gr-qc',
            8:'hep-ph',
            9:'hep-th',
            10:'math.AP',
            11:'math.CO',
            12:'physics.optics',
            13:'quant-ph',
            14:'stat.ML'
        }
        
        # Submission
        ALPHASTAR = 0.15
        b.fit(X=train_abstracts, y=train['Category'], alpha=ALPHASTAR)
        
        print(">> Training done. Generating submission")
        
        y_test = b.predict_matrix(test_abstracts)
        y_test = np.vectorize(cat_mapping.get)(y_test)
        test['Category'] = y_test
        test.drop(columns=["Abstract"], inplace=True)
        test.to_csv(OUTPATH, index=False) ## LEADERBOARD SCORE: 0.76822 w/ 0.3 (FULL DATA: 0.77755 w/ 0.3)
        print(f"Bernoulli NB classification done and output to {OUTPATH}")
    else:
        print("INVALID algorithm. Choose from ['random', 'nbbernoulli']")
        sys.exit(1)


# In[28]:


if __name__ == "__main__":
    main()


# In[1]:


###### Code to produce comparison plots.
# import seaborn as sns
# import matplotlib.pyplot as plt

# alpha_errors = {}
# alpha_errors[a] = np.mean(y_validation == validation['Category'])

# # Organise data into dataframes.
# scores = {"Alpha": list(alpha_errors.keys()) + list(alpha_errors.keys()), 
#           "Accuracy": list(alpha_errors.values()) + [0.7733333333333333, 0.7813333333333333, 0.7893333333333333, 0.7933333333333333, 0.792, 0.7933333333333333, 0.792], 
#           "Source": ["Bernoulli", "Bernoulli", "Bernoulli", "Bernoulli", "Bernoulli", "Bernoulli", "Bernoulli", 
#                      "Multinomial", "Multinomial", "Multinomial", "Multinomial", "Multinomial", "Multinomial", "Multinomial"]}
# df = pd.DataFrame(scores)

# # Generate graph
# plt.figure(figsize=(10, 7))
# plt.title('Validation Set Prediction Accuracy of Naive Bayes Classification on Arxiv Article Abstracts\n Using 70/30 Train/Validation Split')
# sns.lineplot(x="Alpha", y="Accuracy", hue="Source", data=df)
# plt.savefig('compare.png')
# plt.show();

