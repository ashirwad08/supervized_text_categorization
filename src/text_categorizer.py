# author: ash chakraborty
# date: 08/21/2016

# load a set of messages (whispers) and train on their categories

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import pickle
import re
import sys

class TextCat(object):
    """
    Loads a training set of messages and their categories;
    has methods to build new features, convert to a Bag Of Words model, 
    train on some classifiers and store the best one 
    to an HDF5 store. This class is inherited by the TextTest class
    to fit against the trained models and to cross validate.
    """
    
    def __init__(self, trainfile='./../data/train.csv', testfile='./../data/test.csv'):
        """return training data path, and training dataset"""
        self.trainfile = trainfile
        self.testfile = testfile
        self.traindat, self.traintargets = self.read_file(self.trainfile)
        self.testdat, self.testtargets = self.read_file(self.testfile)
        
    
    def read_file(self, filepath):
        """
        read train or testing set into a pandas dataframe, return
        text and target categories as series
        """
        print 'reading {} data...\n'.format(filepath)
        
    
        #dat = pd.read_csv('./../data/{}.csv'.format(name))
        dat = pd.read_csv(filepath)
        
        print 'success\n', dat.shape
        
        return pd.DataFrame(dat.text), dat.category
    
    def eng_new_features(self, dat):
        """
        Takes in a dataframe of text messages and adds to it previously explored
        features like message length, sentiment polarity and subjectivity, POS
        counts, emoji counts, is_hookup flag, and finally imputes all missing msgs
        as 'msngvals'.
        
        Returns a dataframe
        """
        #print "dat shape: \n",dat.shape
        print 'engineering message lengths feature... \n'
        #message lengths
        dat['msg_len'] = self.eng_msg_len(dat)
        
        print 'engineering message sentiments/subjectivity features... \n'
        #message sentiments
        dat['polarity'], dat['subjectivity'] = self.eng_msg_senti(dat)
        
        print 'engineering message POS features... \n'
        #POS tagging
        dat = self.eng_msg_POS(dat)
        
        print 'engineering message emoticons features... \n'
        #emoticons counts
        dat = self.eng_msg_emo(dat)
        
        print 'engineering message hookup indicator feature... \n'
        #hookup indicator
        dat['is_hookup'] = self.eng_msg_hookup_flag(dat)
        
        #impute missing
        # will be performed in clean_features
        return dat
    
        
        
    # -----------------------------------
    # Engineered Features
    # -----------------------------------    
    def eng_msg_len(self, dat):
        """
        Takes in a dataframe with messages in the "text" column, 
        returns lengths of each as new column, including all first person refernces,
        and stopwords. 
        """
        return dat['text'].apply(lambda msg: len(str(msg).decode('utf8').split(' ')))
        
        
        
        
    def eng_msg_senti(self, dat):
        """
        Takes in a dataframe with messages in a "text" column,
        returns sentiment polarity and subjectivity scores from a propietary lexicon
        from the TextBlob module.
        """
        
        return dat.text.apply(lambda msg: TextBlob(str(msg).decode('utf8')).sentiment[0]),  dat.text.apply(lambda msg: TextBlob(str(msg).decode('utf8')).sentiment[1])
    
    
    
    
    def eng_msg_POS(self, dat):
        """
        Takes in a dataframe with messages in a "text" column,
        returns new columns corresponding to Parts of Speech counts in the msg.
        
        Note: Takes approx. 4 minutes to run
        
        Note: Returns the entire dataframe with all columns corresponding to all POS
        
        """
        print "NOTE: Please wait approx. 4 mintues for Parts Of Speech tagging to complete!"
        parts={
        'CC':0,
        'CD':0,
        'DT':0,
        'EX':0,
        'FW':0,
        'IN':0,
        'JJ':0,
        'JJR':0,
        'JJS':0,
        'LS':0,
        'MD':0,
        'NN':0,
        'NNS':0,
        'NNP':0,
        'NNPS':0,
        'PDT':0,
        'POS':0,
        'PRP':0,
        'PRP$':0,
        'RB':0,
        'RBR':0,
        'RBS':0,
        'RP':0,
        'SYM':0,
        'TO':0,
        'UH':0,
        'VB':0,
        'VBD':0,
        'VBG':0,
        'VBN':0,
        'VBP':0,
        'VBZ':0,
        'WDT':0,
        'WP':0,
        'WP$':0,
        'WRB':0}
        
        for key in parts.keys():
            dat[key]=0
        
        for i in np.arange(0,dat.shape[0]):
            for word,pos in TextBlob(str(dat.text[i]).decode('utf-8')).tags:
                dat.loc[i,pos] += 1
    
    
        return dat
        
        
        
    def eng_msg_emo(self, dat):
        """
        Takes in a dataframe with messages in a "text" column,
        returns new columns corresponding to emoticon counts (grouped into 10 types)
        
        Note: Returns the entire dataframe with all columns corresponding to all POS
        
        """
        
        #create emoji regex dict
        emo_dict={'smile':':[-\)]',
        'wink':';[-\)p]',
        'sad':':[-\(]',
        'cool':'B\)|:>',
        'laugh':':[-dD]',
        'cry':'D:|:[-s]',
        'silly':':[-p]',
        'other':':[-ox\|\/\[@]|>:-\)',
        'pos_emos':':[\*]|o:\)|:3|\(y\)|<3',
        'neg_emos':':[-!]|>:-o|\(n\)'}
        
        #initialize emo columns
        for key in emo_dict.keys():
            dat[key] = 0
        
        #populate counts. For every message, check for key match and update count
        for i in np.arange(0, dat.shape[0]):
            msg = str(dat['text'][i]).decode('utf-8')
            
            for k, v in emo_dict.iteritems():
                pattern = re.compile(v)
                num_matches = len(pattern.findall(msg))
                if num_matches:
                    dat.loc[i, k] += num_matches
        
        return dat
        
        
    def eng_msg_hookup_flag(self, dat):
        """
        Takes in a dataframe with messages in a "text" column,
        returns a Series of flag value counts that indicate whether or not the message
        contains regex patterns indicating possible "hookup" styled messages.
        E.g. 21 F, 17male, 34f, etc.
        """
      
        return dat.text.str.count(r'\s*\d{2,2}\s*[MmFf]|\s*[MmFf]\s*\d{1,2}|\d\d-\d\d')
        
    
    
    #---------------------
    #PRE-PROCESS MESSAGES: impute missing, lowercase, alphabets only,
    #remove stops, stem!
    #note: emoji vocabulary and lengths have already been computed
    #at the end of get_corpus() messages wil be ready for tokenization
    # and BOW
    #--------------------
    def get_corpus(self, dat):
        """
        Takes a dataframe with the text in "text" column,
        extracts alphabets only, lowercase, stopped, stemmed, and tokenize-able. 
        Also imputes missing text to "msngval".
        
        returns a list of 'Bag of Words'-ready sentences
        """
        
        corpus =[]
        
        for i in np.arange(0,dat.shape[0]):
            corpus.append(self.clean_msgs(dat['text'][i]))
        
        return corpus
    
    
    def clean_msgs(self, msg):
       """
       Supports get_corpus() by performing the following actions on each 
       message and returning the cleaned, tokeniz-able version
       """
       
       #if missing, impute with "msngval"
       if not msg:
           msg=u'msngval'
       else:
           msg=str(msg).decode('utf8')
           
           #alphabets only
           msg = re.sub('[^a-zA-Z]',' ', msg)
           
           #lowercase, make list
           msg = msg.lower().split()
    
           #remove stopwords
           stops = set(stopwords.words('english'))
           msg = [word for word in msg if not word in stops]
    
           #stem
           p=PorterStemmer()
           msg = [p.stem(word) for word in msg]
       
       return " ".join(msg).decode('utf8')
        
        
    def get_BOW(self, corpus, idf=True, sublinear_tf=True, ngram=(1,1)):
        """
        Takes in a corpus list of text sentences (pre-processed) and 
        returns a Bag Of Words model. Also pickles the trained vectorizer
        for test set to pick up later.
        """
        print 'TFIDF Vectorizing...\n'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram, stop_words=None, use_idf=idf, smooth_idf=True, sublinear_tf=sublinear_tf)
        
        vectorizer.fit(corpus)
        
        # #print 'pickling trained vectorizer...\n'
        # #try:
        #     #pickle this vectorizer so test set can use to transform
        #     pickle.dump(vectorizer, './../pickles/trained_vectorizer.p')
        #     print 'success!\n'
        # except PickleError as e:
        #     print 'pickling error! {}'.format(e)
        
        
        return vectorizer, (vectorizer.transform(corpus)).toarray()
    
    
    
    def merge_BOW_engfeats(self, bow, dat):
        """
        Merge the Bag Of Words model with the engineered features
        to return a final dataset to train or test.
        """
        print 'Bag Of Words dim:{}, Engineered features dim:{}\n'.format(bow.shape, dat.shape)
        assert (bow.shape[0] == dat.shape[0]),"BOW rows mismatch with dataset!"
        
        print 'Imputing NaNs and Normalizing \n'
        dat_feats = dat.loc[:, dat.columns != 'text']
        #impute missing values to 0 (NaNs resulted from empty strings)
        temp=dat_feats.apply(lambda val: val.fillna(0), axis=1)
        ##NORMALIZE ONLY THE CONCATENATED FEATURES to 0,1 range 
        dat_feats=preprocessing.MinMaxScaler().fit_transform(temp)
        
        print 'Concatenating BOW features with engineered features... \n'
        #we now have BOW model, append engineered features to this matrix
        #all features passed into dat that are not "text" are engineered
        
        return   np.concatenate((bow, dat_feats), axis=1)
        
    
    
    def model_pipe_fit(self, finaldat, targets, params = {
            #'svd__n_components':[10, 50, 100, 500],
            'clf_LR__C': [10],
            'clf_LR__penalty': ['l2'],
            'clf_LR__class_weight':['balanced'],
            }, cv=5):
        """
        Pipeline the prepared dataset through dimensionality reduction
        and a classifier. Perform a grid search cross vlidation on this
        classifier using the macro f1 score for imbalanced classes.
        
        Using Logistic Regression because it has performed the better than Random Forests
        and SVMs.
        
        Pickle the best estimators to predict with later.
        
        """
        
        pipe = Pipeline([#('svd', TruncatedSVD()),
        ('clf_LR', LogisticRegression())])
        
        
        print 'starting grid search on pipeline: SVD + Logistic Regression (tuning with the Macro F1 score)\n'
        gs = GridSearchCV(pipe, params, scoring='f1_macro', verbose=10, cv=cv)
        gs.fit(finaldat, targets)
        
        print "Best training macro f1 score: {:.05f}\n".format(gs.best_score_)
        print "Best estimators: {}\n".format(gs.best_estimator_)
        print "Best parameters: {}\n".format(gs.best_params_)
        
        # print 'pickling best model \n'
        # try:
        #     #pickle this model so test set can use to transform
        #     pickle.dump(gs, './../pickles/trained_model.p',protocol=2)
        #     print 'success!\n'
        # except PickleError as e:
        #     print 'pickling error! {}'.format(e)
            
        
        return gs



def main():
    
    trainfile = './../data/train.csv'
    testfile = './../data/test.csv'
    
    if len(sys.argv)>1:
        trainfile = sys.argv[1]
    else:
        print 'WARNING: No train path provided, will use default "./../data/train.csv" as train location\n'
    
    if len(sys.argv)>=2:
        testfile = sys.argv[2]
    else:
        print 'WARNING: No test path provided, will use default "./../data/test.csv" as test location\n'

    tt = TextCat(trainfile, testfile)
    
    #get new features
    print 'Building new TRAINING set features: \n'
    trainfeats = tt.eng_new_features(tt.traindat)
    print '=========================================\n'
    print 'Building new TEST set features: \n'
    testfeats = tt.eng_new_features(tt.testdat)
    
    #preprocess corpus
    print 'Pre-processing training messages: \n'
    traincorp = tt.get_corpus(tt.traindat)
    print 'Pre-processing test messages: \n'
    testcorp = tt.get_corpus(tt.testdat)
    
    #get BOWs
    print 'Generating training Bag Of Words: \n'
    vectorizer, trainBOW = tt.get_BOW(traincorp, idf=True, sublinear_tf=False, ngram=(1,1))
    ##get test BOW by transforming on trained vectorizer
    #trainedBOW, vect = pickle.load('./../pickles/trained_vectorizer.p')
    print 'Generating test Bag Of Words: \n'
    testBOW = vectorizer.transform(testcorp).toarray()
    print 'Test BOW dimensions: ',testBOW.shape
    
    #merge BOWs and Eng. Features
    print 'TRAIN: Merging Engineered Features and Bag Of Words: \n'
    train = tt.merge_BOW_engfeats(trainBOW, trainfeats)
    print 'TEST: Merging Engineered Features and Bag Of Words: \n'
    test = tt.merge_BOW_engfeats(testBOW, testfeats)
    
    #Final train and test ready!
    #Train model
    print 'Begin Training! \n'
        
    #Best parameters: {'clf_LR__class_weight': 'balanced', 'clf_LR__penalty': 'l2', 'clf_LR__C': 10, 'svd__n_components': 3000}
    mod = tt.model_pipe_fit(train, tt.traintargets, #params = {
        #'svd__n_components':[100, 500, 1000, 3000],
        #'clf_LR__C': [0.01, 0.1, 1, 10, 50, 100],
        #'clf_LR__penalty': ['l2','l1'],
        #'clf_LR__class_weight':[None,'balanced'],}, 
        cv=10)
    
    print '============================================\n'
    print 'Predicting on TEST SET...'
    print '============================================\n'
    print 'OUT OF SAMPLE RESULTS: \n'
    #predict
    preds = mod.predict(test)
    
    mf1 = f1_score(tt.testtargets, preds, average='macro')
    
    print "Out of sample Macro F1 Score: {:.03f}".format(mf1)
    print "Classification Report..."
    print (classification_report(tt.testtargets, preds))
    
       
        
        
if __name__=="__main__":
    main()
