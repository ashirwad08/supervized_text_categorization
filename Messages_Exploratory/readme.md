

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

Reading in the training data, 1.1MB file so let's just read in the whole thing to take a look. _baseline.py_ has a nice function to read it into a dict but I prefer pandas.


```python
train = pd.read_csv('./../data/train.csv')
```


```python
train.sample(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9292</th>
      <td>misc</td>
      <td>People are people So why should it be You and ...</td>
    </tr>
    <tr>
      <th>10726</th>
      <td>meetup</td>
      <td>Any girls want to read a story ? Send me a pic...</td>
    </tr>
    <tr>
      <th>2403</th>
      <td>personal</td>
      <td>I am so bored and lonely.</td>
    </tr>
    <tr>
      <th>10625</th>
      <td>personal</td>
      <td>Can not understand when I comment on friends F...</td>
    </tr>
    <tr>
      <th>2266</th>
      <td>meetup</td>
      <td>Any cute/hoy guys or girls wanna skype? 20f se...</td>
    </tr>
    <tr>
      <th>9830</th>
      <td>personal</td>
      <td>I started smoking to fit in, now my favorite p...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>misc</td>
      <td>Shaved today... Not getting laid. Grrr</td>
    </tr>
    <tr>
      <th>6704</th>
      <td>personal</td>
      <td>3 months of hard work is starting to pay off.</td>
    </tr>
    <tr>
      <th>7069</th>
      <td>misc</td>
      <td>Shannon where are you</td>
    </tr>
    <tr>
      <th>11457</th>
      <td>school</td>
      <td>I paid off $665 and some cents on my student l...</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (14048, 2)



14,048 messages with a category outcome. How many categories in the training set?


```python
len(train.category.unique())
```




    17



And what's the category prevalence in the training set like, for these 17 categories...


```python
train.category.value_counts(normalize=True).plot(kind='bar', figsize=(15,5))
plt.title('Message category prevalence in dataset');
```


![png](output_8_0.png)


We have an imbalance problem! 5 categories account for almost 80% of the training set. Any classifier we train won't learn to distinguish the remaining 12 categories very well.

## 1. Message Length  

Do message lengths (# of words) clue us into categories?


```python
#with stopwords
train['num_words']=train['text'].apply(lambda msg: len(str(msg).decode('utf8').split(' ')))
#train.text[5].split(' ')
```


```python
g = sns.FacetGrid(train, col='category', size=3, aspect=2, col_wrap=3, sharey=False)
g = g.map(plt.hist, 'num_words', bins=np.arange(0,60,2))
```


![png](output_12_0.png)


We see that the distribution of the number of words (including stopwords) does vary by categories, some more than the other. This might be a decent predictor at least between categories that are prone to short messages (ex. "misc", "meetup", or "tattoos" versus those that tend towards being more wordy such as "school", "family", or "relationships". 

---  

## 2. Emoji Prevalence  

What about the prevalence of common **emojis**? Can the usage of emojis clue us in to certain categories?


```python
train['num_emojis']=train.text.str.count(':[-\)]|:[-\(]|:[-Dd]|:[-Pp]|:[-o0O]|:[-\\\/]|:[-$*@]|:smile:|:hug:|[<>]3|;[\)]')
```

Calculate the prevalence of emojis across the total number of messages within each class.


```python
(train.groupby(by='category')['num_emojis'].sum()/train.groupby(by='category').size()).sort_values(ascending=False).plot(kind='bar', figsize=(15,5))
plt.title('Emoji prevalence by class')
```




    <matplotlib.text.Text at 0x11fa7bed0>




![png](output_17_1.png)


Interesting results! We notice that classes tending toward social interaction, such as "military", "meetup", "relationships", "lgbtq", "misc", "personal", "school" tend to see a higher usage of emojis. Perhaps this alludes to the dominant age group of the class or the nature of the interactions, or both. Nevertheless, Emoji prevalence might be a very good predictor. 

> Although this is an interesting insight, *emoji prevalence* **cannot** be used as-is as a feature in this exercise because of problems with validating the prevalence against unseen holdout data. **One possible way to incorporate this into the ML exercise could be to add a bunch of features during the tokenization exercise that represent a *bag of emojis* - a tedious exercise, but one that might help the classifier discern classes well, as seen in the figure above.**  

**For the final model, I've incorporated an emoji vocabulary of 10 common emojis.**

--- 

## 3. Sentiment Analysis  

Does message *sentiment* clue us into the categories? How about their subjectivity? Let's use the *TextBlob* package's generic sentiment scorer, to start with.


```python
from textblob import TextBlob
```


```python
train['polarity']=train.text.apply(lambda msg: TextBlob(str(msg).decode('utf-8')).sentiment[0])
train['subjectivity']=train.text.apply(lambda msg: TextBlob(str(msg).decode('utf-8')).sentiment[1])
train['sentiment']=np.where(train['polarity']>0, 'positive','negative') #for visualizing
train['sentiment']=np.where(train['polarity']==0, 'neutral',train['sentiment']) #for visualizing
```


```python
pd.set_option('max_colwidth',150)
train[['text','polarity','subjectivity','sentiment']].sample(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1490</th>
      <td>Trying to do this rubiks cube I'm getting mad</td>
      <td>-0.625000</td>
      <td>1.000000</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>661</th>
      <td>Any ladies wanna drink?</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>6917</th>
      <td>I met a guy on here a couple weeks ago and I really want to meet him in person.</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>10847</th>
      <td>So bored in hartlepool, there'sfuck all to do</td>
      <td>-0.500000</td>
      <td>1.000000</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>11501</th>
      <td>I wonder if it's possible to create a new Playstation Network Account.</td>
      <td>0.068182</td>
      <td>0.727273</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>12026</th>
      <td>Stop acting like the world owes you shit the world don't woe you shit EARN what you want Stop asking for fucking handouts You lazy fucks</td>
      <td>-0.250000</td>
      <td>0.680000</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>I can't remember the last time I left the house without makeup on. I'm a guy..</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>4769</th>
      <td>I don't understand why I never had a bf or a first kiss...</td>
      <td>0.250000</td>
      <td>0.333333</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>9878</th>
      <td>I'm nothing but a letdown</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>5595</th>
      <td>If I take care of myself, why doesn't the urge go away?</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
  </tbody>
</table>
</div>



While not perfect, the package sentiment and subjectivity scorer does a decent job. Subsequent iterations can train mroe personalizes sentiment models. How does sentiment (polarity) of messages vary by class?


```python
kws = dict(s=100, linewidth=.5, edgecolor="black")

g = sns.FacetGrid(train, col='category', size=4, aspect=2, col_wrap=3, 
                  sharey=True, sharex=True, hue='sentiment', palette=['blue','red','green'])
g = g.map(plt.scatter, 'polarity', 'subjectivity', **kws)
g.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x1235dd8d0>




![png](output_25_1.png)


There are some catgories where sentiment (not a perfect score itself) seems to be able to discren well, for example: "lgbtq" sees more positive sentiments that are moderately subjective, as does "tatoos", "faith" and "sports". The sparse classes seem to see a clearer distinction of sentiment. Subjectivity isn't very helpful as polarity is. Improving the sentiment accuracy by training on a curated lexicon specific to this space might sharply improve the discerning power of this feature.

---  

## 4. Parts of Speech Analysis


```python
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
    train[key]=0


for i in np.arange(0,train.shape[0]-1):
    for word,pos in TextBlob(str(train.text[i]).decode('utf-8')).tags:
        train.loc[i,pos] += 1

```


```python
pd.set_option('max_columns',100)
train.sample(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>text</th>
      <th>num_words</th>
      <th>num_punct</th>
      <th>num_emojis</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>sentiment</th>
      <th>WRB</th>
      <th>PRP$</th>
      <th>VBG</th>
      <th>FW</th>
      <th>CC</th>
      <th>PDT</th>
      <th>RBS</th>
      <th>PRP</th>
      <th>CD</th>
      <th>WP$</th>
      <th>VBP</th>
      <th>VBN</th>
      <th>EX</th>
      <th>JJ</th>
      <th>IN</th>
      <th>WP</th>
      <th>VBZ</th>
      <th>DT</th>
      <th>MD</th>
      <th>NNPS</th>
      <th>RP</th>
      <th>NN</th>
      <th>POS</th>
      <th>RBR</th>
      <th>VBD</th>
      <th>JJS</th>
      <th>JJR</th>
      <th>SYM</th>
      <th>VB</th>
      <th>TO</th>
      <th>UH</th>
      <th>LS</th>
      <th>RB</th>
      <th>WDT</th>
      <th>NNS</th>
      <th>NNP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8885</th>
      <td>relationships</td>
      <td>I'm fucking done with my boyfriend ... First of all I he is cheating on me and then he says I'm fucking cheating on him...</td>
      <td>25</td>
      <td>123.0</td>
      <td>0.0</td>
      <td>-0.316667</td>
      <td>0.644444</td>
      <td>negative</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8358</th>
      <td>personal</td>
      <td>Its to the point i don't want to hang out with my friends They just get hit on or talk about boys who flirt with them I'm to embarrassed to say no...</td>
      <td>42</td>
      <td>191.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4929</th>
      <td>meetup</td>
      <td>Any ladies want to just chat</td>
      <td>6</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['adjectives'] = np.sum(train[['JJ','JJR','JJS']], axis=1)
train['verbs'] = np.sum(train[['VB','VBG','VBD','VBN','VBP','VBZ']], axis=1)
train['prop_nouns'] = np.sum(train[['NNP','NNPS']], axis=1)
```


```python
#visualize parts of speech densities by class
#kws = dict(s=100, linewidth=.5, edgecolor="black")

#plot hist adjectives, verbs, nouns
g = sns.FacetGrid(train, col='category', size=3, aspect=2, col_wrap=3, 
                  sharey=False, sharex=True)
g = g.map(plt.hist, 'verbs', bins=np.arange(0,20,1))
#g.add_legend()
```


![png](output_31_0.png)


Adjectives: Mostly right skewed but "fashion", "family", "military" do tend to see more adjectives per message.  
Verbs (base form): definitely sees some variation between classes
Proper Nouns: "pop_culture", "sports", "meetups" do see the use of more proper nouns 

Did not check other POS such as *adverbs*, *prepositions*, etc. but the POS composition of messages might clue us into the category. **We'll preserve each POS as a feature in our vector space.**


## Future Implementation for POS Tagging: 

On subsequent iterations, I'd like to use the Parts of Speech to formulate more meaningful n-gram structures instead of brute-force n-gram permutations (which, later analysis reveals, don't increase the overall performance by much anyway).

---  

# 5. FUTURE IMPLEMENTATION! Concept Space: Latent Semantic Analysis or Latent Dirichlect Analysis (Keyword/Topic Extraction) 

Can we incorporate an *unsupervised* approach - by extracting the main concepts around each class - into our *supervised* learning training? One idea could be to extract the main keywords/phrases *for the classes* and incorporate this into the training set vector space for each record.  

If a test set has multiple records, then a similar exercise might be performed on the test set vector space, with 0's for unseen and missing features and missing classes.  



```python
#not incorporated
```

# 6. Flag "Hookup" Style Messages

The "meetup" category seems to have a lot of messages indicating age and sex of the person, usually of format "23 m" or "f 20". Let's create a flag to identify this in the text and see how well it does to discern the "meetup" class.


```python
import re
pattern = re.compile(r'\s*\d{2,2}\s*[MmFf]|\s*[MmFf]\s*\d{1,2}|\d\d-\d\d')
```


```python
pattern.findall(' girls 15-17 22m')
```




    ['15-17', ' 22m']




```python
train['is_hookup']=train.text.str.count(r'\s*\d{2,2}\s*[MmFf]|\s*[MmFf]\s*\d{1,2}|\d\d-\d\d')
```


```python

(train.groupby(by='category')['is_hookup'].sum()/train.groupby(by='category').size()).sort_values(ascending=False).plot(kind='bar', 
                                                                                           figsize=(15,5))
plt.title('Prevalence of "Hookup" Type Messages in Classes')
```




    <matplotlib.text.Text at 0x11bf9d910>




![png](output_40_1.png)


Clearly, the is_hookup flag is a **strong** indicator of "meetup" and "lgbtq" categories.

# 7. Missing Values


```python
train[train['text'].isnull()].sample(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>text</th>
      <th>is_hookup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6047</th>
      <td>lgbtq</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9084</th>
      <td>misc</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11744</th>
      <td>misc</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7057</th>
      <td>meetup</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9319</th>
      <td>faith</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7627</th>
      <td>pop_culture</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12785</th>
      <td>misc</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>meetup</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4228</th>
      <td>misc</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>726</th>
      <td>misc</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
(train[train['text'].isnull()].groupby(by='category').size()/train.groupby(by='category').size()).sort_values(ascending=False).plot(kind='bar',
                                                                                                      figsize=(15,5))
plt.title("Missing Messages Prevalence Amongst Training Categories")
```




    <matplotlib.text.Text at 0x11ba91c90>




![png](output_44_1.png)


Messages that are missing are much more prevalent in "military", "faith", "misc", and "Tatoos" categories. Instead of dropping, let's mark these strings as "msngval" to indicate missing message. Assuming this is *not* human error, this might serve to indicate a systematic or behavioral anomaly pertaining to these categories. The imputed string should also survive stemming and BOW.

---  

# Result Summary  

I proceed with building a non-parametric model with 6 out of 7 features explored above, concatenated to a Bag Of Words model weighted by Term Document Inverse Document Frequency. Multiple n-grams were tested but didn't make a great deal of difference in the final output, so I stuck with a unigram model. 

Upon training with multiple classifiers (see rough work notebook "Build_Test_Train.ipynb"), Logistic Regression delivers the best performance on a *non-reduced dimensional* space of approximately 7,678 feaetures. The final results are printed below.  

**Run the attached script text_categorizer.py by passing the train and test file paths and names in the following format (note: paths will default to *./../data/train.csv* and *./../data/test.csv* if nothing provided): **  
> text_categorizer.py /filepath/train.csv /filepath/test.csv  


--- 

*Copying outputs (out of sample) on performing a *grid search cross validation* and fitting the best parameters are...*


Best parameters: {'clf_LR__class_weight': 'balanced', 'clf_LR__penalty': 'l2', 'clf_LR__C': 10}

============================================

============================================

OUT OF SAMPLE RESULTS: 

Out of sample Macro F1 Score: 0.531
Classification Report...  

             precision    recall  f1-score   support

    animals       0.56      0.73      0.63        26
      faith       0.36      0.36      0.36        11
     family       0.57      0.78      0.66       110
    fashion       0.30      0.31      0.30        42
       food       0.51      0.65      0.57        63
      lgbtq       0.64      0.67      0.66       147
     meetup       0.76      0.80      0.78       737  
   military       0.78      0.50      0.61        14
       misc       0.52      0.42      0.46       683
   personal       0.57      0.53      0.55       910
pop_culture       0.44      0.36      0.39       150
        qna       0.29      0.43      0.35       221
relationships       0.56      0.54      0.55       337
     school       0.58      0.58      0.58        74
     sports       0.24      0.40      0.30        10
     tatoos       0.70      0.93      0.80        15
       work       0.42      0.55      0.48        49

avg / total       0.57      0.57      0.57      3599



![Out of sample results](./../visuals/oos_results.png)


```python

```
