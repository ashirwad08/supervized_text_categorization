# Predict Chat Message Category    

This is a one off project that involves exploring a sample dataset of chat messages that are labeled according to the category of the "channel" they belong to. I perform a quick data exploration to generatesome meaningful features that can be quickly put through a supervised learning routine to get an idea of predictive performance about how well the algorithm can classify the messages.  
  

Objective: Build a quick MVP to demonstrate a classifier's predictive performance on categorizing messages.   

* Data Exploration is [here](./Messages_Exploratory/readme.md)  
* A self contained sklearn preprocessing and prediction pipeline that outputs _out of sample_ classifier performance is [here](./src/text_categorizer.py)  
* I use the _Macro F1 Score_ to get an idea of classifier performance across the various outcome classes despite their imbalances. On subsequent iterations, I'd like to generate a weighted score to investigate performance on each of the categories in depth.  
* Other to-dos for subsequent iterations: Use LSA or LDA to extract latent concepts within categories, these could perhaps be fed in as additional features to the feature space.    
