# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 15:33:15 2021

@author: ruben
"""
import requests
import pandas as pd
from datetime import datetime
import datetime as dt
from urllib import parse
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import statsmodels.api as sm
import nltk
import spacy
#from spacy.cli.download import download
#download(model="en_core_web_sm")
import re
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
from collections import Counter
from nltk.util import ngrams

#######  Scrape the CDC website for their COVID updates  #######

def get_info_old(url):
    #send request   
    response = requests.get(url)
    #parse    
    soup = BeautifulSoup(response.text)
    #get information we need
    news = soup.find('div', attrs={'class': 'card-body bg-white'}).text
   
    parse_result = parse.urlparse(url)
    parse_splitted = parse_result.path.split("/")
    date_raw = parse_splitted[len(parse_splitted)-1]
    date = datetime.strptime(date_raw.split(".")[0], "%m%d%Y")

    columns = [news,date]
    column_names = ['News','Date']
    return dict(zip(column_names, columns))

def get_info_new(url):
    #send request   
    response = requests.get(url)
    #parse    
    soup = BeautifulSoup(response.text)
    #get information we need
    news = soup.find('div', attrs={'class': 'row mb-3 bg-white'}).text
   
    parse_result = parse.urlparse(url)
    parse_splitted = parse_result.path.split("/")
    date_raw = parse_splitted[len(parse_splitted)-1]
    date = datetime.strptime(date_raw.split(".")[0], "%m%d%Y")
    columns = [news,date]
    column_names = ['News','Date']
    return dict(zip(column_names, columns))

dates = [dt.datetime(2020, 4, 10, 0, 0)]
date_list = []
links = []
time_add = dt.timedelta(days=7)  
for i in range(1,78):
    new_date = dates[i-1]+time_add
    dates.append(new_date)
    date_list.append(dates[i-1].strftime("%m%d%Y"))
    links.append("https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/"+date_list[i-1]+".html")

#Manual correction for Thanksgiving, Christmas and New Years Eve:
date_list[33] = '11302020'
links[33] = "https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/"+date_list[33]+".html"
date_list[37] = '12282020'
links[37] = "https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/"+date_list[37]+".html"
date_list[38] = '01042021'
links[38] = "https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/"+date_list[38]+".html"

#And a rather unfortunate mistake by the CDC, where they mispelled 2021 into 2121       
date_list[53] = '04162121'
links[53] = "https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/"+date_list[53]+".html"

#And a day off
date_list.remove('06182021')
links.remove("https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/past-reports/06182021.html")

df_old = [get_info_old(url) for url in links[:44]]
df_new =  [get_info_new(url) for url in links[44:] ]
df = df_old+df_new
news_df = pd.DataFrame(df) #Final dataframe with the articles and dates

#save the articles dataset
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
news_df.to_csv('out.zip', index=False,
          compression=compression_opts) 

#######  preprocessing the data  #######
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

#import other lists of stopwords
with open('StopWords_GenericLong.txt', 'r') as f:
 x_gl = f.readlines()
with open('StopWords_DatesandNumbers.txt', 'r') as f:
 x_d = f.readlines()
#import nltk stopwords
stopwords = nltk.corpus.stopwords.words('english')
#combine all stopwords
[stopwords.append(x.rstrip()) for x in x_gl]
[stopwords.append(x.rstrip()) for x in x_d]
#change all stopwords into lowercase
stopwords_lower = [s.lower() for s in stopwords]

def text_preprocessing(str_input): 
     #tokenization, remove punctuation, lemmatization
     words=[token.lemma_ for token in nlp(str_input) if not token.is_punct]
 
     # remove symbols, websites, email addresses 
     words = [re.sub(r"[^A-Za-z@]", "", word) for word in words] 
     words = [re.sub(r"\S+com", "", word) for word in words]
     words = [re.sub(r"\S+@\S+", "", word) for word in words] 
     words = [word for word in words if word!=' ']
     words = [word for word in words if len(word)!=0] 
 
     #remove stopwords     
     words=[word.lower() for word in words if word.lower() not in stopwords_lower]
     #combine a list into one string   
     string = " ".join(words)
     return string            

def wordcount(words, dct):
    counting = Counter(words)
    count = []

    for key, value in counting.items():
        if key in dct:
            count.append([key, value])
    return count

def negwordcount(words, dct, negdct, lngram):
    mid = int(lngram / 2)
    ng = ngrams(words, lngram)
    nglist = []
    for grams in ng:
        nglist.append(grams)
    keeper = []
    n = len(nglist)
    i = 1
    for grams in nglist:
        if n - i < int(lngram / 2):
            mid = mid + 1
        if grams[mid] in dct:
            for j in grams:
                if j in negdct:
                    keeper.append(grams[mid])
                    break
        i = i + 1
    count = wordcount(keeper, dct)

    return count

def lexcnt(txt, txt_raw, pos_dct, neg_dct, negat_dct, lngram):
    #txt is the preprocessed text to save computation time. The raw text is only used for seeing if negations are present. 
    txt = word_tokenize(txt)
    # Count words in lexicon
    pos_wc = wordcount(txt, pos_dct)
    pos_wc = [cnt[1] for cnt in pos_wc]
    pos_wc = sum(pos_wc)

    neg_wc = wordcount(txt, neg_dct)
    neg_wc = [cnt[1] for cnt in neg_wc]
    neg_wc = sum(neg_wc)

    # Count negated words in lexicon
    pos_wcneg = negwordcount(txt_raw, pos_dct, negat_dct, lngram)
    pos_wcneg = [cnt[1] for cnt in pos_wcneg]
    pos_wcneg = sum(pos_wcneg)

    neg_wcneg = negwordcount(txt_raw, neg_dct, negat_dct, lngram)
    neg_wcneg = [cnt[1] for cnt in neg_wcneg]
    neg_wcneg = sum(neg_wcneg)

    pos = pos_wc - (pos_wcneg) + neg_wcneg
    neg = neg_wc - (neg_wcneg) + pos_wcneg

    if pos > neg:
        out = 1
    elif pos < neg:
        out = -1
    else:
        out = 0
    
    return pos, neg, out


news_df['news_cleaned']=news_df['News'].apply(text_preprocessing)

negat_dct = ["n't", "not", "never", "no", "neither", "nor", "none"]
lngram = 7

# Dictionaries
# negative dictionary 
neg_dct = ""
with io.open("negativemaster.txt", "r", encoding = "utf-8", errors = "ignore") as infile:
    for line in infile:
        neg_dct = neg_dct + line
# saved the lm_negative dictionary in variable neg_dct
neg_dct = neg_dct.split("\n")
neg_dct = [e.lower() for e in neg_dct]   # converted uppercase words to lowercase

# positive dictionary
pos_dct = ""
with io.open("positivemaster.txt", "r", encoding = "utf-8", errors = "ignore") as infile:
    for line in infile:
        pos_dct = pos_dct + line

pos_dct = pos_dct.split("\n")
pos_dct = [e.lower() for e in pos_dct]

pred = [lexcnt(news_df["News"][i], news_df["news_cleaned"][i], pos_dct, neg_dct, negat_dct, lngram) for i in range(0,news_df.shape[0])]
pred = pd.DataFrame(pred, columns=('p','n', 'out'))

news_df['Sentiment']= pred['out']


######  plotting  ######

#correct CDC's mistake of the 2121 date:
news_df['Date'][53] = dt.datetime(2021, 4, 16, 0, 0)

#Load financial data
nq = yf.Ticker("^IXIC")
nqhist_full = nq.history(start='2020-04-16', end='2021-09-24')["Open"]
nqhist = []
time_add1=dt.timedelta(days=1)

for i in range(0, len(news_df['Date'])):
   nqhist.append(nq.history(start=news_df['Date'][i], end=news_df['Date'][i]+time_add1)["Open"][0])

nq_daily_returns_full = pd.DataFrame(nqhist_full).pct_change()
nq_daily_returns_full = nq_daily_returns_full[1:]
nq_daily_returns = pd.DataFrame(nqhist).pct_change()
nq_daily_returns = nq_daily_returns[0][1:]



fig=plt.figure()
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax3=fig.add_subplot(111, label="3", frame_on=False)

ax.plot(nq_daily_returns_full.index.values, nq_daily_returns_full, color="C0")
ax.set_xlabel("Date", color="C0")
ax.set_ylabel("%", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

ax2.plot(news_df['Date'], news_df['Sentiment'], color="C1")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('', color="C1") 
ax2.set_ylabel('Sentiment', color="C1")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="C1")
ax2.tick_params(axis='y', colors="C1")
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)


ax3.set_xticks([])
ax3.set_yticks([])

plt.show()

#Scaling the articles' sentiment
p_train = np.array(pred['p'])
p_train = p_train.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
p_scaled = scaler.fit_transform(p_train) 
pred['p_scaled'] = p_scaled

n_train = np.array(pred['n'])
n_train = n_train.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
n_scaled = scaler.fit_transform(n_train)
pred['n_scaled'] = -n_scaled

pred['sent_body'] = pred[['p_scaled','n_scaled']].mean(axis=1)

s_train = np.array(pred['sent_body'])
s_train = s_train.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(-1,1))
s_scaled = scaler.fit_transform(s_train) 
pred['sent_scaled'] = s_scaled

pred.hist(column='sent_scaled',bins=40)

df_comp = news_df.join(pred['sent_scaled'], how='outer')
df_comp = df_comp.join(pred['p'], how='outer')
df_comp = df_comp.join(pred['n'], how='outer')


#Plotting with scaled sentiment
fig=plt.figure()
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax3=fig.add_subplot(111, label="3", frame_on=False)

ax.plot(nq_daily_returns_full.index.values, nq_daily_returns_full, color="C0")
ax.set_xlabel("Date", color="C0")
ax.set_ylabel("%", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

ax2.plot(df_comp['Date'], df_comp['sent_scaled'], color="C1")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('', color="C1") 
ax2.set_ylabel('Sentiment', color="C1")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="C1")
ax2.tick_params(axis='y', colors="C1")
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)


ax3.set_xticks([])
ax3.set_yticks([])

plt.show()

######  Correlations  ######
corr, _ = kendalltau(df_comp['Sentiment'][:75], nq_daily_returns)
corr

corr, _ = kendalltau(df_comp['sent_scaled'][:75], nq_daily_returns)
corr

nq_var = []
for i in range(0, (len(df_comp['Date'])-1)):
    nq_data = nq.history(start=df_comp['Date'][i], end=df_comp['Date'][i+1])["Open"]
    nq_ret = pd.DataFrame(nq_data).pct_change()
    nq_var.append(np.var(nq_ret[1:]))


corr, _ = kendalltau(df_comp['Sentiment'][:75], nq_var)
corr

corr, _ = kendalltau(df_comp['sent_scaled'][:75], nq_var)
corr

senti = np.array(df_comp['Sentiment'][:75]).reshape((-1,1))
senti2 = sm.add_constant(senti)
est = sm.OLS(nq_daily_returns, senti2)
est2 = est.fit()
print(est2.summary())

senti_scaled = np.array(df_comp['sent_scaled'][:75]).reshape((-1,1))
senti_scaled2 = sm.add_constant(senti_scaled)
est = sm.OLS(nq_daily_returns, senti_scaled2)
est2 = est.fit()
print(est2.summary())

senti3 = np.array(df_comp['Sentiment'][:75]).reshape((-1,1))
senti4 = sm.add_constant(senti3)
est = sm.OLS(nq_var, senti4)
est2 = est.fit()
print(est2.summary())

senti_scaled3 = np.array(df_comp['sent_scaled'][:75]).reshape((-1,1))
senti_scaled4 = sm.add_constant(senti_scaled3)
est = sm.OLS(nq_var, senti_scaled4)
est2 = est.fit()
print(est2.summary())

###### Prediction ######
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X = df_comp['sent_scaled'][:75]
X = X.tolist()
y = nq_daily_returns
y = y.tolist()
y1= np.asarray(nq_var)

y_dummy = []
for i in range(0,len(nq_daily_returns)):
    if(y[i]>0):
        y_dummy.append(1)
    else:
        y_dummy.append(0)
        
X_train,X_test,y_train,y_test=train_test_split(X,y_dummy,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

logreg.fit(X_train.reshape(-1,1),y_train)

#
X_test = np.asarray(X_test)
y_pred=logreg.predict(X_test.reshape(-1,1))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test.reshape(-1,1))[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#Now for variance
y1_dummy = []
for i in range(1,len(nq_var)):
    if(y1[i]>y1[i-1]):
        y1_dummy.append(1)
    else:
        y1_dummy.append(0)        

X1 = X[1:]

X_train,X_test,y_train,y_test=train_test_split(X1,y1_dummy,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

logreg.fit(X_train.reshape(-1,1),y_train)

#
X_test = np.asarray(X_test)
y_pred=logreg.predict(X_test.reshape(-1,1))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test.reshape(-1,1))[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#Repeat to calculate distribution of variables
n = 10000
acc = []
prec = []
rec = []
auc1 = []
for i in range(0,n):
    X_train,X_test,y_train,y_test=train_test_split(X,y_dummy,test_size=0.30)
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    logreg.fit(X_train.reshape(-1,1),y_train)
    
    #
    X_test = np.asarray(X_test)
    y_pred=logreg.predict(X_test.reshape(-1,1))
    
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix
    
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Precision:",metrics.precision_score(y_test, y_pred))
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    
    acc.append(metrics.accuracy_score(y_test, y_pred))
    prec.append(metrics.precision_score(y_test, y_pred))
    rec.append(metrics.recall_score(y_test, y_pred))
    
    y_pred_proba = logreg.predict_proba(X_test.reshape(-1,1))[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    auc1.append(auc)
    #plt.plot(fpr,tpr,label="data, auc="+str(auc))
    #plt.legend(loc=4)
    #plt.show()
    
    
np.mean(acc)    
np.sqrt(np.var(acc))

np.mean(prec)    
np.sqrt(np.var(prec))

np.mean(rec)    
np.sqrt(np.var(rec))

np.mean(auc1)    
np.sqrt(np.var(auc1))


acc = []
prec = []
rec = []
auc1 = []
for i in range(0,n):
    X_train,X_test,y_train,y_test=train_test_split(X1,y1_dummy,test_size=0.30)
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    logreg.fit(X_train.reshape(-1,1),y_train)
    
    #
    X_test = np.asarray(X_test)
    y_pred=logreg.predict(X_test.reshape(-1,1))
    
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix
    
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Precision:",metrics.precision_score(y_test, y_pred))
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    
    acc.append(metrics.accuracy_score(y_test, y_pred))
    prec.append(metrics.precision_score(y_test, y_pred))
    rec.append(metrics.recall_score(y_test, y_pred))
    
    y_pred_proba = logreg.predict_proba(X_test.reshape(-1,1))[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    auc1.append(auc)
    #plt.plot(fpr,tpr,label="data, auc="+str(auc))
    #plt.legend(loc=4)
    #plt.show()
    
    
np.mean(acc)    
np.sqrt(np.var(acc))

np.mean(prec)    
np.sqrt(np.var(prec))

np.mean(rec)    
np.sqrt(np.var(rec))

np.mean(auc1)    
np.sqrt(np.var(auc1))

np.mean(y_dummy)
