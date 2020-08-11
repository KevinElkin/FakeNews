from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd
import re
from collections import defaultdict
import json

import newspaper
from newspaper import Article
from newspaper import Source
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from tqdm import tqdm

def scrapeMain(num):

    link_data = []
    authDate_list = []
    type_list = []
    articleName_list = []
    date_list = []
    auth_list = []
    truth_list = []
            
    # add one to num since we want the number of pages passed in 
    for pageNum in tqdm(range(num+1)):

        # No page zero 
        if pageNum == 0:
            continue
        try:    
            url = 'https://www.politifact.com/factchecks/list/?page='+str(pageNum)    
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")

            
            #get information for each box link on main page 
            articleBlocks = soup.find_all('li',attrs={'class':'o-listicle__item'})

            #iterate through each quote block on a page 
            for block in articleBlocks:
                link_data.append("https://www.politifact.com" + str(block.find('div', attrs={'class': 'm-statement__quote'}).find('a').attrs['href']))
                authDate_list.append(block.find('div', attrs={'class': 'm-statement__body'}).find('footer', attrs={'class': 'm-statement__footer'}).text.strip())
                type_list.append(block.find('div', attrs={'class': 'm-statement__meta'}).find('a').attrs['title'])
                articleName_list.append(block.find('div', attrs={'class': 'm-statement__quote'}).find('a').text.strip())
                truth_list.append(block.find_all('img', attrs={'class': 'c-image__original'})[1].attrs['alt'])


        except Exception as e:
            
            #extract error information if exception is thrown while scraping
            err_type, err_obj, err_info = sys.exc_info()
            print('ERROR FOR LINK:', url)
            print(err_type)

    #seperate date and author into 2 seperate lists
    for elem in authDate_list:
        date_auth = elem.split('•')
        auth_list.append(date_auth[0][3:])
        date_list.append(date_auth[1][1:])
            
    #create dataframe and output to csv
    data = {'Quote': articleName_list, 'Source': type_list, 'Date': date_list, 'Post Author': auth_list, 'Link': link_data, 'Label': truth_list}    
    df = pd.DataFrame.from_dict(data)
    df.to_csv('politifact_data.csv', index=False)
    
    
def format_data(csv):
    
    df = pd.read_csv(csv)
    df['Year'] = pd.to_datetime(df['Date'])
    df['Year'] = pd.DatetimeIndex(df['Year']).year
    df = df.dropna()

    # Drop 'flip' values from 'Label column
    flip = ['full-flop', 'half-flip', 'no-flip']
    df = df[~df['Label'].isin(flip)]

    df['Label'] = df['Label'].replace(['mostly-true','pants-fire','barely-true','half-true'],
                                                  ['true','false','false','true'])
    
    # Stopwords - remove common words that do not contribute to meaning of text (i.e. 'the', 'a')
    stop_words = stopwords.words('english')

    # Lemmatization - process of grouping together the different forms of same root word (i.e. 'swimming' --> 'swim')
    lemmatizer = WordNetLemmatizer()

    df['Quote'] = df['Quote'].apply(lambda x : x.lower()).apply(lambda x : re.sub(r'[^\w\s]', '', x))
    
    for index, row in df.iterrows():

        filter_sentence = ''
        sentence = row['Quote']

        # Cleaning the sentence with regex
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Tokenization
        words = nltk.word_tokenize(sentence)   

        # Stopwords removal
        words = [w for w in words if not w in stop_words]

        # Lemmatization
        for word in words:
            filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(word))

        df.loc[index, 'Quote'] = filter_sentence

    return df

#call method on the first 590 pages 
#scrapeMain(590)
    
