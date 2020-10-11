import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

def plot_wordcloud(wordcloud,title):
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.savefig(title)

def plotTop10Words(tokens,title):
    words = []
    for i in tokens.keys():
        words.append((tokens[i],i))
        
    words.sort(reverse = True)
    X = []
    Y = []
    for i in words:
        X.append(i[1])
        if len(X) ==10:
            break
        
    for i in words:
        Y.append(i[0])
        if len(Y) == 10:
            break
        
    label_X = 'words'
    label_Y = 'frequency'
    draw(X,Y,label_X,label_Y,title)


def draw(X,Y,label_X,label_Y,title):            #Plotting the graph

    plt.bar(X, Y, tick_label = X, width = 0.5, color = ['orange', 'black']) 
    plt.xlabel(label_X) 
    plt.ylabel(label_Y) 
    plt.title(title) 
    plt.show()
    plt.savefig(title)


def plotRelationShip(tokens,title):             #Relationship between word length and frequency
    word_lengths = {}
    for i in tokens.keys():
        if len(i) not in word_lengths.keys():
            word_lengths[len(i)] = tokens[i]
        else:
            word_lengths[len(i)] += tokens[i]

    X = []
    Y = []

    for i in word_lengths.keys():
        X.append(i)

    X.sort()

    for i in X:
        Y.append(word_lengths[i])

    label_X = 'word length'
    label_Y = 'frequency'
    draw(X,Y,label_X,label_Y,title)

contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he shall",
        "he'll've": "he shall have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has",
        "i'd": "I had",
        "i'd've": "I would have",
        "i'll": "I shall",
        "i'll've": "I shall have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it shall",
        "it'll've": "it shall have",
        "it's": "it has",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she shall",
        "she'll've": "she shall have",
        "she's": "she has",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that has",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there has",
        "they'd": "they had",
        "they'd've": "they would have",
        "they'll": "they shall",
        "they'll've": "they shall have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall",
        "what'll've": "what shall have",
        "what're": "what are",
        "what's": "what has",
        "what've": "what have",
        "when's": "when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has",
        "where've": "where have",
        "who'll": "who shall",
        "who'll've": "who will have",
        "who's": "who has",
        "who've": "who have",
        "why's": "why has",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
}

#--------------------------------------For removing contractions from the book---------------------------------------------------------------------------------

fnew = open("book.txt", 'w')
with open('book_original.txt', errors= 'ignore') as fin:
    for w in fin:
        w = w.split(" ")
        for word in w:
            if word in contractions.keys():
                word = contractions[word]
                
            print( ''.join(word), end=' ', file=fnew)
            
#--------------------------------------Pre-precessing, Tokenozation, Stemming, Lemmatization----------------------------------------------------------------------------------
         
stop_words = set(stopwords.words('english'))            #getting the set of all stopwords in english language

stemmer = PorterStemmer()                               #creating a Porter Stemmer for stemming
lemmatizer = WordNetLemmatizer()                        #creating a WordNet Lemmatizer for Lemmatization

                                                        #file handles for various files
ftoken = open('token.txt', 'w')                         #Tokens with stopwords -> token.txt
ftoken_without_sw = open('token_without_sw.txt', 'w')   #Tokens without stopwords -> token_without_sw.txt
fstem = open('stemmed.txt','w')                         #Stemmend words ->  stemmed.txt
flemmatize = open('lemmatized.txt','w')                 #Lemmatized words -> lemmatized.txt



with open('book.txt', errors= 'ignore') as fin:
    for line in fin:                                    #reading a line from book
        line = line.lower()                             #converting words into lower case
        tokens = word_tokenize(line)                    #Tokenizing words
        for w in tokens:
            w = re.sub(r'https?:\/\/.[\r\n]', '', w)    #removing links (if any)
            w = re.sub(r'\S*@\S*\s?','',w)              #removing mail ids (if any)
            w = re.sub(r'[^a-z]+', '', w)               #removing all words with characters other than alphabets 
            w = re.sub(r' ', '', w)                     #removing white spaces
            if w == "chapter":                          #removing word "chapter" (due to multiple useless occurance)
                w = ''
            
            print( ''.join(w), end=' ', file=ftoken)    #writing processed tokenized words into "token.txt" file
            
            if w not in stop_words:                                     #passing non-stem words
                s = stemmer.stem(w)                                     #stemming of tokenized word
                l = lemmatizer.lemmatize(s)                             #lemmatization of stemmed words
                print( ''.join(s), end=' ', file=fstem)                 #writing stemmed words into "stemmed.txt" file 
                print( ''.join(l), end=' ', file=flemmatize)            #writing lemmatized words into "lemmatized.txt" file
                print( ''.join(w), end=' ', file=ftoken_without_sw)     #writing non-stem tokenized words into "token_without_sw.txt" file
                

#--------------------------------------Frequency Distribution of Tokens (with stopwords)---------------------------------------------------------------------------------

fdist_token=FreqDist()
ftoken = open('token.txt', 'r')
for word in ftoken:
    words = word.split(" ")
    for w in words:
        if w == '':
            continue
        fdist_token[w]+=1

#Plotting of the Frequency distribution
plotRelationShip(fdist_token,"Relationship between word length and frequency (with stopwords)")

#--------------------------------------Frequency Distribution of Tokens (without stopwords)------------------------------------------------------------------------------

fdist_token_without_sw=FreqDist()
ftoken_without_sw = open('token_without_sw.txt', 'r')

for word in ftoken_without_sw:
    words = word.split(" ")
    for w in words:
        if w == '':
            continue
        fdist_token_without_sw[w]+=1
        
#Plotting of the Frequency distribution
plotRelationShip(fdist_token_without_sw, "Relationship between word length and frequency (without stopwords)")
      
#---------------------------------------PoSTagging and Distribution of Tags----------------------------------------------------------------------------------------------


lemmatized = list()
flem = open('lemmatized.txt', 'r')

for word in flem:
    word = word.split()
    for w in word:
        if w == "":
            continue
        lemmatized.append(w)

brown_tags = brown.tagged_sents(categories=['fiction','romance','adventure','mystery','humor','science_fiction'])

size = int(len(brown_tags) * 0.9)
train = brown_tags[:size]
test = brown_tags[size:]

train[:1000]
test[:1000]

tag0 = nltk.DefaultTagger('NN')
tag1 = nltk.UnigramTagger(train, backoff=tag0)
tag2 = nltk.BigramTagger(train, backoff=tag1)
tag2.evaluate(test)

Tagged_Book1 = tag2.tag(lemmatized)

X = []
Y = []
for i in Tagged_Book1:
    X.append(i[0])
    Y.append(i[1])

Tags = []
for word in lemmatized:
    for idx, i in enumerate(X):
        if word == i:
            Tags.append(Y[idx])
            break
            
fdist_tags=FreqDist()
for word in Tags:
    fdist_tags[word]+=1

Tag_Freq = [(v,k) for k,v in fdist_tags.items()]
Tag_Freq.sort(reverse=True)

tag = [x[0] for x in Tag_Freq]
freq = [x[1] for x in Tag_Freq]

#draw(freq,tag,"Tags","Frequency","Relationship between Tags and Frequency")


#---------------------------------------Creation of Word Cloud (with stopwords)-----------------------------------------------------------------------------------------------------------
ftoken = open('token.txt','r')
for line in ftoken:
    text = line
    
wordcloud=WordCloud(stopwords={}).generate(text)                #Words excluding stopwords
plot_wordcloud(wordcloud,'Cloud.png')    

#---------------------------------------Creation of Word Cloud (without stopwords)-----------------------------------------------------------------------------------------------------------

ftoken = open('token.txt','r')
for line in ftoken:
    text = line


wordcloud=WordCloud().generate(text)                #Words excluding stopwords
plot_wordcloud(wordcloud,'Cloud_without_sw.png')

#---------------------------------------Analysis of Top 10 words (with stopwords)-----------------------------------------------------------------------------------------------------------

plotTop10Words(fdist_token,'Frequency of Top 10 words (with stopwords)')

#---------------------------------------Analysis of Top 10 words (without stopwords)-----------------------------------------------------------------------------------------------------------

plotTop10Words(fdist_token_without_sw, 'Frequency of Top 10 words (without stopwords)')