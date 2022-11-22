import langid
import pymorphy2
from stop_words import get_stop_words

import nltk
'''
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
'''

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob


class TextPreproc():

    def __init__(self, test_size=0.2, rebalance=True):

        '''
        input data must be a Dataframe with columns:
            - message: str
            - category: int (0/1)
        '''

        self.data = self.load_data()
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=0)

        # print(len(self.train_data[self.train_data.target == 1]) / len(self.train_data))
        # --> 0.13406317300789664 --> classes are unbalansed!
        # Another way to detect that classes are unbalanced: self.train_data.target.value_counts()

        # Need to upsample:
        if rebalance:
            ratio = len(self.train_data.loc[self.train_data['category'] == 0]) // \
                    len(self.train_data.loc[self.train_data['category'] == 1])
            df_1 = self.train_data.loc[self.train_data['category'] == 1]
            df_1 = df_1.loc[df_1.index.repeat(ratio)]
            self.train_data = pd.concat([self.train_data.loc[self.train_data['category'] == 0], df_1]).sample(frac=1)
            self.train_data = self.train_data.reset_index(drop=True)

        # fitting tfidf vectorizer on the train data
        self.vectorizer = TfidfVectorizer()
        train_vect = self.vectorizer.fit_transform(self.preproc_corpus(self.train_data['message']))
        self.vect_len = train_vect.toarray().shape[1]
        self.train_features_df = pd.DataFrame(train_vect.toarray())
        self.train_features_df['target'] = [tmp for tmp in self.train_data['category']]

        # using tfidf vectorizer to vectorize test data
        test_vect = self.vectorizer.transform(self.preproc_corpus(self.test_data['message']))
        self.test_features_df = pd.DataFrame(test_vect.toarray())
        self.test_features_df['target'] = [tmp for tmp in self.test_data['category']]

    def load_data(self):

        '''0 - ham, 1 - spam'''

        df = pd.read_csv(glob("*.csv")[0])
        df.columns = [name.lower() for name in df.columns]

        df.category = pd.Categorical(df.category)
        df["category"] = df.category.cat.codes

        return df

    def get_initial_data(self):
        return self.data

    def get_train_test_preprocd(self):
        return self.train_features_df, \
               self.test_features_df

    def get_vector_len(self):
        return self.vect_len
    def get_vectorizer(self):
        return self.vectorizer

    def preproc_corpus(self, text_corpus):

        '''
            text_corpus - list of messages. Each message have str type
            out - list of preprocessed messages
        '''

        tokenizer = RegexpTokenizer(r'\w+')
        morph = pymorphy2.MorphAnalyzer()
        sentTokenizer = PunktSentenceTokenizer()
        lemmatizer = WordNetLemmatizer()

        langid.set_languages(['en', 'ru'])
        stopWordsEn = set().union(get_stop_words('en'), stopwords.words('english'))
        stopWordsRu = set().union(get_stop_words('ru'), stopwords.words('russian'))
        stopWords = list(set().union(stopWordsEn, stopWordsRu))
        stopWords.sort()

        out = []
        for i, text in enumerate(text_corpus):

            sentList = [sent for sent in sentTokenizer.tokenize(text)]
            tokens = [word for sent in sentList for word in tokenizer.tokenize(sent.lower())]
            lemmedTokens = []

            for token in tokens:

                if langid.classify(token)[0] == 'en':
                    lemmedTokens.append(lemmatizer.lemmatize(token))

                elif langid.classify(token)[0] == 'ru':
                    lemmedTokens.append(morph.parse(token)[0].normal_form)

            out.append(" ".join([token for token in lemmedTokens if not token in stopWords]))

        return out

    def preproc_letter(self, letter):

        letter_vect = self.vectorizer.transform(self.preproc_corpus([letter]))
        return letter_vect.toarray()


if __name__ == "__main__":

    model = TextPreproc(rebalance=True)
    '''train_data, test_data = model.get_train_test_preprocd()
    print(test_data)'''

    letter_arr = [
        "Hi, how are you feeling? You haven't written for a long time, so I thought something might have happened.",
        'Only today! buy one king-size pizza, get one cola for free! Hurry up!',
        'love you sweetie! ;)',
        "hey, do you want to get rich? do you want to afford everything you've been dreaming about for a long time? "
        "Buy my book and I'll tell you how to become rich!",
        'bae i cannot wait anymore. I want you now!',
        'You’ve won!',
        'The IRS is trying to contact you',
        'You have a refund coming',
        'Verify your bank account',
        'You have a package delivery',
        'Verify your Apple iCloud ID',
        'Bitcoin, anyone?',
        'A family member needs help',
        'Reactivate your account',
        'You have a new billing statement']

    for letter in letter_arr:
        print(model.preproc_letter(letter))
