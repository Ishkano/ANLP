import langid
import pymorphy2
from stop_words import get_stop_words

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
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

        # print(len(self.data[self.data.category == 1]) / len(self.data))
        # --> 0.13406317300789664 --> classes are unbalansed!
        # Another way to detect that classes are unbalanced: self.data.category.value_counts()

        # Need to upsample:
        if rebalance:
            ratio = len(self.data.loc[self.data['category'] == 0]) // \
                    len(self.data.loc[self.data['category'] == 1])
            df_1 = self.data.loc[self.data['category'] == 1]
            df_1 = df_1.loc[df_1.index.repeat(ratio)]
            self.data = pd.concat([self.data.loc[self.data['category'] == 0], df_1]).sample(frac=1)
            self.data = self.data.reset_index(drop=False)

        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=42)

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
    #train_data, test_data = model.get_train_test_preprocd()
    #print(test_data)

    letter = ['Hello, how r u?']
    print(model.preproc_letter(letter))
