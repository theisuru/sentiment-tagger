import math
from collections import defaultdict

import numpy as np
from gensim.models import word2vec


class W2VVectorizer:
    """vectorize the words using predefined W2V model"""

    def __init__(self, w2v_model_name, tfidf):
        self.model = word2vec.Word2Vec.load(w2v_model_name)
        self.idf_dict = None
        self.max_idf_score = 1
        self.tfidf = tfidf

    def fit_transform(self, train_documents):
        if self.tfidf:
            self.idf_dict, self.max_idf_score = self.calculate_idf(train_documents)
        clean_train_comments = []
        for comment in train_documents:
            clean_train_comments.append(comment.split())
        return self.get_avg_feature_vectors(clean_train_comments)

    def transform(self, test_documents):
        clean_test_comments = []
        for comment in test_documents:
            clean_test_comments.append(comment.split())
        return self.get_avg_feature_vectors(clean_test_comments)

    def calculate_idf(self, train_comments):
        df_dict = defaultdict(int)
        for comment in train_comments:
            words = comment.split()
            for word in set(words):
                df_dict[word] += 1

        idf_dict = dict()
        for word in df_dict:
            idf_dict[word] = math.log(train_comments.size / float(df_dict[word]))
        max_idf_score = idf_dict[max(idf_dict, key=lambda key: idf_dict[key])]
        return idf_dict, max_idf_score

    # get a list of feature vectors for all comments
    def get_avg_feature_vectors(self, reviews):
        counter = 0
        review_feature_vecs = np.zeros((len(reviews), self.model.vector_size), dtype="float32")
        for review in reviews:
            review_feature_vecs[int(counter)] = self.comment_to_feature_vector(review)
            counter = counter + 1.
        return review_feature_vecs

    # make a feature vector from a single comment
    def comment_to_feature_vector(self, words):
        feature_vec = np.zeros((self.model.vector_size,), dtype="float32")
        nwords = 0.
        index2word_set = set(self.model.wv.index2word)
        for word in words:
            if word in index2word_set:
                if self.tfidf:
                    if word in self.idf_dict:
                        nwords = nwords + self.idf_dict[word]
                        feature_vec = np.add(feature_vec, self.model[word] * self.idf_dict[word])
                    else:
                        nwords = nwords + self.max_idf_score
                        feature_vec = np.add(feature_vec, self.model[word] * self.max_idf_score)
                else:
                    nwords = nwords + 1.
                    feature_vec = np.add(feature_vec, self.model[word])
            # else:
            #     print("Word is not in W2V model: " + word)

        # we have some one word comments that is not included in original model, todo expand original model or ignore them
        if nwords != 0:
            feature_vec = np.divide(feature_vec, nwords)

        return feature_vec
