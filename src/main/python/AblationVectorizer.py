import numpy as np


class AblationVectorizer:
    def __init__(self, vectorizers):
        self.vectorizers = vectorizers

    def fit_transform(self, train_documents):
        vectors = []
        for vectorizer in self.vectorizers:
            if len(vectors) == 0:
                vectors = vectorizer.fit_transform(train_documents)
                if type(vectors).__module__ != np.__name__:
                    vectors = vectors.toarray()
            else:
                v = vectorizer.fit_transform(train_documents)
                if type(v).__module__ != np.__name__:
                    v = v.toarray()
                vectors = np.concatenate((vectors, v), axis=1)

        return vectors

    def transform(self, test_documents):
        vectors = []
        for vectorizer in self.vectorizers:
            if len(vectors) == 0:
                vectors = vectorizer.transform(test_documents)
                if type(vectors).__module__ != np.__name__:
                    vectors = vectors.toarray()
            else:
                v = vectorizer.transform(test_documents)
                if type(v).__module__ != np.__name__:
                    v = v.toarray()
                vectors = np.concatenate((vectors, v), axis=1)

        return vectors
