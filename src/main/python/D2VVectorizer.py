from gensim.models.doc2vec import Doc2Vec


class D2VVectorizer:
    """vectorize the documents using predefined D2V model"""

    def __init__(self, d2v_model_name):
        self.model = Doc2Vec.load(d2v_model_name)

    def fit_transform(self, train_documents):
        train_vector = []
        for comment in train_documents:
            comment_as_word_list = self.get_word_list(comment)
            train_vector.append(self.model.infer_vector(comment_as_word_list))
        return train_vector

    def transform(self, test_documents):
        test_vector = []
        for comment in test_documents:
            comment_as_word_list = self.get_word_list(comment)
            test_vector.append(self.model.infer_vector(comment_as_word_list))
        return test_vector

    # split a comment into sentences of words
    def get_word_list(self, comment):
        word_list = []
        sentences = str(comment).split(".")
        for sentence in sentences:
            if len(sentence) > 2:
                word_list.extend(sentence.split())
        return word_list
