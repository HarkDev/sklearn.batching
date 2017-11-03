class DummiesTransformer(TransformerMixin):
    def __init__(self, columns, drop_first=False):
        self.columns = columns
        self.drop_first = drop_first
        return
    
    def transform(self, X, *_):
        X = pd.get_dummies(X, columns=self.columns, drop_first=self.drop_first)
        
        return X
    
    def fit(self, *_):
        return self
