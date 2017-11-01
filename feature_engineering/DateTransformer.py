class DateTransformer(TransformerMixin):
	"""
	Converts a datetime type column, into 3 separate columns that represent:
	* year
	* month
	* day
	* day_of_week
	
	Parameters
	**********
	
	* date_column: Name of the date column.
	* include_day_of_week: Defaults True. Set to False to avoid creating this column.
	* drop_date_column: Default True. Either to drop the source column or not.
	"""

    def __init__(self, date_column, include_day_of_week=True, drop_date_column=True):
        self.date_column = date_column
        self.drop_date_column = drop_date_column
        self.include_day_of_week = include_day_of_week
        
    def transform(self, X, *_):
        # Get each part of the date onto a separate column
        X['year'] = X[self.date_column].dt.month.astype(np.uint16)
        X['month'] = X[self.date_column].dt.month.astype(np.int8)
        X['day'] = X[self.date_column].dt.day.astype(np.int8)
        
        if self.include_day_of_week:
            # Get the day of week
            X['day_of_week'] = X[self.date_column].dt.dayofweek.astype(np.int8)
        
        # Drop the date column if requested
        if self.drop_date_column:
            X.drop([self.date_column], axis=1, inplace=True)
        
        return X
    
    def fit(self, *_):
        return self