from sklearn.preprocessing import StandardScaler

class BatchStandardScaler(TransformerMixin):
    
    def __init__(self, columns=None, batch_size=100000):
        self.batch_size = batch_size
        self.columns = columns
        self.scaler = StandardScaler()
        
    def drawProgressBar(self, percent, barLen = 20):
        """
        Displays a progressbar on the output.
        """
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()
        
        
    def fit(self, X, *_):
        """
        Fits the entire X on the Standard Scaler, by applying batching.
        X should be a Pandas Dataframe object that contains the entire data to scale. Batching is applied internally.
        """
        
        total_rows = X.shape[0]
        index = 0

        print('\n Partially fitting standard scaler...')
        # FIT 
        while index < total_rows:

            # Get current partial size
            partial_size = min(self.batch_size, total_rows - index)  # needed because last loop is possibly incomplete

            # Get the partial block to fit
            partial_x = X[self.columns].iloc[index:index+partial_size]

            # Fit partial
            self.scaler.partial_fit(partial_x)

            # Add the current partial block size to the processed index
            index += partial_size
            
            # Draw progressbar
            self.drawProgressBar(index / total_rows, 50)
            
        return self
            
    def transform(self, X, *_):
        total_rows = X.shape[0]
        index = 0

        print('\n Partially transforming...')
        # TRANSFORM    
        while index < total_rows:

            # Get the current partial size
            partial_size = min(self.batch_size, total_rows - index)  # needed because last loop is possibly incomplete
            
            # Get the partial block of data
            partial_x = X[self.columns].iloc[index:index+partial_size]

            # Apply transforming
            X[self.columns].iloc[index:index+partial_size] = self.scaler.transform(partial_x)
    
            # Increment the current index
            index += partial_size
            
            # Draw progressbar
            self.drawProgressBar(index / total_rows, 50)
            
        return 