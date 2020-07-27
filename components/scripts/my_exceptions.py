class InvalidDataFrameFormat(Exception):
    def __init__(self, dataframe, message="DataFrame not in correct format"):
        self.dataframe = dataframe
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dataframe} -> {self.message}'