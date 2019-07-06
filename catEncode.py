def getObjectFeature(df, col, datalength=1460):
    '''# 0 = most common category, highest int = least common.'''
    if df[col].dtype != "object":  # if it's not categorical..
        print("feature", col, "is not an object feature.")
        return df
    else:
        df1 = df
        counts = df1[
            col
        ].value_counts()  # get the counts for each label for the feature
        #         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels
        df1[col] = [
            counts.index.tolist().index(i) if i in counts.index.tolist() else 0
            for i in df1[col]
        ]  # do the conversion
        return df1  # make the new (integer) column from the conversion