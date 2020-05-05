def new_pc(df):
    df['postcodes'] = df['text'].str.findall(r'\b(?:[A-Za-z][A-HJ-Ya-hj-y]?[0-9][0-9A-Za-z]? ?[0-9][A-Za-z]{2}|[Gg][Ii][Rr] ?0[Aa]{2})\b').apply(' | '.join)
    return df.drop("text", axis=1)

