def return_post_codes(df):
    """
    Write a function that takes a pandas DataFrame with one column, text, that
    contains an arbitrary text. The function should extract all post-codes that
    appear in that text and concatenate them together with " | ". The result is
    a new dataframe with a column "postcodes" that contains all concatenated
    postcodes.

    Example input:
                                                                            text
    0  Great Doddington, Wellingborough NN29 7TA, UK\nTaylor, Leeds LS14 6JA, UK
    1  This is some text, and here is a postcode CB4 9NE

    Expected output:

                postcodes
    0  NN29 7TA | LS14 6JA
    1              CB4 9NE

    Note: Postcodes, in the UK, are of one of the following form where `X` means
    a letter appears and `9` means a number appears:

    X9 9XX
    X9X 9XX
    X99 9XX
    XX9 9XX
    XX9X 9XX
    XX99 9XX

    Even though the standard layout is to include one single space
    in between the two halves of the post code, there are occasional formating
    errors where an arbitrary number of space is included (0, 1, or more). You
    should parse those codes as well.

    :param df: a DataFrame with the text column
    :return: new DataFrame with the postcodes column
    """

    raise NotImplementedError

#custom                                                                                                                                               
print re.findall(r'\b[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2}\b', s)

#regex from #http://en.wikipedia.orgwikiUK_postcodes#Validation                                                                                            
print re.findall(r'[A-Z]{1,2}[0-9R][0-9A-Z]? [0-9][A-Z]{2}', s)