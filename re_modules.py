import re
import my_modules as mm


def contains_phone(text):
    phonePattern=re.compile(r'''
                # don't match beginning of string,number can start anywhere
    (\d{3})     # area code is 3 digits (e.g. '800')
    \D*         # optional separator is any number of non-digits
    (\d{3})     # trunk is 3 digits (e.g. '555')
    \D*         # optional separator
    (\d{4})     # rest of number is 4 digits (e.g. '1212')
    \D*         # optional separator
    (\d*)       # extension is optional and can be any number of digits
    $           # end of string
    ''',re.VERBOSE)
    # return len(phonePattern.findall(text))
    if len(phonePattern.findall(text)) > 0:
        return "phonenumber"
    else :
        return text


emoticons_str=r'''
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )'''
regex_str=[
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)" ,# hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f]['
    r'0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)'  # anything else
]
tokens_re  =re.compile(r'('+'|'.join(regex_str)+')',re.VERBOSE | re.IGNORECASE)
emoticon_re=re.compile(r'^'+emoticons_str+'$',re.VERBOSE | re.IGNORECASE)


def preprocess(s,lowercase=False):
    # print("Method: preprocess(s,lowercase=False)")
    tokens=tokens_re.findall(str(s))
    if lowercase:
        tokens=[token if emoticon_re.search(token) else token.lower() for token in tokens]
    tokens = mm.remove_stopwords(tokens)
    return tokens


def main():
    pass


if __name__ == "__main__": main()
