# src/preprocess.py
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# downloads (first-run safe)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')
HTML_PATTERN = re.compile(r'<.*?>')
EMOJI_PATTERN = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def normalize_text(text, lower=True, remove_urls=True, remove_html=True, remove_emoji=True,
                   remove_punct=True, keep_apostrophe=False, remove_digits=False):
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if remove_urls:
        text = URL_PATTERN.sub(' ', text)
    if remove_html:
        text = HTML_PATTERN.sub(' ', text)
    if remove_emoji:
        text = EMOJI_PATTERN.sub(' ', text)
    if remove_punct:
        if keep_apostrophe:
            punct = ''.join(ch for ch in string.punctuation if ch != "'")
            trans = str.maketrans(punct, ' '*len(punct))
            text = text.translate(trans)
        else:
            text = re.sub(r'[^\w\s]', ' ', text)
    if remove_digits:
        text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text, remove_stopwords=True):
    text = normalize_text(text)
    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

# convenience for vectorizers that expect a function returning list or string
def identity_tokenizer(text):
    # if text is already tokenized earlier
    return text
