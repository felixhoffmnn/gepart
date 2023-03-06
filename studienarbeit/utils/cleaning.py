import re

import nltk
import nltk.corpus
import spacy
from loguru import logger
from num2words import num2words
from thinc.api import prefer_gpu


class Cleaning:
    """This class provides various methods to clean texts and tweets

    Source
    ------
    https://www.solvistas.com/blog/python-nlp-pipeline-fuer-die-extraktion-von-themen-aus-nachrichten/
    """

    def __init__(self) -> None:
        """In order to clean text some models and lists are needed which gets loaded within the constructor"""
        nltk.download("stopwords")
        nltk.download("punkt")
        self.stopwords_ger = nltk.corpus.stopwords.words("german")
        self.spacy_nlp_ger = spacy.load(
            "de_core_news_lg", exclude=["tagger", "morphologizer", "parser", "senter", "ner"]
        )

        logger.info("Using GPU for cleaning." if prefer_gpu() else "Using CPU for cleaning.")

        logger.debug("Initialized CleanText class.")

    def clean_text(self, text: str) -> str:
        """Receives a text and removes all special characters, icons and usernames and returns the cleaned text

        Parameters
        ----------
        text : str
            The text which should be cleaned
        isTweet : bool
            A boolean which indicates if the text is a tweet or not

        Returns
        -------
        str
            The cleaned text
        """
        text = text.replace("\n", " ")  # remove newlines

        text = re.sub(r"&and;|&", " und ", text)  # replace &and; with "und"
        text = re.sub(r"&#37;|%", " prozent ", text)  # replace &#37; with "prozent"
        text = re.sub(r"&euro;|€", " euro ", text)  # replace &euro; with "euro"

        text = re.sub(r"<U\+[0-9A-Z]{4,8}>", " ", text)  # remove emojis
        text = re.sub(r"&[0-9A-Z]{4,8};", " ", text)  # remove emojis
        text = re.sub(r"https?:\/\/\S+|www.\S+", " ", text)  # remove http urls
        text = re.sub(r"@\S+", " ", text)  # remove @mentions

        text = re.sub(r"(\d+).(\d+)", r"\1\2", text)  # remove dots and commas from numbers
        text = re.sub(
            r"\s\d+\s", lambda x: " " + num2words(int(x.group(0)), lang="de") + " ", text
        )  # replace numbers with words
        text = re.sub(r"[,.:;?]+", " ", text)  # remove punctuation
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", " ", text)  # remove special characters
        text = re.sub(r"\s[a-zA-ZäöüÄÖÜß]\s", " ", text)  # remove single characters
        text = " ".join(text.split()).lower()  # remove multiple spaces

        logger.debug("Text cleaned.")
        return text

    def lemma_text(self, text: str) -> str:
        """Takes a text and lemmatizes it

        Parameters
        ----------
        text : str
            The text which should be lemmatized

        Returns
        -------
        str
            The lemmatized text
        """
        spacy_doc = self.spacy_nlp_ger(text)
        lemma_tokens = " ".join([token.lemma_.lower() for token in spacy_doc if token.lemma_ != "--"])

        logger.debug("Text stemmed.")
        return lemma_tokens

    def filter_text(self, text: str) -> str:
        """Using the list of tokens all stopwords are being removed and the remaining tokens are joined together

        Parameters
        ----------
        text : str
            The tokens of the text

        Returns
        -------
        str
            A list of the tokens without stopwords
        """
        filtered_text = " ".join([token for token in text.split() if token not in self.stopwords_ger])

        logger.debug("Stopwords removed.")
        return filtered_text

    def pipeline(self, text: str) -> tuple[str, str, str]:
        """A pipeline in order to combine all methods in order to clean a text

        Parameters
        ----------
        text : str
            The text which should be cleaned and tokenized

        Returns
        -------
        tuple[str, str, str]
            A tuple containing the cleaned text, the stemmed tokens and the tokens without stopwords
        """
        text_clean = self.clean_text(text)
        lemmatized_text = self.lemma_text(text_clean)
        removed_stopwords = self.filter_text(lemmatized_text)

        logger.debug("Text cleaned!")
        return (text_clean, lemmatized_text, removed_stopwords)
