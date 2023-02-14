import re

import nltk
import spacy
from loguru import logger

class CleanText:
    """This class provides various methods to clean texts and tweets

    Source
    ------
    https://www.solvistas.com/blog/python-nlp-pipeline-fuer-die-extraktion-von-themen-aus-nachrichten/
    """

    def __init__(self) -> None:
        """In order to clean text some models and lists are needed which gets loaded within the constructor"""
        nltk.download("stopwords")
        nltk.download("punkt")
        self.stopwords_ger = set(nltk.corpus.stopwords.words("german"))
        self.spacy_nlp_ger = spacy.load("de_core_news_md", exclude=["tagger", "morphologizer", "parser", "senter", "ner"]) # Try senter

        logger.debug("Initialized CleanText class.")

    def clean_text(self, text: str, isTweet: bool) -> str:
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
        temp = text.lower()
        if isTweet:
            temp = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|'", "", temp)
            temp = re.sub(r"http[s]?:\/\/\S+|[^\w]\s+", " ", temp)
        temp = temp.strip()

        logger.debug("Text cleaned.")
        return temp

    def stemm_text(self, clean_text: str) -> list[str]:
        """Takes a text and stems each word

        Parameters
        ----------
        clean_text : str
            The text which should be stemmed (cleaned)

        Returns
        -------
        list[str]
            The stemmed tokens
        """
        spacy_doc = self.spacy_nlp_ger(clean_text)
        stemmed_tokens = [token.lemma_ for token in spacy_doc]

        logger.debug("Text stemmed.")
        return stemmed_tokens

    def remove_stopwords(self, tokenize_text: list[str]) -> list[str]:
        """Using the list of tokens all stopwords are being removed and the remaining tokens are joined together

        Parameters
        ----------
        tokenize_text (list[str]): The tokens of the text

        Returns
        -------
        list[str]
            A list of the tokens without stopwords
        """
        filtered_tokens = [token for token in tokenize_text if token not in self.stopwords_ger]

        logger.debug("Stopwords removed.")
        return filtered_tokens

    def pipeline(self, text: str, isTweet: bool) -> tuple[str, list[str]]:
        """A pipeline in order to combine all methods in order to clean a text

        Parameters
        ----------
        text : str
            The text which should be cleaned and tokenized
        isTweet : bool
            A boolean which indicates if the text is a tweet or not

        Returns
        -------
        list[str]
            A List with all tokens of the text except the stopwords
        """
        text_clean = self.clean_text(text, isTweet)
        stemmed_text = self.stemm_text(text_clean)
        removed_stopwords = self.remove_stopwords(stemmed_text)

        logger.debug("Text cleaned!")
        return (text_clean, removed_stopwords)
