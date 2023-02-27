import re

import nltk
import nltk.corpus
import spacy
from loguru import logger


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
            "de_core_news_md", exclude=["tagger", "morphologizer", "parser", "senter", "ner"]
        )

        self.clean_chars = re.compile(r"[^A-Za-züöäÖÜÄß ]", re.MULTILINE)
        self.clean_http_urls = re.compile(r"http[s]?:\/\/\S+|[^\w]\s+", re.MULTILINE)
        self.clean_at_mentions = re.compile(r"@\S+", re.MULTILINE)

        logger.debug("Initialized CleanText class.")

    def _replace_numbers(self, text: str) -> str:
        """Replaces numbers with their german equivalent

        Parameters
        ----------
        text : str
            The text which should be cleaned

        Returns
        -------
        str
            The cleaned text
        """
        return (
            text.replace("0", " null")
            .replace("1", " eins ")
            .replace("2", " zwei ")
            .replace("3", " drei ")
            .replace("4", " vier ")
            .replace("5", " fünf ")
            .replace("6", " sechs ")
            .replace("7", " sieben ")
            .replace("8", " acht ")
            .replace("9", " neun ")
        )

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
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub("", text)
        text = self.clean_at_mentions.sub("", text)
        text = self._replace_numbers(text)
        text = self.clean_chars.sub("", text)
        text = " ".join(text.split())
        text = text.strip().lower()

        logger.debug("Text cleaned.")
        return text

    def stemm_text(self, text: str) -> list[str]:
        """Takes a text and stems each word

        Parameters
        ----------
        text : str
            The text which should be stemmed (cleaned)

        Returns
        -------
        list[str]
            The stemmed tokens
        """
        spacy_doc = self.spacy_nlp_ger(text)
        lemma_tokens = [token.lemma_.lower() for token in spacy_doc]

        logger.debug("Text stemmed.")
        return lemma_tokens

    def filter_text(self, text: list[str]) -> list[str]:
        """Using the list of tokens all stopwords are being removed and the remaining tokens are joined together

        Parameters
        ----------
        text (list[str]): The tokens of the text

        Returns
        -------
        list[str]
            A list of the tokens without stopwords
        """
        filtered_text = [token for token in text if token not in self.stopwords_ger]

        logger.debug("Stopwords removed.")
        return filtered_text

    def pipeline(self, text: str) -> tuple[str, list[str], list[str]]:
        """A pipeline in order to combine all methods in order to clean a text

        Parameters
        ----------
        text : str
            The text which should be cleaned and tokenized

        Returns
        -------
        tuple[str, list[str], list[str]]
            A tuple containing the cleaned text, the stemmed tokens and the tokens without stopwords
        """
        text_clean = self.clean_text(text)
        stemmed_text = self.stemm_text(text_clean)
        removed_stopwords = self.filter_text(stemmed_text)

        logger.debug("Text cleaned!")
        return (text_clean, stemmed_text, removed_stopwords)
