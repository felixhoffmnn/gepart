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
        self.clean_chars_with_punctuation = re.compile(r"[^A-Za-züöäÖÜÄß.,!? ]", re.MULTILINE)
        self.clean_http_urls = re.compile(r"http[s]?:\/\/\S+|www.\s+", re.MULTILINE)
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

    def clean_text(self, text: str, keep_punctuation: bool = False, keep_upper: bool = False) -> str:
        """Receives a text and removes all special characters, icons and usernames and returns the cleaned text

        Parameters
        ----------
        text : str
            The text which should be cleaned
        keep_punctuation : bool
            A boolean which indicates if punctuation should be kept or not
        keep_upper : bool
            A boolean which indicates if upper case letters should be kept or be lowered

        Returns
        -------
        str
            The cleaned text
        """
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub("", text)
        text = self.clean_at_mentions.sub("", text)
        text = self._replace_numbers(text)
        if keep_punctuation:
            text = self.clean_chars_with_punctuation.sub("", text)
        else:
            text = self.clean_chars.sub("", text)
        text = " ".join(text.split())
        text = text.strip()
        if not keep_upper:
            text = text.lower()

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
        lemma_tokens = [token.lemma_.lower() for token in spacy_doc if token.lemma_ != "--"]

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

    def clean_gender(self, text: str, gender_symbols=["*", ":"]) -> str:
        """Removes 'gegenderte' words from a text

        Parameters
        ----------
        text : str
            The text which should be cleaned
        gender_symbols : list, optional
            The symbols used in the text as gender symbols, by default ["*", ":"]

        Returns
        -------
        str
            The cleaned text
        """
        for symbol in gender_symbols:
            text = re.sub(f"([a-zßäöü])\{symbol}innen([a-zßäöü]?)", r"\1\2", text)
            text = re.sub(f"([a-zßäöü])\{symbol}in([a-zßäöü]?)", r"\1\2", text)
            text = text.replace(f"Sinti{symbol}zze und Rom{symbol}nja", "Sinti und Roma")
            text = text.replace(f"der{symbol}die", "der")
            text = text.replace(f"die{symbol}der", "der")
            text = text.replace(f"den{symbol}die", "den")
            text = text.replace(f"dem{symbol}der", "dem")
            text = text.replace(f"der{symbol}s", "des")
            text = text.replace(f"eines{symbol}einer", "eines")
            text = text.replace(f"einer{symbol}s", "eines")
            text = text.replace(f"ihre{symbol}seine", "seine")
            text = text.replace(f"seiner{symbol}ihrer", "seiner")
            text = text.replace(f"jeder{symbol}m", "jedem")
            text = text.replace(f"Sie{symbol}Er", "Er")
            text = text.replace(f"des{symbol}der", "des")
            text = text.replace(f"welchem{symbol}welcher", "welchem")
            text = text.replace(f"{symbol}r", "r")
            text = text.replace(f"{symbol}n", "n")
            text = text.replace(f"{symbol}e", "n")

        logger.debug("Gender removed.")
        return text

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
