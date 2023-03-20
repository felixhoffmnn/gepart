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

        text = self.clean_gender(text)

        text = re.sub(r"[^a-zA-ZäöüÄÖÜß,.:;\?!\s]", " ", text)  # remove special characters
        text = re.sub(r"\s[a-zA-ZäöüÄÖÜß]\s", " ", text)  # remove single characters

        if not keep_punctuation:
            text = re.sub(r"[,.:;\?!]+", " ", text)  # remove punctuation

        text = " ".join(text.split())  # remove multiple spaces

        if not keep_upper:
            text = text.lower()

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

    def clean_gender(self, text: str, gender_symbols: list[str] = ["*", ":"]) -> str:
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
            The cleaned text without gender forms
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

        return text

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
