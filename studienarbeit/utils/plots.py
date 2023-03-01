import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Plots:
    """
    TODO: Plot length of texts in symbole count
    TODO: Wordcloud with most used words per party
    TODO: Count of sentences per text
    TODO: Plot count per day per party
    """

    def __init__(self, document_type: str):
        self.color_palette = "muted"
        sns.set(style="white", palette=self.color_palette, rc={"figure.figsize": (20, 8)})
        self.line_kws = {"color": "r", "alpha": 0.7, "lw": 5}

        self.document_type = document_type

    def _compose_title(self, title: str):
        return f"{title} ({self.document_type})"

    def party_count(self, df: pd.DataFrame, column="party", title="Anzahl an Einträgen pro Partei"):
        fig, axs = plt.subplots()

        sns.countplot(x=column, data=df, ax=axs)

        fig.suptitle(self._compose_title(title))
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def sentiment(self, df: pd.DataFrame, column="sentiment", title="Sentimentverteilung pro Partei"):
        fig, axs = plt.subplots()

        sns.barplot(
            x="party",
            y="count",
            hue=column,
            data=df.groupby([column, "party"]).size().reset_index(name="count"),
            ax=axs,
        )

        fig.suptitle(self._compose_title(title))
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def word_count(self, df: pd.DataFrame, column="filter_word_count", title="Wortanzahl pro Partei", x_lim=100):
        fig, axs = plt.subplots(1, 2)

        sns.kdeplot(data=df, x=column, hue="party", palette=self.color_palette, ax=axs[0])
        sns.boxenplot(data=df, x="party", y=column, ax=axs[1])

        fig.suptitle(self._compose_title(title))
        axs[0].set_xlabel("Wortanzahl (Nach Filterung)")
        axs[0].set_ylabel("Anzahl")
        axs[0].set_xlim(0, x_lim)
        axs[1].set_xlabel("Parteien")
        axs[1].set_ylabel("Wortanzahl")

    def gender(self, df: pd.DataFrame, column="gender", title="Geschlechterverteilung pro Partei"):
        fig, axs = plt.subplots()

        sns.countplot(data=df, x="party", hue=column, ax=axs)

        fig.suptitle(self._compose_title(title))
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")
        axs.legend(title="Geschlecht")

    def user_count(self, df: pd.DataFrame, column="screen_name", title="Anzahl der Nutzer pro Partei"):
        fig, axs = plt.subplots()

        sns.barplot(data=df.groupby("party")[column].nunique().reset_index(name="count"), x="party", y="count", ax=axs)

        fig.suptitle(self._compose_title(title))
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl der Nutzer")
