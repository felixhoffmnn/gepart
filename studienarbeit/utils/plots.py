import ast
from collections import Counter

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud


class Plots:
    """
    TODO: Plot length of texts in symbole count
    TODO: Wordcloud with most used words per party
    TODO: Plot count per day per party
    """

    def __init__(self, party_palette=None):
        self.color_palette = "muted"
        self.party_palette = party_palette
        sns.set(style="white", palette=self.color_palette, rc={"figure.figsize": (20, 8)})
        self.line_kws = {"color": "r", "alpha": 0.7, "lw": 5}
        with open("../../data/stopwords/german_stopwords_full.txt", "r") as f:
            self.stopwords = f.read().splitlines()

    def party_count(self, df: pd.DataFrame, column="party", title="Parteien"):
        fig, axs = plt.subplots()

        sns.countplot(x=column, data=df, ax=axs, palette=self.party_palette, order=df[column].value_counts().index)

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def grouped_party_count(self, df: pd.DataFrame, group_column, title, main_column="party"):
        fig, axs = plt.subplots()

        sns.countplot(x=main_column, data=df, hue=group_column, ax=axs)

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def politician_count(
        self, df: pd.DataFrame, politician_column="politicianId", party_column="party", title="Politiker pro Partei"
    ):
        df_speakers = df[[politician_column, party_column]].drop_duplicates()
        df_speakers = df_speakers[df_speakers[politician_column].notnull()]
        df_speakers = df_speakers[df_speakers[politician_column] != -1]
        df_speakers_grouped = (
            df_speakers.groupby(party_column).count().reset_index().rename(columns={politician_column: "count"})
        )
        fig, axs = plt.subplots()

        sns.barplot(x=party_column, y="count", data=df_speakers_grouped, palette=self.party_palette)

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def sentiment(self, df: pd.DataFrame, column="sentiment", title="Sentiment der Reden nach Partei"):
        fig, axs = plt.subplots()

        sns.barplot(
            x="party",
            y="count",
            hue=column,
            data=df.groupby([column, "party"]).size().reset_index(name="count"),
            ax=axs,
        )

        fig.suptitle(title)
        axs.set_xlabel("Partei")
        axs.set_ylabel("Anzahl an Reden")

    def text_count(
        self, df: pd.DataFrame, column="stemm_word_count", title="Wortanzahl", measure_name="Wortanzahl", x_lim=100
    ):
        fig, axs = plt.subplots(1, 2)

        sns.kdeplot(data=df, x=column, hue="party", palette=self.party_palette, ax=axs[0], legend=False)
        axs[0].legend(title="Partei", loc="upper right", labels=df["party"].unique())

        descending_median_order = df.groupby("party")[column].median().sort_values(ascending=False).index
        sns.boxenplot(
            data=df, x="party", y=column, palette=self.party_palette, ax=axs[1], order=descending_median_order
        )

        fig.suptitle(title)
        axs[0].set_xlabel(measure_name)
        axs[0].set_ylabel("Dichte")
        axs[0].set_xlim(0, x_lim)
        axs[1].set_xlabel("Partei")
        axs[1].set_ylabel(measure_name)

    def gender(self, df: pd.DataFrame, column="gender", title="Geschlechterverteilung pro Partei"):
        fig, axs = plt.subplots()

        sns.countplot(data=df, x="party", hue=column, ax=axs)

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")
        axs.legend(title="Geschlecht")

    def user_count(self, df: pd.DataFrame, column="screen_name", title="Anzahl der Nutzer pro Partei"):
        fig, axs = plt.subplots()

        sns.barplot(df.groupby("party")[column].nunique().reset_index(name="count"), x="party", y="count", ax=axs)

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl der Nutzer")

    def wordclouds(self, df: pd.DataFrame, column="tokenized_text", title="Wordclouds nach Partei"):
        df["tokenized_text_merged"] = df[column].apply(lambda x: " ".join(ast.literal_eval(x)))

        numbers = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
        verbs = ["müssen", "sagen", "sollen", "geben", "gehen", "dürfen", "müssen"]
        speeches = ["kollege", "kollegin", "dame", "herr", "jahr", "antrag", "rede"]
        all_stopwords = self.stopwords + numbers + verbs + speeches

        df["tokenized_text_merged"] = df["tokenized_text_merged"].apply(
            lambda x: " ".join([word for word in x.split() if word not in all_stopwords])
        )

        parties = sorted(df["party"].unique())
        wordclouds = [self._get_get_wordcloud(df, party, 20) for party in parties]

        fig, axs = plt.subplots(2, 3)

        for i, ax in enumerate(axs.flatten()):
            ax.imshow(wordclouds[i], interpolation="bilinear")
            ax.axis("off")
            ax.set_title(parties[i])

        fig.suptitle("Wordclouds nach Partei")

    def _get_get_wordcloud(self, df: pd.DataFrame, party: str, max_words: int = 20) -> WordCloud:
        df_party = df[df["party"] == party]
        wc = WordCloud()
        counts_all = Counter()

        df_party["tokenized_text_merged"].progress_apply(lambda x: counts_all.update(wc.process_text(x)))
        wc.generate_from_frequencies(counts_all)
        wc.background_color = "white"
        wc.max_words = max_words

        return wc
