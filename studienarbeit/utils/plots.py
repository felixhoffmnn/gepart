import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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

    def sentiment(self, df: pd.DataFrame, column="sentiment", title="Sentiment"):
        fig, axs = plt.subplots()

        sns.barplot(
            x="party",
            y="count",
            hue=column,
            data=df.groupby([column, "party"]).size().reset_index(name="count"),
            ax=axs,
        )

        fig.suptitle(title)
        axs.set_xlabel("Parteien")
        axs.set_ylabel("Anzahl")

    def text_count(
        self, df: pd.DataFrame, column="stemm_word_count", title="Wortanzahl", measure_name="Wortanzahl", x_lim=100
    ):
        fig, axs = plt.subplots(1, 2)

        sns.kdeplot(data=df, x=column, hue="party", palette=self.party_palette, ax=axs[0])

        descending_median_order = df.groupby("party")[column].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="party", y=column, palette=self.party_palette, ax=axs[1], order=descending_median_order)

        fig.suptitle(title)
        axs[0].set_xlabel(measure_name)
        axs[0].set_ylabel("Anteil")
        axs[0].set_xlim(0, x_lim)
        axs[1].set_xlabel("Parteien")
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
