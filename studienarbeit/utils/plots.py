import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="white", palette="muted", rc={"figure.figsize": (20, 8)})
line_kws = {"color": "r", "alpha": 0.7, "lw": 5}


def plot_party_count(df, column="party", title="Parteien"):
    fig, axs = plt.subplots()

    sns.countplot(x=column, data=df, ax=axs)

    fig.suptitle(title)
    axs.set_xlabel("Parteien")
    axs.set_ylabel("Anzahl")


def plot_sentiment(df, column="sentiment", title="Sentiment"):
    fig, axs = plt.subplots()

    sns.kdeplot(data=df, x=column, hue="party", ax=axs)

    fig.suptitle(title)
    axs.set_xlabel("Parteien")
    axs.set_ylabel("Anzahl")


def plot_word_count(df, column="word_count", title="Wortanzahl", x_lim=100):
    fig, axs = plt.subplots(1, 2)

    sns.kdeplot(data=df, x=column, hue="party", ax=axs[0])
    sns.boxenplot(data=df, x="party", y=column, ax=axs[1])

    fig.suptitle(title)
    axs[0].set_xlabel("Wortanzahl")
    axs[0].set_ylabel("Anzahl")
    axs[0].set_xlim(0, x_lim)
    axs[1].set_xlabel("Parteien")
    axs[1].set_ylabel("Wortanzahl")


def plot_gender(df, column="gender", title="Geschlechterverteilung pro Partei"):
    _, axs = plt.subplots()

    sns.countplot(data=df, x="party", hue=column, ax=axs)

    axs.set_title(title)
    axs.set_xlabel("Parteien")
    axs.set_ylabel("Anzahl")
    axs.legend(title="Geschlecht")
