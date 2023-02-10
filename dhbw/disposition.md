---
title: "PoliAna: Nachrichten und Politikanalyse mittels Natural Language Processing"
subject: Studienarbeit
author: [Leopold Fuchs, Felix Hoffmann]
date: 2023-01-09
keywords: [nlp]
lang: de
header-left: "PoliAna"
titlepage: true
toc-own-page: true
footnotes-pretty: true
link-citations: true
book: true
classoption: [oneside]
header-includes: |
    \usepackage{graphicx}
    \usepackage{tikz}
---

# Kurzbeschreibung der Arbeit

<!-- Worum geht es in der Arbeit? Wie ist die (aktuelle) Ausgangssituation? Welches
Themenfeld wird bearbeitet? Welche Problemstellung soll angegangen werden? Welche
Grundlagen müssen vorhanden sein und welche Randbedingungen sind gegeben? Welche
Zielsetzungen gibt es in dieser Arbeit? Welche methodische Vorgehensweise wird
gewählt? Dies soll möglichst in einem Fließtext dokumentiert werden. Idealerweise
abschließend mit sehr konkreten Zielbeschreibungen, die auch validierbar sind. -->

Ereignisse wie die Wahl von Donald Trump zum US-Präsidenten, das Brexit-Referendum, als auch der Krieg in der Ukraine werfen die Frage auf, welchen Einfluss die Neuen Medien auf die politische Meinungsbildung haben [@brandon_russia_2022; @lee_how_2020]. Plattformen wie Twitter, Facebook und Instagram haben in den letzten Jahren die Art und Weise, wie Nachrichten erstellt und verbreitet werden, stark verändert. Unter anderem die Möglichkeit eines Nutzers, seine Meinung zu Themen schnell und direkt zu äußern, rückt den konkreten sprachlichen Ausdruck sowie polarisierende Meinungen von Politikern deutlich stärker in den Fokus.

Bisherige Arbeiten beschäftigen sich meist mit englischsprachigen Daten aus den USA oder Großbritannien. Außerdem umfassen Arbeiten wie die von Sältzer und Stier über die Bundestagswahl 2021 lediglich Tweets von Twitter [@saltzer_wahl_2022]. Dennoch wird in der Arbeit von Sältzer und Stier gezeigt, dass es möglich ist, Trends zu analysieren und die Parteizugehörigkeit eines Politikers anhand seiner Tweets zu klassifizieren.

Ziel dieser Arbeit ist es, mittels Natural Language Processing und Machine Learning die Meinung der Bevölkerung zu einem einzelnen Politiker oder Parteien in Deutschland zu analysieren und ein besseres Verständnis in die Reaktionen und Meinungen der Bevölkerung zu bekommen. Dafür sollen im Vergleich zu bisherigen Arbeiten Daten aus mehreren Quellen einbezogen werden.

Mögliche Fragestellungen sind für diese Arbeit sind:

-   Lassen sich Trends in der Semantik einzelner Politiker feststellen?
-   Ist es möglich, auf Basis der Semantik festzustellen, ob ein Politiker zum Flügel einer Partei gehört?
-   Können Politiker anhand ihrer Nachrichten und Daten klassifiziert werden?
-   Wie stark ist die Übereinstimmung der Klassifikation mit der wahrhaftigen Parteizugehörigkeit?
-   Unterstützen Nutzer ebenfalls die Politiker, welche am nächsten an ihrer Meinung sind?
-   Lässt sich die Parteizugehörigkeit aufgrund einzelner Worte oder Phrasen bestimmen?

# Gliederung und Zeitplan

<!-- Identifikation der wesentlichen Arbeitsschritte. Meilensteinplan. Konsequenzen
und Möglichkeiten der Meilensteine. Zeitplan bis zur Beendigung des praktischen
Teils sowie der Dokumentation. Eine erste Gliederung der Arbeit. Benennung von
Kapiteln und Unterkapiteln. Dies gilt als Leitfaden, noch nicht als abschließend. -->

Als erste Gliederung der Arbeit kann die folgende Struktur angenommen werden:

-   Einleitung
    -   Problemstellung
    -   Zielsetzung
    -   Methodik
    -   Struktur der Arbeit
-   Grundlagen
    -   Politikapparat und Parteienlandschaft
    -   Erörterung, welche Medienplattformen und Nachrichtenquellen verwendet werden sollen
    -   Auswahl von Quellen und Sammeln von Daten
    -   NLP-Pipeline
    -   Machine Learning / Clustering
-   Datenbeschaffung und -analyse nach CRISP-DM
    -   _Business Understanding_
        -   Festlegung auf spezifische Ereignisse, die näher untersucht werden sollen
    -   _Data Understanding_
        -   Identifizieren von geeigneten Nachrichtenquellen
        -   Sammeln der Trainingsdaten aus verschiedenen Quellen
    -   _Data Preparation_
    -   _Modeling_ (Unterteilung in verschiedene Thesen und Analysen)
        1. Trainieren eines NLP-Modells, das Texte nach Parteien klassifiziert
        2. Analysieren von Nachrichten und Tweets zu ausgewählten Ereignissen
    -   _Evaluation_
        -   Vergleich und Evaluation der trainierten Modelle
        -   Mathematische Betrachtung der Ergebnisse zur Trenddetektion
    -   _Deployment_
        -   Bereitstellung des besten Modells für die weiteren Analysen
        -   (grafische) Darstellung der Analyseergebnisse
-   Fazit
    -   Zusammenfassung
    -   Ausblick

Der Hauptteil der Arbeit wird nach dem CRISP-DM Prozessmodell aufgebaut. Die einzelnen Schritte stellen dabei die wesentlichen Meilensteine dar, die während des Projekts erreicht werden sollen.

In den kommenden drei Monaten soll die praktische Implementierung des Projekts abgeschlossen werden. Dabei sollen auch parallel die wesentlichen Inhalte, die praktisch umgesetzt werden, in der schriftlichen Arbeit dokumentiert werden. Nach Abschluss der praktischen Arbeit soll bis zum Ende der Gesamt-Projektlaufzeit die Vollendung und Optimierung der schriftlichen Arbeit erfolgen.

# Grundlegende Literatur

<!-- Belegen der Ausgangssituation. Wer hat auf ähnlichem Themenfeld bereits
gearbeitet? Wie passt die Studienarbeit in die aktuelle wissenschaftliche
Landschaft und was ist neu (dies wird oben dargelegt und hier belegt). Was wird
durch die erstellte Lösung verbessert und wie wird dies nachgewiesen? -->

Eine gute Übersicht über die notwendigen politikwissenschaftlichen Hintergründe bieten Bukow et al., die die Organisation und Funktion der deutschen Parteien beschreiben [@bukow_2013].

Kalyanam et al. untersuchen die Auswirkungen von Events in der realen Wert auf die Social Media Aktivität [@kalyanam_2016]. Ebenso untersuchen Tsytsarau et al. diesen Zusammenhang, mit einem Fokus auf die Verbindung der medialen Berichterstattung und dem Sentiment, der durch die Social Media Aktivitäten dargestellt wird [@tsytsarau_2014].
Gimpel et al. nähern sich der Thematik der Social Media Nutzung durch eine Cluster-Analyse zu verschiedenen Rollen in Twitter-Diskussionen [@gimpel_2018].
Zudem untersucht Sältzer die Positionen von Bundestagskandidaten auf Twitter und betrachtet diese einerseits im Vergleich innerhalb der Parteien sowie andererseits auf einem allgemeinen politischen Koordinatensystem [@saltzer_wahl_2022; @saltzer_finding_2022].

Li et al. sowie Kowsari et al. bieten je einen umfassenden Überblick sowie Vor- und Nachteile verschiedener Arten von Text-Klassifikation und gehen dabei sowohl auf traditionelle als auch auf Deep-Learning-Ansätze ein [@li_2021; @kowsari_2019].
Minaee et al. untersuchen und vergleichen die Verwendung von Deep-Learning-Modellen für die Aufgabe der Text-Klassifikation [@minaee_2022].

Wong et al. bestimmen die politische Ausrichtung aufgrund des Verhaltens von Personen auf Twitter durch die Betrachtung, welche Accounts ähnliche andere Accounts retweeten [@wong_2016].
Zudem vergleichen Doan et al. verschiedene Sprachmodelle zur Klassifizierung von Reden nach Parteien und führen dies für verschiedene Länder bzw. Sprachen durch [@doan_2022].
Auch Biessmann et al. klassifizieren Reden nach Parteien und nutzen zum Trainieren Parlaments-Debatten des Bundestages. Zudem wenden sie den Klassifikator auf andere Arten von Texten wie Social Media Posts an [@biessmann_2016].

Die Darstellung der Literatur zeigt, dass sich bereits einige Arbeiten mit der Analyse von Social Media Aktivitäten im Zusammenhang mit realen Events sowie mit dem Problem, Texte nach Parteizugehörigkeit zu klassifizieren, beschäftigen.
In unserer Studienarbeit wollen wir neue Klassifikationsverfahren nutzen und vergleichen, um eine höhere Performance als vergangene Arbeiten zu erreichen.
Zudem wollen wir nicht nur die Texte von Reden nutzen, sondern auch Social Media Posts und Parteiprogramme als Trainingsdaten für den Klassifikator einbeziehen.
Für die darauf aufbauenden Analysen wollen wir zusätzlich zu den Social Media Aktivitäten, die mit bestimmten Ereignissen in Verbindung stehen, auch die jeweilige politische Einstellung der Nutzer, bestimmt durch den trainierten Klassifikator, nutzen.

# Literaturverzeichnis

&nbsp;
