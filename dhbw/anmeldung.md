# PoliAna: Nachrichten und Politikanalyse mittels Natural Language Processing

- [PoliAna: Nachrichten und Politikanalyse mittels Natural Language Processing](#poliana-nachrichten-und-politikanalyse-mittels-natural-language-processing)
  - [Problemstellung und Ziel der Arbeit](#problemstellung-und-ziel-der-arbeit)
  - [Geplantes Vorgehen](#geplantes-vorgehen)
  - [Literatur](#literatur)

## Problemstellung und Ziel der Arbeit

An Beispielen wie der Wahl von Trump im Jahr 2016 und aktuell dem Krieg in der Ukraine deutet sich eine Korrelation zwischen auf der einen Seite politischen Ereignissen und auf der anderen Seite Nachrichten sowie Aktivitäten in sozialen Medien in Form von beispielsweise Tweets auf Twitter ab. Dabei kommt es unter anderem auch zu Fake-News oder zahlreichen ausfallenden Kommentaren von Nutzern in sozialen Medien.

Mittels dieses Projektes sollen Methoden des Natural Language Processing (NLP) verwendet werden, um zunächst Nachrichten, Tweets oder andere Kommentare zu analysieren und anschließend im Kontext von politischen Ereignissen auszuwerten. Die Auswertung mittels NLP kann zum Beispiel durch Sentiment-Analysen erfolgen. Dabei gilt es zu untersuchen, in welchem Verhältnis die Nachrichten- und Social Media-Aktivitäten zu ausgewählten politisch relevanten Geschehnissen stehen.

## Geplantes Vorgehen

Der Prozess der Datensammlung und Auswertung teilt sich in drei Abschnitte auf:

1. Analyse und Validierung von Nachrichtenagenturen und -plattformen, um festzustellen, welche davon über eine API für die Datensammlung verfügen oder legal gescrapt werden können.
2. Script-basierte Datensammlung und Verarbeitung in Form einer NLP-Pipeline und anschließender Speicherung in einer Datenbank, um die Daten zu persistieren und zugänglich für die spätere Analyse zu machen.
3. Datenanalyse, Trenderkennung und Interpretation auf Basis der in der Datenbank gespeicherten Nachrichten. Da das eigene Trainieren von Machine Learning Modellen sehr zeit- und kostenaufwändig ist, soll für die Analyse und Datenverarbeitung zunächst auf vortrainierte Modelle zurückgegriffen werden.

## Literatur

[1] M. Tsytsarau, T. Palpanas, und M. Castellanos, „Dynamics of news events and social media reaction“, in Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, New York, NY, USA, Aug. 2014, S. 901–910. doi: 10.1145/2623330.2623670.

[2] J. Kalyanam, M. Quezada, B. Poblete, und G. Lanckriet, „Prediction and Characterization of High-Activity Events in Social Media Triggered by Real-World News“, PLOS ONE, Bd. 11, Nr. 12, S. e0166694, Dez. 2016, doi: 10.1371/journal.pone.0166694.

[3] B. Batrinca und P. C. Treleaven, „Social media analytics: a survey of techniques, tools and platforms“, AI & Soc, Bd. 30, Nr. 1, S. 89–116, Feb. 2015, doi: 10.1007/s00146-014-0549-4.

[4] R. Bandari, S. Asur, und B. Huberman, „The Pulse of News in Social Media: Forecasting Popularity“, Proceedings of the International AAAI Conference on Web and Social Media, Bd. 6, Nr. 1, Art. Nr. 1, 2012.
