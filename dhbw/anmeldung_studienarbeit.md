# Studienarbeitsthema

- [Studienarbeitsthema](#studienarbeitsthema)
    - [Problemstellung und Ziel der Arbeit](#problemstellung-und-ziel-der-arbeit)
    - [Geplantes Vorgehen](#geplantes-vorgehen)

## Problemstellung und Ziel der Arbeit

An Beispielen wie der Wahl von Trump in ... und aktuell dem Krieg in der Ukraine ist ein Anstieg von Fakenews und toxischen Nachrichten in sozialen Medien zu verzeichnen. Mittels dieses Projektes sollen NLP-Methoden verwendet werden, um zunächst Nachrichten, Tweets oder Kommentare zu analysieren und anschließen auszuwerten.

Bei der Datenanalyse soll der Fokus auf Fakenews, toxischen Nachrichten und dem Nachrichtensentiment gelegt werden. Hierfür können NLP Modelle von Huggingface oder spaCy verwendet werden. Besonders interessant für die Analyse ist eine Auswertung vor und nach einem politischen Event.

## Geplantes Vorgehen
Der Prozess der Datensammlung und Auswertung teilt sich in drei Abschnitte auf.

1. Analyse der Nachrichtenagenturen und -plattformen um festzusetllen welche davon eine API für die Datensammlung besitzen oder welche legal gescrapt werden können.
2. Datensammlung, Verarbeitung in Form einer NLP pipline und anschließender Speicherung in einer Datenbank um die Daten zu persistieren und zugänglich für die spätere Analyse zu machen.
3. Datenanalyse und Trenderkennung auf Basis der in der Datenbank gespeicherten Nachrichten.

Da das eigene Trainieren von Machine Learining Modellen sehr zeit- und kostenaufwändig ist, sollen zunächst vortrainierte Modelle verwendent werden.

TODO: Konzept nach Crisp?

Die Umsetzung für die Analyse der Nachritenplattformen soll der ... Methode folgen. Bei dieser Methode ensteht eine Evaluationsmatrix welche die Plattformen bewertet und kategorisiert. Die Sammlung und Analyse soll Script basiert ablaufen. Diese sollen dann auf einem Server automatisiert ausgeführt werden um kontinuirlich Daten zu sammeln.

Folgend ist einige Literatur gelistet, welche NLP und Fake News Detection behandeln.

-   Lorem Ipsum
