import logging

import pandas as pd
from nltk import sent_tokenize
from transformers import AutoTokenizer


class SplitText:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    def split_dataframe_texts(self, df, text_col_name, max_length=512):
        df_split = pd.DataFrame(columns=df.columns)
        for _, row in df.iterrows():
            texts = self._split_text(row[text_col_name], max_length)
            for text in texts:
                new_row = row.copy()
                new_row[text_col_name] = text
                df_split = df_split.append(new_row, ignore_index=True)

        return df_split

    def _split_text(self, text, max_length):
        sentences = sent_tokenize(text)

        if len(sentences) < 3:
            num_check_sentences = len(sentences)
        else:
            num_check_sentences = 3
        for i in range(num_check_sentences):
            sentences[i] = self._remove_greeting_sentence(sentences[i])
            sentences[len(sentences) - i - 1] = self._remove_greeting_sentence(sentences[len(sentences) - i - 1])
        sentences = [sentence for sentence in sentences if sentence is not None]

        filtered_text = " ".join(sentences).strip()
        tokens = self.tokenizer.tokenize(filtered_text)

        if len(tokens) > max_length:
            num_new_texts = (len(tokens) // max_length) + 1
            new_texts = []

            while sentences:
                current_text = ""
                current_num_tokens = 0
                while (current_num_tokens < (len(tokens) / num_new_texts)) and sentences:
                    current_text += " " + sentences.pop(0)
                    current_num_tokens = len(self.tokenizer.tokenize(current_text))

                new_texts.append(current_text.strip())

            return new_texts
        else:
            return [filtered_text]

    def _remove_greeting_sentence(self, sentence: str):
        if (sentence is None) or any(
            keyword in sentence.lower() for keyword in ["herr", "frau", "sehr", "geehrte", "kolleg", "präsid", "dank"]
        ):
            return None
        else:
            return sentence