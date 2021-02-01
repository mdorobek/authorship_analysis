import numpy as np
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
from nltk.tokenize import sent_tokenize
from itertools import groupby
import copy
import re
import sys
import textstat


# Method to create a matrix with contains only zeroes and a index starting by 0
def create_matrix_index_zeros(rows, columns):
    arr = np.zeros((rows, columns))
    for r in range(0, rows):
        arr[r, 0] = r
    return arr


# Method to get all authors with a given number of texts. Used in chapter 5.1 to get a corpus with 100 Texts for 25
# authors
def get_balanced_df_all_authors(par_df, par_num_text):
    author_count = par_df["author"].value_counts()
    author_list = []
    df_balanced_text = pd.DataFrame(columns=['label_encoded', 'author', 'genres', 'release_date', 'text'])
    for i in range(0, len(author_count)):
        if author_count[i] >= par_num_text and not author_count.index[i] == "Gast-Rezensent":
            author_list.append(author_count.index[i])
    texts = [par_num_text for i in range(0, len(author_count))]
    for index, row in par_df.iterrows():
        if row['author'] in author_list:
            if texts[author_list.index(row['author'])] != 0:
                d = {'author': [row['author']], 'genres': [row['genres']],
                     'release_date': [row['release_date']], 'text': [row['text']]}
                df_balanced_text = df_balanced_text.append(pd.DataFrame.from_dict(d), ignore_index=True)
                texts[author_list.index(row['author'])] -= 1
            if sum(texts) == 0:
                break
    # Label encoding and delete author column after
    dic_author_mapping = author_encoding(df_balanced_text)
    df_balanced_text['label_encoded'] = get_encoded_author_vector(df_balanced_text, dic_author_mapping)[:, 0]
    df_balanced_text.drop("author", axis=1, inplace=True)
    # Print author mapping in file
    original_stdout = sys.stdout
    with open('author_mapping.txt', 'w') as f:
        sys.stdout = f
        print(dic_author_mapping)
        sys.stdout = original_stdout
    for i in range(0, len(author_list)):
        print(f"Autor {i+1}: {par_num_text - texts[i]} Texte")
    return df_balanced_text


# Method to get a specific number of authors with a given number of texts. Used later on to get results for different
# combinations of authors and texts
def get_balanced_df_by_texts_authors(par_df, par_num_text, par_num_author):
    author_count = par_df["author"].value_counts()
    author_list = []
    df_balanced_text = pd.DataFrame(columns=['label_encoded', 'author', 'genres', 'release_date', 'text'])
    loop_count, loops = 0, par_num_author
    while loop_count < loops:
        if author_count[loop_count] >= par_num_text and not author_count.index[loop_count] == "Gast-Rezensent":
            author_list.append(author_count.index[loop_count])
        # Skip the Author "Gast-Rezensent" if its not the last round and increase the loops by 1
        elif author_count.index[loop_count] == "Gast-Rezensent":
            loops += 1
        loop_count += 1
    texts = [par_num_text for i in range(0, len(author_list))]
    for index, row in par_df.iterrows():
        if row['author'] in author_list:
            if texts[author_list.index(row['author'])] != 0:
                d = {'author': [row['author']], 'genres': [row['genres']],
                     'release_date': [row['release_date']], 'text': [row['text']]}
                df_balanced_text = df_balanced_text.append(pd.DataFrame.from_dict(d), ignore_index=True)
                texts[author_list.index(row['author'])] -= 1
            if sum(texts) == 0:
                break
    # Label encoding and delete author column after
    dic_author_mapping = author_encoding(df_balanced_text)
    df_balanced_text['label_encoded'] = get_encoded_author_vector(df_balanced_text, dic_author_mapping)[:, 0]
    df_balanced_text.drop("author", axis=1, inplace=True)
    # Print author mapping in file
    original_stdout = sys.stdout
    with open('author_mapping.txt', 'w') as f:
        sys.stdout = f
        print(dic_author_mapping)
        sys.stdout = original_stdout
    for i in range(0, len(author_list)):
        print(f"Autor {i+1}: {par_num_text - texts[i]} Texte")
    return df_balanced_text


# Feature extraction of the feature described in chapter 5.6.1
def get_bow_matrix(par_df):
    nlp = spacy.load("de_core_news_sm")
    d_bow = {}
    d_bow_list = []
    function_pos = ["ADP", "AUX", "CONJ", "CCONJ", "DET", "PART", "PRON", "SCONJ"]
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        tokens = [word for word in tokens if not word.is_punct and not word.is_space and not
                  word.is_digit and word.lemma_ not in STOP_WORDS and word.pos_ not in function_pos]
        for word in tokens:
            try:
                d_bow["bow:"+word.lemma_.lower()] += 1
            except KeyError:
                d_bow["bow:"+word.lemma_.lower()] = 1
        d_bow_list.append(copy.deepcopy(d_bow))
        d_bow.clear()
    return pd.DataFrame(d_bow_list)


# Feature extraction of the feature described in chapter 5.6.2
def get_word_n_grams(par_df, n):
    nlp = spacy.load("de_core_news_sm")
    d_word_ngram = {}
    d_word_ngram_list = []
    function_pos = ["ADP", "AUX", "CONJ", "CCONJ", "DET", "PART", "PRON", "SCONJ"]
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        tokens = [word for word in tokens if not word.is_punct and not word.is_space and not
                  word.is_digit and word.lemma_ not in STOP_WORDS and word.pos_ not in function_pos]
        tokens = [token.lemma_.lower() for token in tokens]
        for w in range(0, len(tokens)):
            if w + n <= len(tokens):
                try:
                    d_word_ngram["w" + str(n) + "g" + ":" + '|'.join(tokens[w:w + n])] += 1
                except KeyError:
                    d_word_ngram["w" + str(n) + "g" + ":" + '|'.join(tokens[w:w + n])] = 1
        d_word_ngram_list.append(copy.deepcopy(d_word_ngram))
        d_word_ngram.clear()
    return pd.DataFrame(d_word_ngram_list)


# Feature extraction of the feature described in chapter 5.6.3
def get_word_count(par_df):
    arr_wordcount = np.zeros((len(par_df), 1))
    nlp = spacy.load("de_core_news_sm")
    only_words = []
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for t in tokens:
            if not t.is_punct and not t.is_space:
                only_words.append(t)
        arr_wordcount[index] = len(only_words)
        only_words.clear()
    return pd.DataFrame(data=arr_wordcount, columns=["word_count"])


# Feature extraction of the feature described in chapter 5.6.4 with some variations
# Count all word lengths individually
def get_word_length_matrix(par_df):
    nlp = spacy.load("de_core_news_sm")
    d_word_len = {}
    d_word_len_list = []
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        tokens = [word for word in tokens if not word.is_punct and not word.is_space and not word.is_digit]
        for word in tokens:
            try:
                d_word_len["w_len:"+str(len(word.text))] += 1
            except KeyError:
                d_word_len["w_len:"+str(len(word.text))] = 1
        d_word_len_list.append(copy.deepcopy(d_word_len))
        d_word_len.clear()
    return pd.DataFrame(d_word_len_list)


# Count word lengths and set 2 intervals
def get_word_length_matrix_with_interval(par_df, border_1, border_2):
    arr_wordcount_with_interval = np.zeros((len(par_df), border_1 + 2))
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for word in tokens:
            if len(word.text) <= border_1 and not word.is_punct and not word.is_space and not word.is_digit:
                arr_wordcount_with_interval[index, len(word.text) - 1] += 1
            elif border_1 < len(
                    word.text) <= border_2 and not word.is_punct and not word.is_space and not word.is_digit:
                arr_wordcount_with_interval[index, -2] += 1
            elif not word.is_punct and not word.is_space and not word.is_digit:
                arr_wordcount_with_interval[index, -1] += 1

    word_length_labels = [str(i) for i in range(1, border_1+1)]
    word_length_labels.append(f"{border_1+1}-{border_2}")
    word_length_labels.append(f">{border_2}")
    return pd.DataFrame(data=arr_wordcount_with_interval, columns=word_length_labels)


# Count word lengths and sum all above a defined margin
def get_word_length_matrix_with_margin(par_df, par_margin):
    arr_wordcount_with_interval = np.zeros((len(par_df), par_margin + 1))
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for word in tokens:
            if len(word.text) <= par_margin and not word.is_punct and not word.is_space and not word.is_digit:
                arr_wordcount_with_interval[index, len(word.text) - 1] += 1
            elif par_margin < len(word.text) and not word.is_punct and not word.is_space and not word.is_digit:
                arr_wordcount_with_interval[index, -1] += 1

    word_length_labels = [str(i) for i in range(1, par_margin+1)]
    word_length_labels.append(f">{par_margin}")
    return pd.DataFrame(data=arr_wordcount_with_interval, columns=word_length_labels)


# Count the average word length of the article
def get_average_word_length(par_df):
    arr_avg_word_len_vector = np.zeros((len(par_df), 1))
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        symbol_sum = 0
        words = 0
        tokens = nlp(row['text'])
        for word in tokens:
            if not word.is_punct and not word.is_space and not word.is_digit:
                symbol_sum += len(word.text)
                words += 1
        arr_avg_word_len_vector[index, 0] = symbol_sum / words
    return pd.DataFrame(data=arr_avg_word_len_vector, columns=["avg_word_length"])


# Feature extraction of the feature described in chapter 5.6.5
def get_yules_k(par_df):
    d = {}
    nlp = spacy.load("de_core_news_sm")
    arr_yulesk = np.zeros((len(par_df), 1))
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for t in tokens:
            if not t.is_punct and not t.is_space and not t.is_digit:
                w = t.lemma_.lower()
                try:
                    d[w] += 1
                except KeyError:
                    d[w] = 1
        s1 = float(len(d))
        s2 = sum([len(list(g)) * (freq ** 2) for freq, g in groupby(sorted(d.values()))])
        try:
            k = 10000 * (s2 - s1) / (s1 * s1)
            arr_yulesk[index] = k
        except ZeroDivisionError:
            pass
        d.clear()
    return pd.DataFrame(data=arr_yulesk, columns=["yulesk"])


# Feature extraction of the feature described in chapter 5.6.6
# Get a vector of all special characters
def get_special_char_label_vector(par_df):
    nlp = spacy.load("de_core_news_sm")
    special_char_label_vector = []
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for t in tokens:
            chars = ' '.join([c for c in t.text])
        chars = nlp(chars)
        for c in chars:
            if c.is_punct and c.text not in special_char_label_vector:
                special_char_label_vector.append(c.text)
    return special_char_label_vector


# Get a matrix of all special character by a given vector of special chars
def get_special_char_matrix(par_df, par_special_char_label_vector):
    nlp = spacy.load("de_core_news_sm")
    arr_special_char = np.zeros((len(par_df), len(par_special_char_label_vector)))
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for t in tokens:
            chars = ' '.join([c for c in t.text])
        chars = nlp(chars)
        for c in chars:
            if c.text in par_special_char_label_vector:
                arr_special_char[index, par_special_char_label_vector.index(c.text)] += 1
    return arr_special_char


# Feature extraction of the feature described in chapter 5.6.7
# Get the char-affix-n-grams by a defined n
def get_char_affix_n_grams(par_df, n):
    d_prefix_list, d_suffix_list, d_space_prefix_list, d_space_suffix_list = [], [], [], []
    d_prefix, d_suffix, d_space_prefix, d_space_suffix = {}, {}, {}, {}
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for w in range(0, len(tokens)):
            # Prefix
            if len(tokens[w].text) >= n + 1:
                try:
                    d_prefix["c" + str(n) + "_p: " + tokens[w].text.lower()[0:n]] += 1
                except KeyError:
                    d_prefix["c" + str(n) + "_p: " + tokens[w].text.lower()[0:n]] = 1
            # Suffix
            if len(tokens[w].text) >= n + 1:
                try:
                    d_suffix["c" + str(n) + "_s: " + tokens[w].text.lower()[-n:]] += 1
                except KeyError:
                    d_suffix["c" + str(n) + "_s: " + tokens[w].text.lower()[-n:]] = 1
        d_prefix_list.append(copy.deepcopy(d_prefix))
        d_suffix_list.append(copy.deepcopy(d_suffix))
        d_prefix.clear()
        d_suffix.clear()
        for i in range(0, len(row['text'])):
            if row['text'][i] == " " and i + n <= len(row['text']) and i - n >= 0:
                # Space-prefix
                try:
                    d_space_prefix["c" + str(n) + "_sp: " + row['text'].lower()[i:n + i]] += 1
                except KeyError:
                    d_space_prefix["c" + str(n) + "_sp: " + row['text'].lower()[i:n + i]] = 1
                # Space-suffix
                try:
                    d_space_suffix["c" + str(n) + "_ss: " + row['text'].lower()[i - n + 1:i + 1]] += 1
                except KeyError:
                    d_space_suffix["c" + str(n) + "_ss: " + row['text'].lower()[i - n + 1:i + 1]] = 1
        d_space_prefix_list.append(copy.deepcopy(d_space_prefix))
        d_space_suffix_list.append(copy.deepcopy(d_space_suffix))
        d_space_prefix.clear()
        d_space_suffix.clear()
    df_pre = pd.DataFrame(d_prefix_list)
    df_su = pd.DataFrame(d_suffix_list)
    df_s_pre = pd.DataFrame(d_space_prefix_list)
    df_s_su = pd.DataFrame(d_space_suffix_list)
    df_affix = pd.concat([df_pre, df_su, df_s_pre, df_s_su], axis=1)
    return df_affix


# Get the char-word-n-grams by a defined n
def get_char_word_n_grams(par_df, n):
    d_whole_word_list, d_mid_word_list, d_multi_word_list = [], [], []
    d_whole_word, d_mid_word, d_multi_word = {}, {}, {}
    match_list = []
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for w in range(0, len(tokens)):
            # Whole-word
            if len(tokens[w].text) == n:
                try:
                    d_whole_word["c" + str(n) + "_ww: " + tokens[w].text.lower()] += 1
                except KeyError:
                    d_whole_word["c" + str(n) + "_ww: " + tokens[w].text.lower()] = 1
            # Mid-word
            if len(tokens[w].text) >= n + 2:
                for i in range(1, len(tokens[w].text) - n):
                    try:
                        d_mid_word["c" + str(n) + "_miw: " + tokens[w].text.lower()[i:i + n]] += 1
                    except KeyError:
                        d_mid_word["c" + str(n) + "_miw: " + tokens[w].text.lower()[i:i + n]] = 1
        d_whole_word_list.append(copy.deepcopy(d_whole_word))
        d_mid_word_list.append(copy.deepcopy(d_mid_word))
        d_whole_word.clear()
        d_mid_word.clear()
        # Multi-word
        # ignore special character
        trimmed_text = re.sub(r'[\s]+', ' ', re.sub(r'[^\w ]+', '', row['text']))
        match_list.clear()
        for i in range(1, n - 1):
            regex = r"\w{" + str(i) + r"}\s\w{" + str(n - 1 - i) + r"}"
            match_list += re.findall(regex, trimmed_text.lower())
        for match in match_list:
            try:
                d_multi_word["c" + str(n) + "_mw: " + match] += 1
            except KeyError:
                d_multi_word["c" + str(n) + "_mw: " + match] = 1
        d_multi_word_list.append(copy.deepcopy(d_multi_word))
        d_multi_word.clear()
    df_ww = pd.DataFrame(d_whole_word_list)
    df_miw = pd.DataFrame(d_mid_word_list)
    df_mw = pd.DataFrame(d_multi_word_list)
    df_word = pd.concat([df_ww, df_miw, df_mw], axis=1)
    return df_word


# Get the char-punct-n-grams by a defined n
def get_char_punct_n_grams(par_df, n):
    d_beg_punct_list, d_mid_punct_list, d_end_punct_list = [], [], []
    d_beg_punct, d_mid_punct, d_end_punct = {}, {}, {}
    for index, row in par_df.iterrows():
        for c in range(0, len(row['text'])):
            if row['text'][c] in ["!", "„", "“", "(", ")", "?", "{", "}", "[", "]", "‚", "‘", "-", "_", ".", ",", ";",
                                  "/", "\\", ":"]:
                if c <= len(row['text']) - n + 1:
                    # beg-punct
                    try:
                        d_beg_punct["c" + str(n) + "_bp: " + row['text'].lower()[c:c + n]] += 1
                    except KeyError:
                        d_beg_punct["c" + str(n) + "_bp: " + row['text'].lower()[c:c + n]] = 1
                if c >= n - 1:
                    # end-punct
                    try:
                        d_end_punct["c" + str(n) + "_ep: " + row['text'].lower()[c - n + 1:+1]] += 1
                    except KeyError:
                        d_end_punct["c" + str(n) + "_ep: " + row['text'].lower()[c - n + 1:c + 1]] = 1
                # Mid-punct
                # Run through all combinations of summands around the special char
                for i in range(1, n - 1):
                    if len(row['text']) - i + 1 >= c >= i - 1:
                        try:
                            d_mid_punct["c" + str(n) + "_mp: " + row['text'].lower()[c - i:c + n - i]] += 1
                        except KeyError:
                            d_mid_punct["c" + str(n) + "_mp: " + row['text'].lower()[c - i:c + n - i]] = 1

        d_beg_punct_list.append(copy.deepcopy(d_beg_punct))
        d_end_punct_list.append(copy.deepcopy(d_end_punct))
        d_mid_punct_list.append(copy.deepcopy(d_mid_punct))
        d_beg_punct.clear()
        d_end_punct.clear()
        d_mid_punct.clear()
    df_bp = pd.DataFrame(d_beg_punct_list)
    df_mp = pd.DataFrame(d_mid_punct_list)
    df_ep = pd.DataFrame(d_end_punct_list)
    df_punct = pd.concat([df_bp, df_mp, df_ep], axis=1)
    return df_punct


# Feature extraction of the feature described in chapter 5.6.8
# Count all digits individually
def get_digits(par_df):
    d_digits = {}
    d_digits_list = []
    for index, row in par_df.iterrows():
        match_list = re.findall(r"\d", row['text'])
        for match in match_list:
            try:
                d_digits[f"digit:{match}"] += 1
            except KeyError:
                d_digits[f"digit:{match}"] = 1
        d_digits_list.append(copy.deepcopy(d_digits))
        d_digits.clear()
    return pd.DataFrame(d_digits_list)


# sum the count of all digits
def get_sum_digits(par_df):
    sum_digits_list = []
    for index, row in par_df.iterrows():
        match_list = re.findall(r"\d", row['text'])
        sum_digits_list.append(len(match_list))
    return pd.DataFrame(data=sum_digits_list, columns=['sum_digits'])


# Feature extraction of the feature described in chapter 5.6.9
def get_function_words(par_df):
    d_fwords = {}
    d_fwords_list = []
    nlp = spacy.load("de_core_news_sm")
    function_pos = ["ADP", "AUX", "CONJ", "CCONJ", "DET", "PART", "PRON", "SCONJ"]
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for token in tokens:
            if token.pos_ in function_pos:
                try:
                    d_fwords[f"f_word:{token.lemma_.lower()}"] += 1
                except KeyError:
                    d_fwords[f"f_word:{token.lemma_.lower()}"] = 1
        d_fwords_list.append(copy.deepcopy(d_fwords))
        d_fwords.clear()
    return pd.DataFrame(d_fwords_list)


# Feature extraction of the feature described in chapter 5.6.10
def get_pos_tags(par_df):
    d_pos_tags = {}
    d_pos_tags_list = []
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        for token in tokens:
            if not token.pos_ == "SPACE" and not token.pos_ == "PUNCT" and not token.pos_ == "SYMBOL" \
                    and not token.pos_ == "X":
                try:
                    d_pos_tags[f"pos_tag:{token.pos_}"] += 1
                except KeyError:
                    d_pos_tags[f"pos_tag:{token.pos_}"] = 1
        d_pos_tags_list.append(copy.deepcopy(d_pos_tags))
        d_pos_tags.clear()
    return pd.DataFrame(d_pos_tags_list)


# Feature extraction of the feature described in chapter 5.6.11
def get_pos_tags_n_grams(par_df, n):
    d_pos_tags_gram = {}
    d_pos_tags_gram_list = []
    nlp = spacy.load("de_core_news_sm")

    for index, row in par_df.iterrows():
        tokens = nlp(row['text'])
        tokens = [t.pos_ for t in tokens if not t.pos_ == "SPACE" and not t.pos_ == "PUNCT" and not t.pos_ == "SYMBOL"
                  and not t.pos_ == "X"]
        for i in range(0, len(tokens)):
            if i + n <= len(tokens):
                try:
                    d_pos_tags_gram[f"pt_{n}g:{'|'.join(tokens[i:i + n])}"] += 1
                except KeyError:
                    d_pos_tags_gram[f"pt_{n}g:{'|'.join(tokens[i:i + n])}"] = 1
        d_pos_tags_gram_list.append(copy.deepcopy(d_pos_tags_gram))
        d_pos_tags_gram.clear()
    return pd.DataFrame(d_pos_tags_gram_list)


# Feature extraction of the feature described in chapter 5.6.12
def get_sentence_end_start(par_df):
    d_s_end_pos = {}
    d_s_end_pos_list = []
    d_s_start_pos = {}
    d_s_start_pos_list = []
    nlp = spacy.load("de_core_news_sm")
    for index, row in par_df.iterrows():
        sentences = sent_tokenize(row['text'], language='german')
        for s in sentences:
            token = nlp(s)
            token = [t.pos_ for t in token if
                     not t.pos_ == "SPACE" and not t.pos_ == "PUNCT" and not t.pos_ == "SYMBOL" and not t.pos_ == "X"]
            if len(token) > 1:
                # if token[-1] == "CCONJ":
                #    print(token)
                #    print(s)
                try:
                    d_s_start_pos["s_start:" + str(token[0])] += 1
                    d_s_end_pos["s_end:" + str(token[-1])] += 1
                except KeyError:
                    d_s_start_pos["s_start:" + str(token[0])] = 1
                    d_s_end_pos["s_end:" + str(token[-1])] = 1
        d_s_start_pos_list.append(copy.deepcopy(d_s_start_pos))
        d_s_end_pos_list.append(copy.deepcopy(d_s_end_pos))
        d_s_start_pos.clear()
        d_s_end_pos.clear()
    return pd.DataFrame(d_s_start_pos_list), pd.DataFrame(d_s_end_pos_list)


# Feature extraction of the feature described in chapter 5.6.13
def get_flesch_reading_ease_vector(par_df):
    arr_fre_vector = np.zeros((len(par_df), 1))
    textstat.set_lang("de")
    for index, row in par_df.iterrows():
        arr_fre_vector[index, 0] = textstat.flesch_reading_ease(row['text'])
    return pd.DataFrame(data=arr_fre_vector[:, 0], columns=['fre'])


# Encode the authors as described in chapter 6.9
# dictionary for the mapping of author and label
def author_encoding(par_df):
    dic_author_mapping = {}
    number = 1
    for index, row in par_df.iterrows():
        if row['author'] not in dic_author_mapping:
            dic_author_mapping[row['author']] = number
            number += 1
    return dic_author_mapping


# get vector with all author labels in the DataFrame
def get_encoded_author_vector(par_df, par_dic_author_mapping):
    arr_encoded_author = np.zeros((len(par_df), 1))
    for index, row in par_df.iterrows():
        arr_encoded_author[index] = par_dic_author_mapping.get(row['author'])
    return arr_encoded_author


# Additional length metrics for the calculation of the individual relative frequency in chapter 7.1.4
# number of character in the document
def get_char_count(par_df):
    list_text_char_count = []
    for index, row in par_df.iterrows():
        list_text_char_count.append(len(row['text']))
    return pd.DataFrame(data=list_text_char_count, columns=['char_count'])


# number of sentences in the document
def get_sentence_count(par_df):
    list_sentence_count = []
    for index, row in par_df.iterrows():
        # nlp has a much more efficient sentence parsing as described in chapter 4.1
        list_sentence_count.append(len(sent_tokenize(row['text'], language='german')))
    return pd.DataFrame(data=list_sentence_count, columns=['sentence_count'])


# Method for the main extraction. Extract all the features of a given DataFrame and print them to a csv
def print_data_to_csv(par_df):
    # Length Metrics
    df_text_length_metrics = pd.concat([get_char_count(par_df), get_sentence_count(par_df),
                                        get_word_count(par_df)], axis=1)
    df_text_length_metrics.to_csv(f"daten/1_raw/length_metrics.csv", index=False)

    # BOW chapter 5.6.1
    get_bow_matrix(par_df).to_csv("daten/1_raw/bow.csv", index=False)

    # Word n-grams chapter 5.6.2
    for n in range(2, 7):
        get_word_n_grams(par_df, n).to_csv(f"daten/1_raw/word_{n}_gram.csv", index=False)

    # Word count chapter 5.6.3
    get_word_count(par_df).to_csv(f"daten/1_raw/word_count.csv", index=False)

    # Word length chapter 5.6.4
    get_word_length_matrix(par_df).to_csv(f"daten/1_raw/word_length.csv", index=False)
    # Average word length
    get_average_word_length(par_df).to_csv(f"daten/1_raw/average_word_length.csv", index=False)
    # Word length with bins
    get_word_length_matrix_with_interval(par_df, 20, 30).to_csv(f"daten/1_raw/word_length_bin.csv", index=False)

    # Yules K chapter 5.6.5
    get_yules_k(par_df).to_csv(f"daten/1_raw/yules_k.csv", index=False)

    # Special Char chapter 5.6.6
    sc_label_vector = ["!", "„", "“", "§", "$", "%", "&", "/", "(", ")", "=", "?", "{", "}", "[", "]", "\\", "@", "#",
                       "‚", "‘", "-", "_", "+", "*", ".", ",", ";"]
    special_char_matrix = get_special_char_matrix(par_df, sc_label_vector)
    sc_label_vector = ["s_char:" + sc for sc in sc_label_vector]
    df_special_char = pd.DataFrame(data=special_char_matrix, columns=sc_label_vector)
    df_special_char.to_csv(f"daten/1_raw/special_char.csv", index=False)

    # Char n-grams chapter 5.6.7
    for n in range(2, 6):
        get_char_affix_n_grams(par_df, n).to_csv(f"daten/1_raw/char_affix_{n}_gram.csv", index=False)
        get_char_word_n_grams(par_df, n).to_csv(f"daten/1_raw/char_word_{n}_gram.csv", index=False)
        get_char_punct_n_grams(par_df, n).to_csv(f"daten/1_raw/char_punct_{n}_gram.csv", index=False)

    # Digits chapter 5.6.8
    get_digits(par_df).to_csv(f"daten/1_raw/digits.csv", index=False)

    # Function Words chapter 5.6.9
    get_function_words(par_df).to_csv(f"daten/1_raw/function_words.csv", index=False)

    # Pos Tags chapter 5.6.10
    get_pos_tags(par_df).to_csv(f"daten/1_raw/pos_tags.csv", index=False)

    # Pos Tag n-grams chapter 5.6.11
    for n in range(2, 6):
        get_pos_tags_n_grams(par_df, n).to_csv(f"daten/1_raw/pos_tag_{n}_gram.csv", index=False)

    # Sentence start/end Pos chapter 5.6.12
    df_start_pos, df_end_pos = get_sentence_end_start(par_df)
    pd.concat([df_start_pos, df_end_pos], axis=1).to_csv(f"daten/1_raw/pos_tag_start_end.csv", index=False)

    # FRE chapter 5.6.13
    get_flesch_reading_ease_vector(par_df).to_csv(f"daten/1_raw/fre.csv", index=False)


# Method to get the time between the first and the last written article of an author. Used in chapter 5.1
def get_time_span_by_label(par_df):
    par_df['release_date'] = pd.to_datetime(par_df['release_date'], format='%d.%m.%Y', errors='ignore')
    min_date = par_df.groupby(['label_encoded']).min()['release_date']
    max_date = par_df.groupby(['label_encoded']).max()['release_date']
    df_test = pd.DataFrame(data=min_date.values, columns=['from_date'])
    df_test['to_date'] = max_date.values
    df_test['diff'] = (df_test['to_date'] - df_test['from_date'])/np.timedelta64(1, 'D')
    for index, row in df_test.iterrows():
        years, months, days = time_format(row['diff'])
        print(f"{index+1}. Jahre: {years}, Monate {months}, Tage: {days}")

    years, months, days = time_format(df_test['diff'].values.mean())
    print(f"Durchschnittlicher Zeitraum = Jahre: {years}, Monate {months}, Tage: {days}")


# Parse the time format from days to a readable format years, months, days
def time_format(par_days):
    years = par_days / 365
    # get the number of months, by multiply the number after the dot by 365 and divide by 30.
    months = (years - int(years)) * 365 / 30
    # accordingly, the number after the dot * 30 for the days
    days = int((months - int(months)) * 30)
    years = int(years)
    months = int(months)
    return years, months, days


# The following calculations were used in the chapters for the analysis of the data.
# They are all commented to be able to use them separate

# Global options to see all lines and columns of a printed DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# # # Create the balanced Corpus with 100 article from 25 authors
# chapter 5.1
# df_reviews = pd.read_csv("musikreviews.csv", sep=',', encoding="ansi")
# df_balanced = get_balanced_df_all_authors(df_reviews, 100, 25)
# df_balanced.to_csv("musikreviews_balanced_authors.csv", index=False)


# Read the balanced DataFrame with 25 authors with 100 texts
df_reviews = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8", nrows=2500)

# # # Analysis of the features # # #

# Chapter 5.1: Average word count of all article
"""df_word_count = get_word_count(df_reviews)
print(df_word_count.mean())"""


# Chapter 5.1: Count genres by author
"""original_stdout = sys.stdout
with open('testoutput.txt', 'w', encoding="ansi") as f:
    sys.stdout = f
    print(df_reviews['genres'].value_counts())
    sys.stdout = original_stdout"""

# Chapter 5.6.1 BoW
# Get the count of the top 5 used word per author in average, data for table 2
"""df_bow = get_bow_matrix(df_reviews)
df_bow = df_bow.fillna(value=0)
df_bow['label_encoded'] = df_reviews['label_encoded']
print(df_bow.shape)
for i in range(1, 26):
    avg_values = df_bow.groupby(['label_encoded']).mean().loc[i, :].sort_values(ascending=False)
    print(avg_values[0:5])"""

# Chapter 5.6.2 word-n-grams
# Get the data for table 3: Count of n-grams, top 5 n-grams by sum and count of n-grams with sum > 1
"""for i in range(2, 7):
    word_n_grams = get_word_n_grams(df_reviews, i)
    print(f"len N{i}: {len(word_n_grams.columns)}")
    sums = word_n_grams.sum().sort_values(ascending=False)
    print(sums[0:5])
    print(f"N{i} > 1: {len([i for i in sums if i > 1])}")"""

# Chapter 5.6.3 Word count
# Get the average Text_length scaled and unscaled for picture 2
"""df_word_count = get_word_count(df_reviews)
scaler = StandardScaler()
scaler.fit(df_word_count)
word_count_vector_scaled = scaler.transform(df_word_count)
df_text_length = pd.DataFrame()
df_text_length['label_encoded'] = df_reviews['label_encoded']
df_text_length["word_count"] = df_word_count['word_count']
df_text_length["word_count_scaled"] = word_count_vector_scaled[:, 0]
print(df_text_length.groupby(['label_encoded']).mean())"""

# Chapter 5.6.4 word length
# Get the data for picture 3, sum of words with a margin of >20
# print(get_word_length_matrix_with_margin(df_reviews, 20).sum())

# Chapter 5.6.5 Yules K
# Get the data for picture 4, average Yules K for every author
"""df_yule = get_yules_k(df_reviews)
df_yule['label_encoded'] = df_reviews['label_encoded']
print(df_yule.groupby(['label_encoded']).mean())"""

# Get the data for picture 5, all article by label and Yules k
"""df_reviews["yules_k"] = get_yules_k(df_reviews)['yulesk']
# Get all values for Yules K
original_stdout = sys.stdout
with open('yules_k_all_normalized.csv', 'w') as f:
    sys.stdout = f
    for index, row in df_reviews.iterrows():
        print(f"{row['label_encoded']}, {row['yules_k']}")
    sys.stdout = original_stdout"""

# Chapter 5.6.6 Special char
# define the special chars as in the chapter
"""sc_label_vector = ["!", "„", "“", "§", "$", "%", "&", "/", "(", ")", "=", "?", "{", "}", "[", "]", "\\", "@", "#", 
                      "‚", "‘", "-", "_", "+", "*", ".", ",", ";"]
# get the data for table 4: average count of special char per author
special_char_matrix = get_special_char_matrix(df_reviews, sc_label_vector)
df_special_char_scaled = pd.DataFrame(data=special_char_matrix, columns=sc_label_vector)
df_special_char_scaled['label_encoded'] = df_reviews['label_encoded']
print(df_special_char_scaled.groupby(['label_encoded']).mean())"""

# Chapter 5.6.7 char-n-grams
# Get the data for table 5, top 5 ngrams, sum of ngrams, sum all ngrams for n from 2-5 (inclusive)
"""for i in range(2, 6):
    df_char_affix_grams = get_char_affix_n_grams(df_reviews, i)
    df_char_word_grams = get_char_word_n_grams(df_reviews, i)
    df_char_punct_grams = get_char_punct_n_grams(df_reviews, i)

    df_char_grams = pd.concat([df_char_affix_grams, df_char_word_grams, df_char_punct_grams], axis=1)

    print(f"n{i} all:")
    print(df_char_grams.mean().sort_values(ascending=False)[0:5])
    print(f"n{i} affix:")
    print(df_char_affix_grams.mean().sort_values(ascending=False)[0:5])
    print(f"n{i} word:")
    print(df_char_word_grams.mean().sort_values(ascending=False)[0:5])
    print(f"n{i} punct:")
    print(df_char_punct_grams.mean().sort_values(ascending=False)[0:5])

    print(f"n{i}  Gesamt: {df_char_grams.count(axis=1)[0]}")
    print(f"n{i}a: {df_char_affix_grams.count(axis=1)[0]}")
    print(f"n{i}w: {df_char_word_grams.count(axis=1)[0]}")
    print(f"n{i}p: {df_char_punct_grams.count(axis=1)[0]}")"""

# Chapter 5.6.8 digits
# Get the data for table 6, average count of the digits for every author
"""df_digit = get_digits(df_reviews).fillna(value=0)
df_digit['label_encoded'] = df_reviews['label_encoded']
print(df_digit.groupby(['label_encoded']).mean())"""

# Chapter 5.6.9 Function Words
# Get the data for table 7, count of top 5 function words per author
"""df_fwords = get_function_words(df_reviews)
print(df_fwords.shape)
df_fwords['label_encoded'] = df_reviews['label_encoded']
for i in range(1, 26):
    test = df_fwords.groupby(['label_encoded']).mean().loc[i, :].sort_values(ascending=False)
    print(test[0:5])"""

# Chapter 5.6.10 PoS-Tags
# Get the data for table 6, average count of the PoS-Tags for every author
"""df_pos_tags = get_pos_tags(df_reviews).fillna(value=0)
df_pos_tags['label_encoded'] = df_reviews['label_encoded']
print(df_pos_tags.groupby(['label_encoded']).mean())"""

# Chapter 5.6.11 PoS-Tag-n-grams
# Get data for table 9, top 5 ngrams, count of ngrams for n from 2-5 (included)
"""for n in range(2, 6):
    pos_tags_n_grams = get_pos_tags_n_grams(df_reviews, n)
    print(f"N{n} Länge: {pos_tags_n_grams.count(axis=1)[0]}")
    print(pos_tags_n_grams.mean().sort_values(ascending=False)[0:5])"""


# Chapter 5.6.12 Pos Sentence Start End
# Get the data for table 10, count of all pos sentence start and end per author
"""df_start_pos, df_end_pos = get_sentence_end_start(df_reviews)
df_start_pos = df_start_pos.fillna(value=0)
df_end_pos = df_end_pos.fillna(value=0)
df_start_pos['label_encoded'] = df_reviews['label_encoded']
print(df_start_pos.groupby(['label_encoded']).mean())
df_end_pos['label_encoded'] = df_reviews['label_encoded']
print(df_end_pos.groupby(['label_encoded']).mean())"""

# Chapter 5.6.13 FRE
"""df_fre = get_flesch_reading_ease_vector(df_reviews)
# Get the min max and average FRE
print(f"avg: {df_fre.mean()}")
print(f"max: {df_fre.max()}")
print(f"min: {df_fre.min()}")

# Get the data for picture 6, average FRE per author
df_fre['label_encoded'] = df_reviews['label_encoded']
print(df_fre.groupby(['label_encoded']).mean())"""
