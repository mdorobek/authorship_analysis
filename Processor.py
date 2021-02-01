import pandas as pd
import numpy as np
import copy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from os import listdir
from os.path import isfile, join
import sys
import math
from sklearn.metrics import accuracy_score, f1_score
import re
from Extractor import get_word_length_matrix, get_word_length_matrix_with_interval, get_average_word_length, \
    get_word_length_matrix_with_margin, get_char_count, get_digits, get_sum_digits, get_word_n_grams, \
    get_char_affix_n_grams, get_char_word_n_grams, get_char_punct_n_grams, get_pos_tags_n_grams, get_bow_matrix, \
    get_yules_k, get_special_char_matrix, get_function_words, get_pos_tags, get_sentence_end_start, \
    get_flesch_reading_ease_vector, get_sentence_count, get_word_count

from sklearn.preprocessing import StandardScaler, Normalizer


# Chapter 7.1.1. method to trim a feature with low sum e.g. ngrams lower then 5
def trim_df_sum_feature(par_df, par_n):
    par_df = par_df.fillna(value=0)
    columns = par_df.columns.to_numpy()
    data_array = par_df.to_numpy(dtype=float)
    sum_arr = data_array.sum(axis=0)
    # reduce n if 0 features would be returned
    while len(par_df.columns) - len(np.where(sum_arr < par_n)[0]) == 0:
        par_n -= 1
    positions = list(np.where(sum_arr < par_n))
    columns = np.delete(columns, positions)
    data_array = np.delete(data_array, positions, axis=1)
    return pd.DataFrame(data=data_array, columns=columns)


# Chapter 7.1.1. method to trim feature with low occurrence over all article
def trim_df_by_occurrence(par_df, n):
    df_masked = par_df.notnull().astype('int')
    word_rate = df_masked.sum()
    columns = []
    filtered_bow = pd.DataFrame()
    for i in range(0, len(word_rate)):
        if word_rate[i] > n:
            columns.append(word_rate.index[i])
    for c in columns:
        filtered_bow[c] = par_df[c]
    return filtered_bow


# Chapter 7.1.1. Process of filtering the data with low occurrence and save the filtered features in a new file
def filter_low_occurrence():
    df_bow = pd.read_csv("daten/raw/bow.csv", sep=',', encoding="utf-8", nrows=2500)
    print(f"BOW before: {len(df_bow.columns)}")
    df_bow = trim_df_by_occurrence(df_bow, 1)
    print(f"BOW after: {len(df_bow.columns)}")
    df_bow.to_csv(f"daten/2_filter_low_occurrence/bow.csv", index=False)

    for n in range(2, 7):
        word_n_gram = pd.read_csv(f"daten/raw/word_{n}_gram.csv", sep=',', encoding="utf-8", nrows=2500)
        print(f"Word_{n}_gram before: {len(word_n_gram.columns)}")
        word_n_gram = trim_df_by_occurrence(word_n_gram, 1)
        print(f"Word_{n}_gram after: {len(word_n_gram.columns)}")
        word_n_gram.to_csv(f"daten/2_filter_low_occurrence/word_{n}_gram.csv", index=False)

    for n in range(2, 6):
        char_affix_n_gram = pd.read_csv(f"daten/trimmed_occ_greater_one/char_affix_{n}_gram_1.csv", sep=',',
                                        encoding="utf-8", nrows=2500)
        print(f"char_affix_{n}_gram before: {len(char_affix_n_gram.columns)}")
        char_affix_n_gram = trim_df_sum_feature(char_affix_n_gram, 5)
        print(f"char_affix_{n}_gram after: {len(char_affix_n_gram.columns)}")
        char_affix_n_gram.to_csv(f"daten/2_filter_low_occurrence/char_affix_{n}_gram.csv", index=False)

        char_word_n_gram = pd.read_csv(f"daten/trimmed_occ_greater_one/char_word_{n}_gram_1.csv", sep=',',
                                       encoding="utf-8", nrows=2500)
        print(f"char_word_{n}_gram before: {len(char_word_n_gram.columns)}")
        char_word_n_gram = trim_df_sum_feature(char_word_n_gram, 5)
        print(f"char_word_{n}_gram after: {len(char_word_n_gram.columns)}")
        char_word_n_gram.to_csv(f"daten/2_filter_low_occurrence/char_word_{n}_gram.csv", index=False)

        char_punct_n_gram = pd.read_csv(f"daten/trimmed_occ_greater_one/char_punct_{n}_gram_1.csv", sep=',',
                                        encoding="utf-8", nrows=2500)
        print(f"char_punct_{n}_gram before: {len(char_punct_n_gram.columns)}")
        char_punct_n_gram = trim_df_sum_feature(char_punct_n_gram, 5)
        print(f"char_punct_{n}_gram after: {len(char_punct_n_gram.columns)}")
        char_punct_n_gram.to_csv(f"daten/2_filter_low_occurrence/char_punct_{n}_gram.csv", index=False)

    df_f_word = pd.read_csv("daten/raw/function_words.csv", sep=',', encoding="utf-8", nrows=2500)
    print(f"Function Words before: {len(df_f_word.columns)}")
    df_f_word = trim_df_by_occurrence(df_f_word, 1)
    print(f"Function Words after: {len(df_f_word.columns)}")
    df_f_word.to_csv(f"daten/2_filter_low_occurrence/function_words.csv", index=False)

    for n in range(2, 6):
        pos_tags_n_gram = pd.read_csv(f"daten/raw/pos_tag_{n}_gram.csv", sep=',', encoding="utf-8", nrows=2500)
        print(f"pos_tag_{n}_gram before: {len(pos_tags_n_gram.columns)}")
        pos_tags_n_gram = trim_df_by_occurrence(pos_tags_n_gram, 1)
        print(f"pos_tag_{n}_gram after: {len(pos_tags_n_gram.columns)}")
        pos_tags_n_gram.to_csv(f"daten/2_filter_low_occurrence/pos_tag_{n}_gram.csv", index=False)


# Chapter 7.1.2. method to filter words based on document frequency
def trim_df_by_doc_freq(par_df, par_doc_freq):
    df_masked = par_df.notnull().astype('int')
    word_rate = df_masked.sum() / len(par_df)
    columns = []
    filtered_bow = pd.DataFrame()
    for i in range(0, len(word_rate)):
        if word_rate[i] < par_doc_freq:
            columns.append(word_rate.index[i])
    for c in columns:
        filtered_bow[c] = par_df[c]
    return filtered_bow


# Chapter 7.1.2 Process of filtering the data with high document frequency and save the filtered features in a new file
def filter_high_document_frequency():
    # Filter words with high document frequency
    df_bow = pd.read_csv("daten/2_filter_low_occurrence/bow.csv", sep=',', encoding="utf-8", nrows=2500)
    print(f"BOW before: {len(df_bow.columns)}")
    df_bow = trim_df_by_doc_freq(df_bow, 0.5)
    print(f"BOW after: {len(df_bow.columns)}")
    df_bow.to_csv(f"daten/3_fiter_high_frequency/bow.csv", index=False)

    df_f_word = pd.read_csv("daten/2_filter_low_occurrence/function_words.csv", sep=',', encoding="utf-8", nrows=2500)
    print(f"Function Word before: {len(df_f_word.columns)}")
    df_f_word = trim_df_by_doc_freq(df_f_word, 0.5)
    print(f"Function Word after: {len(df_f_word.columns)}")
    df_f_word.to_csv(f"daten/3_fiter_high_frequency/function_words.csv", index=False)

    for n in range(2, 7):
        word_n_gram = pd.read_csv(f"daten/2_filter_low_occurrence/word_{n}_gram.csv", sep=',', encoding="utf-8",
                                  nrows=2500)
        print(f"Word_{n}_gram before: {len(word_n_gram.columns)}")
        word_n_gram = trim_df_by_doc_freq(word_n_gram, 0.5)
        print(f"Word_{n}_gram after: {len(word_n_gram.columns)}")
        word_n_gram.to_csv(f"daten/3_fiter_high_frequency/word_{n}_gram.csv", index=False)


# Chapter 7.1.4. get the relative frequency based on a length metric (char, word, sentence)
def get_rel_frequency(par_df_count, par_df_len_metric_vector):
    df_rel_freq = pd.DataFrame(columns=par_df_count.columns)
    for index, row in par_df_count.iterrows():
        df_rel_freq = df_rel_freq.append(row.div(par_df_len_metric_vector[index]))
    return df_rel_freq


# Chapter 7.1.4. whole process of the chapter. Get the individual relative frequency of a feature and compare
# the correlation to the article length from the absolute and relative feature, save the feature with the estimated
# relative frequency in a new file
def individual_relative_frequency():
    df_len_metrics = pd.read_csv(f"daten/1_raw/length_metrics.csv", sep=',', encoding="utf-8", nrows=2500)
    # different metrics for individual relative frequencies
    metrics = ['word_count', 'char_count', 'sentence_count']

    for m in metrics:
        # The csv is placed in a folder based on the metric for the individual relative frequency
        path = f'daten/4_relative_frequency/{m}'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for f in files:
            x = pd.read_csv(f"daten/4_relative_frequency/{m}/{f}",
                            sep=',', encoding="utf-8", nrows=2500).fillna(value=0)
            x_rel = get_rel_frequency(x, df_len_metrics[m])

            # Save the CSV with relative frequency
            x_rel.to_csv(
                f"daten/4_relative_frequency/{f.split('.')[0]}"
                f"_rel.csv", index=False)

            # Correlation is always between the metrics and the word_count
            x['word_count'] = df_len_metrics['word_count']
            x_rel['word_count'] = df_len_metrics['word_count']

            # only on the test data 60/40 split
            x_train, x_test = train_test_split(x, test_size=0.4, random_state=42)
            x_train_rel, x_test_rel = train_test_split(x_rel, test_size=0.4, random_state=42)

            # Calculate the median correlation
            print(f"{f}_abs: {x_train.corr(method='pearson', min_periods=1)['word_count'].iloc[:-1].mean()}")
            print(f"{f}_rel: {x_train_rel.corr(method='pearson', min_periods=1)['word_count'].iloc[:-1].mean()}")


# Chapter 7.2.1 First step of the iterative filter: Rank the features
def sort_features_by_score(par_x, par_y, par_select_metric):
    # Get a sorted ranking of all features by the selected metric
    selector = SelectKBest(par_select_metric, k='all')
    selector.fit(par_x, par_y)
    # Sort the features by their score
    return pd.DataFrame(dict(feature_names=par_x.columns, scores=selector.scores_)).sort_values('scores',
                                                                                                ascending=False)


# Chapter 7.2.1 method to get the best percentile for GNB
def get_best_percentile_gnb(par_x_train, par_y_train, par_iter, par_df_sorted_features, step):
    result_list = []
    gnb = GaussianNB()
    best_perc_round = par_iter - 1  # If no other point is found, highest amount of features (-1 starts to count from 0)
    # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
    if len(par_y_train.index) / len(np.unique(par_y_train.values).tolist()) < 10:
        cv = int(len(par_y_train.index) / len(np.unique(par_y_train.values).tolist())) - 1
    else:
        cv = 10
    for perc_features in np.arange(step, par_iter + 1, step):
        start_time = datetime.now()
        # 1%*i best features to keep and create new dataframe with those only
        number_of_features = int(perc_features * (len(par_x_train.columns) / 100))
        # minimum one feature
        number_of_features = 1 if number_of_features < 1 else number_of_features
        feature_list = par_df_sorted_features['feature_names'][: number_of_features].tolist()
        x_new_training = copy.deepcopy(par_x_train[feature_list])

        # GNB Training
        result_list.append(
            cross_val_score(gnb, x_new_training, par_y_train, cv=cv, n_jobs=-1, scoring='accuracy').mean())
        # Compares the accuracy with the 5 following points => needs 6 points minimum
        if len(result_list) > 5:
            # list starts to count at 0, subtract one more from len
            difference_list_p2p = [result_list[p + 1] - result_list[p] for p in
                                   range(len(result_list) - 6, len(result_list) - 1)]
            difference_list_1p = [result_list[p + 1] - result_list[len(result_list) - 6] for p in
                                  range(len(result_list) - 6, len(result_list) - 1)]
            # Find the best percent if 5 following points were lower then the point before or had a deviation <= 0.5%
            # or all points are 2% lower then the first point
            if all(point_y <= 0 for point_y in difference_list_p2p) or \
                    all(-0.005 <= point_y <= 0.005 for point_y in difference_list_1p) or \
                    all(point_y < -0.02 for point_y in difference_list_1p):
                # the best perc is the results - 6 point in the result list
                best_perc_round = len(result_list) - 6
                break

        # Console Output
        print(f"GNB Round {perc_features / step}: {datetime.now() - start_time}")

    # Optimization of the best percent
    # If any point with a lower percent is higher, it is the new optimum
    if any(point_y > result_list[best_perc_round] for point_y in result_list[:len(result_list) - 5]):
        best_perc_round = result_list.index(max(result_list[:len(result_list) - 5]))

    # Tradeoff of 0.5% accuracy for lesser percent of features
    # As long as there is a lesser maximum with 1% lesser accuracy, which has a minimum of 2% less percent features
    better_perc_exists = True
    best_accuracy_tradeoff = result_list[best_perc_round] - 0.01
    # If there are no 5% left for the tradeoff there is no better perc
    if best_perc_round - int(2 / step) < 0:
        better_perc_exists = False
    while better_perc_exists:
        earliest_pos = best_perc_round - int(2 / step)
        # if its less then 0 it starts to count backside
        earliest_pos = 0 if earliest_pos < 0 else earliest_pos
        if any(point_y > best_accuracy_tradeoff for point_y in result_list[:earliest_pos]):
            best_perc_round = result_list.index(max(result_list[:earliest_pos]))
        else:
            better_perc_exists = False

    # the best percent of the features is calculated by the percent start plus the rounds * step
    best_perc = step + step * best_perc_round
    print(best_perc)
    return best_perc, best_perc_round, result_list


# Chapter 7.2.1 method to get the best percentile for SVC
def get_best_percentile_svc(par_x_train, par_y_train, par_iter, par_df_sorted_features, step):
    result_list = []
    # Parameter for SVC
    param_grid_svc = {'C': (0.001, 0.01, 0.1, 1, 10),
                      'kernel': ('linear', 'poly', 'rbf'),
                      'gamma': ('scale', 'auto')}
    # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
    if len(par_y_train.index) / len(np.unique(par_y_train.values).tolist()) < 10:
        cv = int(len(par_y_train.index) / len(np.unique(par_y_train.values).tolist())) - 1
    else:
        cv = 10
    best_perc_round = par_iter - 1  # If no other point is found, highest amount of features (-1 starts to count from 0)
    for perc_features in np.arange(step, par_iter + 1, step):
        start_time = datetime.now()
        # 1%*i best features to keep and create new dataframe with those only
        number_of_features = int(perc_features * (len(par_x_train.columns) / 100))
        # minimum one feature
        number_of_features = 1 if number_of_features < 1 else number_of_features
        feature_list = par_df_sorted_features['feature_names'][: number_of_features].tolist()
        x_new_training = copy.deepcopy(par_x_train[feature_list])

        # SVC Test
        grid_search = GridSearchCV(svm.SVC(), param_grid_svc, cv=cv, n_jobs=-1, scoring='accuracy')
        grid_results = grid_search.fit(x_new_training, par_y_train)
        result_list.append(grid_results.best_score_)

        # Compares the accuracy with the 5 following points => needs 6 points minimum
        if len(result_list) > 5:
            # list starts to count at 0, subtract one more from len
            difference_list_p2p = [result_list[p + 1] - result_list[p] for p in
                                   range(len(result_list) - 6, len(result_list) - 1)]
            difference_list_1p = [result_list[p + 1] - result_list[len(result_list) - 6] for p in
                                  range(len(result_list) - 6, len(result_list) - 1)]
            # Find the best percent if 5 following points were lower then the point before or had a deviation <= 0.5%
            # or all points are 2% lower then the first point
            if all(point_y <= 0 for point_y in difference_list_p2p) or \
                    all(-0.005 <= point_y <= 0.005 for point_y in difference_list_1p) or \
                    all(point_y < -0.02 for point_y in difference_list_1p):
                # the best perc is the results - 6 point in the result list
                best_perc_round = len(result_list) - 6
                break

        # Console Output
        print(f"SVC Round {perc_features / step}: {datetime.now() - start_time}")

    # Optimization of the best percent
    # If any point with a lower percent is higher, it is the new optimum
    if any(point_y > result_list[best_perc_round] for point_y in result_list[:len(result_list) - 5]):
        best_perc_round = result_list.index(max(result_list[:len(result_list) - 5]))

    # Tradeoff of 1% accuracy for lesser percent of features
    # As long as there is a lesser maximum with 1% lesser accuracy, which has a minimum of 2% less percent features
    better_perc_exists = True
    best_accuracy_tradeoff = result_list[best_perc_round] - 0.01
    # If there are no 5% left for the tradeoff there is no better perc
    if best_perc_round - int(2 / step) < 0:
        better_perc_exists = False
    while better_perc_exists:
        earliest_pos = best_perc_round - int(2 / step)
        # if its less then 0 it starts to count backside
        earliest_pos = 0 if earliest_pos < 0 else earliest_pos
        if any(point_y > best_accuracy_tradeoff for point_y in result_list[:earliest_pos]):
            best_perc_round = result_list.index(max(result_list[:earliest_pos]))
        else:
            better_perc_exists = False

    # the best percent of the features is calculated by the percent start plus the rounds * step
    best_perc = step + step * best_perc_round
    print(best_perc)
    return best_perc, best_perc_round, result_list


# Chapter 7.2.1 method to get the best percentile for KNN
def get_best_percentile_knn(par_x_train, par_y_train, par_iter, par_df_sorted_features, step):
    result_list = []
    best_perc_round = par_iter - 1  # If no other point is found, highest amount of features (-1 starts to count from 0)
    # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
    if len(par_y_train.index) / len(np.unique(par_y_train.values).tolist()) < 10:
        cv = int(len(par_y_train.index) / len(np.unique(par_y_train.values).tolist())) - 1
    else:
        cv = 10
    for perc_features in np.arange(step, par_iter + 1, step):
        start_time = datetime.now()
        # 1%*i best features to keep and create new dataframe with those only
        number_of_features = int(perc_features * (len(par_x_train.columns) / 100))
        # minimum one feature
        number_of_features = 1 if number_of_features < 1 else number_of_features
        feature_list = par_df_sorted_features['feature_names'][: number_of_features].tolist()
        x_new_training = copy.deepcopy(par_x_train[feature_list])

        # Parameter for KNN
        # Some Values from 3 to square of samples
        neighbors = [i for i in range(3, int(math.sqrt(len(x_new_training.index))), 13)]
        neighbors += [1, 3, 5, 11, 19, 36]
        if int(math.sqrt(len(feature_list))) not in neighbors:
            neighbors.append(int(math.sqrt(len(x_new_training.index))))
        # Not more neighbors then samples-2
        neighbors = [x for x in neighbors if x < len(x_new_training.index) - 2]
        # remove duplicates
        neighbors = list(set(neighbors))
        param_grid_knn = {'n_neighbors': neighbors,
                          'weights': ['uniform', 'distance'],
                          'metric': ['euclidean', 'manhattan']}

        # KNN Training
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, n_jobs=-1, scoring='accuracy')
        grid_results = grid_search.fit(x_new_training, par_y_train)
        result_list.append(grid_results.best_score_)

        # Compares the accuracy with the 5 following points => needs 6 points minimum
        if len(result_list) > 5:
            # list starts to count at 0, subtract one more from len
            difference_list_p2p = [result_list[p + 1] - result_list[p] for p in
                                   range(len(result_list) - 6, len(result_list) - 1)]
            difference_list_1p = [result_list[p + 1] - result_list[len(result_list) - 6] for p in
                                  range(len(result_list) - 6, len(result_list) - 1)]
            # Find the best percent if 5 following points were lower then the point before or had a deviation <= 0.5%
            # or all points are 2% lower then the first point
            if all(point_y <= 0 for point_y in difference_list_p2p) or \
                    all(-0.005 <= point_y <= 0.005 for point_y in difference_list_1p) or \
                    all(point_y < -0.02 for point_y in difference_list_1p):
                # the best perc is the results - 6 point in the result list
                best_perc_round = len(result_list) - 6
                break

        # Console Output
        print(f"KNN Round {perc_features / step}: {datetime.now() - start_time}")

    # Optimization of the best percent
    # If any point with a lower percent is higher, it is the new optimum
    if any(point_y > result_list[best_perc_round] for point_y in result_list[:len(result_list) - 5]):
        best_perc_round = result_list.index(max(result_list[:len(result_list) - 5]))

    # Tradeoff of 1% accuracy for lesser percent of features
    # As long as there is a lesser maximum with 1% lesser accuracy, which has a minimum of 2% less percent features
    better_perc_exists = True
    best_accuracy_tradeoff = result_list[best_perc_round] - 0.01
    # If there are no 5% left for the tradeoff there is no better perc
    if best_perc_round - int(2 / step) < 0:
        better_perc_exists = False
    while better_perc_exists:
        earliest_pos = best_perc_round - int(2 / step)
        # if its less then 0 it starts to count backside
        earliest_pos = 0 if earliest_pos < 0 else earliest_pos
        if any(point_y >= best_accuracy_tradeoff for point_y in result_list[:earliest_pos]):
            best_perc_round = result_list.index(max(result_list[:earliest_pos]))
        else:
            better_perc_exists = False

    # the best percent of the features is calculated by the percent start plus the rounds * step
    best_perc = step + step * best_perc_round
    print(best_perc)
    return best_perc, best_perc_round, result_list


# Chapter 7.2.1 Filter the feature based on the estimated best percentile and save it into a new file
def print_filter_feature_percentile(par_path, par_df_sorted_features, par_percent, par_x, par_file_name):
    # select the 1 percent of the features (len/100) multiplied by par_best_percent
    number_features = round(par_percent * (len(par_x.columns) / 100))
    # If the 1st percent is less then 1
    number_features = 1 if number_features < 1 else number_features
    feature_list = par_df_sorted_features['feature_names'][:number_features].tolist()

    # print the name of the features in a file
    original_stdout = sys.stdout
    with open(f'{par_path}selected_features/{par_file_name}_filtered.txt', 'w', encoding="utf-8") as f:
        sys.stdout = f
        print(f"Features: {len(feature_list)}")
        print(f"{feature_list}")
        sys.stdout = original_stdout

    # select the best features from the original dataset
    par_x[feature_list].to_csv(f"{par_path}csv_after_filter/{par_file_name}_filtered.csv", index=False)


# Chapter 7.2.1 Complete process of the iterative Filter
def iterative_filter_process(par_path, par_df, par_num_texts, par_num_authors):
    y = par_df['label_encoded']
    path = f'{par_path}csv_before_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # Filter the files for author and text numbers if 'all' is not set.
    if par_num_authors != "all":
        r = re.compile(f"a{par_num_authors}_")
        files = list(filter(r.match, files))
    if par_num_authors != "all":
        r = re.compile(f".*t{par_num_texts}_")
        files = list(filter(r.match, files))
    step_perc = 1.0
    for f in files:
        filename = f.split(".")[0]
        print(f)
        x = pd.read_csv(f"{par_path}csv_before_filter/{f}", sep=',', encoding="utf-8", nrows=2500)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
        # Get sorted features
        df_sorted_features = sort_features_by_score(x_train, y_train, mutual_info_classif)
        # Calculate the best percentiles of the data for the different classifier
        best_perc_gnb, best_round_gnb, result_list_gnb = get_best_percentile_gnb(x_train, y_train, 50,
                                                                                 df_sorted_features, step_perc)
        best_perc_svc, best_round_svc, result_list_svc = get_best_percentile_svc(x_train, y_train, 50,
                                                                                 df_sorted_features, step_perc)
        best_perc_knn, best_round_knn, result_list_knn = get_best_percentile_knn(x_train, y_train, 50,
                                                                                 df_sorted_features, step_perc)

        # select the beast features from the original dataset
        print_filter_feature_percentile(par_path, df_sorted_features, best_perc_gnb, x, "gnb_" + filename)
        print_filter_feature_percentile(par_path, df_sorted_features, best_perc_svc, x, "svc_" + filename)
        print_filter_feature_percentile(par_path, df_sorted_features, best_perc_knn, x, "knn_" + filename)

        # print best perc to a file
        original_stdout = sys.stdout
        with open(f'{par_path}best_perc/{filename}.txt', 'w') as f:
            sys.stdout = f
            print(f"best_perc_gnb: ({best_perc_gnb}|{result_list_gnb[best_round_gnb]})\n"
                  f"best_perc_svc: ({best_perc_svc}|{result_list_svc[best_round_svc]})\n"
                  f"best_perc_knn: ({best_perc_knn}|{result_list_knn[best_round_knn]})")
            sys.stdout = original_stdout

        # draw diagram
        len_list = [len(result_list_gnb), len(result_list_svc), len(result_list_knn)]
        plt.plot([i * step_perc for i in range(1, len(result_list_gnb) + 1)], result_list_gnb, 'r-', label="gnb")
        plt.plot(best_perc_gnb, result_list_gnb[best_round_gnb], 'rx')
        plt.plot([i * step_perc for i in range(1, len(result_list_svc) + 1)], result_list_svc, 'g-', label="svc")
        plt.plot(best_perc_svc, result_list_svc[best_round_svc], 'gx')
        plt.plot([i * step_perc for i in range(1, len(result_list_knn) + 1)], result_list_knn, 'b-', label="knn")
        plt.plot(best_perc_knn, result_list_knn[best_round_knn], 'bx')
        plt.axis([step_perc, (max(len_list) + 1) * step_perc, 0, 1])
        plt.xlabel('Daten in %')
        plt.ylabel('Genauigkeit')
        plt.legend()
        plt.savefig(f"{par_path}/diagrams/{filename}")
        plt.cla()

        # print accuracy to file
        df_percent = pd.DataFrame(data=[i * step_perc for i in range(1, max(len_list) + 1)], columns=['percent'])
        df_gnb = pd.DataFrame(data=result_list_gnb, columns=['gnb'])
        df_svc = pd.DataFrame(data=result_list_svc, columns=['svc'])
        df_knn = pd.DataFrame(data=result_list_knn, columns=['knn'])
        df_accuracy = pd.concat([df_percent, df_gnb, df_svc, df_knn], axis=1)
        df_accuracy = df_accuracy.fillna(value="")
        df_accuracy.to_csv(f'{par_path}accuracy/{filename}_filtered.csv', index=False)


# Chapter 8.1. and later, basically the process of the iterative filter only with the svc classifier
def iterative_filter_process_svm(par_path, par_df, par_num_texts, par_num_authors):
    y = par_df['label_encoded']
    path = f'{par_path}csv_before_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # Filter the files for author and text numbers if 'all' is not set.
    if par_num_authors != "all":
        r = re.compile(f"a{par_num_authors}_")
        files = list(filter(r.match, files))
    if par_num_authors != "all":
        r = re.compile(f".*t{par_num_texts}_")
        files = list(filter(r.match, files))
    step_perc = 1.0
    for f in files:
        filename = f.split(".")[0]
        print(f)
        x = pd.read_csv(f"{par_path}csv_before_filter/{f}", sep=',', encoding="utf-8", nrows=2500)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
        # Get sorted features
        df_sorted_features = sort_features_by_score(x_train, y_train, mutual_info_classif)
        # Calculate the best percentiles of the data for svc
        best_perc_svc, best_round_svc, result_list_svc = get_best_percentile_svc(x_train, y_train, 50,
                                                                                 df_sorted_features, step_perc)

        # select the beast features from the original dataset
        print_filter_feature_percentile(par_path, df_sorted_features, best_perc_svc, x, filename)

        # print best perc to a file
        original_stdout = sys.stdout
        with open(f'{par_path}best_perc/{filename}.txt', 'w') as out_f:
            sys.stdout = out_f
            print(f"best_perc_svc: ({best_perc_svc}|{result_list_svc[best_round_svc]})\n")
            sys.stdout = original_stdout

        # draw diagram
        plt.plot([i * step_perc for i in range(1, len(result_list_svc) + 1)], result_list_svc, 'g-', label="svc")
        plt.plot(best_perc_svc, result_list_svc[best_round_svc], 'gx')
        plt.axis([step_perc, (len(result_list_svc) + 1) * step_perc, 0, 1])
        plt.xlabel('Daten in %')
        plt.ylabel('Genauigkeit')
        plt.legend()
        plt.savefig(f"{par_path}/diagrams/{filename}")
        plt.cla()

        # print accuracy to file
        df_percent = pd.DataFrame(data=[i * step_perc for i in range(1, len(result_list_svc) + 1)], columns=['percent'])
        df_svc = pd.DataFrame(data=result_list_svc, columns=['svc'])
        df_accuracy = pd.concat([df_percent, df_svc], axis=1)
        df_accuracy = df_accuracy.fillna(value="")
        df_accuracy.to_csv(f'{par_path}accuracy/{filename}_filtered.csv', index=False)


# Chapter 7.2.1. Get the accuracy of the features before the iterative filter, results in table 18
def get_accuracy_before_iterative_filter():
    gnb_result_list, svc_result_list, knn_result_list, gnb_time_list, svc_time_list, knn_time_list \
        = [], [], [], [], [], []
    y = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8", nrows=2500)['label_encoded']
    path = f'daten/5_iterative_filter/csv_before_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    gnb = GaussianNB()
    param_grid_svc = {'C': (0.001, 0.01, 0.1, 1, 10),
                      'kernel': ('linear', 'poly', 'rbf'),
                      'gamma': ('scale', 'auto')}
    # Get the feature names for the table
    feature_list = [re.search("(.+?(?=_rel))", f).group(1) for f in files]
    for f in files:
        print(f)
        x = pd.read_csv(f"daten/5_iterative_filter/csv_before_filter/{f}", sep=',', encoding="utf-8", nrows=2500)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

        # GNB fit
        start_time = datetime.now()
        gnb.fit(x_train, y_train)
        # score on test data
        score = accuracy_score(gnb.predict(x_test), y_test)
        time_taken = datetime.now() - start_time
        print(f"GNB test score for {f}: {score}")
        print(f"GNB time for {f}: {time_taken}")
        gnb_result_list.append(score)
        gnb_time_list.append(time_taken)

        # SVC parameter optimization
        grid_search = GridSearchCV(svm.SVC(), param_grid_svc, cv=10, n_jobs=-1, scoring='accuracy')
        grid_results = grid_search.fit(x_train, y_train)
        svc = svm.SVC(C=grid_results.best_params_['C'], gamma=grid_results.best_params_['gamma'],
                      kernel=grid_results.best_params_['kernel'])
        start_time = datetime.now()
        # fit on train data
        svc.fit(x_train, y_train)
        # predict test data
        score = accuracy_score(svc.predict(x_test), y_test)
        time_taken = datetime.now() - start_time
        print(f"SVC test score for {f}: {score}")
        print(f"SVC time for {f}: {time_taken}")
        svc_result_list.append(score)
        svc_time_list.append(time_taken)

        # Parameter for KNN
        # Some Values from 3 to square of k
        neighbors = [i for i in range(3, int(math.sqrt(len(x.columns))), 13)]
        neighbors += [5, 11, 19, 36]
        if int(math.sqrt(len(x.columns))) not in neighbors:
            neighbors.append(int(math.sqrt(len(x.columns))))
        param_grid_knn = {'n_neighbors': neighbors,
                          'weights': ['uniform', 'distance'],
                          'metric': ['euclidean', 'manhattan']}

        # KNN parameter optimization
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10, n_jobs=-1, scoring='accuracy')
        grid_results = grid_search.fit(x_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=grid_results.best_params_['n_neighbors'],
                                   metric=grid_results.best_params_['metric'],
                                   weights=grid_results.best_params_['weights'])
        # fit on train data
        knn.fit(x_train, y_train)
        # KNN predict test data
        start_time = datetime.now()
        # predict test data
        score = accuracy_score(knn.predict(x_test), y_test)
        time_taken = datetime.now() - start_time
        print(f"KNN test score for {f}: {score}")
        print(f"KNN time for {f}: {time_taken}")
        knn_result_list.append(score)
        knn_time_list.append(time_taken)

    # create dataframe with the scores and times
    df_results = pd.DataFrame()
    df_results['feature'] = feature_list
    df_results['score_gnb'] = gnb_result_list
    df_results['time_gnb'] = gnb_time_list
    df_results['score_svc'] = svc_result_list
    df_results['time_svc'] = svc_time_list
    df_results['score_knn'] = knn_result_list
    df_results['time_knn'] = knn_time_list
    return df_results


# Chapter 7.2.1. Get the accuracy of the features after the iterative filter, results in table 18
def get_accuracy_after_iterative_filter():
    df_gnb_result = pd.DataFrame(columns=['feature', 'score_gnb', 'time_gnb'])
    df_svc_result = pd.DataFrame(columns=['feature', 'score_svc', 'time_svc'])
    df_knn_result = pd.DataFrame(columns=['feature', 'score_knn', 'time_knn'])
    y = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8", nrows=2500)['label_encoded']
    # path = f'daten/5_iterative_filter/csv_after_filter'
    path = f'daten/5_iterative_filter/5_iterative_filter/csv_after_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    gnb = GaussianNB()
    param_grid_svc = {'C': (0.001, 0.01, 0.1, 1, 10),
                      'kernel': ('linear', 'poly', 'rbf'),
                      'gamma': ('scale', 'auto')}

    for f in files:
        print(f)
        # Get the feature name for the table
        feature = re.search(".{4}(.+?(?=_rel))", f).group(1)
        # x = pd.read_csv(f"daten/5_iterative_filter/csv_after_filter/{f}", sep=',', encoding="utf-8", nrows=2500)
        x = pd.read_csv(f"daten/5_iterative_filter/5_iterative_filter/csv_after_filter/{f}", sep=',', encoding="utf-8",
                        nrows=2500)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

        # Select the classifier by the start of the filename
        if f.split("_")[0] == "gnb":
            # GNB fit
            start_time = datetime.now()
            gnb.fit(x_train, y_train)
            # score on test data
            score = accuracy_score(gnb.predict(x_test), y_test)
            time_taken = datetime.now() - start_time
            print(f"GNB test score for {f}: {score}")
            print(f"GNB time for {f}: {time_taken}")
            df_gnb_result = df_gnb_result.append(pd.DataFrame(data={'feature': [feature], 'score_gnb': [score],
                                                                    'time_gnb': [time_taken]}), ignore_index=True)

        elif f.split("_")[0] == "svc":
            # SVC parameter optimization
            grid_search = GridSearchCV(svm.SVC(), param_grid_svc, cv=10, n_jobs=-1, scoring='accuracy')
            grid_results = grid_search.fit(x_train, y_train)
            svc = svm.SVC(C=grid_results.best_params_['C'], gamma=grid_results.best_params_['gamma'],
                          kernel=grid_results.best_params_['kernel'])
            start_time = datetime.now()
            # fit on train data
            svc.fit(x_train, y_train)
            # predict test data
            score = accuracy_score(svc.predict(x_test), y_test)
            time_taken = datetime.now() - start_time
            print(f"SVC test score for {f}: {score}")
            print(f"SVC training time for {f}: {time_taken}")
            df_svc_result = df_svc_result.append(pd.DataFrame(data={'feature': [feature], 'score_svc': [score],
                                                                    'time_svc': [time_taken]}), ignore_index=True)

        elif f.split("_")[0] == "knn":
            # Parameter for KNN
            # Some Values from 3 to square of k
            neighbors = [i for i in range(3, int(math.sqrt(len(x.columns))), 13)]
            neighbors += [5, 11, 19, 36]
            if int(math.sqrt(len(x.columns))) not in neighbors:
                neighbors.append(int(math.sqrt(len(x.columns))))
            param_grid_knn = {'n_neighbors': neighbors,
                              'weights': ['uniform', 'distance'],
                              'metric': ['euclidean', 'manhattan']}

            # KNN parameter optimization
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10, n_jobs=-1, scoring='accuracy')
            grid_results = grid_search.fit(x_train, y_train)
            knn = KNeighborsClassifier(n_neighbors=grid_results.best_params_['n_neighbors'],
                                       metric=grid_results.best_params_['metric'],
                                       weights=grid_results.best_params_['weights'])
            start_time = datetime.now()
            # fit on train data
            knn.fit(x_train, y_train)
            # KNN predict test data
            start_time = datetime.now()
            # predict test data
            score = accuracy_score(knn.predict(x_test), y_test)
            time_taken = datetime.now() - start_time
            print(f"KNN test score for {f}: {score}")
            print(f"KNN test time for {f}: {time_taken}")
            df_knn_result = df_knn_result.append(pd.DataFrame(data={'feature': [feature], 'score_knn': [score],
                                                                    'time_knn': [time_taken]}), ignore_index=True)

    df_merge = pd.merge(df_gnb_result, df_knn_result, on="feature", how='outer')
    df_merge = pd.merge(df_merge, df_svc_result, on="feature", how='outer')
    return df_merge


# Get n article for a given number of authors. Required for setups with different numbers of authors and article
def get_n_article_by_author(par_df, par_label_count, par_article_count):
    df_articles = pd.DataFrame(columns=['label_encoded', 'text'])
    # only keep entries of the "par_label_count" first labels
    par_df = par_df.where(par_df['label_encoded'] <= par_label_count).dropna()
    labels = np.unique(par_df['label_encoded'].values).tolist()
    list_article_count = [par_article_count for i in labels]
    for index, row in par_df.iterrows():
        if list_article_count[labels.index(row['label_encoded'])] != 0:
            d = {'label_encoded': [row['label_encoded']], 'text': [row['text']]}
            df_articles = df_articles.append(pd.DataFrame.from_dict(d), ignore_index=True)
            list_article_count[labels.index(row['label_encoded'])] -= 1
        if sum(list_article_count) == 0:
            break
    return df_articles


# Return indices for n article for a given number of authors. Required for setups with different
# numbers of authors and article
def get_n_article_index_by_author(par_df, par_label_count, par_article_count):
    index_list = []
    # only keep entries of the "par_label_count" first labels
    par_df = par_df.where(par_df['label_encoded'] <= par_label_count).dropna()
    labels = np.unique(par_df['label_encoded'].values).tolist()
    list_article_count = [par_article_count for i in labels]
    for index, row in par_df.iterrows():
        if row['label_encoded'] in labels:
            if list_article_count[labels.index(row['label_encoded'])] != 0:
                index_list.append(index)
                list_article_count[labels.index(row['label_encoded'])] -= 1
        if sum(list_article_count) == 0:
            break
    return index_list


# Method to estimate the f1 score of the test data for GNB
def get_f1_for_gnb(par_x_train, par_x_test, par_y_train, par_y_test):
    gnb = GaussianNB()
    # GNB fit
    gnb.fit(par_x_train, par_y_train)
    # score on test data
    gnb_score = f1_score(gnb.predict(par_x_test), par_y_test, average='micro')
    return gnb_score


# Method to estimate the f1 score of the test data for SVC
def get_f1_for_svc(par_x_train, par_x_test, par_y_train, par_y_test, par_cv):
    # Param Grid SVC
    param_grid_svc = {'C': (0.001, 0.01, 0.1, 1, 10),
                      'kernel': ('linear', 'poly', 'rbf'),
                      'gamma': ('scale', 'auto')}
    # SVC parameter optimization
    grid_search = GridSearchCV(svm.SVC(), param_grid_svc, cv=par_cv, n_jobs=-1, scoring='f1_micro')
    grid_results = grid_search.fit(par_x_train, par_y_train)
    svc = svm.SVC(C=grid_results.best_params_['C'], gamma=grid_results.best_params_['gamma'],
                  kernel=grid_results.best_params_['kernel'])
    # fit on train data
    svc.fit(par_x_train, par_y_train)
    # predict test data
    svc_score = f1_score(svc.predict(par_x_test), par_y_test, average='micro')
    return svc_score


# Method to estimate the f1 score of the test data for KNN
def get_f1_for_knn(par_x_train, par_x_test, par_y_train, par_y_test, par_cv):
    # define param grid for knn, neighbors has the be lower than samples
    neighbors = [1, 3, 5, 11, 19, 36, 50]
    # number of neighbors must be less than number of samples
    neighbors = [x for x in neighbors if x < len(par_x_test)]
    param_grid_knn = {'n_neighbors': neighbors,
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan']}
    # KNN parameter optimization
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=par_cv, n_jobs=-1, scoring='f1_micro')
    grid_results = grid_search.fit(par_x_train, par_y_train)
    knn = KNeighborsClassifier(n_neighbors=grid_results.best_params_['n_neighbors'],
                               metric=grid_results.best_params_['metric'],
                               weights=grid_results.best_params_['weights'])
    # fit on train data
    knn.fit(par_x_train, par_y_train)
    # predict test data
    knn_score = f1_score(knn.predict(par_x_test), par_y_test, average='micro')

    return knn_score


# Method to estimate the accuracy of the test data for SVC
def get_accuracy_for_svc(par_x_train, par_x_test, par_y_train, par_y_test, par_cv):
    # Param Grid SVC
    param_grid_svc = {'C': (0.001, 0.01, 0.1, 1, 10),
                      'kernel': ('linear', 'poly', 'rbf'),
                      'gamma': ('scale', 'auto')}
    # SVC parameter optimization
    grid_search = GridSearchCV(svm.SVC(), param_grid_svc, cv=par_cv, n_jobs=-1, scoring='f1_micro')
    grid_results = grid_search.fit(par_x_train, par_y_train)
    svc = svm.SVC(C=grid_results.best_params_['C'], gamma=grid_results.best_params_['gamma'],
                  kernel=grid_results.best_params_['kernel'])
    # fit on train data
    svc.fit(par_x_train, par_y_train)
    # predict test data
    svc_score = accuracy_score(svc.predict(par_x_test), par_y_test)
    return svc_score


# Chapter 7.3.1. comparison of the word length feature alternatives
def compare_word_length_features():
    df_all_texts = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")
    # Different values for the texts by authors
    list_author_texts = [10, 15, 25, 50, 75, 100]
    # save the results in a dictionary
    dic_f1_results = {'wl_matrix_gnb': [], 'wl_matrix_svc': [], 'wl_matrix_knn': [],
                      'wl_matrix_bins_20_30_gnb': [], 'wl_matrix_bins_20_30_svc': [], 'wl_matrix_bins_20_30_knn': [],
                      'wl_matrix_bins_10_20_gnb': [], 'wl_matrix_bins_10_20_svc': [], 'wl_matrix_bins_10_20_knn': [],
                      'wl_matrix_20_gnb': [], 'wl_matrix_20_svc': [], 'wl_matrix_20_knn': [],
                      'wl_avg_gnb': [], 'wl_avg_svc': [], 'wl_avg_knn': []}

    for author_texts in list_author_texts:
        # get article for n authors with number of author texts
        df_article = get_n_article_by_author(df_all_texts, 25, author_texts)
        # Get the word count for the individual relative frequency
        word_count = get_word_count(df_article)

        # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
        if author_texts * 0.4 < 10:
            cv = int(author_texts * 0.4)
        else:
            cv = 10
        # Get the scores for every feature
        for feature in ["wl_matrix", "wl_matrix_bins_20_30", "wl_matrix_bins_10_20", "wl_avg", "wl_matrix_20"]:
            # select the test/train data by the feature name and calculate the individual relative frequency
            if feature == "wl_matrix":
                x = get_rel_frequency(get_word_length_matrix(df_article).fillna(value=0), word_count['word_count'])
            elif feature == "wl_matrix_bins_20_30":
                x = get_rel_frequency(get_word_length_matrix_with_interval(df_article, 20, 30).fillna(value=0),
                                      word_count['word_count'])
            elif feature == "wl_matrix_bins_10_20":
                x = get_rel_frequency(get_word_length_matrix_with_interval(df_article, 10, 20).fillna(value=0),
                                      word_count['word_count'])
            elif feature == "wl_avg":
                x = get_average_word_length(df_article)
            elif feature == "wl_matrix_20":
                x = get_word_length_matrix_with_margin(df_article, 20)
            # Scale the data, else high counter in wl_matrix can dominate and hyperparameter optimization for svc
            # takes a while because of small differences from average
            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)
            y = df_article['label_encoded']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

            y = df_article['label_encoded']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

            # calculate scores
            gnb_score = get_f1_for_gnb(x_train, x_test, y_train, y_test)
            svc_score = get_f1_for_svc(x_train, x_test, y_train, y_test, cv)
            knn_score = get_f1_for_knn(x_train, x_test, y_train, y_test, cv)

            # Append scores to dictionary
            dic_f1_results[f'{feature}_gnb'].append(gnb_score)
            dic_f1_results[f'{feature}_svc'].append(svc_score)
            dic_f1_results[f'{feature}_knn'].append(knn_score)

            # Console output
            print(f"GNB-Score for {feature} with {author_texts}: {gnb_score}")
            print(f"SVC-Score for {feature} with {author_texts}: {svc_score}")
            print(f"KNN-Score for {feature} with {author_texts}: {knn_score}")

    df_results = pd.DataFrame(dic_f1_results)
    df_results['number_article'] = list_author_texts
    return df_results


# Chapter 7.3.2. comparison of the digit feature alternatives
def compare_digit_features():
    df_all_texts = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")

    # Different values for the texts by authors
    list_author_texts = [10, 15, 25, 50, 75, 100]
    # save the results in a dictionary
    dic_f1_results = {'digit_sum_gnb': [], 'digit_sum_svc': [], 'digit_sum_knn': [],
                      'digits_gnb': [], 'digits_svc': [], 'digits_knn': []}

    for author_texts in list_author_texts:
        # get article for n authors with number of author texts
        df_article = get_n_article_by_author(df_all_texts, 25, author_texts)
        # Get the word count for the individual relative frequency
        char_count = get_char_count(df_article)

        # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
        if author_texts * 0.4 < 10:
            cv = int(author_texts * 0.4)
        else:
            cv = 10

        # Get the scores for every feature
        for feature in ["digit_sum", "digits"]:
            # select the test/train data by the feature name and calculate the individual relative frequency
            if feature == "digit_sum":
                x = get_rel_frequency(get_sum_digits(df_article).fillna(value=0), char_count['char_count'])
            elif feature == "digits":
                x = get_rel_frequency(get_digits(df_article).fillna(value=0), char_count['char_count'])

            y = df_article['label_encoded']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

            # calculate scores
            gnb_score = get_f1_for_gnb(x_train, x_test, y_train, y_test)
            svc_score = get_f1_for_svc(x_train, x_test, y_train, y_test, cv)
            knn_score = get_f1_for_knn(x_train, x_test, y_train, y_test, cv)

            # Append scores to dictionary
            dic_f1_results[f'{feature}_gnb'].append(gnb_score)
            dic_f1_results[f'{feature}_svc'].append(svc_score)
            dic_f1_results[f'{feature}_knn'].append(knn_score)

            # Console output
            print(f"GNB-Score for {feature} with {author_texts}: {gnb_score}")
            print(f"SVC-Score for {feature} with {author_texts}: {svc_score}")
            print(f"KNN-Score for {feature} with {author_texts}: {knn_score}")

    df_results = pd.DataFrame(dic_f1_results)
    df_results['number_article'] = list_author_texts
    return df_results


# Chapter 7.3.3. comparison of the word ngrams with n 4-6
def compare_word_4_6_grams():
    df_all_texts = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")

    # Different values for the texts by authors
    list_author_texts = [10, 15, 25, 50, 75, 100]
    # save the results in a dictionary
    dic_f1_results = {'w4g_gnb': [], 'w4g_svc': [], 'w4g_knn': [],
                      'w5g_gnb': [], 'w5g_svc': [], 'w5g_knn': [],
                      'w6g_gnb': [], 'w6g_svc': [], 'w6g_knn': []}

    # load the data
    df_w4g = pd.read_csv("daten/6_feature_analysis/input_data/word_4_gram_rel.csv", sep=',', encoding="utf-8")
    df_w5g = pd.read_csv("daten/6_feature_analysis/input_data/word_5_gram_rel.csv", sep=',', encoding="utf-8")
    df_w6g = pd.read_csv("daten/6_feature_analysis/input_data/word_6_gram_rel.csv", sep=',', encoding="utf-8")

    for author_texts in list_author_texts:
        # indices for article for n authors with m texts
        index_list = get_n_article_index_by_author(df_all_texts, 25, author_texts)

        # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
        if author_texts * 0.4 < 10:
            cv = int(author_texts * 0.4)
            # CV between 5 and 10 is unusual
            cv = 5 if cv > 5 else cv
        else:
            cv = 10

        # Get the scores for every feature
        for feature in ["w4g", "w5g", "w6g"]:
            # select the indices from the article rows by the given indices
            if feature == "w4g":
                x = df_w4g.iloc[index_list]
            elif feature == "w5g":
                x = df_w5g.iloc[index_list]
            elif feature == "w6g":
                x = df_w6g.iloc[index_list]

            # Delete features which only occur once
            x = trim_df_by_occurrence(x, 1)

            # reset the indices to have a order from 0 to authors * text per author - 1
            x = x.reset_index(drop=True)

            y = df_all_texts.iloc[index_list]['label_encoded']
            y = y.reset_index(drop=True)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

            # calculate scores
            gnb_score = get_f1_for_gnb(x_train, x_test, y_train, y_test)
            svc_score = get_f1_for_svc(x_train, x_test, y_train, y_test, cv)
            knn_score = get_f1_for_knn(x_train, x_test, y_train, y_test, cv)

            # Append scores to dictionary
            dic_f1_results[f'{feature}_gnb'].append(gnb_score)
            dic_f1_results[f'{feature}_svc'].append(svc_score)
            dic_f1_results[f'{feature}_knn'].append(knn_score)

            # Console output
            print(f"GNB-Score for {feature} with {author_texts}: {gnb_score}")
            print(f"SVC-Score for {feature} with {author_texts}: {svc_score}")
            print(f"KNN-Score for {feature} with {author_texts}: {knn_score}")

    df_results = pd.DataFrame(dic_f1_results)
    df_results['number_article'] = list_author_texts
    return df_results


# Chapter 7.3.3. comparison of the word ngrams with n 2-3
def compare_word_2_3_grams():
    df_all_texts = pd.read_csv("musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")

    # Different values for the texts by authors
    list_author_texts = [10, 15, 25, 50, 75, 100]
    # save the results in a dictionary
    dic_f1_results = {'w2g_gnb': [], 'w2g_svc': [], 'w2g_knn': [],
                      'w3g_gnb': [], 'w3g_svc': [], 'w3g_knn': []}

    for author_texts in list_author_texts:
        print(f"Texte pro Autor: {author_texts}")
        # indices for article for n authors with m texts
        index_list = get_n_article_index_by_author(df_balanced, 25, author_texts)

        # define the splits for the hyperparameter tuning, cannot be greater than the number of members in each class
        if author_texts * 0.4 < 10:
            cv = int(author_texts * 0.4)
            # CV between 5 and 10 is unusual
            cv = 5 if cv > 5 else cv
        else:
            cv = 10

        # select the indices from the article rows by the given indices
        df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)
        print(f"Artikel: {len(df_balanced.index)}")

        # extract the features
        df_w2g = get_word_n_grams(df_balanced, 2)
        df_w3g = get_word_n_grams(df_balanced, 3)

        # Preprocessing steps
        word_count = get_word_count(df_balanced)
        df_w2g = preprocessing_steps_pos_tag_n_grams(df_w2g, word_count['word_count'])
        df_w3g = preprocessing_steps_pos_tag_n_grams(df_w3g, word_count['word_count'])

        # Scaler, else SVM need a lot of time with very low numbers.
        scaler = StandardScaler()
        df_w2g[df_w2g.columns] = scaler.fit_transform(df_w2g[df_w2g.columns])
        df_w3g[df_w3g.columns] = scaler.fit_transform(df_w3g[df_w3g.columns])

        label = df_balanced['label_encoded']

        # Train/Test 60/40 split
        df_w2g_train, df_w2g_test, df_w3g_train, df_w3g_test, label_train, label_test = \
            train_test_split(df_w2g, df_w3g, label, test_size=0.4, random_state=42, stratify=label)
        # Get the scores for every feature
        for feature in ["w2g", "w3g"]:
            # select the indices from the article rows by the given indices
            # iterative filter
            # returns df_x_train_gnb, df_x_test_gnb, df_x_train_svc, df_x_test_svc, df_x_train_knn, df_x_test_knn
            if feature == "w2g":
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test = \
                    feature_selection_iterative_filter(df_w2g_train, df_w2g_test, label_train, 1.0, mutual_info_classif)
            elif feature == "w3g":
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test = \
                    feature_selection_iterative_filter(df_w3g_train, df_w3g_test, label_train, 1.0, mutual_info_classif)
                # Do not use iterative filter for gnb train caused by bad results
                x_gnb_train, x_gnb_test, label_train, label_test = \
                    train_test_split(df_w3g, label, test_size=0.4, random_state=42, stratify=label)

            print(f"cv: {cv}")
            print(f"Train Labels: {label_train.value_counts()}")
            print(f"Test Labels: {label_test.value_counts()}")
            # calculate scores
            gnb_score = get_f1_for_gnb(x_gnb_train, x_gnb_test, label_train, label_test)
            svc_score = get_f1_for_svc(x_svc_train, x_svc_test, label_train, label_test, cv)
            knn_score = get_f1_for_knn(x_knn_train, x_knn_test, label_train, label_test, cv)

            # Append scores to dictionary
            dic_f1_results[f'{feature}_gnb'].append(gnb_score)
            dic_f1_results[f'{feature}_svc'].append(svc_score)
            dic_f1_results[f'{feature}_knn'].append(knn_score)

            # Console output
            print(f"GNB-Score for {feature} with {author_texts}: {gnb_score}")
            print(f"SVC-Score for {feature} with {author_texts}: {svc_score}")
            print(f"KNN-Score for {feature} with {author_texts}: {knn_score}")

    df_results = pd.DataFrame(dic_f1_results)
    df_results['number_article'] = list_author_texts
    return df_results


# Chapter 7.3.4. comparison of the different lengths of char ngrams
# Chapter 7.3.4. whole process of the comparison of the char-n-gram features
def compare_char_n_grams_process(par_base_path):
    df_all_texts = pd.read_csv(f"musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")
    author_counts = [25]
    text_counts = [10, 15, 25, 50, 75, 100]
    for number_authors in author_counts:
        for number_texts in text_counts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            extract_n_gram_features_to_csv(df_balanced, par_base_path, number_authors, number_texts)
            iterative_filter_process(par_base_path, df_balanced, number_texts, number_authors)

    compare_char_affix_ngrams(text_counts, author_counts, par_base_path, df_all_texts) \
        .to_csv(f"{par_base_path}results/char_affix_n_grams.csv", index=False)
    compare_char_word_ngrams(text_counts, author_counts, par_base_path, df_all_texts) \
        .to_csv(f"{par_base_path}results/char_word_n_grams.csv", index=False)
    compare_char_punct_ngrams(text_counts, author_counts, par_base_path, df_all_texts) \
        .to_csv(f"{par_base_path}results/char_punct_n_grams.csv", index=False)


# Chapter 7.3.4. char-affix-ngrams
def compare_char_affix_ngrams(par_author_texts, par_authors, par_base_path, par_df):
    # save the results in a dictionary
    dic_f1_results = {'c_affix_2_gnb': [], 'c_affix_2_svc': [], 'c_affix_2_knn': [],
                      'c_affix_3_gnb': [], 'c_affix_3_svc': [], 'c_affix_3_knn': [],
                      'c_affix_4_gnb': [], 'c_affix_4_svc': [], 'c_affix_4_knn': [],
                      'c_affix_5_gnb': [], 'c_affix_5_svc': [], 'c_affix_5_knn': [],
                      'number_authors': [], 'number_texts': []}
    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(par_df, number_authors, number_texts)
            df_balanced = par_df.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Get the scores for every feature
            # Append authors and texts
            dic_f1_results['number_authors'].append(number_authors)
            dic_f1_results['number_texts'].append(number_texts)
            for feature in ["c_affix_2", "c_affix_3", "c_affix_4", "c_affix_5"]:
                # read the data based on n, texts and authors
                if feature == "c_affix_2":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_affix_2_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_affix_2_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_affix_2_gram_filtered.csv")
                elif feature == "c_affix_3":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_affix_3_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_affix_3_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_affix_3_gram_filtered.csv")
                elif feature == "c_affix_4":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_affix_4_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_affix_4_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_affix_4_gram_filtered.csv")
                elif feature == "c_affix_5":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_affix_5_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_affix_5_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_affix_5_gram_filtered.csv")

                # Scaler, else SVM need a lot of time with very low numbers.
                scaler = StandardScaler()
                df_gnb[df_gnb.columns] = scaler.fit_transform(df_gnb[df_gnb.columns])
                df_svc[df_svc.columns] = scaler.fit_transform(df_svc[df_svc.columns])
                df_knn[df_knn.columns] = scaler.fit_transform(df_knn[df_knn.columns])

                # Train/Test 60/40 split
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test, label_train, label_test = \
                    train_test_split(df_gnb, df_svc, df_knn, label, test_size=0.4, random_state=42, stratify=label)

                # calculate scores
                gnb_score = get_f1_for_gnb(x_gnb_train, x_gnb_test, label_train, label_test)
                svc_score = get_f1_for_svc(x_svc_train, x_svc_test, label_train, label_test, cv)
                knn_score = get_f1_for_knn(x_knn_train, x_knn_test, label_train, label_test, cv)

                # Append scores to dictionary
                dic_f1_results[f'{feature}_gnb'].append(gnb_score)
                dic_f1_results[f'{feature}_svc'].append(svc_score)
                dic_f1_results[f'{feature}_knn'].append(knn_score)

                # Console output
                print(f"GNB-Score for {feature} with {number_authors} authors and {number_texts} texts: {gnb_score}")
                print(f"SVC-Score for {feature} with {number_authors} authors and {number_texts} texts: {svc_score}")
                print(f"KNN-Score for {feature} with {number_authors} authors and {number_texts} texts: {knn_score}")
    return pd.DataFrame(dic_f1_results)


# Chapter 7.3.4. char-word-ngrams
def compare_char_word_ngrams(par_author_texts, par_authors, par_base_path, par_df):
    # save the results in a dictionary
    dic_f1_results = {'c_word_2_gnb': [], 'c_word_2_svc': [], 'c_word_2_knn': [],
                      'c_word_3_gnb': [], 'c_word_3_svc': [], 'c_word_3_knn': [],
                      'c_word_4_gnb': [], 'c_word_4_svc': [], 'c_word_4_knn': [],
                      'c_word_5_gnb': [], 'c_word_5_svc': [], 'c_word_5_knn': [],
                      'number_authors': [], 'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(par_df, number_authors, number_texts)
            df_balanced = par_df.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Get the scores for every feature
            # Append authors and texts
            dic_f1_results['number_authors'].append(number_authors)
            dic_f1_results['number_texts'].append(number_texts)
            for feature in ["c_word_2", "c_word_3", "c_word_4", "c_word_5"]:
                # read the data based on n, texts and authors
                if feature == "c_word_2":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_word_2_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_word_2_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_word_2_gram_filtered.csv")
                elif feature == "c_word_3":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_word_3_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_word_3_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_word_3_gram_filtered.csv")
                elif feature == "c_word_4":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_word_4_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_word_4_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_word_4_gram_filtered.csv")
                elif feature == "c_word_5":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_word_5_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_word_5_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_word_5_gram_filtered.csv")

                # Scaler, else SVM need a lot of time with very low numbers.
                scaler = StandardScaler()
                df_gnb[df_gnb.columns] = scaler.fit_transform(df_gnb[df_gnb.columns])
                df_svc[df_svc.columns] = scaler.fit_transform(df_svc[df_svc.columns])
                df_knn[df_knn.columns] = scaler.fit_transform(df_knn[df_knn.columns])

                # Train/Test 60/40 split
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test, label_train, label_test = \
                    train_test_split(df_gnb, df_svc, df_knn, label, test_size=0.4, random_state=42, stratify=label)

                # calculate scores
                gnb_score = get_f1_for_gnb(x_gnb_train, x_gnb_test, label_train, label_test)
                svc_score = get_f1_for_svc(x_svc_train, x_svc_test, label_train, label_test, cv)
                knn_score = get_f1_for_knn(x_knn_train, x_knn_test, label_train, label_test, cv)

                # Append scores to dictionary
                dic_f1_results[f'{feature}_gnb'].append(gnb_score)
                dic_f1_results[f'{feature}_svc'].append(svc_score)
                dic_f1_results[f'{feature}_knn'].append(knn_score)

                # Console output
                print(f"GNB-Score for {feature} with {number_authors} authors and {number_texts} texts: {gnb_score}")
                print(f"SVC-Score for {feature} with {number_authors} authors and {number_texts} texts: {svc_score}")
                print(f"KNN-Score for {feature} with {number_authors} authors and {number_texts} texts: {knn_score}")
    return pd.DataFrame(dic_f1_results)


# Chapter 7.3.4. char-punct-ngrams
def compare_char_punct_ngrams(par_author_texts, par_authors, par_base_path, par_df):
    # save the results in a dictionary
    dic_f1_results = {'c_punct_2_gnb': [], 'c_punct_2_svc': [], 'c_punct_2_knn': [],
                      'c_punct_3_gnb': [], 'c_punct_3_svc': [], 'c_punct_3_knn': [],
                      'c_punct_4_gnb': [], 'c_punct_4_svc': [], 'c_punct_4_knn': [],
                      'c_punct_5_gnb': [], 'c_punct_5_svc': [], 'c_punct_5_knn': [],
                      'number_authors': [], 'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(par_df, number_authors, number_texts)
            df_balanced = par_df.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Get the scores for every feature
            # Append authors and texts
            dic_f1_results['number_authors'].append(number_authors)
            dic_f1_results['number_texts'].append(number_texts)
            for feature in ["c_punct_2", "c_punct_3", "c_punct_4", "c_punct_5"]:
                # read the data based on n, texts and authors
                if feature == "c_punct_2":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_punct_2_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_punct_2_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_punct_2_gram_filtered.csv")
                elif feature == "c_punct_3":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_punct_3_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_punct_3_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_punct_3_gram_filtered.csv")
                elif feature == "c_punct_4":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_punct_4_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_punct_4_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_punct_4_gram_filtered.csv")
                elif feature == "c_punct_5":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_char_punct_5_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_char_punct_5_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_char_punct_5_gram_filtered.csv")

                # Scaler, else SVM need a lot of time with very low numbers.
                scaler = StandardScaler()
                df_gnb[df_gnb.columns] = scaler.fit_transform(df_gnb[df_gnb.columns])
                df_svc[df_svc.columns] = scaler.fit_transform(df_svc[df_svc.columns])
                df_knn[df_knn.columns] = scaler.fit_transform(df_knn[df_knn.columns])

                # Train/Test 60/40 split
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test, label_train, label_test = \
                    train_test_split(df_gnb, df_svc, df_knn, label, test_size=0.4, random_state=42, stratify=label)

                # calculate scores
                gnb_score = get_f1_for_gnb(x_gnb_train, x_gnb_test, label_train, label_test)
                svc_score = get_f1_for_svc(x_svc_train, x_svc_test, label_train, label_test, cv)
                knn_score = get_f1_for_knn(x_knn_train, x_knn_test, label_train, label_test, cv)

                # Append scores to dictionary
                dic_f1_results[f'{feature}_gnb'].append(gnb_score)
                dic_f1_results[f'{feature}_svc'].append(svc_score)
                dic_f1_results[f'{feature}_knn'].append(knn_score)

                # Console output
                print(f"GNB-Score for {feature} with {number_authors} authors and {number_texts} texts: {gnb_score}")
                print(f"SVC-Score for {feature} with {number_authors} authors and {number_texts} texts: {svc_score}")
                print(f"KNN-Score for {feature} with {number_authors} authors and {number_texts} texts: {knn_score}")
    return pd.DataFrame(dic_f1_results)


# Chapter 7.3.4. Print the char-n-gram features in different files
def extract_n_gram_features_to_csv(par_df, par_base_path, par_number_authors, par_number_texts):
    char_count = get_char_count(par_df)
    # n from 2-5
    for n in range(2, 6):
        ca_ng = get_char_affix_n_grams(par_df, n)
        preprocessing_steps_char_n_grams(ca_ng, char_count['char_count'])\
            .to_csv(f"{par_base_path}csv_before_filter/a{par_number_authors}_t{par_number_texts}"
                    f"_char_affix_{n}_gram.csv", index=False)
        cw_ng = get_char_word_n_grams(par_df, n)
        preprocessing_steps_char_n_grams(cw_ng, char_count['char_count'])\
            .to_csv(f"{par_base_path}csv_before_filter/a{par_number_authors}_t{par_number_texts}"
                    f"_char_word_{n}_gram.csv", index=False)
        cp_ng = get_char_punct_n_grams(par_df, n)
        preprocessing_steps_char_n_grams(cp_ng, char_count['char_count'])\
            .to_csv(f"{par_base_path}csv_before_filter/a{par_number_authors}_t{par_number_texts}"
                    f"_char_punct_{n}_gram.csv", index=False)
        print(f"Extraction Round {n - 1} done")
    return True


# combined preprocessing steps of the pos-tag-n-grams
def preprocessing_steps_pos_tag_n_grams(par_feature, length_metric):
    # Filter features which only occur once
    par_feature = trim_df_by_occurrence(par_feature, 1)
    # Individual relative frequency
    par_feature = get_rel_frequency(par_feature.fillna(value=0), length_metric)
    return par_feature


# combined preprocessing steps of the char-n-grams
def preprocessing_steps_char_n_grams(par_feature, length_metric):
    # Filter features which only occur once
    par_feature = trim_df_sum_feature(par_feature, 5)
    # Individual relative frequency
    par_feature = get_rel_frequency(par_feature.fillna(value=0), length_metric)
    return par_feature


# Feature selection with the iterative filter without printing the results in a file
def feature_selection_iterative_filter(par_x_train, par_x_test, par_y_train, par_step, par_classif):
    df_sorted_features = sort_features_by_score(par_x_train, par_y_train, par_classif)
    # Calculate the best percentiles of the data for the different classifier
    best_perc_gnb = get_best_percentile_gnb(par_x_train, par_y_train, 50, df_sorted_features, par_step)[0]
    best_perc_svc = get_best_percentile_svc(par_x_train, par_y_train, 50, df_sorted_features, par_step)[0]
    best_perc_knn = get_best_percentile_knn(par_x_train, par_y_train, 50, df_sorted_features, par_step)[0]

    # select the 1 percent of the features (len/100) multiplied by par_best_percent
    # select the best features from the original dataset
    df_x_train_gnb = par_x_train[
        df_sorted_features['feature_names'][: round(best_perc_gnb * (len(par_x_train.columns) / 100))].tolist()]
    df_x_test_gnb = par_x_test[
        df_sorted_features['feature_names'][: round(best_perc_gnb * (len(par_x_train.columns) / 100))].tolist()]
    df_x_train_svc = par_x_train[
        df_sorted_features['feature_names'][: round(best_perc_svc * (len(par_x_train.columns) / 100))].tolist()]
    df_x_test_svc = par_x_test[
        df_sorted_features['feature_names'][: round(best_perc_svc * (len(par_x_train.columns) / 100))].tolist()]
    df_x_train_knn = par_x_train[
        df_sorted_features['feature_names'][: round(best_perc_knn * (len(par_x_train.columns) / 100))].tolist()]
    df_x_test_knn = par_x_test[
        df_sorted_features['feature_names'][: round(best_perc_knn * (len(par_x_train.columns) / 100))].tolist()]
    return df_x_train_gnb, df_x_test_gnb, df_x_train_svc, df_x_test_svc, df_x_train_knn, df_x_test_knn


# Chapter 7.3.5. function to compare the pos-tag-n-grams
def compare_pos_tag_ngrams(par_author_texts, par_authors, par_base_path, par_df):
    # save the results in a dictionary
    dic_f1_results = {'pos_2_gnb': [], 'pos_2_svc': [], 'pos_2_knn': [],
                      'pos_3_gnb': [], 'pos_3_svc': [], 'pos_3_knn': [],
                      'pos_4_gnb': [], 'pos_4_svc': [], 'pos_4_knn': [],
                      'pos_5_gnb': [], 'pos_5_svc': [], 'pos_5_knn': [],
                      'number_authors': [], 'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(par_df, number_authors, number_texts)
            df_balanced = par_df.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Get the scores for every feature
            # Append authors and texts
            dic_f1_results['number_authors'].append(number_authors)
            dic_f1_results['number_texts'].append(number_texts)
            for feature in ["pos_2", "pos_3", "pos_4", "pos_5"]:
                # read the data based on n, texts and authors
                if feature == "pos_2":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_pos_tag_2_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_pos_tag_2_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_pos_tag_2_gram_filtered.csv")
                elif feature == "pos_3":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_pos_tag_3_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_pos_tag_3_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_pos_tag_3_gram_filtered.csv")
                elif feature == "pos_4":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_pos_tag_4_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_pos_tag_4_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_pos_tag_4_gram_filtered.csv")
                elif feature == "pos_5":
                    df_gnb = pd.read_csv(
                        f"{par_base_path}csv_after_filter/gnb_a{number_authors}_t{number_texts}"
                        f"_pos_tag_5_gram_filtered.csv")
                    df_svc = pd.read_csv(
                        f"{par_base_path}csv_after_filter/svc_a{number_authors}_t{number_texts}"
                        f"_pos_tag_5_gram_filtered.csv")
                    df_knn = pd.read_csv(
                        f"{par_base_path}csv_after_filter/knn_a{number_authors}_t{number_texts}"
                        f"_pos_tag_5_gram_filtered.csv")

                # Scaler, else SVM need a lot of time with very low numbers.
                scaler = StandardScaler()
                df_gnb[df_gnb.columns] = scaler.fit_transform(df_gnb[df_gnb.columns])
                df_svc[df_svc.columns] = scaler.fit_transform(df_svc[df_svc.columns])
                df_knn[df_knn.columns] = scaler.fit_transform(df_knn[df_knn.columns])

                # Train/Test 60/40 split
                x_gnb_train, x_gnb_test, x_svc_train, x_svc_test, x_knn_train, x_knn_test, label_train, label_test = \
                    train_test_split(df_gnb, df_svc, df_knn, label, test_size=0.4, random_state=42, stratify=label)

                # calculate scores
                gnb_score = get_f1_for_gnb(x_gnb_train, x_gnb_test, label_train, label_test)
                svc_score = get_f1_for_svc(x_svc_train, x_svc_test, label_train, label_test, cv)
                knn_score = get_f1_for_knn(x_knn_train, x_knn_test, label_train, label_test, cv)

                # Append scores to dictionary
                dic_f1_results[f'{feature}_gnb'].append(gnb_score)
                dic_f1_results[f'{feature}_svc'].append(svc_score)
                dic_f1_results[f'{feature}_knn'].append(knn_score)

                # Console output
                print(f"GNB-Score for {feature} with {number_authors} authors and {number_texts} texts: {gnb_score}")
                print(f"SVC-Score for {feature} with {number_authors} authors and {number_texts} texts: {svc_score}")
                print(f"KNN-Score for {feature} with {number_authors} authors and {number_texts} texts: {knn_score}")
    return pd.DataFrame(dic_f1_results)


# Chapter 7.3.5. complete process of the pos-tag-n-grams comparison
def compare_pos_n_grams_process(par_base_path):
    df_all_texts = pd.read_csv(f"musikreviews_balanced_authors.csv", sep=',', encoding="utf-8")
    author_counts = [25]
    text_counts = [10, 15, 25, 50, 75, 100]
    for number_authors in author_counts:
        for number_texts in text_counts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)
            word_count = get_word_count(df_balanced)
            # extract features and preprocessing
            for n in range(2, 6):
                pt_ng = get_pos_tags_n_grams(df_balanced, n)
                preprocessing_steps_pos_tag_n_grams(pt_ng, word_count['word_count']) \
                    .to_csv(f"{par_base_path}csv_before_filter/"
                            f"a{number_authors}_t{number_texts}_pos_tag_{n}_gram.csv", index=False)
            iterative_filter_process(par_base_path, df_balanced, number_texts, number_authors)
            # 2 grams for svc get not filtered, overwrite unfiltered for svc
            pt_ng = get_pos_tags_n_grams(df_balanced, 2)
            preprocessing_steps_pos_tag_n_grams(pt_ng, word_count['word_count']) \
                .to_csv(f"{par_base_path}csv_after_filter/"
                        f"svc_a{number_authors}_t{number_texts}_pos_tag_2_gram_filtered.csv", index=False)
    compare_pos_tag_ngrams(text_counts, author_counts, par_base_path, df_all_texts) \
        .to_csv(f"{par_base_path}results/pos_tag_n_grams.csv", index=False)


# Method to print all features for different counts of authors and texts
# Including all Preprocessing steps and filtering
def print_all_features_svc(par_base_path, par_article_path):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    author_counts = [2, 3, 4, 5, 10, 15, 25]
    text_counts = [5, 10, 15, 25, 50, 75, 100]
    for number_authors in author_counts:
        for number_texts in text_counts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # get all the features
            df_bow = get_bow_matrix(df_balanced)
            df_word_2g = get_word_n_grams(df_balanced, 2)
            df_word_count = get_word_count(df_balanced)
            df_word_length = get_word_length_matrix_with_margin(df_balanced, 20)
            df_yules_k = get_yules_k(df_balanced)

            sc_label_vector = ["!", "„", "“", "§", "$", "%", "&", "/", "(", ")", "=", "?", "{", "}", "[", "]", "\\",
                               "@", "#",
                               "‚", "‘", "-", "_", "+", "*", ".", ",", ";"]
            special_char_matrix = get_special_char_matrix(df_balanced, sc_label_vector)
            sc_label_vector = ["s_char:" + sc for sc in sc_label_vector]
            df_special_char = pd.DataFrame(data=special_char_matrix, columns=sc_label_vector)

            df_char_affix_4g = get_char_affix_n_grams(df_balanced, 4)
            df_char_word_3g = get_char_word_n_grams(df_balanced, 3)
            df_char_punct_3g = get_char_punct_n_grams(df_balanced, 3)
            df_digits = get_sum_digits(df_balanced)
            df_fwords = get_function_words(df_balanced)
            df_pos_tags = get_pos_tags(df_balanced)
            df_pos_tag_2g = get_pos_tags_n_grams(df_balanced, 2)

            df_start_pos, df_end_pos = get_sentence_end_start(df_balanced)
            df_start_end_pos = pd.concat([df_start_pos, df_end_pos], axis=1)

            df_fre = get_flesch_reading_ease_vector(df_balanced)

            # 7.1.1 Remove low occurrence
            df_bow = trim_df_by_occurrence(df_bow, 1)
            df_word_2g = trim_df_by_occurrence(df_word_2g, 1)
            df_fwords = trim_df_by_occurrence(df_fwords, 1)
            df_pos_tag_2g = trim_df_by_occurrence(df_pos_tag_2g, 1)
            df_char_affix_4g = trim_df_sum_feature(df_char_affix_4g, 5)
            df_char_word_3g = trim_df_sum_feature(df_char_word_3g, 5)
            df_char_punct_3g = trim_df_sum_feature(df_char_punct_3g, 5)

            # 7.1.2 Remove high frequency
            df_bow = trim_df_by_doc_freq(df_bow, 0.5)
            df_word_2g = trim_df_by_doc_freq(df_word_2g, 0.5)
            df_fwords = trim_df_by_doc_freq(df_fwords, 0.5)

            # 7.1.4 individual relative frequency
            df_len_metrics = pd.concat([get_char_count(df_balanced), get_sentence_count(df_balanced),
                                        df_word_count], axis=1)
            df_bow = get_rel_frequency(df_bow.fillna(value=0), df_len_metrics['word_count'])
            df_word_2g = get_rel_frequency(df_word_2g.fillna(value=0), df_len_metrics['word_count'])
            df_word_length = get_rel_frequency(df_word_length.fillna(value=0), df_len_metrics['word_count'])
            df_special_char = get_rel_frequency(df_special_char.fillna(value=0), df_len_metrics['char_count'])
            df_char_affix_4g = get_rel_frequency(df_char_affix_4g.fillna(value=0), df_len_metrics['char_count'])
            df_char_word_3g = get_rel_frequency(df_char_word_3g.fillna(value=0), df_len_metrics['char_count'])
            df_char_punct_3g = get_rel_frequency(df_char_punct_3g.fillna(value=0), df_len_metrics['char_count'])
            df_digits = get_rel_frequency(df_digits.fillna(value=0), df_len_metrics['char_count'])
            df_fwords = get_rel_frequency(df_fwords.fillna(value=0), df_len_metrics['word_count'])
            df_pos_tags = get_rel_frequency(df_pos_tags.fillna(value=0), df_len_metrics['word_count'])
            df_pos_tag_2g = get_rel_frequency(df_pos_tag_2g.fillna(value=0), df_len_metrics['word_count'])
            df_start_end_pos = get_rel_frequency(df_start_end_pos.fillna(value=0), df_len_metrics['sentence_count'])

            # Print to CSV
            # Files for iterative filter
            df_bow.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}_bow.csv", index=False)
            df_word_2g.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}"
                              f"_word_2_gram.csv", index=False)
            df_char_affix_4g.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}"
                                    f"_char_affix_4_gram.csv", index=False)
            df_char_word_3g.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}"
                                   f"_char_word_3_gram.csv", index=False)
            df_char_punct_3g.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}"
                                    f"_char_punct_3_gram.csv", index=False)
            df_fwords.to_csv(f"{par_base_path}csv_before_filter/a{number_authors}_t{number_texts}"
                             f"_function_words.csv", index=False)

            # Files not for iterative filter directly in after filter folder
            df_word_count.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                                 f"_word_count.csv", index=False)
            df_word_length.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                                  f"_word_length.csv", index=False)
            df_yules_k.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                              f"_yules_k.csv", index=False)
            df_special_char.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                                   f"_special_char.csv", index=False)
            df_digits.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                             f"_digits.csv", index=False)
            df_pos_tags.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                               f"_pos_tag.csv", index=False)
            df_pos_tag_2g.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                                 f"_pos_tag_2_gram.csv", index=False)
            df_start_end_pos.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}"
                                    f"_pos_tag_start_end.csv", index=False)
            df_fre.to_csv(f"{par_base_path}csv_after_filter/a{number_authors}_t{number_texts}_fre.csv", index=False)
            print(f"Extraction for {number_authors} authors with {number_texts} texts done. Starting iterative filter")
            # Run the iterative filter
            iterative_filter_process_svm(par_base_path, df_balanced, number_texts, number_authors)


# create a dataframe with the combined features for a specific number of authors and texts
# features can be excluded by name
def create_df_combined_features(par_path, par_num_texts, par_num_authors, par_exclude):
    path = f'{par_path}csv_after_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # Filter the files for author and text numbers
    r = re.compile(f"a{par_num_authors}_")
    files = list(filter(r.match, files))
    r = re.compile(f".*t{par_num_texts}_")
    files = list(filter(r.match, files))
    # exclude a feature by regex
    regex = re.compile(f'.*{par_exclude}')
    files = [i for i in files if not regex.match(i)]
    df_all = pd.DataFrame()
    # combine all features
    for feature in files:
        df_feature = pd.read_csv(f"{par_path}csv_after_filter/{feature}", sep=',', encoding="utf-8")
        df_all = pd.concat([df_all, df_feature], axis=1)
    return df_all


# Chapter 8.4. comparison of normalization and standardization
def compare_normalization_standardization(par_article_path, par_feature_path, par_author_texts, par_authors):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    dic_f1_results = {'without': [], 'standard': [], 'normal': [],
                      'number_authors': [], 'number_texts': []}
    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Append authors and texts
            dic_f1_results['number_authors'].append(number_authors)
            dic_f1_results['number_texts'].append(number_texts)

            df_features = create_df_combined_features(par_feature_path, number_texts, number_authors, "nothing")

            # standardization of features
            df_features_stand = copy.deepcopy(df_features)
            scaler = StandardScaler()
            df_features_stand[df_features_stand.columns] = \
                scaler.fit_transform(df_features_stand[df_features_stand.columns])

            # normalization of features
            df_features_norm = copy.deepcopy(df_features)
            normalizer = Normalizer()
            df_features_norm[df_features_norm.columns] = \
                normalizer.fit_transform(df_features_norm[df_features_norm.columns])

            x_train, x_test, x_train_stand, x_test_stand, x_train_norm, x_test_norm, label_train, label_test = \
                train_test_split(df_features, df_features_stand, df_features_norm, label,
                                 test_size=0.4, random_state=42, stratify=label)

            # append the results
            dic_f1_results['without'].append(get_f1_for_svc(x_train, x_test, label_train, label_test, cv))
            dic_f1_results['standard'].append(get_f1_for_svc(x_train_stand, x_test_stand, label_train,
                                                             label_test, cv))
            dic_f1_results['normal'].append(get_f1_for_svc(x_train_norm, x_test_norm, label_train,
                                                           label_test, cv))
            print(f"Scores for {number_authors} authors with {number_texts} texts created.")

    return pd.DataFrame(dic_f1_results)


# Chapter 8.5.1. Comparison of the individual features, data for table 21
def compare_single_features(par_article_path, par_feature_path, par_author_texts, par_authors):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    dic_results = {'number_authors': [], 'number_texts': []}

    path = f'{par_feature_path}csv_after_filter'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # get unique values for the list of the features
    feature_list = list(set([re.search(r"a\d+_t\d+_(.+?(?=$))", f).group(1) for f in files]))

    for feature in feature_list:
        dic_results[feature] = []

    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Append authors and texts
            dic_results['number_authors'].append(number_authors)
            dic_results['number_texts'].append(number_texts)

            for feature in feature_list:
                df_feature = pd.read_csv(
                    f"{par_feature_path}csv_after_filter/a{number_authors}_t{number_texts}_{feature}")

                # standardization of features
                scaler = StandardScaler()
                df_feature[df_feature.columns] = \
                    scaler.fit_transform(df_feature[df_feature.columns])
                x_train, x_test, label_train, label_test = \
                    train_test_split(df_feature, label, test_size=0.4, random_state=42, stratify=label)
                dic_results[feature].append(
                    get_f1_for_svc(x_train, x_test, label_train, label_test, cv))
            print(f"Scores for {number_authors} authors with {number_texts} texts created.")
    return pd.DataFrame(dic_results)


# Chapter 8.5.2. Get the values of the difference functions, data for table 22
def get_feature_function_difference(par_article_path, par_feature_path, par_author_texts, par_authors):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    dic_f1_wo_feature = {'wo_bow': [], 'wo_word_2_gram': [], 'wo_word_count': [], 'wo_word_length': [],
                         'wo_yules_k': [], 'wo_special_char': [], 'wo_char_affix': [], 'wo_char_word': [],
                         'wo_char_punct': [], 'wo_digits': [], 'wo_function_words': [], 'wo_pos_tag.csv': [],
                         'wo_pos_tag_2_gram': [], 'wo_pos_tag_start_end': [], 'wo_fre': [], 'number_authors': [],
                         'number_texts': []}
    dic_f1_diff_feature = {'diff_bow': [], 'diff_word_2_gram': [], 'diff_word_count': [], 'diff_word_length': [],
                           'diff_yules_k': [], 'diff_special_char': [], 'diff_char_affix': [], 'diff_char_word': [],
                           'diff_char_punct': [], 'diff_digits': [], 'diff_function_words': [], 'diff_pos_tag.csv': [],
                           'diff_pos_tag_2_gram': [], 'diff_pos_tag_start_end': [], 'diff_fre': [],
                           'number_authors': [],
                           'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:

            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Append authors and texts
            dic_f1_wo_feature['number_authors'].append(number_authors)
            dic_f1_wo_feature['number_texts'].append(number_texts)
            dic_f1_diff_feature['number_authors'].append(number_authors)
            dic_f1_diff_feature['number_texts'].append(number_texts)

            # Read the f1 Score from the previous calculations
            df_score_all = pd.read_csv(f"{par_feature_path}/results/compared_stand_normal.csv")
            f1_score_all = df_score_all.loc[(df_score_all['number_authors'] == number_authors) &
                                            (df_score_all['number_texts'] == number_texts)]['standard'].iloc[0]

            for key in dic_f1_diff_feature:
                if key != "number_authors" and key != "number_texts":
                    key = re.search(r'.+?(?=_)_(.*)', key).group(1)
                    # exclude the specific feature
                    df_features = create_df_combined_features(par_feature_path, number_texts, number_authors, key)
                    # standardization of features
                    scaler = StandardScaler()
                    df_features[df_features.columns] = \
                        scaler.fit_transform(df_features[df_features.columns])

                    x_train, x_test, label_train, label_test = \
                        train_test_split(df_features, label, test_size=0.4, random_state=42, stratify=label)

                    # append the results
                    score_wo = get_f1_for_svc(x_train, x_test, label_train, label_test, cv)
                    dic_f1_wo_feature[f'wo_{key}'].append(score_wo)
                    dic_f1_diff_feature[f'diff_{key}'].append(f1_score_all - score_wo)
                    print(f"{key} done for {number_authors} authors and {number_texts} texts.")

    return pd.DataFrame(dic_f1_wo_feature), pd.DataFrame(dic_f1_diff_feature)


# Chapter 8.5.3. Comparison of the model with or without content features, picture 28
def compare_content_features(par_article_path, par_feature_path, par_author_texts, par_authors):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    dic_results = {'wo_content_features': [], 'with_content_features': [], 'number_authors': [], 'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:
            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Append authors and texts
            dic_results['number_authors'].append(number_authors)
            dic_results['number_texts'].append(number_texts)

            # calculate f1 with all features
            df_all = create_df_combined_features(par_feature_path, number_texts, number_authors, "nothing")
            # standardization of features
            scaler = StandardScaler()
            df_all[df_all.columns] = \
                scaler.fit_transform(df_all[df_all.columns])
            x_train, x_test, label_train, label_test = \
                train_test_split(df_all, label, test_size=0.4, random_state=42, stratify=label)

            dic_results['with_content_features'].append(get_f1_for_svc(x_train, x_test, label_train, label_test, cv))

            # calculate f1 without content features
            df_wo_content = create_df_combined_features(par_feature_path, number_texts, number_authors,
                                                        "(word_count|word_2_gram|char_word_3_gram|bow)")
            # standardization of features
            scaler = StandardScaler()
            df_wo_content[df_wo_content.columns] = \
                scaler.fit_transform(df_wo_content[df_wo_content.columns])
            x_train, x_test, label_train, label_test = \
                train_test_split(df_wo_content, label, test_size=0.4, random_state=42, stratify=label)
            dic_results['wo_content_features'].append(
                get_f1_for_svc(x_train, x_test, label_train, label_test, cv))
            print(f"{number_authors} authors with {number_texts} texts compared.")

    return pd.DataFrame(dic_results)


# Chapter 8.5.3. Get the difference functions without content features, table 23
def get_feature_function_difference_without_content(par_article_path, par_feature_path, par_author_texts, par_authors):
    df_all_texts = pd.read_csv(f"{par_article_path}", sep=',', encoding="utf-8")
    dic_f1_wo_feature = {'wo_word_length': [], 'wo_yules_k': [], 'wo_special_char': [], 'wo_char_affix': [],
                         'wo_char_punct': [], 'wo_digits': [], 'wo_function_words': [], 'wo_pos_tag.csv': [],
                         'wo_pos_tag_2_gram': [], 'wo_pos_tag_start_end': [], 'wo_fre': [], 'number_authors': [],
                         'number_texts': []}
    dic_f1_diff_feature = {'diff_word_length': [], 'diff_yules_k': [], 'diff_special_char': [], 'diff_char_affix': [],
                           'diff_char_punct': [], 'diff_digits': [], 'diff_function_words': [], 'diff_pos_tag.csv': [],
                           'diff_pos_tag_2_gram': [], 'diff_pos_tag_start_end': [], 'diff_fre': [],
                           'number_authors': [], 'number_texts': []}

    for number_authors in par_authors:
        for number_texts in par_author_texts:

            index_list = get_n_article_index_by_author(df_all_texts, number_authors, number_texts)
            df_balanced = df_all_texts.iloc[index_list].reset_index(drop=True)

            # define the splits for the hyperparameter tuning, cannot be greater than number of members in each class
            if number_texts * 0.4 < 10:
                cv = int(number_texts * 0.4)
                # CV between 5 and 10 is unusual
                cv = 5 if cv > 5 else cv
            else:
                cv = 10

            label = df_balanced['label_encoded']
            # Append authors and texts
            dic_f1_wo_feature['number_authors'].append(number_authors)
            dic_f1_wo_feature['number_texts'].append(number_texts)
            dic_f1_diff_feature['number_authors'].append(number_authors)
            dic_f1_diff_feature['number_texts'].append(number_texts)

            # Read the f1 Score from the previous calculations
            df_score_all = pd.read_csv(f"{par_feature_path}/results/compare_content_feature.csv")
            f1_score_all = df_score_all.loc[(df_score_all['number_authors'] == number_authors) &
                                            (df_score_all['number_texts'] == number_texts)]['wo_content_features'].iloc[0]

            for key in dic_f1_diff_feature:
                if key != "number_authors" and key != "number_texts":
                    key = re.search(r'.+?(?=_)_(.*)', key).group(1)
                    # exclude the specific feature and all content features
                    df_features = create_df_combined_features(par_feature_path, number_texts, number_authors,
                                                              f"({key}|word_count|word_2_gram|char_word_3_gram|bow)")
                    # standardization of features
                    scaler = StandardScaler()
                    df_features[df_features.columns] = \
                        scaler.fit_transform(df_features[df_features.columns])

                    x_train, x_test, label_train, label_test = \
                        train_test_split(df_features, label, test_size=0.4, random_state=42, stratify=label)

                    # append the results
                    score_wo = get_f1_for_svc(x_train, x_test, label_train, label_test, cv)
                    dic_f1_wo_feature[f'wo_{key}'].append(score_wo)
                    dic_f1_diff_feature[f'diff_{key}'].append(f1_score_all - score_wo)
                    print(f"{key} done for {number_authors} authors and {number_texts} texts.")

    return pd.DataFrame(dic_f1_wo_feature), pd.DataFrame(dic_f1_diff_feature)


# Global options to see all lines and columns of a printed DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# The following calculations were used in the chapters for the preprocessing and modeling of the data.
# They are all commented to be able to use them separate

# # Preprocessing
# Chapter 7.1.1. filter features with low occurrence
# filter_low_occurrence()

# Chapter 7.1.2. filter features with high document frequency
# filter_high_document_frequency()

# Chapter 7.1.4. individual relative frequency
# individual_relative_frequency()

# # Feature Selection
# Chapter 7.2.1. iterative filter process
"""iterative_filter_process('daten/5_iterative_filter/', pd.read_csv("musikreviews_balanced_authors.csv", sep=',', 
                                                                  encoding="utf-8", nrows=2500), "all", "all")"""
# Chapter 7.2.2. evaluation of the iterative filter
# get_accuracy_after_iterative_filter()).to_csv(f"daten/5_iterative_filter/results/accuracy_after_filter.csv", index=False)
# get_accuracy_before_iterative_filter().to_csv(f"daten/5_iterative_filter/results/accuracy_before_filter.csv", index=False)

# # Feature alternatives
# Chapter 7.3.1. alternatives word length
# compare_word_length_features().to_csv("daten/6_feature_analysis/results/compared_word_length.csv", index=False)

# Chapter 7.3.2. alternatives digit features
# compare_digit_features().to_csv("daten/6_feature_analysis/results/compared_digit_features.csv", index=False)

# Chapter 7.3.3. alternatives word-n-grams
# compare_word_4_6_grams().to_csv("daten/6_feature_analysis/results/compared_4_6_word_grams.csv", index=False)
# compare_word_2_3_grams().to_csv("daten/6_feature_analysis/results/compared_2_3_word_grams.csv", index=False)

# Chapter 7.3.4. alternatives char-n-grams
# compare_char_n_grams_process("daten/6_feature_analysis_char_n_grams/")

# Chapter 7.3.5. alternatives pos-tag-n-grams
# compare_pos_n_grams_process("daten/6_feature_analysis_pos_tag_n_grams/")


# # Modeling
# Get all Features for different count of authors and texts. Including all preprocessing and selection steps.
# Needed for further steps in chapter 8.
# print_all_features_svc("daten/7_modeling/", "musikreviews_balanced_authors.csv")
# Chapter 8.4. compare scaling methods
"""compare_normalization_standardization("musikreviews_balanced_authors.csv","daten/7_modeling/", 
                                      [5, 10, 15, 25, 50, 75, 100], [2, 3, 4, 5, 10, 15, 25])\
    .to_csv("daten/7_modeling/results/compared_stand_normal.csv", index=False)"""

# Chapter 8.5.1. compare single features
"""compare_single_features("musikreviews_balanced_authors.csv", "daten/7_modeling/", [5, 10, 15, 25, 50, 75, 100],
                        [25]).to_csv("daten/7_modeling/results/compared_single_features.csv", index=False)"""

# Chapter 8.5.2. difference functions with content features
"""df_wo_feature, df_diff = \
    get_feature_function_difference("musikreviews_balanced_authors.csv", "daten/7_modeling/", 
                                    [5, 10, 15, 25, 50, 75, 100], [25])
df_wo_feature.to_csv("daten/7_modeling/results/f1_without_feature.csv", index=False)
df_diff.to_csv("daten/7_modeling/results/f1_difference_function", index=False)"""

# Chapter 8.5.2. compare model with and without content features
"""compare_content_features("musikreviews_balanced_authors.csv", "daten/7_modeling/",[5, 10, 15, 25, 50], [15])\
    .to_csv("daten/7_modeling/results/compare_content_feature.csv", index=False)"""

# Chapter 8.5.2. difference functions without content features
"""df_wo_feature, df_diff \
    = get_feature_function_difference_without_content("musikreviews_balanced_authors.csv", 
                                                      "daten/7_modeling/", [5, 10, 15, 25, 50, 75, 100], [25])
df_wo_feature.to_csv("daten/7_modeling/results/f1_without_feature_no_content.csv", index=False)
df_diff.to_csv("daten/7_modeling/results/f1_difference_function_no_content", index=False)"""
