# Data profile: results/q1_common_preprocess/v1/a1_preprocessed.csv.gz

**Shape:** 51230 rows × 140 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `dataset` | unknown | 51230 | 0 |  |
| `source_domain` | unknown | 51230 | 0 |  |
| `id` | unknown | 51230 | 0 |  |
| `sub_path` | boolean | 0 | 51230 (100%) | 0 levels: ; min_group_n=0 |
| `scalar_fineweb_edu` | continuous | 51230 | 0 | mean=1.31, sd=0.647, range=[-0.652, 4.38], skew=0.78 (moderately skewed); outliers=1610 (IQR) |
| `scalar_fluency_en` | continuous | 51230 | 0 | mean=0.751, sd=0.297, range=[0.00198, 0.999], skew=-1.05 (highly skewed); -> log axis |
| `scalar_modernbert_cleanliness` | continuous | 51230 | 0 | mean=0.726, sd=0.24, range=[0.2, 1], skew=-0.61 (moderately skewed) |
| `scalar_modernbert_readability` | continuous | 51230 | 0 | mean=0.751, sd=0.22, range=[0.0162, 1], skew=-1.07 (highly skewed); outliers=1893 (IQR) |
| `scalar_modernbert_reasoning` | continuous | 51217 | 13 (0%) | mean=0.354, sd=0.25, range=[1.82e-05, 1], skew=1.27 (highly skewed); outliers=5952 (IQR); -> log axis |
| `scalar_modernbert_professionalism` | continuous | 51225 | 5 (0%) | mean=0.57, sd=0.28, range=[0.2, 1], skew=0.15 (approximately symmetric) |
| `scalar_dsir_books` | continuous | 51230 | 0 | mean=-4.7e+03, sd=1.54e+04, range=[-1.24e+06, -24], skew=-22.23 (highly skewed); outliers=4781 (IQR) |
| `scalar_dsir_wiki` | continuous | 51230 | 0 | mean=-4.83e+03, sd=1.64e+04, range=[-1.56e+06, -25], skew=-27.27 (highly skewed); outliers=4780 (IQR) |
| `scalar_dsir_math` | continuous | 51230 | 0 | mean=-4.38e+03, sd=1.41e+04, range=[-1.67e+06, -22.9], skew=-40.69 (highly skewed); outliers=4689 (IQR) |
| `scalar_qurater` | continuous | 51230 | 0 | mean=0.158, sd=2, range=[-6.97, 7.78], skew=0.45 (approximately symmetric); outliers=1113 (IQR) |
| `scalar_ad_en` | continuous | 51230 | 0 | mean=0.919, sd=0.194, range=[0.000793, 1], skew=-3.21 (highly skewed); outliers=8359 (IQR); -> log axis |
| `scalar_rps_doc_word_count` | continuous | 51230 | 0 | mean=952, sd=6.37e+03, range=[1, 5.53e+05], skew=33.91 (highly skewed); outliers=5326 (IQR); -> log axis |
| `scalar_rps_doc_num_sentences` | continuous | 51230 | 0 | mean=72.6, sd=428, range=[1, 2.51e+04], skew=25.67 (highly skewed); outliers=5598 (IQR); -> log axis |
| `scalar_rps_doc_unigram_entropy` | continuous | 51230 | 0 | mean=4.54, sd=0.868, range=[0, 7.85], skew=-0.52 (moderately skewed); outliers=958 (IQR) |
| `scalar_rps_doc_frac_unique_words` | continuous | 51230 | 0 | mean=56.3, sd=16.3, range=[5.09, 100], skew=0.04 (approximately symmetric); outliers=903 (IQR) |
| `scalar_rps_doc_frac_no_alph_words` | continuous | 51230 | 0 | mean=28.3, sd=13.9, range=[4.35, 100], skew=3.26 (highly skewed); outliers=2559 (IQR) |
| `scalar_rps_doc_frac_chars_top_2gram` | continuous | 51230 | 0 | mean=3.59, sd=4.34, range=[0, 135], skew=4.35 (highly skewed); outliers=3729 (IQR) |
| `scalar_rps_doc_frac_chars_top_3gram` | continuous | 51230 | 0 | mean=2.92, sd=4.36, range=[0, 202], skew=6.59 (highly skewed); outliers=3469 (IQR) |
| `scalar_rps_lines_uppercase_letter_fraction` | continuous | 51230 | 0 | mean=20.1, sd=14.6, range=[0, 87], skew=0.48 (approximately symmetric); outliers=24 (IQR) |
| `scalar_rps_lines_ending_with_terminal_punctution_mark` | continuous | 51230 | 0 | mean=54.6, sd=30.4, range=[0, 100], skew=0.24 (approximately symmetric) |
| `scalar_rps_lines_numerical_chars_fraction` | continuous | 51230 | 0 | mean=2.5, sd=4.12, range=[0, 72.3], skew=5.17 (highly skewed); outliers=3783 (IQR) |
| `scalar_rps_doc_mean_word_length` | continuous | 51230 | 0 | mean=6.21, sd=2.69, range=[1.93, 83], skew=4.57 (highly skewed); outliers=5712 (IQR) |
| `norm_fineweb_edu` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_fluency_en` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_modernbert_cleanliness` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_modernbert_readability` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_modernbert_reasoning` | continuous | 51217 | 13 (0%) | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_modernbert_professionalism` | continuous | 51225 | 5 (0%) | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_dsir_books` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_dsir_wiki` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_dsir_math` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_qurater` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_ad_en` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=-0.00 (approximately symmetric) |
| `norm_rps_doc_word_count` | continuous | 51230 | 0 | mean=0.516, sd=0.261, range=[0.00178, 1], skew=0.04 (approximately symmetric); -> log axis |
| `norm_rps_doc_num_sentences` | continuous | 51230 | 0 | mean=0.516, sd=0.257, range=[0.00189, 1], skew=0.08 (approximately symmetric); -> log axis |
| `norm_rps_doc_unigram_entropy` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_rps_doc_frac_unique_words` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 0.995], skew=-0.00 (approximately symmetric) |
| `norm_rps_doc_frac_no_alph_words` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0.005, 1], skew=0.00 (approximately symmetric); -> log axis |
| `norm_rps_doc_frac_chars_top_2gram` | continuous | 51230 | 0 | mean=0.495, sd=0.283, range=[0, 0.917], skew=-0.05 (approximately symmetric) |
| `norm_rps_doc_frac_chars_top_3gram` | continuous | 51230 | 0 | mean=0.471, sd=0.255, range=[0, 0.775], skew=-0.25 (approximately symmetric) |
| `norm_rps_lines_uppercase_letter_fraction` | continuous | 51230 | 0 | mean=0.5, sd=0.289, range=[0, 1], skew=0.00 (approximately symmetric) |
| `norm_rps_lines_ending_with_terminal_punctution_mark` | continuous | 51230 | 0 | mean=0.523, sd=0.317, range=[0.00896, 0.995], skew=0.15 (approximately symmetric); -> log axis |
| `norm_rps_lines_numerical_chars_fraction` | continuous | 51230 | 0 | mean=0.5, sd=0.274, range=[1.77e-19, 1], skew=-0.29 (approximately symmetric); -> log axis |
| `norm_rps_doc_mean_word_length` | continuous | 51230 | 0 | mean=0.488, sd=0.294, range=[1.78e-35, 1], skew=-0.18 (approximately symmetric); -> log axis |
| `missing_fineweb_edu` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_fluency_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_modernbert_cleanliness` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_modernbert_readability` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_modernbert_reasoning` | boolean | 51230 | 0 | 2 levels: 0(51217), 1(13); min_group_n=13 |
| `missing_modernbert_professionalism` | boolean | 51230 | 0 | 2 levels: 0(51225), 1(5); min_group_n=5 |
| `missing_dsir_books` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_dsir_wiki` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_dsir_math` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_qurater` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_ad_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_word_count` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_num_sentences` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_unigram_entropy` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_frac_unique_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_frac_no_alph_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_frac_chars_top_2gram` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_frac_chars_top_3gram` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_lines_uppercase_letter_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_lines_ending_with_terminal_punctution_mark` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_lines_numerical_chars_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `missing_rps_doc_mean_word_length` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_fineweb_edu` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_fluency_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_modernbert_cleanliness` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_modernbert_readability` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_modernbert_reasoning` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_modernbert_professionalism` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_dsir_books` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_dsir_wiki` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_dsir_math` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_qurater` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_ad_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_word_count` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_num_sentences` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_unigram_entropy` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_frac_unique_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_frac_no_alph_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_frac_chars_top_2gram` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_frac_chars_top_3gram` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_lines_uppercase_letter_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_lines_ending_with_terminal_punctution_mark` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_lines_numerical_chars_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `invalid_rps_doc_mean_word_length` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_fineweb_edu` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_fluency_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_modernbert_cleanliness` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_modernbert_readability` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_modernbert_reasoning` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_modernbert_professionalism` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_dsir_books` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_dsir_wiki` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_dsir_math` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_qurater` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_ad_en` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_word_count` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_num_sentences` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_unigram_entropy` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_frac_unique_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_frac_no_alph_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_frac_chars_top_2gram` | boolean | 51230 | 0 | 2 levels: 0(51228), 1(2); min_group_n=2 |
| `unit_suspect_rps_doc_frac_chars_top_3gram` | boolean | 51230 | 0 | 2 levels: 0(51228), 1(2); min_group_n=2 |
| `unit_suspect_rps_lines_uppercase_letter_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_lines_ending_with_terminal_punctution_mark` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_lines_numerical_chars_fraction` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_rps_doc_mean_word_length` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `outlier_fineweb_edu` | boolean | 51230 | 0 | 2 levels: 0(50537), 1(693); min_group_n=693 |
| `outlier_fluency_en` | boolean | 51230 | 0 | 2 levels: 0(42909), 1(8321); min_group_n=8321 |
| `outlier_modernbert_cleanliness` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `outlier_modernbert_readability` | boolean | 51230 | 0 | 2 levels: 0(51106), 1(124); min_group_n=124 |
| `outlier_modernbert_reasoning` | boolean | 51230 | 0 | 2 levels: 0(45101), 1(6129); min_group_n=6129 |
| `outlier_modernbert_professionalism` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `outlier_dsir_books` | boolean | 51230 | 0 | 2 levels: 0(46861), 1(4369); min_group_n=4369 |
| `outlier_dsir_wiki` | boolean | 51230 | 0 | 2 levels: 0(46851), 1(4379); min_group_n=4379 |
| `outlier_dsir_math` | boolean | 51230 | 0 | 2 levels: 0(46930), 1(4300); min_group_n=4300 |
| `outlier_qurater` | boolean | 51230 | 0 | 2 levels: 0(51112), 1(118); min_group_n=118 |
| `outlier_ad_en` | boolean | 51230 | 0 | 2 levels: 0(36952), 1(14278); min_group_n=14278 |
| `outlier_rps_doc_word_count` | boolean | 51230 | 0 | 2 levels: 0(50949), 1(281); min_group_n=281 |
| `outlier_rps_doc_num_sentences` | boolean | 51230 | 0 | 2 levels: 0(50918), 1(312); min_group_n=312 |
| `outlier_rps_doc_unigram_entropy` | boolean | 51230 | 0 | 2 levels: 0(50934), 1(296); min_group_n=296 |
| `outlier_rps_doc_frac_unique_words` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `outlier_rps_doc_frac_no_alph_words` | boolean | 51230 | 0 | 2 levels: 0(47442), 1(3788); min_group_n=3788 |
| `outlier_rps_doc_frac_chars_top_2gram` | boolean | 51230 | 0 | 2 levels: 0(47468), 1(3762); min_group_n=3762 |
| `outlier_rps_doc_frac_chars_top_3gram` | boolean | 51230 | 0 | 2 levels: 0(48731), 1(2499); min_group_n=2499 |
| `outlier_rps_lines_uppercase_letter_fraction` | boolean | 51230 | 0 | 2 levels: 0(51229), 1(1); min_group_n=1 |
| `outlier_rps_lines_ending_with_terminal_punctution_mark` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `outlier_rps_lines_numerical_chars_fraction` | boolean | 51230 | 0 | 2 levels: 0(47097), 1(4133); min_group_n=4133 |
| `outlier_rps_doc_mean_word_length` | boolean | 51230 | 0 | 2 levels: 0(45305), 1(5925); min_group_n=5925 |
| `missing_count` | boolean | 51230 | 0 | 2 levels: 0(51212), 1(18); min_group_n=18 |
| `invalid_count` | boolean | 51230 | 0 | 1 levels: 0(51230); min_group_n=51230 |
| `unit_suspect_count` | ordinal | 51230 | 0 | 2 levels: 0(51228), 2(2); min_group_n=2 |
| `outlier_count` | continuous | 51230 | 0 | mean=1.24, sd=1.4, range=[0, 9], skew=1.56 (highly skewed); outliers=784 (IQR) |

## Group structure
- Grouped by: `source_domain`
- Number of groups: 7
- Group size: min=171, median=10000, max=10000

## Correlations (Pearson, sorted by |r|)
- `norm_dsir_books` ↔ `norm_dsir_wiki` : r = 0.999 (very strong)
- `norm_dsir_books` ↔ `norm_dsir_math` : r = 0.997 (very strong)
- `norm_dsir_wiki` ↔ `norm_dsir_math` : r = 0.997 (very strong)
- `scalar_rps_lines_ending_with_terminal_punctution_mark` ↔ `norm_rps_lines_ending_with_terminal_punctution_mark` : r = 0.996 (very strong)
- `scalar_modernbert_professionalism` ↔ `norm_modernbert_professionalism` : r = 0.992 (very strong)
- `scalar_dsir_books` ↔ `scalar_dsir_wiki` : r = 0.988 (very strong)
- `scalar_rps_lines_uppercase_letter_fraction` ↔ `norm_rps_lines_uppercase_letter_fraction` : r = -0.982 (very strong)
- `scalar_rps_doc_frac_unique_words` ↔ `norm_rps_doc_frac_unique_words` : r = 0.972 (very strong)
- `scalar_qurater` ↔ `norm_qurater` : r = 0.964 (very strong)
- `scalar_rps_doc_unigram_entropy` ↔ `norm_rps_doc_unigram_entropy` : r = 0.961 (very strong)
- ... +980 more pairs

## Warnings
- 列 'sub_path' 缺失率 100% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'missing_modernbert_professionalism' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。
- 列 'unit_suspect_rps_doc_frac_chars_top_2gram' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。
- 列 'unit_suspect_rps_doc_frac_chars_top_3gram' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。
- 列 'outlier_rps_lines_uppercase_letter_fraction' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。
- 列 'unit_suspect_count' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['scalar_fineweb_edu', 'scalar_fluency_en', 'scalar_modernbert_cleanliness', 'scalar_modernbert_readability', 'scalar_modernbert_reasoning']）或 pairplot 散点矩阵
- 分类维度组合数 = 16777216（sub_path, missing_fineweb_edu, missing_fluency_en, missing_modernbert_cleanliness, missing_modernbert_readability, missing_modernbert_reasoning, missing_modernbert_professionalism, missing_dsir_books, missing_dsir_wiki, missing_dsir_math, missing_qurater, missing_ad_en, missing_rps_doc_word_count, missing_rps_doc_num_sentences, missing_rps_doc_unigram_entropy, missing_rps_doc_frac_unique_words, missing_rps_doc_frac_no_alph_words, missing_rps_doc_frac_chars_top_2gram, missing_rps_doc_frac_chars_top_3gram, missing_rps_lines_uppercase_letter_fraction, missing_rps_lines_ending_with_terminal_punctution_mark, missing_rps_lines_numerical_chars_fraction, missing_rps_doc_mean_word_length, invalid_fineweb_edu, invalid_fluency_en, invalid_modernbert_cleanliness, invalid_modernbert_readability, invalid_modernbert_reasoning, invalid_modernbert_professionalism, invalid_dsir_books, invalid_dsir_wiki, invalid_dsir_math, invalid_qurater, invalid_ad_en, invalid_rps_doc_word_count, invalid_rps_doc_num_sentences, invalid_rps_doc_unigram_entropy, invalid_rps_doc_frac_unique_words, invalid_rps_doc_frac_no_alph_words, invalid_rps_doc_frac_chars_top_2gram, invalid_rps_doc_frac_chars_top_3gram, invalid_rps_lines_uppercase_letter_fraction, invalid_rps_lines_ending_with_terminal_punctution_mark, invalid_rps_lines_numerical_chars_fraction, invalid_rps_doc_mean_word_length, unit_suspect_fineweb_edu, unit_suspect_fluency_en, unit_suspect_modernbert_cleanliness, unit_suspect_modernbert_readability, unit_suspect_modernbert_reasoning, unit_suspect_modernbert_professionalism, unit_suspect_dsir_books, unit_suspect_dsir_wiki, unit_suspect_dsir_math, unit_suspect_qurater, unit_suspect_ad_en, unit_suspect_rps_doc_word_count, unit_suspect_rps_doc_num_sentences, unit_suspect_rps_doc_unigram_entropy, unit_suspect_rps_doc_frac_unique_words, unit_suspect_rps_doc_frac_no_alph_words, unit_suspect_rps_doc_frac_chars_top_2gram, unit_suspect_rps_doc_frac_chars_top_3gram, unit_suspect_rps_lines_uppercase_letter_fraction, unit_suspect_rps_lines_ending_with_terminal_punctution_mark, unit_suspect_rps_lines_numerical_chars_fraction, unit_suspect_rps_doc_mean_word_length, outlier_fineweb_edu, outlier_fluency_en, outlier_modernbert_cleanliness, outlier_modernbert_readability, outlier_modernbert_reasoning, outlier_modernbert_professionalism, outlier_dsir_books, outlier_dsir_wiki, outlier_dsir_math, outlier_qurater, outlier_ad_en, outlier_rps_doc_word_count, outlier_rps_doc_num_sentences, outlier_rps_doc_unigram_entropy, outlier_rps_doc_frac_unique_words, outlier_rps_doc_frac_no_alph_words, outlier_rps_doc_frac_chars_top_2gram, outlier_rps_doc_frac_chars_top_3gram, outlier_rps_lines_uppercase_letter_fraction, outlier_rps_lines_ending_with_terminal_punctution_mark, outlier_rps_lines_numerical_chars_fraction, outlier_rps_doc_mean_word_length, missing_count, invalid_count, unit_suspect_count 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。
- scalar_fluency_en 跨数个量级（0.00198 ~ 0.999）→ 用对数 y 轴
- scalar_modernbert_readability 高度偏态（skew=-1.07）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_modernbert_reasoning 跨数个量级（1.82e-05 ~ 1）→ 用对数 y 轴
- scalar_dsir_books 高度偏态（skew=-22.23）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_dsir_wiki 高度偏态（skew=-27.27）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_dsir_math 高度偏态（skew=-40.69）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_ad_en 跨数个量级（0.000793 ~ 1）→ 用对数 y 轴
- scalar_rps_doc_word_count 跨数个量级（1 ~ 5.53e+05）→ 用对数 y 轴
- scalar_rps_doc_num_sentences 跨数个量级（1 ~ 2.51e+04）→ 用对数 y 轴
- scalar_rps_doc_frac_no_alph_words 高度偏态（skew=3.26）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_rps_doc_frac_chars_top_2gram 高度偏态（skew=4.35）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_rps_doc_frac_chars_top_3gram 高度偏态（skew=6.59）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_rps_lines_numerical_chars_fraction 高度偏态（skew=5.17）→ 考虑对数变换或小提琴图代替均值柱图
- scalar_rps_doc_mean_word_length 高度偏态（skew=4.57）→ 考虑对数变换或小提琴图代替均值柱图
- norm_rps_doc_word_count 跨数个量级（0.00178 ~ 1）→ 用对数 y 轴
- norm_rps_doc_num_sentences 跨数个量级（0.00189 ~ 1）→ 用对数 y 轴
- norm_rps_doc_frac_no_alph_words 跨数个量级（0.005 ~ 1）→ 用对数 y 轴
- norm_rps_lines_ending_with_terminal_punctution_mark 跨数个量级（0.00896 ~ 0.995）→ 用对数 y 轴
- norm_rps_lines_numerical_chars_fraction 跨数个量级（1.77e-19 ~ 1）→ 用对数 y 轴
- norm_rps_doc_mean_word_length 跨数个量级（1.78e-35 ~ 1）→ 用对数 y 轴
- outlier_count 高度偏态（skew=1.56）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
