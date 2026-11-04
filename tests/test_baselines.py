import numpy as np
import pandas as pd

from wlsp_bn import baselines as bl

SYNSET = "\n".join(
    [
        "日見出し語: 犬・イヌ",
        "英見出し語: dog",
        "日カテゴリー: 哺乳類",
        "日説明文:",
        "一行目",
        "二行目",
        "英説明文:",
        "a domestic animal",
    ]
)


def test_field_reads_the_lines_up_to_the_next_label():
    assert bl.field(SYNSET, "日説明文") == "一行目 二行目"
    assert bl.field(SYNSET, "日見出し語") == "犬・イヌ"
    assert bl.field(SYNSET, "中項目") == ""


def test_headword_match_needs_a_shared_word():
    pairs = pd.DataFrame(
        {
            "wlsp_full": ["見出し: いぬ\n近義語: 犬", "見出し: 猫\n近義語: "],
            "bn_1step": [SYNSET, SYNSET],
        }
    )
    assert bl.headword_match(pairs).tolist() == [1.0, 0.0]


def test_tied_scores_are_ranked_in_the_order_of_the_pairs():
    pairs = pd.DataFrame({"record_id": [1] * 20})
    assert bl.rank_in_record(pairs, np.zeros(20)).tolist() == [-i for i in range(20)]


def test_tfidf_of_identical_and_disjoint_texts():
    pairs = pd.DataFrame({"w_all": ["犬 動物", "猫"], "b_all": ["犬 動物", "車"]})
    assert np.allclose(bl.tfidf(pairs, 1), [1.0, 0.0])


def test_rank_in_record_counts_within_each_record():
    pairs = pd.DataFrame({"record_id": [1, 1, 1, 2, 2]})
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    assert bl.rank_in_record(pairs, scores).tolist() == [-2, 0, -1, -1, 0]


def test_best_threshold_separates_the_positives():
    threshold, f1 = bl.best_threshold(
        [True, False, True, False], np.array([0.9, 0.1, 0.8, 0.2])
    )
    assert f1 == 1.0
    assert 0.2 <= threshold < 0.8


def test_fit_ngram_takes_the_smallest_n_on_a_tie():
    y = [True, False]
    n, _, f1 = bl.fit_ngram(y, lambda n: np.array([1.0, 0.0]))
    assert (n, f1) == (1, 1.0)
