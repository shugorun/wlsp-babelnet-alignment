from wlsp_bn.cells import surface_match
from wlsp_bn.predictions import majority

WLSP = "見出し: イヌ\n近義語: 犬・狗\n部門: 自然"


def test_majority_needs_more_than_half_of_the_runs():
    runs = [[True, True, False], [True, False, False], [False, True, True]]
    assert majority(runs) == [True, True, False]


def test_surface_match_compares_the_shown_words():
    assert surface_match(WLSP, "日見出し語: 犬")
    assert not surface_match(WLSP, "日見出し語: いぬ")
