import pytest

from wlsp_bn.alignment import Prompt, labels_from_list, labels_from_set, requests

IDS = ["bn:00000001n", "bn:00000002n", "bn:00000003n"]
REQUEST = {
    "record_id": "1",
    "chunk": {"index": 0},
    "wlsp": {"text": ""},
    "candidates": [{"synset_id": s, "text": ""} for s in IDS],
}


def answer(*pairs):
    return {"labels": [{"synset_id": s, "label": label} for s, label in pairs]}


def test_a_repeated_candidate_keeps_its_first_label():
    got = labels_from_list(
        answer((IDS[0], "EQUAL"), (IDS[0], "NONE"), (IDS[1], "NONE"), (IDS[2], "NONE")),
        REQUEST,
    )
    assert got[IDS[0]] == "EQUAL"


def test_one_wrong_id_is_put_right():
    got = labels_from_list(
        answer((IDS[0], "NONE"), (IDS[1], "NONE"), ("bn:0000003n", "EQUAL")),
        REQUEST,
    )
    assert got[IDS[2]] == "EQUAL"


def test_a_missing_candidate_is_labelled_none():
    got = labels_from_list(answer((IDS[0], "EQUAL"), (IDS[1], "NONE")), REQUEST)
    assert got[IDS[2]] == "NONE"


def test_a_missing_candidate_is_not_filled_in_the_second_pass():
    with pytest.raises(ValueError):
        labels_from_list(
            answer((IDS[0], "EQUAL"), (IDS[1], "NONE")), REQUEST, fill=False
        )


def test_two_missing_candidates_reject_the_answer():
    with pytest.raises(ValueError):
        labels_from_list(answer((IDS[0], "EQUAL")), REQUEST)


def test_an_unknown_label_rejects_the_answer():
    with pytest.raises(ValueError):
        labels_from_list(
            answer((IDS[0], "SAME"), (IDS[1], "NONE"), (IDS[2], "NONE")), REQUEST
        )


def test_a_listed_id_matches_without_leading_zeros():
    got = labels_from_set({"equal_synset_ids": ["bn:2n"]}, REQUEST)
    assert got == {IDS[0]: "NONE", IDS[1]: "EQUAL", IDS[2]: "NONE"}


def test_requests_split_a_record_into_chunks():
    config = {"chunk": 2, "wlsp_field": "wlsp_full", "bn_field": "bn_1step"}
    prompt = Prompt("p", "", {}, config)
    record = {
        "record_id": "1",
        "wlsp": {"wlsp_full": "w"},
        "candidates": [{"synset_id": s, "bn_1step": s[-2]} for s in IDS],
    }
    got = requests([record], prompt)
    assert [r["chunk"]["index"] for r in got] == [0, 1]
    assert [c["text"] for r in got for c in r["candidates"]] == ["1", "2", "3"]
    assert got[0]["wlsp"] == {"text": "w"}
