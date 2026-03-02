from maskscomp.change_detection import PairRecord


def test_no_sample_overlap_between_train_val() -> None:
    rows = [
        PairRecord("a", "s1", "p1", "c1", "0", "1", "train"),
        PairRecord("b", "s2", "p2", "c2", "0", "1", "train"),
        PairRecord("c", "s3", "p3", "c3", "0", "1", "val"),
    ]
    train = {r.sample_id for r in rows if r.split == "train"}
    val = {r.sample_id for r in rows if r.split == "val"}
    assert train.isdisjoint(val)
