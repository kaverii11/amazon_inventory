import numpy as np
import pandas as pd
import pytest

from invcount.abid.splits import make_splits


def fake_meta(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.choice([0, 1, 2, 3, 4, 5, 6, 9], size=n, p=[.08, .12, .2, .2, .15, .1, .1, .05])
    return pd.DataFrame({"id": np.arange(1, n + 1), "quantity": q})


def test_disjoint_balanced_and_sized():
    res = make_splits(fake_meta(), per_class=100, n_natural=200, seed=0)
    ids = [set(df.id) for df in res.values()]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            assert not ids[a] & ids[b]
    assert [len(res[k]) for k in ("train", "val", "test")] == [420, 90, 90]
    for k, per in (("train", 70), ("val", 15), ("test", 15)):
        assert res[k].quantity.value_counts().to_dict() == {c: per for c in range(6)}
    assert len(res["test_natural"]) == 200
    assert res["test_natural"].quantity.between(0, 5).all()


def test_deterministic_and_seed_dependent():
    m = fake_meta()
    a = make_splits(m, per_class=100, seed=3)
    b = make_splits(m, per_class=100, seed=3)
    c = make_splits(m, per_class=100, seed=4)
    assert all(a[k].equals(b[k]) for k in a)
    assert not a["test"].equals(c["test"])


def test_too_few_images_raises():
    with pytest.raises(ValueError):
        make_splits(fake_meta(n=500), per_class=100)


def test_ignores_missing_and_out_of_range():
    m = pd.concat([fake_meta(), pd.DataFrame({"id": [99999, 99998], "quantity": [None, 7]})])
    res = make_splits(m, per_class=100, n_natural=50)
    assert not ({99999, 99998} & set(pd.concat(res.values()).id))


def test_natural_set_keeps_pool_mix():
    m = fake_meta(n=20000, seed=1)
    pool = m[m.quantity <= 5].quantity.value_counts(normalize=True)
    nat = make_splits(m, per_class=500, n_natural=4000, seed=0)["test_natural"].quantity.value_counts(normalize=True)
    assert (abs(nat - pool) < 0.02).all()
