from m3bench_repro.inference.llava_med import count_image_token_ids


def test_exactly_one_image_token_is_accepted() -> None:
    assert count_image_token_ids([1, 4, -200, 5], -200) == 1


def test_multiple_image_tokens_are_detectable() -> None:
    assert count_image_token_ids([-200, 4, -200], -200) == 2
