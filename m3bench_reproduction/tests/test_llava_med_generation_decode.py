from m3bench_repro.inference.llava_med import (
    GenerationSequenceContract,
    decode_generation_sequence,
)


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        names = {1: "", 10: "According", 11: " to", 2: ""}
        return "".join(names[token] for token in ids)


def test_continuation_only_sequence_is_decoded_whole() -> None:
    result = decode_generation_sequence(
        ToyTokenizer(), [1, 10, 11, 2], contract=GenerationSequenceContract.CONTINUATION_ONLY
    )
    assert result.decoded_text == "According to"
    assert result.raw_token_ids == (1, 10, 11, 2)


def test_prompt_length_slice_would_delete_continuation_only_answer() -> None:
    result = decode_generation_sequence(
        ToyTokenizer(), [1, 10, 11, 2], contract=GenerationSequenceContract.CONTINUATION_ONLY
    )
    legacy = ToyTokenizer().decode([1, 10, 11, 2][3:], skip_special_tokens=True).strip()
    assert result.decoded_text == "According to"
    assert legacy == ""
