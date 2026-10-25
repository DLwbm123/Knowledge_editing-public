"""A fixed literal source-reference boundary, never a substitute for answer grading."""
import re


def normalize(text):
    return ' '.join(text.casefold().split()).rstrip('.')


def boundary(tokenizer, tokens, reference):
    def decode(ids):
        return normalize(tokenizer.decode(ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False))
    full = decode(tokens)
    reference = normalize(reference)
    assert reference
    match = re.search(r'(?<!\w)' + re.escape(reference) + r'(?!\w)', full)
    if match is None:
        return dict(matched=False, reason='NO_LITERAL_FULL_REFERENCE')
    for k in range(1, len(tokens)):
        prefix = decode(tokens[:k])
        if len(prefix) >= match.end() and full.startswith(prefix):
            if tokenizer.eos_token_id in tokens[:k]:
                return dict(matched=False, reason='EOS_BEFORE_BOUNDARY')
            return dict(matched=True, prefix_tokens=k, original_next_token=tokens[k],
                        normalized_match_start=match.start(), normalized_match_end=match.end())
    return dict(matched=False, reason='NO_NEXT_TOKEN_AFTER_REFERENCE')


def selfcheck():
    class Tokenizer:
        eos_token_id = 99
        def decode(self, ids, **_):
            return ''.join({1: 'The ', 2: 'Lung', 3: 's', 4: ', ', 5: 'Heart',
                            6: ' BRAIN ', 7: 'EDEMA', 8: '.', 99: ''}[i] for i in ids)
    tokenizer = Tokenizer()
    assert not boundary(tokenizer, [1, 2, 3, 99], 'Lung')['matched']
    assert boundary(tokenizer, [1, 2, 4, 5, 99], 'Lung')['prefix_tokens'] == 2
    assert boundary(tokenizer, [6, 7, 8, 99], 'brain edema')['prefix_tokens'] == 2
    assert not boundary(tokenizer, [5], 'Heart')['matched']
    return dict(status='PASS', whole_word=True, first_boundary=True,
                case_whitespace=True, next_token_required=True)


if __name__ == '__main__':
    print(selfcheck())
