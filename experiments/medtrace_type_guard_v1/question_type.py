"""Conservative English answer-interface compatibility, not semantic scope."""
import re

WH = {'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how'}
AUX = {'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did', 'has', 'have', 'had',
       'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}


def question_type(question):
    if not isinstance(question, str):
        raise TypeError('Question must be text')
    text = question.strip().lower().replace('’', "'")
    text = re.sub(r'^(?:please answer the following question:|question:)\s*', '', text)
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    if not words:
        return 'unknown'
    first = {'can\'t': 'can', 'won\'t': 'will', 'shan\'t': 'shall'}.get(words[0], words[0])
    if first.endswith("n't"):
        first = first[:-3]
    # Indirect requests can demand either an open or a polar answer.
    if first in {'can', 'could', 'would', 'will', 'may', 'might'} and words[1:2] == ['you']:
        return 'unknown'
    if first in WH:
        return 'open'
    if first in AUX:
        return 'unknown' if any(w in WH or w == 'or' for w in words[1:]) else 'polar'
    return 'open' if any(w in WH for w in words) else 'unknown'


def incompatible(query, edit_question):
    # ponytail: syntax only; semantic and cross-format scope need separate validation.
    return {question_type(query), question_type(edit_question)} == {'polar', 'open'}


def selfcheck():
    cases = {'Is a lesion visible?': 'polar', 'Does this image show a lesion?': 'polar',
             "Isn't a lesion visible?": 'polar', 'Can a lesion cause pain?': 'polar',
             'What is visible?': 'open', 'This image contains what object?': 'open',
             'Where is the lesion?': 'open', 'Can you identify the lesion?': 'unknown',
             'Could you describe this image?': 'unknown', 'Can you tell me whether it is present?': 'unknown',
             'Was this a contrast CT or a non-contrast CT?': 'unknown',
             'Does this explain why the patient has pain?': 'unknown',
             'Describe this image.': 'unknown', '': 'unknown', '这是什么？': 'unknown',
             'Question: Is a lesion visible?': 'polar',
             'Please answer the following question: What is visible?': 'open'}
    for text, expected in cases.items():
        assert question_type(text) == expected, (text, question_type(text))
    assert incompatible('Is a lesion visible?', 'What disease is visible?')
    assert incompatible('What disease is visible?', 'Is a lesion visible?')
    assert not incompatible('Is a lesion visible?', 'Is this abnormal?')
    assert not incompatible('What is visible?', 'Where is the lesion?')
    assert not incompatible('Can you identify the lesion?', 'What is visible?')
    assert not incompatible('Was this contrast CT or non-contrast CT?', 'What is visible?')
    try:
        question_type(None)
    except TypeError:
        pass
    else:
        raise AssertionError('Non-text question accepted')
    return dict(status='PASS', checks=len(cases)+7)


if __name__ == '__main__':
    print(selfcheck())
