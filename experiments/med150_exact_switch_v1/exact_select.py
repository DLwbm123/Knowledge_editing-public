"""Allow modal switching only on exact normalized candidate-source question."""
import unicodedata


def normalize(value):
    assert isinstance(value,str)
    return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def select(original,modal,question,candidate_sources):
    if original is None:
        assert modal is None
        return None
    if modal==original:return original
    assert modal is not None and len(candidate_sources)==5
    return modal if normalize(question) in {normalize(x) for x in candidate_sources} else original


def selfcheck():
    src=['What is here?','Who?','Which?','Count?','Other?']
    assert select(None,None,'What is here?',src) is None
    assert select('a','a','unknown',src)=='a'
    assert select('a','b',' WHAT  IS HERE? ',src)=='b'
    assert select('a','b','What is here',src)=='a'
    assert select('a','b','Something else?',src)=='a'
    return dict(status='PASS',exact_normalization=True,OFF_preserved=True,punctuation_retained=True)

if __name__=='__main__':print(selfcheck())
