
def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
