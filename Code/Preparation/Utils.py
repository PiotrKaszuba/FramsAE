import re

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_genSuff(representation):
    if representation == 'f4':
        return '/*4*/'
    if representation == 'f9':
        return '/*9*/'
    return ''


def extract_fram(fram):
    fram = ''.join(list(fram))
    s = re.search('S[^ST]*T', fram)
    if s:
        fram = s.group()

    fram = re.sub('[ST]', '', fram)

    return fram