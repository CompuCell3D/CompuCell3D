import chardet
from chardet.universaldetector import UniversalDetector

def decode(filename,max_lines=-1):
    """

    :param filename:
    :return:
    """

    use_max_line = False
    if max_lines > 0:
        use_max_line = True

    detector = UniversalDetector()
    i = 0
    for i, line in enumerate(open(filename, 'rb')):
        detector.feed(line)
        if detector.done: break
        if use_max_line and i > max_lines:
            break
    detector.close()
    print('detector.result')
    print('scanned lines i= ', i)

    return detector.result['encoding']