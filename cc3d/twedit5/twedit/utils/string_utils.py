def remove_n_chars(s, start_pos, n_chars):
    return s[:start_pos] + s[start_pos + n_chars:]


if __name__ == '__main__':
    s = 'Montreal'

    print(remove_n_chars(s, 1, 4))
