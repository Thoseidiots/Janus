# broken Python: syntax error + mutable default + bare except
def bad_func(items=[]):          # mutable default arg
    try:
        result = items[10]
    except:                      # bare except
        pass
    x = y                        # NameError bait
    return result


def divide(a, b):
    return a / b                 # possible ZeroDivisionError

if __name__ == "__main__":
    divide(1, 0)
    bad_func
