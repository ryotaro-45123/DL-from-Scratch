def print_sum(x, y):
    """This function is test function

    Parameters
    ----------
        x: int
        y: int
    
    Return
    ------
        object: int

    Examples
    --------
    >>> print_sum(1, 2)
    3
    >>> print_sum(-1, 2)
    1
    """
    print(x + y)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # print_sum()