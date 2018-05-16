from keras import backend as K


def linspace(*args, **kwargs):
    """
    Keras backend equivalent to numpy and TF's linspace

    Arguments:
        start (float or int): the starting point. If only two values
            are provided, the stop value
        stop (float or int): the stopping point. If only two values
            are provided, the number of points.
        n (int): the number of points to return
    """
    endpoint = kwargs.get('endpoint', True)
    if len(args) == 1:
        raise ValueError('must provide the number of points')
    elif len(args) == 2:
        stop, n = args
        start = 0
    elif len(args) == 3:
        start, stop, n = args
    else:
        raise ValueError('invalid call to linspace')

    range_ = stop - start
    if endpoint:
        step_ = range_ / (n - 1)
    else:
        step_ = range_ / n

    points = K.arange(0, n, dtype=K.floatx())
    points *= step_
    points += start

    return points
