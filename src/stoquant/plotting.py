"""Load the noninteractive plotting backend only when drawing a figure."""


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
