def des_sel():
    # return [22.1, 0.7, None, 1.0]  # Original selection effects
    return [22.4, 0.7, None, 1.0]  # After SMP and cuts


def lowz_sel():
    # return [15.5, 1.0, None, 1.0]  # As a cdf
    # return [13.70, 1.4+0.25, 3.8, 0.2]  # Original selection effect
    return [13.75, 1.45+0.0, 7.5, 0.3]  # Skew normal, but high skewness
