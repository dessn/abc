import numpy as np


def get_blinding_ratios():
    """ Generates uniform ratios between 0.8 and 1.2. 
    
    Integer verification is used on the first three numbers
    to ensure that everyone is playing with the same ratios

    """
    np.random.seed(54321)
    nums = np.random.randint(0, 1000, 100)
    # Make sure that random is working identically on
    # all machines by asserting the first three
    # values on my machine
    assert nums[0] == 593 and nums[1] == 650 and nums[2] == 26, \
        "THESE RANDOM VALUES ARE DIFFERENT BETWEEN MACHINES, LET SAM KNOW ASAP"

    ratios = 0.8 + 0.4 * nums / 1000.0
    return ratios[10:]  # Gives us 90 params to blind with if needed


def blind_om(oms):
    """ Returns a blinded array of oms """
    ratios = get_blinding_ratios()
    return np.array(oms) * ratios[10]


def blind_w(ws):
    """ Returns a blinded array of w """
    ratios = get_blinding_ratios()
    return np.array(ws) * ratios[20]


def blind_ol(ols):
    """ Returns a blinded array of ols """
    ratios = get_blinding_ratios()
    return np.array(ols) * ratios[30]

if __name__ == "__main__":
    get_blinding_ratios()
