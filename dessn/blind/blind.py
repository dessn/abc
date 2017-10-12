import numpy as np


def get_indexes():
    return 10, 20, 30


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
    return np.array(oms) * ratios[get_indexes()[0]]


def unblind_om(oms):
    """ Unblinds an array of oms """
    ratios = get_blinding_ratios()
    return np.array(oms) / ratios[get_indexes()[0]]


def blind_w(ws):
    """ Returns a blinded array of w """
    ratios = get_blinding_ratios()
    return np.array(ws) * ratios[get_indexes()[1]]


def unblind_w(ws):
    """ Unblinds an array of oms """
    ratios = get_blinding_ratios()
    return np.array(ws) / ratios[get_indexes()[1]]


def blind_ol(ols):
    """ Returns a blinded array of ols """
    ratios = get_blinding_ratios()
    return np.array(ols) * ratios[get_indexes()[2]]


def unblind_ol(ols):
    """ Unblinds an array of oms """
    ratios = get_blinding_ratios()
    return np.array(ols) / ratios[get_indexes()[2]]


if __name__ == "__main__":
    get_blinding_ratios()
    test = np.random.normal(loc=1000.0, scale=100, size=10)
    assert np.all(np.isclose(test, unblind_ol(blind_ol(test)))), "Unblind/blind test failed for ols"
    assert np.all(np.isclose(test, unblind_om(blind_om(test)))), "Unblind/blind test failed for oms"
    assert np.all(np.isclose(test, unblind_w(blind_w(test)))), "Unblind/blind test failed for ws"
    print("All tests passed, good to go")
