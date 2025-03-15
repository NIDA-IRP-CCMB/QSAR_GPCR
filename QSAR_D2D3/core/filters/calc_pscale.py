import math

def calc_pscale(value, units):
    if units == 'fM':
        conversion_factor = 10 ** -15
    elif units == 'pM':
        conversion_factor = 10 ** -12
    elif units == 'nM':
        conversion_factor = 10 ** -9
    elif units == 'uM':
        conversion_factor = 10 ** -6
    elif units == 'mM':
        conversion_factor = 10 ** -3
    elif units == 'M':
        conversion_factor = 1
    else:
        print('Unknown units to convert: ', units)
        return -1
    pvalue = (int((math.log10(value * conversion_factor) * -1) * 100)) / 100.0

    return pvalue