# Can x be converted into a numeric?
def isNumeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
