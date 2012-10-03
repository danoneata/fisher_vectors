""" Some standard values that can be easily reused across all the functions and
modules.

"""
DESCS_LEN = {
    'mbh': 192,
    'hog': 96,
    'hof': 108,
    'hoghof': 96 + 108,
    'mfcc': 39,
    'all': 96 + 108 + 192}
NR_PCA_COMPONENTS = 64
IP_TYPE = 'dense5.track15mbh'

def get_descs_len(desc_type):
    if 'mbh' in desc_type:
        return 192
    elif 'hoghof' in desc_type:
        return 96 + 108
    elif 'hog' in desc_type:
        return 96
    elif 'hof' in desc_type:
        return 108
    elif 'mfcc' in desc_type:
        return 39
