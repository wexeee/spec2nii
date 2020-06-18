from pathlib import Path
import re
import numpy as np
import warnings


def read_bruker_series(dir_path):
    """ Extract data from a bruker series
    containing a method and fid file.
    """

    dir_path = Path(dir_path)
    params = read_method_file(dir_path)

    reps = params['PVM_NRepetitions']
    avgs = params['PVM_NAverages']
    acquisitions = int(reps*avgs)
    fid_points = int(params['PVM_DigNp'])
    bandwidth = params['PVM_DigSw']
    dwelltime = 1/bandwidth
    shift = int(params['PVM_DigShift'])
    n_coils = int(params['PVM_EncNReceivers'])
    imaging_frequency = params['PVM_FrqRef'][0]
    echo_time = params['PVM_EchoTime']*1E-3
    points_to_read = int(acquisitions * fid_points * 2 * n_coils)

    cmplx_fid = read_fid_file(dir_path, points_to_read)

    # Reshape to fid_points,n_coils,acquisitions
    cmplx_fid = cmplx_fid.reshape((acquisitions, n_coils, fid_points)).T

    # Remove shift points
    cmplx_fid = cmplx_fid[shift:, :, :]

    affine = affine_from_params(params)

    headers = params
    headers.update({'ImagingFrequency': imaging_frequency})
    headers.update({'Dwelltime': dwelltime})
    headers.update({'EchoTime': echo_time})

    return cmplx_fid, affine, dwelltime, headers


def affine_from_params(params):
    """ Create 4x4 affine from parameter dict"""
    orientation = np.squeeze(params['PVM_VoxArrGradOrient'])
    shift = np.squeeze(params['PVM_VoxArrPosition'])
    warnings.warn('The orientation of bruker data is mostly untested.')
    # shift[0] *= -1
    # shift[1] *= -1
    # shift[2] *= -1    
    size = np.squeeze(params['PVM_VoxArrSize'])
    affine = np.zeros((4, 4))
    affine[3, 3] = 1
    affine[:3, :3] = orientation*size[[0, 2, 1]]
    affine[:3, 3] = shift[[0, 2, 1]]
    affine
    return affine


def read_fid_file(dir_path, points_to_read):
    """ Read fid file in dir"""
    fid_file = dir_path / 'fid'

    fid_file_sz = fid_file.stat().st_size
    fid_file_points = fid_file_sz / 4

    if fid_file_points != points_to_read:
        print(f'Expected size = {points_to_read}'
              f' but file size is {fid_file_points}!')

    fid_raw = np.fromfile(fid_file,
                          dtype=np.int32,
                          count=points_to_read)

    return fid_raw[0::2] + 1j * fid_raw[1::2]


def read_method_file(dir_path):
    """Read method file and parse parameters."""

    mthd_file = dir_path / 'method'

    with open(mthd_file, 'r') as mf:
        lines = mf.readlines()

    params = {}
    for idx, line in enumerate(lines):
        # Read lines with ##$
        if line.startswith('##$'):
            params.update(read_hash_hash_dollar(line, lines, idx))

    return params


def read_hash_hash_dollar(line, lines, index):
    # Deal with those lines containing arrays on subsequent lines
    match = re.search(r'\#\#\$(\w*)=\( ([\d,\s]*) \)', line.strip())
    # Deal with those lines containing arrays inline (can spread to next line)
    match2 = re.search(r'\#\#\$(\w*)=\((.*)', line.strip())
    if match:
        param_name = match[1]
        size = np.fromstring(match[2].strip(), dtype=int, sep=',')

        value = ''
        while not lines[index+1].startswith(('##$', '$$ @vis')):
            value += lines[index+1].strip()
            index += 1
        try:
            num_value = np.fromstring(value, sep=' ')
            value = num_value.reshape(size)
        except ValueError:
            pass

    elif match2:
        param_name = match2[1]
        if match2[2].endswith(',') and \
           not lines[index+1].startswith(('##$', '$$ @vis')):
            value = match2[2]
            # Look over next lines
            while not lines[index+1].startswith(('##$', '$$ @vis')):
                value += lines[index+1].rstrip(')\n')
                index += 1
        else:
            value = match2[2].rstrip(')')

    else:
        match = re.search(r'\#\#\$(\w*)=(.*)', line.strip())
        param_name = match[1]
        try:
            value = float(match[2])
        except ValueError:
            value = match[2]

    return {param_name: value}
