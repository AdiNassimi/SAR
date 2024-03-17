#! /usr/bin/python3
import argparse
import h5py
import numpy as np
import scipy.io
import pathlib
import sys

GSI_HDF5_FORMAT_VERSION = 'GSI-SAR-HDF5-FORMAT-0.1'
_FORMAT_VERSION_DT = h5py.string_dtype('ascii', len(GSI_HDF5_FORMAT_VERSION))

def create_version_attribute(h5_file):
    h5_file.attrs.create('version', GSI_HDF5_FORMAT_VERSION, dtype=_FORMAT_VERSION_DT)

def write_pulses(matlab_obj, pulses_file):
    subdata = matlab_obj['subData'] 
    phdata = np.squeeze(np.ascontiguousarray(subdata[0][0][8].transpose()))
    number_of_pulses = phdata.shape[0]
    number_of_samples = phdata.shape[1]

    deltaF = np.squeeze(subdata[0][0][2])
    minF = np.squeeze(subdata[0][0][3])
    if minF.shape == ():
        minF = np.float32(minF[()])
        minF = np.ones((number_of_pulses,), dtype=np.float32) * minF

    assert minF.shape == (number_of_pulses,), 'bad shape for minimal frequency or error reading it'

    AntX = np.squeeze(subdata[0][0][4]) 
    AntY = np.squeeze(subdata[0][0][5]) 
    AntZ = np.squeeze(subdata[0][0][6]) 
    R0 = np.squeeze(subdata[0][0][7]) 


    pulses_file.create_dataset('pulses', dtype=np.complex64, data=phdata)
    pulses_file.create_dataset('range', dtype=np.float32, data=R0)
    pulses_file.create_dataset('frequency_delta', (), dtype=np.float32, data=deltaF)
    pulses_file.create_dataset('minimal_frequencies', dtype=np.float32, data=minF) 

    pulses_file.create_dataset('x', dtype=np.float32, data=AntX) 
    pulses_file.create_dataset('y', dtype=np.float32, data=AntY) 
    pulses_file.create_dataset('z', dtype=np.float32, data=AntZ) 

    create_version_attribute(pulses_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This tool converts Matlab pulses files into GSI\'s format')
    parser.add_argument('pulses', help=f'Path to a non-existing file, compatible with version {GSI_HDF5_FORMAT_VERSION} of the format')
    parser.add_argument('matlab_file', help='Path to a Matlab file containing the phase history')

    args = parser.parse_args()

    if pathlib.Path(args.pulses).exists():
        sys.exit(f'{args.pulses} already exits!')
    elif not pathlib.Path(args.matlab_file).exists():
        sys.exit(f'could not find {args.matlab_file}')
    else:
        with h5py.File(args.pulses, 'w') as pulses_file:
            write_pulses(scipy.io.loadmat(args.matlab_file), pulses_file)

            sys.exit()
