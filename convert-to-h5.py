#! /usr/bin/python3

import numpy as np
import h5py
import ghdf5
import argparse
import sys
import pathlib

GSI_HDF5_FORMAT_VERSION = 'GSI-SAR-HDF5-FORMAT-0.2'
_FORMAT_VERSION_DT = h5py.string_dtype('ascii', len(GSI_HDF5_FORMAT_VERSION))

def create_version_attribute(h5_file):
    h5_file.attrs.create('version', GSI_HDF5_FORMAT_VERSION, dtype=_FORMAT_VERSION_DT)

def write_pulses(input_file, pulses_file):
    real, imaginary = np.array(input_file['SyntheticPulses'])
    number_of_pulses, number_of_samples = real.shape

    pulses = real + imaginary * 1j

    mid = number_of_pulses // 2
    range_ = np.array(input_file['SyntheticPulseData1']['Range']).flatten()
    range_offset = np.int32(range_[mid])
    range_ -= range_offset


    deltaF = input_file.attrs['Band_Width'] / number_of_samples
    deltaF = deltaF[0]

    min_freq = input_file.attrs['RF_Frequency'] - deltaF * (number_of_samples - 1) / 2
    minF = np.ones(number_of_pulses) * np.squeeze(min_freq)

    x, y, z = np.array(input_file['PulsePositions'])

    antenna_base = np.array([coord[mid] for coord in (x, y, z)], dtype=np.int32)
    x -= antenna_base[0]
    y -= antenna_base[1]
    z -= antenna_base[2]

    pulses_file.pulses = pulses.astype(np.complex64)
    pulses_file.range = range_.astype(np.float32)
    pulses_file.range_offset = range_offset
    pulses_file.frequency_delta = deltaF.astype(np.float32)
    pulses_file.minimal_frequencies = minF.astype(np.float32)
    pulses_file.positions(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    pulses_file.antenna_base = antenna_base


def write_dtm(input_file, dtm_file):

    x = np.array(input_file['DTM']['x'])
    y = np.array(input_file['DTM']['y'])
    z = np.array(input_file['DTM']['z'])

    assert x.dtype == y.dtype == z.dtype == np.float64
    assert x.shape == y.shape == z.shape

    mid_rows = x.shape[0]//2
    mid_cols = x.shape[1]//2

    dtm_base = np.array([coord[mid_rows, mid_cols] for coord in (x, y, z)], dtype=np.int32)
    x -= dtm_base[0]
    y -= dtm_base[1]
    z -= dtm_base[2]

    dtm_file.positions(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    dtm_file.dtm_base = dtm_base

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This tool converts input HDF5 file to GSI\'s format')
    parser.add_argument('pulses', help=f'Path to a non-existing file, compatible with version {GSI_HDF5_FORMAT_VERSION} of the format')
    parser.add_argument('dtm', help=f'Path to a non-existing file, compatible with version {GSI_HDF5_FORMAT_VERSION} of the format')
    parser.add_argument('input_file', help='Path to an input HDF5 file')
    args = parser.parse_args()
    input_file_path = pathlib.Path(args.input_file)
    pulses_file_path = pathlib.Path(args.pulses)
    dtm_file_path = pathlib.Path(args.dtm)
    # TODO(opereg): catch all input errors
    if not input_file_path.exists():
        sys.exit(f'{input_file_path} doesn\'t exist!')
    elif pulses_file_path.exists():
        sys.exit(f'{pulses_file_path} already exists!')
    elif dtm_file_path.exists():
        sys.exit(f'{dtm_file_path} already exists!')
    else:
        with h5py.File(args.input_file, 'r') as input_h5_file:

            pulses_file = ghdf5.pulsesFile(args.pulses, 'w')
            dtm_file = ghdf5.dtmFile(args.dtm, 'w')

            write_pulses(input_h5_file, pulses_file)
            write_dtm(input_h5_file, dtm_file)

            pulses_file.close()
            dtm_file.close()

        sys.exit()
