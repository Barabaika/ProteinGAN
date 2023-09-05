import os
import pandas as pd
import numpy as np
from common.preprocessing.dataframe import save_as_tfrecords_multithreaded
from common.bio.amino_acid import fasta_to_numpy

SPLITS = ['val', 'train']
PATH_TO_DATA_FOLDER = '../data/diff_data_train'

for split in SPLITS:
    fasta_path = os.path.join(PATH_TO_DATA_FOLDER,split, split+'.fasta')
    out_tfrecords_path = os.path.join(PATH_TO_DATA_FOLDER, split)
    # create numpy array of ids from fasta
    ids_ndarray = fasta_to_numpy(fasta_path, 254)
    # create a dataframe and store seqence_ids to it
    data = pd.DataFrame(index = range(ids_ndarray.shape[0]))
    data['sequence'] = list(ids_ndarray)
    data['Label'] = 0

    save_as_tfrecords_multithreaded(
        path = out_tfrecords_path,
        original_data= data,
        columns = ['sequence'],
        group_by_col= 'Label'
    )


