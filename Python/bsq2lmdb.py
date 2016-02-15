#!/usr/bin/python
import readBSQ
import numpy as np
import lmdb
import caffe


N = 1000

# Example data

def bsq2lmdb(decodedBSQ, outFileName, debug=False):
    iData = decodedBSQ[0] # dcb data
    iLabl = decodedBSQ[1] # wavelength names
    map_size = iData.nbytes * 10

    env = lmdb.open(outFileName, map_size=map_size)

    with env.begin(write=True) as txn:
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = iData.shape[3]
            datum.height = iData.shape[2]
            datum.width = iData.shape[1]
            datum.data = iData[i].tostring()
            datum.label = int(iLabl[i])
            str_id = '{:08}'.format(i)  # not sure what this does

    return


if __name__=="__main__":
    print('bsq2lmdbMain')


# We need to prepare the database fo rthe size.  We'll set it to 10
# times greater than what we theoretically need.  There is little drawback to 
# setting this too big.  If you still run ito problem after raising this, you might want to try saving fewer entries in a single transaction.

   # txn is a Transaction object 
    # or .tostring() if numpy < 1.9 otherwise .tobytes()
        #the encode is only essential in python 3
        # txn.put(str_id.encode('ascii'), datum.SerializeToString())
