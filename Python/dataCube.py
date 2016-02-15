__author__ = 'Christo Robison'
#fun fact: unlike matlab, pre-allocation of vectors/lists isn't as important in python.
#O(n^2) [appending in matlab] vs O(n) [appending in python]
import LBP_thesis as LT


class dataCube:
    """An object to store HS and Metadata from a BSQ file"""
    type = 'dataStruct'
    def __init__(self, name, hSData = None, lambdas = None, maskData = None):
        self.Name = name
        self.HSdata = hSData
        self.MaskData = maskData
        self.Lambdas = lambdas
        self.lbpData = None
        self.radius = 3
        self.n_points = 8 * self.radius

    def setMaskData(self, maskData):
        self.MaskData = maskData


    def extractVects(self, mask = None):
        if mask is None:
            _mask = self.setMaskData()
        else:
            _mask = mask




    def find_LBP(self, mode = 'uniform', radius = None, points = None):
        if radius is None | points is None:
            if radius is None:
                radius = self.radius
            if points is None:
                points = self.n_points

        #need to run for loop to generate lbp for each band
        #for x in range(len(self.HSdata)):
        lbp = LT.gen_LBP(self.HSdata, points, radius)
        self.LBPData = lbp
