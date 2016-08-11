def runSpectral(dcb, gt):
    (classes, gmlc, clmap) = runGauss(dcb, gt)
    v0 = imshow(classes=clmap)
    (gtresults, gtErrors) = genResults(clmap, gt)
    v1 = imshow(classes=gtresults)
    v2 = imshow(classes=gtErrors)
    return (gtresults, gtErrors)

def runPCA(dcb, gt):
    pc = principal_components(dcb)
    pc_0999 = pc.reduce(fraction=0.999)
    img_pc = pc_0999.transform(dcb)
    (classes, gmlc, clmap) = runGauss(img_pc, gt)


def genResults(clmap, gt):
    gtresults = clmap * (gt!=0)
    gtErrors = gtresults * (gtresults !=gt)
    return (gtResults, gtErrors)

def runGauss(dcb, gt):
    classes = create_training_classes(dcb, gt)
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(dcb)
    return (classes, gmlc, clmap)

def displayPlots(clmap, gt, gtresults = None, gtErrors = None):
    if (gtresults and gtErrors is None): 
        (gtresults, gtErrors) = genResults(clmap, gt)
    v0 = imshow(classes=clmap)
    v1 = imshow(classes = gtresults)
    v2 = imshow(classes = gtErrors)