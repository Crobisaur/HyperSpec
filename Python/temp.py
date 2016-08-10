def hsfft(dcb):
	'''Takes in a datacube and 2d mask'''
	temp_dcb = pyfftw.n_byte_align(dcb, 16, dtype'complex128')
	dcb_fft = pyfftw.interfaces.numpy_fft.fftn(temp_dcb)
	return dcb_fft

def genMask(h=443, w=313, offset=41, debug=False):
	'''Generates a fft mask to eliminate HS system artifacts from image'''
	mask = np.ones((h, w), dtype='float32')
	mask[:, (w/2):((w+2//2)//2)] = 0
	mask[((h/2)-offset):((h/2)+offset), :] = 1
	if debug: print(mask)
	return mask

def applyMask(dcb_fft, mask=genMask, debug=False):
	'''Applies mask to spatial information of each band in datacube.'''
	masked_fft = np.multiply(np.rollaxis(dcb_fft, 2, 0,), ft.ifftshift(mask))
	if debug: print("Shape of FFT:" + str(np.shape(masked_fft)))
	return masked_fft

def hsifft(dcb_fft, debug=False):
	'''Performs inverse fourier transform on datacube'''
	dcb_out = pyfftw.interfaces.numpy_fft.ifftn(dcb_fft)
	if debug: print("Shape of iFFT:" + str(np.shape(dcb_out)))
	return dcb_out

def dcbFilter(dcb, h=443, w=313, offset=41, debug=False):
	'''A helper function to simplify masking process and reduce confusion'''
	masked_fft = applyMask(hsfft(dcb), genMask(h, w, offset, debug))
	dcb_out = hsifft(masked_fft, debug)
	return dcb_out