import torch
from scipy.interpolate import interp1d
import numpy as np 

def sinusoidal_noise(length, Amplitude_cap=0.01 , frequency_cap= 2):
    """
    Generate a 1D sinusoidal noise vector.
    - length: number of points in the spectrum
    - Amplitude_cap: max deviation
    - frequency_cap: number of max cycles over the spectrum
    """
    x = torch.linspace(300, 1942, length)  # evenly spaced points

    noise = torch.rand(1)*Amplitude_cap * torch.sin(torch.rand(1)*frequency_cap * x + torch.rand(1)*2*torch.pi)
    return noise


def left_right_shift(wavelengths, intensities, maxshift):#not super fast
    """
    Shift the spectrum randomly left or right.
    """
    if torch.is_tensor(wavelengths):
        wavelengths = wavelengths.numpy() 

    if torch.is_tensor(intensities):
        intensities = intensities.numpy() 

    f = interp1d(wavelengths, intensities, kind='quadratic', fill_value='extrapolate')
    shift_val = (torch.rand(1).item() * 2 - 1) * maxshift
    shifted = f(wavelengths + shift_val)

    # Convert back to torch.Tensor
    return torch.tensor(shifted, dtype=torch.float32)

def reversible_scaler(array,max,reverse = False):
    """Applys a reversible scaling to the data [0,1] based on the max concentration of that compound

    Reverse = True for reverse
    max= max of the original data
    don't input max if reversing
    """

    if max is None and reverse==False:
        maximum=np.max(array,axis=0)#returns 3 element array

        for i in range(len(array)):
            array[i]=array[i]/maximum
        return array, maximum

    else:
        for i in range(len(array)):
            array[i]=array[i]*max
            array=np.maximum(array,0)
    
        return array






# def left_right_shift(wavelengths,intensities,maxshift):
#     """
#     Shift the plot to the right or to the left randomly. 
#     Mimics calibration and peaks not being in the exact same place in different machines
#     wavelengths - the x in the extrapolation
#     intensities -  the intensities in the spectrum
#     maxshift - maximum possible shift to the left or right in the spectrum.
#     some extrapolation will happen to one of the sides.
#     """
#     shift=[]
#     f = interp1d(wavelengths, intensities, kind='quadratic', fill_value='extrapolate')
#     shift.append(f(wavelengths+(torch.rand(1) * 2 - 1) * maxshift))
    
#     return shift[0]

# def left_right_shift_torch(wavelengths, intensities, maxshift):
#     """
#     Shift the spectrum left or right using PyTorch only (linear interpolation).
#     wavelengths: 1D tensor of x values
#     intensities: 1D tensor of y values
#     maxshift: max shift amount in wavelength units
#     """
#     # Random shift between -maxshift and +maxshift
#     shift_val = (torch.rand(1, device=wavelengths.device) * 2 - 1) * maxshift
    
#     # Apply shift
#     shifted_wavelengths = wavelengths + shift_val
    
#     # Linear interpolation using torch.interp
#     shifted_intensities = torch.interp(
#         shifted_wavelengths,
#         wavelengths,
#         intensities
#     )
#     return shifted_intensities


print("0")
