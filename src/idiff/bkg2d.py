'''
Module: idiff.bkg2d
-------------------

Background subtraction for 2D-arrays/images.
'''


import skimage as sk
import os
import numpy as np

def rolling_ball(arr, radius=20):
    '''
    Subtract background from an array using *rolling ball* algorithm.

    Parameters
    ----------
    arr : numpy array
        Original array.
        Usually 2D-array supplied from package stemdiff.
    radius : int, optional, default is 20
        Radius of the rolling ball

    Returns
    -------
    arr_bcorr : numpy array
        The array with the subtracted background.
    '''
    # Get background from RollingBall algorithm in sk = skimage
    background = sk.restoration.rolling_ball(arr, radius=radius)
    # Subtract background from original array
    arr_bcorr = arr - background
    # Return array with subtracted background
    return(arr_bcorr)


def deep_enhance(data, model_path, output_path=None):
    """
    Performs AI-based enhancement on diffraction data using Keras / PyTorch 
    models.

    This function automatically detects the model framework based on the file 
    extension.

    Parameters
    ----------
    data : numpy.ndarray
        Input stack of diffraction patterns
        
    model_path : str
        Path to the saved model file. 
        Supported extensions: .keras, .h5 (Keras) or .pt, .pth (PyTorch).
        
    output_path : str, optional
        Destination path for the result as a .npy file. 
        If None, results are returned but not saved to disk.

    Returns
    -------
    numpy.ndarray 
        The enhanced/predicted image in (B, H, W, C) format.


    Notes
    -----
    For PyTorch (.pt/.pth): This function assumes the model was saved as 
    a complete object (torch.save(model)). If only the state_dict was saved, 
    the architecture must be instantiated before loading.
    """
    extension = os.path.splitext(model_path)[1].lower()
    
    # (1) KERAS / TENSORFLOW
    if extension in ['.keras', '.h5']:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        prediction = model.predict(data)

    # (2) PYTORCH 
    elif extension in ['.pt', '.pth']:
        import torch
        # Note: PyTorch expects [Batch, Channels, Height, Width]
        # Swapping from [B, H, W, C] to [B, C, H, W]
        data_torch = torch.from_numpy(data).permute(0, 3, 1, 2).float()
        
        # Loading the model
        # Note: This assumes the model was saved via torch.save(model_obj)
        # If only state_dict was saved, the architecture must be instantiated 
        # first.
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        
        with torch.no_grad():
            output = model(data_torch)
            
            # Convert back to Numpy [B, H, W, C]
            prediction = output.permute(0, 2, 3, 1).cpu().numpy()

    else:
        raise ValueError(f"Unsupported model extension: {extension}")

    # Save results if path is provided
    if output_path:
        np.save(output_path, prediction)
        print(f"Predictions saved to: {output_path}")

    return prediction

