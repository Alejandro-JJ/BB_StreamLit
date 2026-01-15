"""
Small functions needed in several places
in BeadBuddy Streamlit
Many of these functions will be cached inside the script, 
so that they only execute if params are changed
"""
import tifffile
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.cm as cm



def load_3d_tiff(file):
    """
    It will be called from 
    "Browse files..." in ST
    """
    
    volume = tifffile.imread(file)
    if volume.ndim != 3:
        st.error(f"Invalid TIFF shape: {volume.shape}. Must be 3D (Z,Y,X).")
        return None
    return volume

def normalize_for_display(slice_2d):
    """Scale 2D slice to uint8 [0,255] for safe display.
    The images are not modified for analysis, since
    we need to take proper pixel values"""
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    if slice_max > slice_min:
        slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        slice_norm = slice_2d * 0
    return (slice_norm * 255).astype(np.uint8)

def prepare_slices_for_display_and_color(volume, canvas_display_width=400, colormap='gray'):
    """
    Takes whole volume, normalizes to (0,255)
    and resizes to respect canvas dimensions.
    If colormap is provided, it is applied to normalized slices.
    
    colormap: None or name of a matplotlib colormap (e.g. "magma", "viridis")
    """
    slices_display = []

    for z in range(volume.shape[0]):
        slice_2d = volume[z]

        # ----- Normalize to 0–255 -----
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            norm = (slice_2d - slice_min) / (slice_max - slice_min)
        else:
            norm = np.zeros_like(slice_2d, dtype=float)

        # ----- Apply colormap or grayscale -----
        if colormap is not None:
            cmap = cm.get_cmap(colormap)
            colored = cmap(norm)[:, :, :3]               # ignore alpha
            slice_disp = (colored * 255).astype(np.uint8)
        else:
            slice_disp = (norm * 255).astype(np.uint8)

        # ----- Resize -----
        img = Image.fromarray(slice_disp)
        w, h = img.size
        scale = canvas_display_width / w
        img_resized = img.resize((canvas_display_width, int(h * scale)))

        slices_display.append(np.array(img_resized))

    return np.array(slices_display)

def get_pixel_value(volume, z, x, y):
    if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
        return volume[z, y, x]
    return None

'''
Colletion of small functions necessary for the SH analysis
'''
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as sh
from scipy.spatial.transform import Rotation as R


def cart2sph(x, y, z):
    '''
    Simple transformation between cartesian and spherical coordinates
    '''
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)         # Elevation     [-pi/2, +pi/2]
    az = np.arctan2(y, x)           # Azimuth       [-pi, +pi]
    return az, el, r


def sph2cart(az, el, r):
    '''
    Simple transformation between spherical and cartesian coordinates
    '''
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def sphereFit(spX,spY,spZ):
    '''
    Fits a cloud of points to a best-fitting sphere, 
    returning its radius and center
    '''
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C

def SpherePlot(x,y,z):
    '''
    Three dimensional plot of a cloud of points
    '''
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')   
    ax.set_zlabel('Z axis')  
    
    ax.scatter(x, y, z, c='r', marker='.')   
    plt.show()

        
def SH2NP(coefficient_table):
    '''
     This function takes a Spherical Harmonics coefficient table from SHTools
     and it gives you back a 3-column matrix with the cartesian coordinates of
     the volume that they define.
     This allows an easier later manipulation of the data
     Aditionally, it can plot the 3D shape as a cloud of points
     The input is a SHGrid table, just like the internal variable coeff obtained
     after running sh.expand.SHExpandLSQ(d,lat,lon,lmax)
    '''

    # hard-coded: BAD:
    
    grid=coefficient_table.expand(lmax=20) # Evaluation in grid
    np_grid=grid.to_array()
    
    latitudes=grid.lats()   # [-90°, 90°]
    longitudes=grid.lons()  # [ 0°, 360°]
    N_lats=np.size(latitudes)
    N_lons=np.size(longitudes)
    
    a_latitudes=np.repeat(latitudes, N_lons)    # [-90°, 90°]
    a_longitudes=np.tile(longitudes, N_lats)    # [ 0°, 360°]
    a_radius=np.ndarray.flatten(np_grid) 
    spher_coord=np.vstack((a_latitudes, a_longitudes, a_radius))
    
    # Back to cartessian coordinates and plot
#    r_offset=0
    x,y,z=sph2cart(np.radians(a_longitudes-180), np.radians(a_latitudes), a_radius)#+r_offset)
#    SpherePlot(x,y,z)
#    plt.title('Spherical Harmonics expansion')
    
    return x,y,z

def Rotate(x,y,z,rot_x,rot_y,rot_z):
    '''
     Rotate a set of (x,y,z) cartesian coordinates around the origin
     The proper input are vertical arrays for the coordinates
     The rotation angles will be given in degrees
     The order of the rotation series is defined as X, then Y, then Z
    '''
    
    x=x.reshape(np.size(x),1)    
    y=y.reshape(np.size(y),1)
    z=z.reshape(np.size(z),1)
    
    coord=np.concatenate((x,y,z), axis=1)
    rotation=R.from_euler('xyz', [rot_x,rot_y,rot_z], degrees='True')
    coord_rot=rotation.apply(coord)

    x_rot=coord_rot[:,0]
    x_rot=x_rot.reshape(np.size(x_rot),1)
    
    y_rot=coord_rot[:,1]
    y_rot=y_rot.reshape(np.size(y_rot),1)

    z_rot=coord_rot[:,2]
    z_rot=z_rot.reshape(np.size(z_rot),1)
    
    return x_rot, y_rot, z_rot
    
def RotateForceLine(x,y,z,rot_x,rot_y,rot_z):
    
    x=x.reshape(np.size(x),1)    
    y=y.reshape(np.size(y),1)
    z=z.reshape(np.size(z),1)
    
    coord=np.concatenate((x,y,z), axis=1)
    rotation=R.from_euler('yxz', [rot_x,rot_y,rot_z], degrees='True')
    coord_rot=rotation.apply(coord)

    x_rot=coord_rot[:,0]
    x_rot=x_rot.reshape(np.size(x_rot),1)
    
    y_rot=coord_rot[:,1]
    y_rot=y_rot.reshape(np.size(y_rot),1)

    z_rot=coord_rot[:,2]
    z_rot=z_rot.reshape(np.size(z_rot),1)
    
    return x_rot, y_rot, z_rot  


def TransparentAxes(ax):
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        