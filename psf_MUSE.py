# Purpose of this code
# 1. Fit the PSF profile of a NFM observation, including the wings
# 2. Find the fractional energy in the core

# Call: run XX.py DATACUBE.fits CHANNEL_NUMBER "FIT_FUNCTION"

# FIT FUNCTION = moffat1, moffat2, moffat1_PL, moffat2_PL

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import sys
import os
import time
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

# input_file           = "./DATACUBE_FINAL_30aug2019_AO.fits"#"./DATACUBE_STD_0001.fits"
input_file = sys.argv[1]
chan_cent = sys.argv[2]
#fit_function = sys.argv[3]
rad_pixel = 175
channel_halfwidth = 100

### Functions ###
# Extract a spectrum


def spec_circ_aper(fits_cube, x0, y0, radius):
    Nz, Ny, Nx = fits_cube.shape
    x, y = np.arange(Nx), np.arange(Ny)
    xgrid, ygrid = np.meshgrid(x, y)
    rgrid = ((xgrid-x0)**2+(ygrid-y0)**2)**0.5
    w = (rgrid <= radius) & (rgrid > radius-1)
    spec = np.zeros(Nz)
    for k in range(Nz):
        img = fits_cube[k, :, :]
        spec[k] = img[w].sum()
    return spec

# Returns a sigmaclipped sum of spaxels in an annular ring around the source
# MUSE IFU artifacts pop up more clearly in linemaps, so it's better to collapse linemap first and then clip pixels
def spec_circ_aper_sigmaclip(fits_cube, chan_range, x0, y0, radius):
    Nz, Ny, Nx = fits_cube.shape
    x, y = np.arange(Nx), np.arange(Ny)
    xgrid, ygrid = np.meshgrid(x, y)
    rgrid = ((xgrid-x0)**2+(ygrid-y0)**2)**0.5
    annular_mask = (rgrid <= radius) & (rgrid > radius-1)
    linemap = np.sum(fits_cube[chan_range[0]:chan_range[1],:,:], 0)
    pixels = linemap[annular_mask]
    clipped_pixels = sigmaclip(pixels, 1000, 3)[0]
    return np.sum(clipped_pixels)/pixels.shape[0]


# Returns a masked sum of spaxels in an annular ring around the source
# MUSE IFU artifacts pop up more clearly in linemaps, so it's better to collapse linemap first and then clip pixels
def spec_circ_aper_mask(fits_cube, chan_range, x0, y0, radius):
    if radius == 0:
        pixels = np.sum(fits_cube[chan_range[0]:chan_range[1],y0,x0])
        return pixels
    Nz, Ny, Nx = fits_cube.shape
    x, y = np.arange(Nx), np.arange(Ny)
    xgrid, ygrid = np.meshgrid(x, y)
    rgrid = ((xgrid-x0)**2+(ygrid-y0)**2)**0.5
    annular_mask = (rgrid <= radius) &\
                   (rgrid > radius-1) &\
                   ~((ygrid >= 153) & (ygrid < 166) & (xgrid >= 84) & (xgrid < 147)) &\
                   ~((ygrid >= 153) & (ygrid < 166) & (xgrid >= 172) & (xgrid < 235))
    linemap = np.sum(fits_cube[chan_range[0]:chan_range[1],:,:], 0)
    pixels = linemap[annular_mask]
    return np.sum(pixels)/pixels.shape[0]


# Returns a masked sum of spaxels in an annular ring around the source
# MUSE IFU artifacts pop up more clearly in linemaps, so it's better to collapse linemap first and then clip pixels
def spec_circ_aper_masksigma(fits_cube, chan_range, x0, y0, radius):
    Nz, Ny, Nx = fits_cube.shape
    x, y = np.arange(Nx), np.arange(Ny)
    xgrid, ygrid = np.meshgrid(x, y)
    rgrid = ((xgrid-x0)**2+(ygrid-y0)**2)**0.5
    annular_mask = (rgrid <= radius) &\
                   (rgrid > radius-1) &\
                   ~((ygrid >= 153) & (ygrid < 166) & (xgrid >= 84) & (xgrid < 147)) &\
                   ~((ygrid >= 153) & (ygrid < 166) & (xgrid >= 172) & (xgrid < 235))
    linemap = np.sum(fits_cube[chan_range[0]:chan_range[1], :, :], 0)
    pixels = linemap[annular_mask]
    clipped_pixels = sigmaclip(pixels, 1000, 3)[0]
    return np.sum(clipped_pixels) / pixels.shape[0]


def moffat1_fit(r, p0, alpha, beta):
    func = p0*pow(1+(r**2/alpha**2), -beta)
    return func.ravel()


def moffat1_pl_fit(r, p0a, alpha, beta, p0c, nu):
    component_1 = p0a*pow(1+(r**2/alpha**2), -beta)
    component_2 = p0c*pow(r, nu)
    func = component_1+component_2
    return func.ravel()


def moffat2_fit(r, p0a, alpha_1, beta_1, p0b, alpha_2, beta_2):
    component_1 = p0a*pow(1+(r**2/alpha_1**2), -beta_1)
    component_2 = p0b*pow(1+(r**2/alpha_2**2), -beta_2)
    func = component_1+component_2
    return func.ravel()


def moffat2_pl_fit(r, p0a, alpha_1, beta_1, p0b, alpha_2, beta_2, p0c, nu):
    component_1 = p0a*pow(1+(r**2/alpha_1**2), -beta_1)
    component_2 = p0b*pow(1+(r**2/alpha_2**2), -beta_2)
    component_3 = p0c*pow(r, nu)
    func = component_1+component_2+component_3
    return func.ravel()


def gauss_fit(r, p0, sig):
    exp_component = pow(r, 2)/(2*sig**2)
    return p0*np.exp(-exp_component)


def draw_plot(rad, fit_function, model_spacing, model_sample_points):
    # Initiate plot
    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    frame2 = fig1.add_axes((.1, .1, .8, .2))
    frame1.set_ylabel("Normalized counts")
    frame2.set_xlabel("Radial distance [arcsec]")
    frame2.set_ylabel("Residuals")
    obs_mode = header["HIERARCH ESO INS AO FOCU1 CONFIG"]
    if header["HIERARCH ESO AOS JIT LOOP ST"]:
        obs_AO = "AO"
    else:
        obs_AO = "noAO"
    fig1.suptitle(obs_mode + "  " + obs_AO + "  " + header["DATE-OBS"] + " " + fit_function + " " + method, fontsize=20)

    # Different kinds of fit
    #component_1 = np.zeros(rad.shape)
    #component_2 = np.zeros(rad.shape)
    #component_3 = np.zeros(rad.shape)
    #model = np.zeros(rad.shape)

    # Plot the data and the model
    if header["HIERARCH ESO INS AO FOCU1 CONFIG"] == "WFM":
        rad = rad * 0.2
        frame1.set_xlim([0.2, rad[len(rad)-1]])
        frame2.set_xlim([0.2, rad[len(rad)-1]])

    elif header["HIERARCH ESO INS AO FOCU1 CONFIG"] == "NFM":
        rad = rad * 0.025
        frame1.set_xlim([0.025, rad[len(rad)-1]])
        frame2.set_xlim([0.025, rad[len(rad)-1]])

    frame1.plot(rad, fl_r_norm, "o", color="green", label="Data")
    frame1.plot(model_spacing, component_1, color="blue", label="Component 1")
    frame1.plot(model_spacing, component_2, color="green", label="Component 2")
    frame1.plot(model_spacing, component_3, color="black", label="Component 3")
    frame1.plot(model_spacing, model, color="red", label="model")
    frame1.legend()
    frame1.set_yscale("log")
    frame1.set_xscale("log")

    # Residual plot
    difference = (fl_r_norm-model_sample_points)/fl_r_norm
    frame2.plot(rad, difference, "o", color="green")
    frame2.axhline(0.1, color="black")
    frame2.set_xscale("log")
    if not os.path.exists("./" + header["OBJECT"]):
        os.mkdir("./" + header["OBJECT"])
    plt.savefig("./" + header["OBJECT"] + "/" + obs_mode + "  " + obs_AO + "  " + header["DATE-OBS"] + " " + fit_function + " " + method + ".png")
    #plt.show()
    plt.clf()

start = time.time()
####################################
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20}
plt.rcParams.update(params)
###################################

std_cube = pyfits.getdata(input_file, 1)
header = pyfits.getheader(input_file)
std_cube[np.isnan(std_cube)] = 0.0
Nz, Ny, Nx = std_cube.shape
chan_range = [int(chan_cent)-channel_halfwidth, int(chan_cent)+channel_halfwidth]

###################################
# Find the center automatically
center_find_img = np.zeros((Nx, Ny))
for i1 in range(Nx):
    for j1 in range(Ny):
        center_find_img[i1, j1] = np.sum(std_cube[chan_range[0]:chan_range[1], j1, i1])
center_find_img[np.isnan(center_find_img)] = 0.0
#plt.imshow(center_find_img);plt.show();sys.exit()  # CHECK IF YOU MADE THE CORRECT IMAGE
target_center = np.where(center_find_img == np.max(center_find_img))
#print(target_center[0], target_center[1])

# Make the radial profile
R = np.arange(1, rad_pixel, 1)
xc, yc = target_center[0], target_center[1]
fl_r = np.zeros(len(R))
rad = np.zeros(len(R))

# Calculate the background level to be subtracted from the final profiles.
background_image = np.zeros((10, 10))
for i in range(0, 10):
    for j in range(0, 10):
        background_image[j, i] = np.sum(std_cube[chan_range[0]:chan_range[1], j+250, i+50])
background = np.mean(background_image)

#for k in range(len(R)):
#    fl = spec_circ_aper(std_cube, xc, yc, R[k+1])
#    fl_linemap = np.sum(fl[chan_range[0]:chan_range[1]])
#    fl_r[k] = np.sum(fl_linemap)/(np.pi*(R[k+1]**2-R[k]**2)) - background
#    rad[k] = (R[k+1]+R[k])/2
#    print(len(R)-k)

functions = [spec_circ_aper_sigmaclip, spec_circ_aper_mask, spec_circ_aper_masksigma]
methods = ["sigma", "mask", "mask + sigma"]
for i in range(len(functions)):
    for k in range(len(R)-1):
        fl = functions[i](std_cube, chan_range, xc, yc, R[k]-1)
        method = methods[i]
        #fl = spec_circ_aper_sigmaclip(std_cube, chan_range, xc, yc, R[k+1])
        #method = "sigma"
        #fl = spec_circ_aper_mask(std_cube, chan_range, xc, yc, R[k+1])
        #method = "mask"
        #fl = spec_circ_aper_masksigma(std_cube, chan_range, xc, yc, R[k+1])
        #method = "mask + sigma"
        fl_r[k] = fl - background#np.sum(fl_linemap)/(np.pi*(R[k+1]**2-R[k]**2)) - background
        rad[k] = (R[k+1]+R[k])/2
        #print(len(R)-k)




    center_value = [np.sum(std_cube[chan_range[0]:chan_range[1], yc, xc])]
    print(center_value)
    fl_r = np.array([*center_value, *fl_r])
    fl_r[np.isnan(fl_r)] = 0.0
    fl_r_norm = np.abs(fl_r/np.max(fl_r))
    rad = np.array([0, *rad])
    err = np.sqrt(fl_r_norm)  # This can potentially be changed so we use the error from the data
    fl_r_norm = fl_r_norm[0:len(rad)-1]
    err = err[0:len(rad)-1]
    rad = rad[0:len(rad)-1]

    end = time.time()
    print(end - start)

    model_spacing = np.arange(0, 45, 0.001)

    print("Fitting 1 Moffat to the PSF profile")
    fit_function = "Moffat1"
    initial_guess = [np.max(fl_r_norm), 1.5, 1.4]
    popt, pcov = curve_fit(moffat1_fit, rad, fl_r_norm, sigma=err, p0=initial_guess)
    print(popt)
    model_sample_points = moffat1_fit(rad, popt[0], popt[1], popt[2])
    model = moffat1_fit(model_spacing, popt[0], popt[1], popt[2])
    component_1 = moffat1_fit(model_spacing, popt[0], popt[1], popt[2])
    component_2 = np.zeros(len(model_spacing))
    component_3 = np.zeros(len(model_spacing))
    draw_plot(rad, fit_function, model_spacing, model_sample_points)

    print("Fitting 1 Moffat & power law to the PSF profile")
    fit_function = "Moffat1 + PL"
    initial_guess = [1.0, 1.5, 1.1, 1.0e-2, 2.4]
    popt, pcov = curve_fit(moffat1_pl_fit, rad[1:len(rad) - 1], fl_r_norm[1:len(rad) - 1],
                           sigma=err[1:len(rad) - 1], p0=initial_guess)
    print(popt)

    model_sample_points = moffat1_pl_fit(rad, popt[0], popt[1], popt[2], popt[3], popt[4])
    model = moffat1_pl_fit(model_spacing, popt[0], popt[1], popt[2], popt[3], popt[4])
    component_1 = moffat1_fit(model_spacing, popt[0], popt[1], popt[2])
    component_2 = popt[3] * pow(model_spacing, popt[4])
    component_3 = np.zeros(len(model_spacing))
    draw_plot(rad, fit_function, model_spacing, model_sample_points)

    # For NFM
    print("Fitting 2 Moffat to the PSF profile")
    fit_function = "Moffat2"
    initial_guess = [1.0, 1.5, 1.1, 0.6, 2.0, 2.0]
    popt, pcov = curve_fit(moffat2_fit, rad, fl_r_norm, sigma=err, p0=initial_guess)
    print(popt)
    model_sample_points = moffat2_fit(rad, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    model = moffat2_fit(model_spacing, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    component_1 = moffat1_fit(model_spacing, popt[0], popt[1], popt[2])
    component_2 = moffat1_fit(model_spacing, popt[3], popt[4], popt[5])
    component_3 = np.zeros(len(model_spacing))
    draw_plot(rad, fit_function, model_spacing, model_sample_points)

    # For NFM
    print("Fitting 2 Moffat & a power law to the PSF profile")
    fit_function = "Moffat2 + PL"
    initial_guess = [1.0, 0.03, 1.1, 0.6, 0.05, 2.0, 0.2, -1.0]
    popt, pcov = curve_fit(moffat2_pl_fit, rad[1:len(rad) - 1], fl_r_norm[1:len(rad) - 1], sigma=err[1:len(rad) - 1],
                           p0=initial_guess,
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0],
                                   [1.0, 10.0, 10.0, 1.0, 10.0, 10.0, np.inf, 10.0]))
    print(popt)
    model_sample_points = moffat2_pl_fit(rad, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
    model = moffat2_pl_fit(model_spacing, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
    component_1 = moffat1_fit(model_spacing, popt[0], popt[1], popt[2])
    component_2 = moffat1_fit(model_spacing, popt[3], popt[4], popt[5])
    component_3 = popt[6] * pow(model_spacing, popt[7])
    draw_plot(rad, fit_function, model_spacing, model_sample_points)
