Data should be stored as:
/%channelName/%Y/%M/%D/{AIA/HMI}%Y%M%D_%h%m_%channelNameZeroPad.pnz

for %Y = year
    %M = month
    %D = day
    %h = hour
    %m = minute
    %channelName / %channelNameZeroPad 
        is the name, zero padded if it's an AIA channel

E.g.
    304/2013/04/18/AIA20130418_0724_0304.npz
    inclination/2013/04/18/HMI20130418_0724_inclination.npz

Pipeline:
    1) dl.py downloads from jsoc to a folder
    2) scan_for_bad_fits.py finds Q!= 0 images and deletes them
    3) resize_from_fits_alt.py converts fits to npy files
    4) checkForNan.py deletes any NaN/inf npy files
    5) make_join.py / make_join_hmi.py takes the EVE data
        and joins it with AIA or HMI data, writing out a 
        per-year CSV with all complete AIA observations and
        their corresponding EVE indexes. The EVE observations
        are wrt a subsampled dataset.
        

