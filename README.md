# sw-irradiance

This is the repository containing code to reproduce the main results from the paper "A Deep Learning Virtual Instrument for Monitoring Extreme UV Solar Spectral Irradiance".  The code included here allows to train the best performing linear+CNN model presented in the paper to map AIA data to 15 channels of EVE MEGS-A spectra, as well as to deploy this model to perform inference on new AIA data.


## Setup the data

- Download data from the stanford repo /!\ Include link to repo /!\
- You should now have a folder with AIA data, 39 channels of EVE data, and separate .npy and .csv files for the integrated EVE MEGS-A irradiance.
- Create another folder for data in which we'll create symlinks for AIA images, so as not to mess up your "clean" data folder. For instance call it "data_30mn_cadence" because that's the cadence we'll be working with here.
- In canonical_data/, run "python link_all.py --data clean_data_folder/ --base experimental_data_folder" . You should now have symlinks to AIA data in the experimental data folder.
- In canonical_data/, run "python make_join.py --eve_root clean_data_folder/EVE/np/ --aia_root clean_data_folder/ --target experimental_data_folder/". In the experimental data folder, you should now have csv files for 2011 through 2014, as well as "iso_10m.npy" and "irradiance_10m.npy".
- Run "python merge_csv.py experimental_data_path/2011.csv experimental_data_path/2012.csv experimental_data_path/2013.csv experimental_data_path/2014.csv experimental_data_path/2011p4.csv" to merge the csv files from 2011 to 2014 into one csv called 2011p4.csv.

We now have a CSV file that puts in correspondence AIA images for a given time step, to 14 channels of EVE MEGS-A spectra for that same time step. We now have to take care of the integrated EVE MEGS-A irradiance which will be the 15th channel. This integrated irradiance is contained in the '201x_eve_megsA_mean_irradiance.csv' and '201x_eve_megsA_mean_irradiance.npy' files. We need to join it to the rest of the channels.

- In canonical_data, run "python concat_EVE_arrays_totirr.py --totirr_root clean_data_folder/ --target experimental_data_folder/".  This script will join the total irradiance arrays for the different years and write it to your experimental data folder.
- Run "python join_irradiances_totirr.py --data_root experimental_data_folder/". You should now have have the files 'irradiance_30mn_14ptot.npy' and 'irradiance_30mn_14ptot.csv' in the experimental data folder. These put in correspondence AIA images for a given time step, with 14 channels of EVE MEGS-A spectra + the channel for integrated MEGS-A irradiance. 
- In canonical_data/, run "python make_splits.py --src experimental_data_folder/irradiance_30mn_14ptot.csv --splits rve" to split the csv file into train test and validation csvs.
- In canonical_data/, run "python make_normalize.py --base experimental_data_folder/ --irradiance experimental_data_folder/irradiance_30mn_14ptot.npy". This script computes normalization quantities on the training set. It should generate npy files for means and stds of AIA and EVE in your experimental data folder.

## Train and test the model

We first need to fit a linear model to output the 15 channels of EVE from the means and stds of the AIA images, with a Huber loss. We can then train a CNN to predict the residuals between EVE and the linear model's predictions of EVE.

- In canonical_code/ run "python setup_residual_totirr --base experimental_data_folder/". This will fit a linear model using means and stds of AIA images and a Huber loss, then save the means and stds as well as the model.
- In canonical_code/, run "python cfg_residual_unified_train_totirr.py --src path_to_config_files/ --data_root experimental_data_folder/ --target path_to_train_results_folder/". This will read the configuration JSON file, create the specified model using the specified parameters, and train it. It will save train and val losses as well as best performing model into the --target folder.
- To test the model, run "python cfg_residual_unified_test_totirr.py --src path_to_config_files/ --models path_to_train_results_folder/ --data_root experimental_data_folder/ --target path_to_test_results_folder/ --eve_root clean_data_folder/EVE/np/ --phase test". This will generate a text file with errors on the test set (or on whichever --phase you specified).

## Use the model for inference

If you want to deploy the model and run inference on new AIA data, you need to do the following.

- Create a data directoy for the year you want to run on, e.g. "data_folder/2015/". 
- In canonical_data/, run "python link_to_all.py --base path_to_clean_data/ --data path_to_year_folder/".
- In canonical_data/, run "python make_csv_inference.csv --data_root year_data_folder/ --target year_data_folder/ --year year_you're_running_on". This will create an index.csv file for that year.
- From the path_to_experimental_data/ you had for the training, you'll need to copy all the normalisation quantities, as well as the linear model. You can do this with "cp path_to_experimental_data/\*.np\* path_to_year_folder/"
- In canonical_code/, run python cfg_residual_unified_inference_totirr.py --src path_to_config_files/ --models path_to_train_results_folder/ --data_root path_to_year_folder/ --target path_to_inference_results_folder/". This will generate a numpy array containing the 15 channels of EVE for each AIA time step in index.csv



