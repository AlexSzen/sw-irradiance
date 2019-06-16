#!/usr/bin/env python3

'''
Helper functions for visualization of AIA and EVE from the net
'''

import matplotlib.pyplot as plt
import torch
import numpy as np

def show_image(image, mean, std):
    ''' takes a pytorch tensor as input, undoes normalization and imshows.
        std and mean are numpy arrays used in transforms'''

    image = std * image.numpy() + mean
    plt.imshow(image)



def visualize_model(model, dataloader, device, num_images = 1):

    ''' switches model to evaluation mode, visualize input + label + output for num_images
        in subplots'''


    ### keep track of plotted images and return at num_images
    plotted_images = 0
    plot_index = 0

    ### save which mode the model is in, to reset that state at the end.
    previous_mode = model.training

    ### switch to evaluation mode.
    model.eval()

    fig = plt.figure()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            for i_image in range(len(inputs)):
                ### subplot with num_images rows and 3 columns for input, output, label.
                fig.add_subplot(num_images, 3, plot_index + 1)                
                plt.title('AIA input')
                plt.imshow(inputs.cpu().numpy()[i_image,0,:,:])

                fig.add_subplot(num_images, 3, plot_index + 2)                
                plt.title('EVE output')
                plt.xlabel(r'$\lambda$ (nm)')
                plt.ylabel(r'Flux $W/m^2$')
                plt.plot(outputs.cpu().numpy()[i_image,:])
   
                fig.add_subplot(num_images, 3, plot_index + 3)
                plt.title('EVE observed')
                plt.xlabel(r'$\lambda$ (nm)')
                plt.ylabel(r'Flux $W/m^2$')
                plt.plot(labels.cpu().numpy()[i_image,:])
                
                plotted_images += 1
                plot_index += 3

                if (plotted_images == num_images):
                    plt.show()
                    model.train(mode = previous_mode)
                    return

        plt.show()
        model.train(mode = previous_mode)