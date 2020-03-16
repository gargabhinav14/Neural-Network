/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.ArrayList;

/**
 *
 * @author abhinav
 */
public class ConvoltionalNeuralNetwork {

    /**
     *
     * About to build a Convolutional Neural Network
     *
     * What we need is
     *
     * => input image
     *
     * => convolutional layer => => => number of filters => => => filter size
     *
     * => pooling layer => => => type of pooling => => => stride (pooling matrix
     * size)
     *
     *
     *
     *
     */
    int numLayers;
    int numFilters;

    public ConvoltionalNeuralNetwork(int number_Of_Convolutional_Layers, int number_Of_Filters) {

        this.numLayers = number_Of_Convolutional_Layers;
        this.numFilters = number_Of_Filters;

    }

    public void feedForward(double[] image_As_Array, int image_Hieght_In_Pixels, int image_Width_In_Pixels, int number_Of_Channels_In_The_Image) {
        double[] imageData = image_As_Array;
        int channels = number_Of_Channels_In_The_Image;
        int[] dimensions = {image_Width_In_Pixels, image_Hieght_In_Pixels};

        /**
         * Here we will have the image as an array Maybe with some metaata at
         * the start , but currently, we must strictly accept only pixel data
         * {r,g,b,a, r,g,b,a, r,g,b,a, r,g,b,a, r,g,b,a, r,g,b,a, r,g,b,a,
         * r,g,b,a,}
         */
        /**
         * Split the array to number_Of_Channels_In_The_Image-Dimensional Array
         */
        ArrayList<Matrix> imageMatrix = createImageMatrixFromArray(imageData, dimensions, channels);
    }

    private ArrayList<Matrix> createImageMatrixFromArray(double[] imageData, int[] dimensions, int channels) {

        //        numberOfPixels = imageData.length
        //Create new arrayList of Matrices of size equal to channels
        ArrayList<ArrayList<Double>> imageMatrix = new ArrayList<ArrayList<Double>>(channels);

        for (int i = 0; i < imageData.length; i++) {
            for (int j = 0; j < channels; j++) {
                imageMatrix.get(j).add(imageData[imageData.length / channels * i + j]);//SET GREEN ARRAYLIST
            }
        }
        
        

        return null;
    }

}
