/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import javax.imageio.ImageIO;

/**
 *
 * @author abhinav
 */
public class ConvoltionalNeuralNetwork {

    //<editor-fold defaultstate="collapsed" desc="misc">
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
     * ArrayList<Matrix> convolvedMatrix =
     * Convolution.convolve(channels,filter);
     *
     * PrintStream out = new PrintStream(new FileOutputStream("output.txt"));
     * System.setOut(out);
     * System.out.println(Arrays.toString(channels.toArray())); // // //
     * m.print();
     *
     * System.out.println("Reading complete.");
     */
    //</editor-fold>
    int numLayers;
    int numFilters;
    int numConvolutionalLayers;
    int numPoolingLayers;
    int filterRows;
    int filterCols;
    ArrayList<Matrix> filterArrayList;

    public ConvoltionalNeuralNetwork(int number_Of_Convolutional_Layers, int number_Of_Filters, int[] filterSize) {

        this.numConvolutionalLayers = number_Of_Convolutional_Layers;
        this.numPoolingLayers = this.numConvolutionalLayers;
        this.numFilters = number_Of_Filters;
        this.filterRows = filterSize[0];
        this.filterCols = filterSize[1];

        for (int i = 0; i < numFilters; i++) {
            this.filterArrayList.add(Matrix.getRandomMatrix(filterRows, filterCols));
        }
    }

    public void feedForward(String filePath) {

        ArrayList<Matrix> imageMatrix = getArrayListFromImage(filePath);

        for (int i = 0; i < this.numFilters; i++) {
            ArrayList<Matrix> convolvedImage = convolve(imageMatrix, this.filterArrayList.get(i));
        }
    }

    public ArrayList<Matrix> convolve(ArrayList<Matrix> channels, Matrix filter) {

        ArrayList<Matrix> resultChannels = new ArrayList<>();

        for (int i = 0; i < channels.size(); i++) {
            Matrix dataMatrix = channels.get(i);
            Matrix resultMatrix = new Matrix(dataMatrix.rows - filter.rows + 1, dataMatrix.cols - filter.cols + 1);
            Matrix[][] miniData2DArray = new Matrix[resultMatrix.rows][resultMatrix.cols];
            for (int j = 0; j < resultMatrix.rows; j++) {
                for (int k = 0; k < resultMatrix.cols; k++) {
                    Matrix miniData = new Matrix(filter.rows, filter.cols);
                    for (int l = 0; l < filter.rows; l++) {
                        for (int m = 0; m < filter.cols; m++) {
                            miniData.data[l][m] = dataMatrix.data[j + l][k + m];
                        }
                    }
                    miniData = Matrix.vectorMultiply(miniData, filter);
                    miniData2DArray[j][k] = miniData;
                }
            }
            Matrix newResultMatrix = Matrix.convert2DArrayToMatrix(miniData2DArray);
            resultChannels.add(newResultMatrix);
        }
        return resultChannels;
    }

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> = getArrayListFromImage(String filePath)">
    private ArrayList<Matrix> getArrayListFromImage(String filePath) {

        BufferedImage image = null;
        ArrayList<Matrix> imageMatrix = new ArrayList<>();

        // READ IMAGE 
        try {
            File input_file = new File(filePath); //image file path 
            /**
             * create an object of BufferedImage type and pass as parameter the
             * width, height and image int type.TYPE_INT_ARGB means that we are
             * representing the Alpha, Red, Green and Blue component of the
             * image pixel using 8 bit integer value.
             *
             */
            BufferedImage imageMeta = ImageIO.read(input_file);
            int imageWidth = imageMeta.getWidth();
            int imageHeight = imageMeta.getHeight();
            int imageType = imageMeta.getType();
            int numberOfComponenets = imageMeta.getColorModel().getColorSpace().getNumComponents();
            if (imageMeta.getColorModel().hasAlpha()) {
                numberOfComponenets++;
            }

            image = new BufferedImage(imageWidth, imageHeight, imageType);

            // Reading input file and saving in buffered image initialized object
            image = ImageIO.read(input_file);

            for (int i = 0; i < numberOfComponenets; i++) {
                //for each type of component ie RGBA craete a new Matrix of image dimension and add to arraylist
                imageMatrix.add(new Matrix(imageWidth, imageHeight));
            }
            for (int i = 0; i < imageWidth; i++) {
                for (int j = 0; j < imageHeight; j++) {
                    Color c = new Color(image.getRGB(i, j));
                    //Need to remove this hard coding
                    if (numberOfComponenets == 4) {
                        imageMatrix.get(0).data[i][j] = c.getRed();
                        imageMatrix.get(1).data[i][j] = c.getGreen();
                        imageMatrix.get(2).data[i][j] = c.getBlue();
                        imageMatrix.get(3).data[i][j] = c.getAlpha();
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
        return imageMatrix;

    }
    //</editor-fold>

}
