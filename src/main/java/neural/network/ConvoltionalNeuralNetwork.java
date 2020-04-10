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

    //<editor-fold defaultstate="collapsed" desc="Global Variables">
    int numLayers;
    int numFilters;
    int numConvolutionalLayers;
    int numPoolingLayers;
    int filterRows;
    int filterCols;
    ArrayList<Matrix> filterArrayList = new ArrayList<>();   ///////////////////////////MAIN GAME PLAYER (This needs the lesson)
    String poolingType;
    int poolingRows;
    int poolingCols;
    int finalPooledImagesLength;

    ArrayList<ArrayList<Matrix>> convoledImageMatrixArrayList = new ArrayList<>();
    ArrayList<ArrayList<Matrix>> pooledImageMatrixArrayList = new ArrayList<>();
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Initialize CNN">
    public ConvoltionalNeuralNetwork(int number_Of_Convolutional_Layers, int number_Of_Filters, int[] filterSize, int[] poolingSize, String poolingType) {

        this.numConvolutionalLayers = number_Of_Convolutional_Layers;
        this.numPoolingLayers = this.numConvolutionalLayers;
        this.numFilters = number_Of_Filters;
        this.filterRows = filterSize[0];
        this.filterCols = filterSize[1];
        this.poolingType = poolingType;
        this.poolingRows = poolingSize[0];
        this.poolingCols = poolingSize[1];

        for (int i = 0; i < numFilters; i++) {
            Matrix m = Matrix.getRandomMatrix(this.filterRows, this.filterCols);
            this.filterArrayList.add(m);
        }

        /**
         * number of convolution layers = 5 number of filter {30,20,10,5,3}
         */
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Feed Forward">
    public void feedForward(String filePath) throws IOException {

        ArrayList<Matrix> imageMatrix = getArrayListFromImage(filePath);

        ArrayList<ArrayList<Matrix>> newImageMatrix = doConvolution(imageMatrix);

        int[] hidden_nodes_array = {2, 2};
//        mlnn.feedForward(newImageMatrix);
        double inputNodes = calculateInputNodesSize(newImageMatrix);

        MultipleLayerNeuralNetwork mlnn = new MultipleLayerNeuralNetwork((int) inputNodes, hidden_nodes_array, 2);

        for (int i = 0; i < newImageMatrix.size(); i++) {
            BufferedImage resultImage = getImageFromArrayList(newImageMatrix.get(i));

            File outputfile = new File("image" + i + ".jpg");
            ImageIO.write(resultImage, "jpg", outputfile);
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> newImageMatrix = doConvolution(ArrayList<Matrix> imageMatrix)">
    private ArrayList<ArrayList<Matrix>> doConvolution(ArrayList<Matrix> imageMatrix) {

        ArrayList<Matrix> newConvolvedImageMatrix = new ArrayList<>();
        ArrayList<Matrix> pooledImageMatrix = new ArrayList<>();
        ArrayList<Matrix> rectifiedImageMatrix = new ArrayList<>();

//        for (int i = 0; i < this.numConvolutionalLayers; i++) {                                       can be used if bug found
//        double loopSize = 0;
        double loopSize = getLoopSize();

//        if (this.numFilters == 1) {
//            loopSize = this.numFilters * this.numConvolutionalLayers;
//        } else {
//            loopSize = Math.pow(this.numFilters, this.numConvolutionalLayers) - 1;
//        }
        int counter = 0;
        for (int i = 0; i <= this.pooledImageMatrixArrayList.size(); i++) {
            for (int j = 0; j < this.numFilters; j++) {
                if (i == 0) {
                    newConvolvedImageMatrix = convolve(imageMatrix, this.filterArrayList.get(j));
                    rectifiedImageMatrix = doRelu(newConvolvedImageMatrix);
                    pooledImageMatrix = pool(rectifiedImageMatrix);
                    this.pooledImageMatrixArrayList.add(pooledImageMatrix);
                    counter++;
                } else {
                    newConvolvedImageMatrix = convolve(this.pooledImageMatrixArrayList.get(i - 1), this.filterArrayList.get(j));
                    rectifiedImageMatrix = doRelu(newConvolvedImageMatrix);
                    pooledImageMatrix = pool(rectifiedImageMatrix);
                    this.pooledImageMatrixArrayList.add(pooledImageMatrix);
                    counter++;
                }
            }
            if (counter >= loopSize) {
                break;
            }
        }
        return this.pooledImageMatrixArrayList;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> convolvedImageMatrix = convolve(ArrayList<Matrix> channels, Matrix filter)">
    public ArrayList<Matrix> convolve(ArrayList<Matrix> channels, Matrix filter) {

        ArrayList<Matrix> resultChannels = new ArrayList<>();
        double filterSum = filter.getSum();

        for (int i = 0; i < channels.size(); i++) {
            Matrix dataMatrix = channels.get(i);
            Matrix resultMatrix = new Matrix(dataMatrix.rows - filter.rows + 1, dataMatrix.cols - filter.cols + 1);
            for (int j = 0; j < resultMatrix.rows; j++) {
                for (int k = 0; k < resultMatrix.cols; k++) {
                    Matrix miniData = new Matrix(filter.rows, filter.cols);
                    for (int l = 0; l < filter.rows; l++) {
                        for (int m = 0; m < filter.cols; m++) {
                            miniData.data[l][m] = dataMatrix.data[j + l][k + m];
                        }
                    }
                    miniData = Matrix.scalarMultiply(miniData, filter);
//                    miniData.fixNegative();
//                    miniData.fixPositive();
                    double sum = miniData.getSum();
                    double val = sum / filterSum;

                    if (val <= 0) {
                        val = 0;
                    }
                    if (val >= 255) {
                        val = 255;
                    }
                    resultMatrix.data[j][k] = val;
                }
            }
            resultChannels.add(resultMatrix);
        }
        return resultChannels;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> rectifiedImageMatrix = doRelu(ArrayList<Matrix> newConvolvedImageMatrix)">
    public ArrayList<Matrix> doRelu(ArrayList<Matrix> newConvolvedImageMatrix) {
        ArrayList<Matrix> rectifiedImageMatrix = new ArrayList<>();
        for (Matrix m : newConvolvedImageMatrix) {
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) {
                    if (m.data[i][j] <= 0) {
                        m.data[i][j] = 0;
                    } else if (m.data[i][j] >= 255) {
                        m.data[i][j] = 255;
                    }
                }
            }
            rectifiedImageMatrix.add(m);
        }
        return rectifiedImageMatrix;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> pooledImageMatrix = pool(ArrayList<Matrix> convolvedImageMatrix)">
    private ArrayList<Matrix> pool(ArrayList<Matrix> convolvedImageMatrix) {

        ArrayList<Matrix> resultChannels = new ArrayList<>();

        int newRows = convolvedImageMatrix.get(0).rows - this.poolingRows + 1;
        int newCols = convolvedImageMatrix.get(0).cols - this.poolingCols + 1;
        for (int i = 0; i < convolvedImageMatrix.size(); i++) {
            Matrix dataMatrix = convolvedImageMatrix.get(i);
            Matrix resultMatrix = new Matrix(newRows, newCols);
            for (int j = 0; j < newRows; j++) {
                for (int k = 0; k < newCols; k++) {

                    Matrix miniData = new Matrix(this.poolingRows, this.poolingCols);
                    for (int l = 0; l < this.poolingRows; l++) {
                        for (int m = 0; m < this.poolingCols; m++) {
                            double abc = dataMatrix.data[j + l][k + m];
                            miniData.data[l][m] = abc;
                        }
                    }
                    int val = 0;
                    if (this.poolingType.equalsIgnoreCase("average")) {
                        val = miniData.getAverageValue();
                    } else if (this.poolingType.equalsIgnoreCase("max")) {
                        val = miniData.getMaximumValue();
                    }
                    if (val <= 0) {
                        val = 0;
                    }
                    if (val >= 255) {
                        val = 255;
                    }
                    resultMatrix.data[j][k] = val;
//                    resultMatrix.fixNegative();
//                    resultMatrix.fixPositive();
                }
            }
            resultChannels.add(resultMatrix);
        }
        return resultChannels;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> = getArrayListFromImage(String filePath)">
    public ArrayList<Matrix> getArrayListFromImage(String filePath) {

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

    //<editor-fold defaultstate="collapsed" desc="BufferedImage resultImage = getImageFromArrayList(ArrayList<Matrix> imageMatrix)">
    private BufferedImage getImageFromArrayList(ArrayList<Matrix> imageMatrix) throws IOException {

        BufferedImage image = new BufferedImage(imageMatrix.get(0).rows, imageMatrix.get(0).cols, BufferedImage.TYPE_3BYTE_BGR);

        if (imageMatrix.size() == 4) {
            for (int j = 0; j < imageMatrix.get(0).rows; j++) {
                for (int k = 0; k < imageMatrix.get(0).cols; k++) {
                    Color c = new Color(
                            (int) imageMatrix.get(0).data[j][k],
                            (int) imageMatrix.get(1).data[j][k],
                            (int) imageMatrix.get(2).data[j][k],
                            (int) imageMatrix.get(3).data[j][k]);
                    image.setRGB(j, k, c.getRGB());
                }
            }
        } else if (imageMatrix.size() == 3) {
            for (int j = 0; j < imageMatrix.get(0).rows; j++) {
                for (int k = 0; k < imageMatrix.get(0).cols; k++) {
                    Color c = new Color(
                            (int) imageMatrix.get(0).data[j][k],
                            (int) imageMatrix.get(1).data[j][k],
                            (int) imageMatrix.get(2).data[j][k]
                    );
                    image.setRGB(j, k, c.getRGB());
                }
            }

        }

        return image;

//        File outputfile = new File("image.jpg");
//        ImageIO.write(image, "jpg", outputfile);
//        image
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="double loopSize = getLoopSize()">
    private double getLoopSize() {
        double loopSize = 0;
        if (this.numFilters == 1) {
            loopSize = this.numFilters * this.numConvolutionalLayers;
            this.finalPooledImagesLength = (int) loopSize;
        } else {
            this.finalPooledImagesLength = (int) Math.pow(this.numFilters, this.numConvolutionalLayers);
            int count = 0;
            for (int i = 1; i <= this.numConvolutionalLayers; i++) {
                loopSize = loopSize + Math.pow(this.numFilters, this.numConvolutionalLayers - count);
                count++;
            }
        }
        return loopSize;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc=" double inputNodes = calculateInputNodesSize(ArrayList<ArrayList<Matrix>> newImageMatrix)">
    private double calculateInputNodesSize(ArrayList<ArrayList<Matrix>> newImageMatrix) {

        double total = 0;

        return newImageMatrix.get(this.pooledImageMatrixArrayList.size() - 1).get(0).cols
                * newImageMatrix.get(0).get(0).rows
                * this.finalPooledImagesLength;

    }
    //</editor-fold>

}
