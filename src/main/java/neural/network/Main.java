/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 *
 * @author abhinav
 */
public class Main {

    public static void main(String[] args) {

//        System.out.println(System.currentTimeMillis());
//        for (long i = 0L; i < 20000000000000000L; i++) {
//
//        }
//        System.out.println(System.currentTimeMillis());

//        BufferedImage image = 
//        Color c = new Color(image.getRGB(j, i));
        int width = 218;    //width of the image 
        int height = 100;   //height of the image 

        BufferedImage image = null;

        // READ IMAGE 
        try {
            File input_file = new File("/home/abhunavgarg/Pictures/aaaa.jpg"); //image file path 

            /* create an object of BufferedImage type and pass 
               as parameter the width,  height and image int 
               type.TYPE_INT_ARGB means that we are representing 
               the Alpha, Red, Green and Blue component of the 
               image pixel using 8 bit integer value. */
            image = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_ARGB);

            // Reading input file 
            image = ImageIO.read(input_file);
            Matrix mRed = new Matrix(width, height);
            Matrix mGreen = new Matrix(width, height);
            Matrix mBlue = new Matrix(width, height);
            Matrix mAlpha = new Matrix(width, height);
//            image.getRGB(width, width)
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {

                    Color c = new Color(image.getRGB(i, j));
                    mRed.data[i][j] = c.getRed();
                    mGreen.data[i][j] = c.getGreen();
                    mBlue.data[i][j] = c.getBlue();
                    mAlpha.data[i][j] = c.getAlpha();
//                    System.out.println("i:"+i + " j:"+j);
                }
            }
            ArrayList<Matrix> channels = new ArrayList<>();

            Matrix filter = Matrix.getRandomMatrix(3, 3);

            channels.add(mRed);
            channels.add(mGreen);
            channels.add(mBlue);
            channels.add(mAlpha);

            ArrayList<Matrix> convolvedMatrix = Convolution.convolve(channels, filter);

            PrintStream out = new PrintStream(new FileOutputStream("output.txt"));
            System.setOut(out);
            System.out.println(Arrays.toString(channels.toArray()));
//            
//
//            m.print();

            System.out.println("Reading complete.");
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }

        /**
         * create a network with input_count, perceptron_count, output_count
         *
         */
        double[][] input = {{0, 1}, {1, 0}, {0, 0}, {1, 1}};
        double[][] output = {{1}, {1}, {0}, {0}};

        /**
         * MyInterface myInterface = (String text) -> { System.out.print(text);
         * };
         */
        ArrayList<Matrix> alm = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            Matrix m = Matrix.getRandomMatrix(9, 9);
            alm.add(m);

        }
//        Convolution conv = new Convolution(alm, new Matrix(3, 3));

//        NeuralNetwork network = new NeuralNetwork(2, 2, 1);
        MultipleLayerNeuralNetwork multiNetwork = new MultipleLayerNeuralNetwork(2, new int[]{2}, 1);

//        for (int i = 0; i < multiNetwork.weightMatrices.size(); i++) {
//            multiNetwork.weightMatrices.get(i).print();
//
//        }
//        network.wieghts_input_hidden.print();
//        network.wieghts_hidden_output.print();
//        }
        for (int i = 0; i < 100000; i++) {
            int rnd = new Random().nextInt(input.length);

            multiNetwork.train(input[rnd], output[rnd]);
//            network.train(input[rnd], output[rnd]);
        }

        System.out.println("+++++NEW WEIGHTS+++++++");

//        for (int i = 0; i < multiNetwork.weightMatrices.size(); i++) {
//            multiNetwork.weightMatrices.get(i).print();
//
//        }
//        network.wieghts_input_hidden.print();
//        network.wieghts_hidden_output.print();
        System.out.println("+++++OUTPUTS+++++++");
        //<editor-fold defaultstate="collapsed" desc="test">
        double[] testa = multiNetwork.feedForward(input[0]);
        double[] testb = multiNetwork.feedForward(input[1]);
        double[] testc = multiNetwork.feedForward(input[2]);
        double[] testd = multiNetwork.feedForward(input[3]);

//</editor-fold>
//        double[] testa = network.feedForward(input[0]);
//        double[] testb = network.feedForward(input[1]);
//        double[] testc = network.feedForward(input[2]);
//        double[] testd = network.feedForward(input[3]);
        Matrix.toMatrix(testa).print();
        Matrix.toMatrix(testb).print();
        Matrix.toMatrix(testc).print();
        Matrix.toMatrix(testd).print();

    }

}
