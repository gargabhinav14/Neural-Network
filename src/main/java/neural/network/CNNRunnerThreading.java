/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author abhunavgarg
 */
public class CNNRunnerThreading extends Thread{

    ArrayList<double[]> finalInput = new ArrayList<>();
    ArrayList<double[]> finalOutput = new ArrayList<>();
    File[] files = null;

    public static void main(String[] args) throws IOException {

        int[] filterSize = {3, 3};
        int[] poolSize = {2, 2};

        ConvoltionalNeuralNetwork cnn = new ConvoltionalNeuralNetwork(2, 2, filterSize, poolSize, "average");

        /**
         * do this for all images i.e. get Data
         */
        //<editor-fold defaultstate="collapsed" desc="TRAIN">
        String filePath = "/home/abhunavgarg/Cody/new Cody/Neural-Network/MNIST TRAIN";

        File folder = new File(filePath);
        CNNRunnerThreading obj1 = new CNNRunnerThreading();
        obj1.forLoop(folder, cnn);

        //</editor-fold>
    }

    public void forLoop(File folder, ConvoltionalNeuralNetwork cnn) throws IOException {

        files = folder.listFiles();

        for (int i = 0; i < files.length; i++) {
//        for (int i = 0; i < 500; i++) {
            if (i % 500 == 0) {
                System.out.println("CONVOLVED : " + i);
            }
            File file = files[i];
            double[] data = cnn.feedForward(file.getPath());
            finalInput.add(data);
            finalOutput.add(getOutput(file.getName()));
        }
    }

    private static double[] getOutput(String path) {
        String[] abc = path.split("_");

        double[] response = new double[1];
        response[0] = Double.parseDouble(abc[0]) / 10;
        return response;

    }

}
