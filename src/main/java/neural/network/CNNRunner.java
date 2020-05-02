/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.Random;
import java.util.TimeZone;

/**
 *
 * @author abhunavgarg
 */
public class CNNRunner {

    public static void main(String[] args) throws IOException {

        int[] filterSize = {3, 3};
        int[] poolSize = {2, 2};
        int[] hiddenNodeArray = {100, 50};
        int outputNodes = 1;

        ConvoltionalNeuralNetwork cnn = new ConvoltionalNeuralNetwork(1, 2, filterSize, poolSize, "max");

        /**
         * do this for all images i.e. get Data
         */
        //<editor-fold defaultstate="collapsed" desc="TRAIN">
        String filePath = "/home/abhunavgarg/Cody/new Cody/Neural-Network/MNIST TRAIN";

        File folder = new File(filePath);
        File[] files = folder.listFiles();

        ArrayList<double[]> finalInput = new ArrayList<>();
        ArrayList<double[]> finalOutput = new ArrayList<>();

        for (int i = 0; i < files.length; i++) {
//        for (int i = 0; i < 1; i++) {
            if (i % 500 == 0) {
                System.out.println("CONVOLVED : " + i);
            }
            File file = files[i];
            double[] data = cnn.feedForward(file.getPath());
            finalInput.add(data);
            finalOutput.add(getOutput(file.getName()));
        }

        System.out.println("INPUT NODES --------------------------------------------------" + finalInput.get(0).length);
//        int[] hiddenNodeArray = {finalInput.get(0).length / 2, finalInput.get(0).length / 4};

        MultipleLayerNeuralNetwork mnn = new MultipleLayerNeuralNetwork(finalInput.get(0).length, hiddenNodeArray, outputNodes);

        for (int j = 0; j < 120000; j++) {
//        for (int j = 0; j < 1; j++) {
            if (j % 500 == 0) {
                System.out.println("TRAINED : " + j);
            }
            int rnd = new Random().nextInt(finalOutput.size());
            mnn.train(finalInput.get(rnd), finalOutput.get(rnd));
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="TEST">
        String filePath1 = "/home/abhunavgarg/Cody/new Cody/Neural-Network/MNIST TEST";

        File folder1 = new File(filePath1);
        File[] files1 = folder1.listFiles();

        ArrayList<double[]> finalInput1 = new ArrayList<>();
        ArrayList<double[]> desiredOutput = new ArrayList<>();

//        ConvoltionalNeuralNetwork cnn1 = new ConvoltionalNeuralNetwork(1, 2, filterSize, poolSize, "max");
        for (int i = 0; i < files1.length; i++) {
//        for (int i = 0; i < 1; i++) {
            if (i % 500 == 0) {
                System.out.println("CONVOLVED : " + i);
            }
            File file = files1[i];
            double[] data = cnn.feedForward(file.getPath());
            finalInput1.add(data);
            desiredOutput.add(getOutput(file.getName()));
        }

//        MultipleLayerNeuralNetwork mnn = new MultipleLayerNeuralNetwork(finalInput1.get(0).length, hiddenNodeArray, outputNodes);
        int counter = 0;
        System.out.println("INPUT NODES --------------------------------------------------" + finalInput1.get(0).length);
        for (int j = 0; j < files1.length; j++) {
            double[] calculatedOutput = mnn.feedForward(finalInput1.get(j));
            if (j % 500 == 0) {
                System.out.println("TRAINED : " + j);
                System.out.println("Desired : " + desiredOutput.get(j)[0] + " || Calculated : " + calculatedOutput[0]);
                if (Math.abs((desiredOutput.get(j)[0] - Math.round(calculatedOutput[0]))) < 0.05) {
                    counter++;
                }
            }
        }
        System.out.println("COUNTER : " + counter);
        int rate = (counter / 10000) * 100;
        System.out.println("RATE : " + rate + "%");
        //</editor-fold>

        Matrix.toMatrix(mnn.feedForward(finalInput1.get(0))).print();
        System.out.println("Output Expected" + desiredOutput.get(0)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(0))).print();
        System.out.println("Output Expected" + desiredOutput.get(11111)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(11111))).print();
        System.out.println("Output Expected" + desiredOutput.get(22222)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(22222))).print();
        System.out.println("Output Expected" + desiredOutput.get(33333)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(33333))).print();
        System.out.println("Output Expected" + desiredOutput.get(44444)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(44444))).print();
        System.out.println("Output Expected" + desiredOutput.get(55555)[0]);
        Matrix.toMatrix(mnn.feedForward(finalInput1.get(55555))).print();
    }

    private static double[] getOutput(String path) {
        String[] abc = path.split("_");
//        String fileName = abc[abc.length - 1];
//        String outputWithType = fileName.split("_")[1];
//        String[] output = outputWithType.split("\\.");
//        String abcd = output[0];

        double[] response = new double[1];
        response[0] = Double.parseDouble(abc[0]) / 10;
        return response;
//        if (response[0] == 1) {
//            response[0] = 0.1;
//            return response;
//        } else {
//            response[0] = 0.9;
//            return response;
//        }
    }

}
