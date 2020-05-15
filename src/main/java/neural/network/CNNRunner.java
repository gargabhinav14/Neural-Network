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
public class CNNRunner {

    public static void main(String[] args) throws IOException {

        int[] filterSize = {3, 3};
        int[] poolSize = {2, 2};
        int[] hiddenNodeArray = {300, 100};
        int outputNodes = 1;

        ConvoltionalNeuralNetwork cnn = new ConvoltionalNeuralNetwork(2, 2, filterSize, poolSize, "average");

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
//        for (int i = 0; i < 500; i++) {
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

        //<editor-fold defaultstate="collapsed" desc="write arrayList to a file">
        try {
            FileOutputStream writeData = new FileOutputStream("finalInput.ser");
            ObjectOutputStream writeStream = new ObjectOutputStream(writeData);

            writeStream.writeObject(finalInput);
            writeStream.flush();
            writeStream.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
//</editor-fold>
        MultipleLayerNeuralNetwork mnn = new MultipleLayerNeuralNetwork(finalInput.get(0).length, hiddenNodeArray, outputNodes);

        for (int j = 0; j < 120000; j++) {
//        for (int j = 0; j < 500; j++) {
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
//        for (int i = 0; i < 500; i++) {
            if (i % 500 == 0) {
                System.out.println("CONVOLVED : " + i);
            }
            File file = files1[i];
            double[] data = cnn.feedForward(file.getPath());
            finalInput1.add(data);
            desiredOutput.add(getOutput(file.getName()));
        }

//        MultipleLayerNeuralNetwork mnn = new MultipleLayerNeuralNetwork(finalInput1.get(0).length, hiddenNodeArray, outputNodes);
        Integer counter = 0;
        System.out.println("INPUT NODES --------------------------------------------------" + finalInput1.get(0).length);
        for (int j = 0; j < files1.length; j++) {
//        for (int j = 0; j < 500; j++) {
            double[] calculatedOutput = mnn.feedForward(finalInput1.get(j));
            if (j % 500 == 0) {
                System.out.println("TRAINED : " + j);
                System.out.println("Desired : " + desiredOutput.get(j)[0] + " || Calculated : " + calculatedOutput[0]);
            }
            double diff = Math.abs(desiredOutput.get(j)[0] - calculatedOutput[0]);
            if (diff < 0.5) {
                counter++;
            }
        }
        System.out.println("COUNTER : " + counter);
        double rate = counter * (100 / finalInput1.size());
        System.out.println("RATE : " + rate + "%");
        //</editor-fold>

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
