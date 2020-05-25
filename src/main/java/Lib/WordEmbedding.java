/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lib;

import Util.Matrix;
import Util.PreProcessTxtFile;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 *
 * @author abhunavgarg
 */
public class WordEmbedding {

    public static void main(String[] args) throws Exception {
        PreProcessTxtFile pptf = new PreProcessTxtFile("/home/abhunavgarg/Cody/new Cody/Neural-Network/got.txt");
        convertToOneHotVectorFile(pptf, "HotVector.ser");
        Map<String, double[]> htoVectorMap = readFileToMap(pptf, "HotVector.ser");

//        convolution(pptf);
        prepareData(htoVectorMap, pptf, "dataSet.ser");

        /**
         * now we have hot vector data dataset
         */
    }

    private static void convertToOneHotVectorFile(PreProcessTxtFile pptf, String outputFilePath) {
        int uniqueWords = pptf.noOfUniqueWords;
        String[] words = pptf.uniqueWordsString.split(" ");
        System.out.println("Vocab Size" + uniqueWords);

        try {
            FileOutputStream writeData = new FileOutputStream(outputFilePath);
            ObjectOutputStream writeStream = new ObjectOutputStream(writeData);

            for (int i = 1; i < uniqueWords; i++) {
                if (i % 500 == 0) {
                    System.out.println("Vectors created " + i);
                }
                Matrix matrix = new Matrix(uniqueWords, 1);
                matrix.data[i - 1][0] = 1;

                Map<String, double[]> map = new HashMap<>();
                map.put(words[i], matrix.toArray());

                writeStream.writeObject(map);
            }
            writeStream.flush();
            writeStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, double[]> readFileToMap(PreProcessTxtFile pptf, String inputFilePath) throws FileNotFoundException, IOException {
        Map<String, double[]> hotVectorMap = new HashMap<>();
        boolean cont = true;
        ObjectInputStream input = new ObjectInputStream(new FileInputStream(inputFilePath));
        int i = 0;
        for (int j = 0; j < pptf.noOfUniqueWords; j++) {
            try {
                i++;
                if (i % 500 == 0) {
                    System.out.println("Parsed " + i);
                }
                Map obj = (Map) input.readObject();
                if (obj != null) {
                    hotVectorMap.putAll(obj);
                } else {
                    cont = false;
                }
            } catch (Exception e) {
//                 System.out.println(e.printStackTrace());
            }
        }
        return hotVectorMap;
    }

    private static void prepareData(Map<String, double[]> map, PreProcessTxtFile pptf, String inputFilePath) throws FileNotFoundException, IOException {

        int[] hiddenNodesArray = {2};

        MultipleLayerNeuralNetwork mlnn = new MultipleLayerNeuralNetwork(pptf.noOfUniqueWords, hiddenNodesArray, pptf.noOfUniqueWords);
//        mlnn.sigmoidRequired(false);
        try {
            for (int i = 0; i < pptf.inputWords.size(); i++) {
//            for (int i = 0; i < 100; i++) {
                if (i % 10000 == 0) {
                    System.out.println("Time : " + System.currentTimeMillis());
                    System.out.println("TRAINED : " + i);
                }
                int rnd = new Random().nextInt(pptf.noOfUniqueWords);
                String inputWord = pptf.inputWords.get(rnd);
                String outputWord = pptf.outputWords.get(rnd);
                double[] inputVector = map.get(inputWord);
                double[] outputVector = map.get(outputWord);
                mlnn.train(inputVector, outputVector);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream("wieghts.ser"));

        outputStream.writeObject(mlnn.weightMatrices);
        outputStream.flush();
        outputStream.close();

        System.out.println("TRAINED");

    }

    private static void convolution(PreProcessTxtFile pptf) {

    }

}
