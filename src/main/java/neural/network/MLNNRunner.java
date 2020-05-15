/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author abhinav
 */
public class MLNNRunner {
    
    public static void main(String[] args) throws IOException {
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
//        ArrayList<Matrix> alm = new ArrayList<>();
//        for (int i = 0; i < 4; i++) {
//            Matrix m = Matrix.getRandomMatrix(9, 9);
//            alm.add(m);
//
//        }
//        Convolution conv = new Convolution(alm, new Matrix(3, 3));
//        NeuralNetwork network = new NeuralNetwork(2, 2, 1);
        int[] hiddenNodeArray = {2};
        int outputNodes = 1;
        
        MultipleLayerNeuralNetwork multiNetwork = new MultipleLayerNeuralNetwork(2, new int[]{2}, 1);

//        for (int i = 0; i < multiNetwork.weightMatrices.size(); i++) {
//            multiNetwork.weightMatrices.get(i).print();
//
//        }
//        network.wieghts_input_hidden.print();
//        network.wieghts_hidden_output.print();
//        }
        multiNetwork.setLearningRate(0.2);
        for (int i = 0; i < 10000; i++) {
            if (i % 500 == 0) {
                System.out.println("Trained : " + i);
            }
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
