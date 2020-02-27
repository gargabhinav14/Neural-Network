/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.Random;

/**
 *
 * @author abhinav
 */
public class Main {

    public static void main(String[] args) {

        /**
         * create a network with input_count, perceptron_count, output_count
         *
         */
        //<editor-fold defaultstate="collapsed" desc="Matrix Demo">
//         int[][] arr = {{1, 2}, {3, 4}};
//         int [][] mat = {{1,2,3},{1,2,3},{1,2,3}};
//        Matrix mat1 = new Matrix(2, 3);
//        Matrix mat2 = new Matrix(3, 2);
//         mat.randomize();
//         mat.print();
//         mat.add(2);
//         mat.print();
//         mat.subtract(2);
//         mat.print();
//        mat.randomize();
//        mat.print();
//        Matrix m3 = Matrix.transpose(mat);
//        m3.print();
//        Matrix m1 = Matrix.scalarMultiply(mat, mat);
//        m1.print();
//        mat1.randomize();
//        mat2.randomize();
//        mat1.print();
//        mat2.print();
//        Matrix m2 = Matrix.vectorMultiply(mat1, mat2);
//        m2.print();
//</editor-fold>
        double[][] input = {{1, 0}, {1, 0}, {0, 0}, {1, 1}};
        double[][] output = {{1}, {1}, {0}, {0}};
        NeuralNetwork network = new NeuralNetwork(2, 2, 1);

        int count0 = 0;
        int count1 = 0;
        int count2 = 0;
        int count3 = 0;

        System.out.println("testing now with learning rate : " + network.learning_rate);
//        double[] totalOutput = network.feedForward(input);

        System.out.println("INPUT HIDDEN WEIGHTS");
        network.wieghts_input_hidden.print();
        System.out.println("HIDDEN OUTPUT WEIGHTS");
        network.wieghts_hidden_output.print();

        System.out.println("BIAS HIDDEN");
        network.bias_hidden.print();
        System.out.println("BIAS OUTPUT");
        network.bias_output.print();

        for (int i = 0; i < 1000000; i++) {
//            Random generator = new Random();
            int rnd = new Random().nextInt(input.length);
//            double k = Math.floor(Math.floor(Math.random() * 10) / 2);
//            int value = (int) k;
//            if (value == 4) {
//                value = 3;
//            }
//            System.out.println(value);
            if (rnd == 0) {
                count0++;
            }
            if (rnd == 1) {
                count1++;
            }
            if (rnd == 2) {
                count2++;
            }if (rnd == 3) {
                count3++;
            }
            
            network.train(input[rnd], output[rnd]);
        }

        System.out.println("INPUT HIDDEN WEIGHTS");
        network.wieghts_input_hidden.print();
        System.out.println("HIDDEN OUTPUT WEIGHTS");
        network.wieghts_hidden_output.print();

        System.out.println("BIAS HIDDEN");
        network.bias_hidden.print();
        System.out.println("BIAS OUTPUT");
        network.bias_output.print();

        System.out.println("+++++OUTPUTS+++++++");
        System.out.println("Count0:"+count0+"||Count1:"+count1+"||Count2:"+count2+"||Count3:"+count3);
        double[] test1 = network.feedForward(input[0]);
        double[] test2 = network.feedForward(input[1]);
        double[] test3 = network.feedForward(input[2]);
        double[] test4 = network.feedForward(input[3]);

        Matrix.toMatrix(test1).print();
        Matrix.toMatrix(test2).print();
        Matrix.toMatrix(test3).print();
        Matrix.toMatrix(test4).print();

    }

}
