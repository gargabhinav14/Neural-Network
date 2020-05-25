/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lib;

import java.util.ArrayList;
import Util.Matrix;

/**
 *
 * @author abhinav
 */
public class NeuralNetwork {

    //<editor-fold defaultstate="collapsed" desc="For Single Layer">
    int inupt_nodes;
    int hidden_nodes;
    int output_nodes;

    Matrix wieghts_input_hidden;
    Matrix wieghts_hidden_output;

    Matrix bias_hidden;
    Matrix bias_output;

    Matrix hidden_outs;
    Matrix output_outs;

    double learning_rate;
//</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="For Multiple Layers">
    ArrayList<Matrix> allWeightMatrices = new ArrayList<>();
    ArrayList<Matrix> allBiasMatrices = new ArrayList<>();

//</editor-fold>
    public NeuralNetwork(int inupt_nodes, int hidden_nodes, int output_nodes) {
        this.inupt_nodes = inupt_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

//        this.wieghts_input_hidden = new Matrix(this.hidden_nodes, this.inupt_nodes);
        this.wieghts_input_hidden = new Matrix(this.hidden_nodes, this.inupt_nodes);
        this.wieghts_hidden_output = new Matrix(this.output_nodes, this.hidden_nodes);

        this.wieghts_input_hidden.randomize();
        this.wieghts_hidden_output.randomize();

        this.bias_hidden = new Matrix(this.hidden_nodes, 1);
        this.bias_output = new Matrix(this.output_nodes, 1);
        this.bias_hidden.randomize();
        this.bias_output.randomize();

        this.learning_rate = 0.1 ;

    }

    public NeuralNetwork(int input_nodes, int[] hidden_nodes_array, int output_nodes) {
        //<editor-fold defaultstate="collapsed" desc="DOC">
        /**
         * layers = (hidden_nodes_array.length);
         *
         * for (int i = 0 ; i < layers ; i ++) { layer[i] =
         * hidden_nodes_array[i]; get number of nodes per layer in layerArray
         *
         * }
         *
         * ArrayList<Matrix> matricesWightsArray ; numberOfMatrices[] = new
         * int[numberOfLayers - 1]
         *
         *
         * for (int i = 0 ; i < hidden_nodes_array.length ; i ++) { itne weight
         * / bias matrices banenge if(i = 0) { Matrix m = new Matrix
         * (hidden_nodes_array[i], input_nodes) } else { Matrix m = new Matrix
         * (hidden_nodes_array[i], hidden_nodes_array[i-1]) } }
         *
         */
        //</editor-fold>

        this.allWeightMatrices = createWeightMatrices(input_nodes, hidden_nodes_array);
        this.allBiasMatrices = createBiasMatrices(hidden_nodes_array);

    }

    public double[] feedForward(double[] inputs_array) {
        //COMPUTATION OF A GUESS
        Matrix input_matrix = Matrix.toMatrix(inputs_array);

        /**
         * calculate the output of Hidden Layer
         */
        this.hidden_outs = Matrix.vectorMultiply(this.wieghts_input_hidden, input_matrix);
        this.hidden_outs.add(this.bias_hidden);
        this.hidden_outs.doSigmoid();

        /**
         * Calculate the output of Output Layer
         */
        this.output_outs = Matrix.vectorMultiply(this.wieghts_hidden_output, this.hidden_outs);
        this.output_outs.add(this.bias_output);
        this.output_outs.doSigmoid();

        return this.output_outs.toArray();
//        return guess;
    }

    public void train(double[] inputs, double[] answer) {

        /**
         * to calculate the errors use the back propagation and calculate the
         * previous errors
         *
         * use the gradients formula to calculate the the gradients i.e. delta
         * Ws the the change in weights
         *
         * calculate the new weights based on the => old W + delW
         *
         */
        Matrix input_matrix = Matrix.toMatrix(inputs);

        /**
         * calculate the output of Hidden Layer
         */
//        System.out.println("Hidden Outs = ");
//        this.hidden_outs.print();
//        System.out.println("Weights input hidden= ");
//        this.wieghts_input_hidden.print();
//        System.out.println("Input Matrix= ");
//        input_matrix.print();
        this.hidden_outs = Matrix.vectorMultiply(this.wieghts_input_hidden, input_matrix);
//        System.out.println("Hidden Outs After Multiplicat8ion");
//        this.hidden_outs.print();

//        System.out.println("Hidden Bias= ");
//        this.bias_hidden.print();
        this.hidden_outs.add(this.bias_hidden);

//        System.out.println("Hidden Outs After Bias Add= ");
//        this.hidden_outs.print();
        this.hidden_outs.doSigmoid();
//        System.out.println("Hidden Outs after Sigmoid =");
//        this.hidden_outs.print();

        /**
         * Calculate the output of Output Layer
         */
        this.output_outs = Matrix.vectorMultiply(this.wieghts_hidden_output, this.hidden_outs);
        this.output_outs.add(this.bias_output);
        this.output_outs.doSigmoid();

//        Matrix outputs = Matrix.toMatrix(outputs_array);
        /**
         * MAIN EQUATION
         *
         * y = Mx + B
         *
         * deltaM = lr*error*X
         *
         *
         * where x is the derivative of Y = sigmoid (W.H + B) using chain rule
         * we get derivative = derivative (sigmoid (W.H + B)) * derivative (W.H
         * + B) => derivative = derivative of (sigmoid (Z)) * H => derivative =
         * Z(1-Z) * H
         *
         * delatB = lr*error
         *
         * deltaW = learningRate * error * (output * (1 - output)) * H
         */
        //Convert Answer Array to Matrix
        Matrix target = Matrix.toMatrix(answer);
        Matrix output_error_matrix = target.subtract(this.output_outs);
//        Matrix output_error_matrix = this.output_outs.subtract(Matrix.toMatrix(answer));
        Matrix gradient_ho = Matrix.doDerivaitveSigmoid(this.output_outs);
        gradient_ho = Matrix.scalarMultiply(gradient_ho, output_error_matrix);
        gradient_ho = Matrix.scalarMultiply(gradient_ho, this.learning_rate);
        Matrix hidden_outs_T = Matrix.transpose(this.hidden_outs);
        Matrix wieghts_ho_deltas = Matrix.vectorMultiply(gradient_ho, hidden_outs_T);
        //change the wieghts and bias
        this.wieghts_hidden_output.add(wieghts_ho_deltas);
        this.bias_output.add(gradient_ho);

        Matrix hidden_error_matrix = Matrix.vectorMultiply(Matrix.transpose(this.wieghts_hidden_output), output_error_matrix);
        Matrix gradient_ih = Matrix.doDerivaitveSigmoid(this.hidden_outs);
        gradient_ih = Matrix.scalarMultiply(gradient_ih, hidden_error_matrix);
        gradient_ih = Matrix.scalarMultiply(gradient_ih, this.learning_rate);
        Matrix input_matrix_T = Matrix.transpose(input_matrix);
        Matrix wieghts_ih_deltas = Matrix.vectorMultiply(gradient_ih, input_matrix_T);

        //change the wieghts and bias
        this.wieghts_input_hidden.add(wieghts_ih_deltas);
        this.bias_hidden.add(gradient_ih);

        /**
         * Adjust the wieghts of input to hidden
         *
         *
         */
    }

    private ArrayList<Matrix> createWeightMatrices(int input_nodes, int[] hidden_nodes_array) {

        ArrayList<Matrix> weightMatrices = new ArrayList<>();

        //for Each Layer
        for (int i = 0; i < hidden_nodes_array.length; i++) {

            //create new ArrayList for each layer to store all matrices
//            ArrayList<Matrix> wieghtMatricesPerLayer = new ArrayList<>();
            Matrix m = null;
            if (i == 0) {
                m = new Matrix(hidden_nodes_array[i], input_nodes);
            } else {
                m = new Matrix(hidden_nodes_array[i], hidden_nodes_array[i - 1]);
            }
            m.randomize();
            weightMatrices.add(m);
        }
//        System.out.println("ALL WIEGHT MATRICES" + this.allWeightMatrices);

        return weightMatrices;
    }

    private ArrayList<Matrix> createBiasMatrices(int[] hidden_nodes_array) {
        ArrayList<Matrix> biasMatrices = new ArrayList<>();
        
        for(int i = 0 ; i < hidden_nodes_array.length ; i ++)
        {
            Matrix m = new Matrix(hidden_nodes_array[i], 1);
            m.randomize();
            biasMatrices.add(m);
            
        }
        

        return biasMatrices;
    }

}
