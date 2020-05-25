/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lib;

import Util.Matrix;
import java.util.ArrayList;

/**
 *
 * @author abhunavgarg
 */
public class RecurrentNeuralNetwork {

    //<editor-fold defaultstate="collapsed" desc="Global Variables">
    ArrayList<Matrix> weightMatrices = new ArrayList<>();   //0 ---> 1 ---> 2 ---> 3
    ArrayList<Matrix> biasMatrices = new ArrayList<>();     //0 ---> 1 ---> 2 ---> 3
    ArrayList<Matrix> outputMatrices = new ArrayList<>();   //3 ---> 2 ---> 1 ---> 0
    ArrayList<Matrix> errorMatrices = new ArrayList<>();    //3 ---> 2 ---> 1 ---> 0

    int inputNodes;
    int outputNodes;
    double learningRate;
    private boolean sigmoidRequired = true;

    int[] hidden_nodes_array;
    int hidden_nodes_array_length;
    
    double[] previousOutput;
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Initialize">
    public RecurrentNeuralNetwork(int input_nodes, int[] hidden_nodes_array, int output_nodes) {
        //set hidden array and its length to a constant
        this.hidden_nodes_array = hidden_nodes_array;
        this.hidden_nodes_array_length = hidden_nodes_array.length;
        this.inputNodes = input_nodes;
        this.outputNodes = output_nodes;
        this.learningRate = 0.1;

        //create all wieght matrices
        this.weightMatrices = createWeightMatrices(input_nodes, hidden_nodes_array, output_nodes);
        //create all bias matrices
        this.biasMatrices = createBiasMatrices(hidden_nodes_array, output_nodes);
        
        this.previousOutput = new Matrix(this.inputNodes, 1).toArray();
    }

    public ArrayList<Matrix> createWeightMatrices(int input_nodes, int[] hidden_nodes_array, int output_nodes) {

        ArrayList<Matrix> weightMatrices = new ArrayList<>();

        //for Each Layer
        for (int i = 0; i <= this.hidden_nodes_array_length; i++) {

            //create new ArrayList for each layer to store all matrices
//            ArrayList<Matrix> wieghtMatricesPerLayer = new ArrayList<>();
            Matrix m = null;
            if (i == 0) {
                m = new Matrix(hidden_nodes_array[i], input_nodes);
                m.randomize();
                weightMatrices.add(m);
            } else if (i == this.hidden_nodes_array_length) {
                m = new Matrix(output_nodes, hidden_nodes_array[i - 1]);
                m.randomize();
                weightMatrices.add(m);
            } else {
                m = new Matrix(hidden_nodes_array[i], hidden_nodes_array[i - 1]);
                m.randomize();
                weightMatrices.add(m);
            }

        }
//        System.out.println("ALL WIEGHT MATRICES" + this.allWeightMatrices);

        return weightMatrices;
    }

    public ArrayList<Matrix> createBiasMatrices(int[] hidden_nodes_array, int output_nodes) {
        ArrayList<Matrix> biasMatrices = new ArrayList<>();

        for (int i = 0; i <= this.hidden_nodes_array_length; i++) {
            Matrix m = null;
            if (i != this.hidden_nodes_array_length) {
                m = new Matrix((int) hidden_nodes_array[i], 1);
                m.randomize();
                biasMatrices.add(m);
            } else {
                m = new Matrix((int) output_nodes, 1);
                m.randomize();
                biasMatrices.add(m);
            }

        }

        return biasMatrices;
    }
    //</editor-fold>

    public double[] feedForward(double[] input) {

        initializeVariables();
        Matrix inputMatrix = Matrix.toMatrix(input);
        Matrix previousOutputMatrix = Matrix.toMatrix(this.previousOutput);
        inputMatrix.add(previousOutputMatrix);
        calculateOutput(inputMatrix.toArray());
        this.previousOutput = this.outputMatrices.get(this.outputMatrices.size()-1).toArray();

        return this.previousOutput;
    }
    
    public void train(double[] input, double[] outputs)
    {
        fixAllWeightsAndBiases(outputs);
    }

    private void calculateOutput(double[] inputs) {
        for (int i = 0; i <= this.hidden_nodes_array_length; i++) {
            if (i == 0) {
                this.outputMatrices.add(calculateOuputOfCurrentLayer(this.weightMatrices.get(i), this.biasMatrices.get(i), Matrix.toMatrix(inputs)));
            } else {
                this.outputMatrices.add(calculateOuputOfCurrentLayer(this.weightMatrices.get(i), this.biasMatrices.get(i), this.outputMatrices.get(i - 1)));

            }

        }
    }

    private Matrix calculateOuputOfCurrentLayer(Matrix wieghtMatrix, Matrix biasMatrix, Matrix inputMatrix) {

        Matrix result = Matrix.vectorMultiply(wieghtMatrix, inputMatrix);
        result.add(biasMatrix);
        if (!this.sigmoidRequired == false) {
            result.doSigmoid();
        }
        return result;

    }

    private void initializeVariables() {
        this.outputMatrices = new ArrayList<>();
        this.errorMatrices = new ArrayList<>();
    }

    private void fixAllWeightsAndBiases(double[] outputs) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
