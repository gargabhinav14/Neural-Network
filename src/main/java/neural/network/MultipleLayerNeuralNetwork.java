/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.ArrayList;
import jdk.nashorn.internal.runtime.PropertyMap;

/**
 *
 * @author abhinav
 */
public class MultipleLayerNeuralNetwork {

    ArrayList<Matrix> weightMatrices = new ArrayList<>();
    ArrayList<Matrix> biasMatrices = new ArrayList<>();
    ArrayList<Matrix> outputMatrices = new ArrayList<>();
    ArrayList<Matrix> errorMatrices = new ArrayList<>();
    int inputNodes;
    int outputNodes;
    double learningRate;

    int[] hidden_nodes_array;
    int hidden_nodes_array_length;

    //<editor-fold defaultstate="collapsed" desc="Set Neural Network Structure">
    public MultipleLayerNeuralNetwork(int input_nodes, int[] hidden_nodes_array, int output_nodes) {

        //create all wieght matrices
        this.weightMatrices = createWeightMatrices(input_nodes, hidden_nodes_array);

        //create all bias matrices
        this.biasMatrices = createBiasMatrices(hidden_nodes_array);

        //set hidden array and its length to a constant
        this.hidden_nodes_array = hidden_nodes_array;
        this.hidden_nodes_array_length = hidden_nodes_array.length;

        this.inputNodes = input_nodes;
        this.outputNodes = output_nodes;

        this.learningRate = 0.1;

    }

    //<editor-fold defaultstate="collapsed" desc="Set Learning Rate">
    void ssetLearningRate(double lr) {
        this.learningRate = lr;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Create Weight and bias Matrices">
    private ArrayList<Matrix> createWeightMatrices(int input_nodes, int[] hidden_nodes_array) {

        ArrayList<Matrix> weightMatrices = new ArrayList<>();

        //for Each Layer
        for (int i = 0; i < this.hidden_nodes_array_length; i++) {

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

        for (int i = 0; i < this.hidden_nodes_array_length; i++) {
            Matrix m = new Matrix((int) hidden_nodes_array[i], 1);
            m.randomize();
            biasMatrices.add(m);

        }

        return biasMatrices;
    }
    //</editor-fold>
    //</editor-fold>

    public void train(double[] inputs, double[] outputs) {

        calulateOutput(inputs);

        fixAllWeightsAndBiases(outputs);

        //timme for back propogation
    }

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> outputMatrices = calulateOutput(inputs)">
    private void calulateOutput(double[] inputs) {

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
        result.doSigmoid();
        return result;

    }

    //</editor-fold>
    private void fixAllWeightsAndBiases(double[] desiredOutput) {

        //calculate Error Matrices
        //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> errorMatrices = calculateErrorMatrices(desiredOutput)">
        calculateErrorMatrices(desiredOutput);
//</editor-fold>

        //fix All Weight Matrices
        fixWieghtMatrices();

        //fix All Bias Matrices
        fixBiasMatrices();
    }

    private void fixWieghtMatrices() {
        //we have to start fixing from the back

//        Matrix previousError = null;
        for (int i = hidden_nodes_array_length; i >= 0; i--) {

//            if (i != hidden_nodes_array_length) {
            this.weightMatrices.set(
                    /**
                     * SIZE*
                     */
                    this.weightMatrices.size() - i,
                    /**
                     * ELEMENT*
                     */
                    fixWeightMatrix(this.errorMatrices.get(i), this.outputMatrices.get(i), this.weightMatrices.get(this.weightMatrices.size() - i))
            );
//            }
//            else
//            {
//                this.weightMatrices.set(
//                        /**
//                         * SIZE*
//                         */
//                        this.weightMatrices.size() - i,
//                        /**
//                         * ELEMENT*
//                         */
//                        fixWeightMatrix(this.errorMatrices.get(i), this.outputMatrices.get(i), this.weightMatrices.get(this.weightMatrices.size() - i))
//                );
//                
//            }
        }

    }

    private Matrix fixWeightMatrix(Matrix error, Matrix output, Matrix weight) {

        return Matrix.vectorMultiply(
                Matrix.scalarMultiply(
                        Matrix.scalarMultiply(
                                Matrix.doDerivaitveSigmoid(output), error
                        ), this.learningRate
                ),
                Matrix.transpose(weight)
        );

    }

    //<editor-fold defaultstate="collapsed" desc="calculateErrorMatrices(desiredOutput)">
    private void calculateErrorMatrices(double[] desiredOutput) {
        for (int i = this.hidden_nodes_array_length - 1; i <= 0; i--) {
            if (i == this.hidden_nodes_array_length - 1) {
                this.errorMatrices.add(this.outputMatrices.get(this.hidden_nodes_array_length - 1).subtract(Matrix.toMatrix(desiredOutput)));
            } else {
                this.errorMatrices.add(Matrix.vectorMultiply(Matrix.transpose(this.weightMatrices.get(i)), this.errorMatrices.get(i)));
            }
        }
    }
    //</editor-fold>

    private void fixBiasMatrices() {

        for (int i = hidden_nodes_array_length; i >= 0; i--) {

//            if (i != hidden_nodes_array_length) {
            this.biasMatrices.set(
                    /**
                     * SIZE*
                     */
                    this.biasMatrices.size() - i,
                    /**
                     * ELEMENT*
                     */
                    fixBiasMatrix(this.errorMatrices.get(i), this.biasMatrices.get(this.biasMatrices.size() - i))
            );
        }
    }

    private Matrix fixBiasMatrix(Matrix error, Matrix bias) {

        return  Matrix.scalarMultiply(
                        Matrix.scalarMultiply(
                                Matrix.doDerivaitveSigmoid(output), error
                        ), this.learningRate
                );
    }
}
