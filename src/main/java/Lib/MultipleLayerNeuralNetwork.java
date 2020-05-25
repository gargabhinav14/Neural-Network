/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lib;

import Util.Matrix;
import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author abhinav
 */
public class MultipleLayerNeuralNetwork {

    //<editor-fold defaultstate="collapsed" desc="Global Variables">
    ArrayList<Matrix> weightMatrices = new ArrayList<>();   //0 ---> 1 ---> 2 ---> 3
    ArrayList<Matrix> biasMatrices = new ArrayList<>();     //0 ---> 1 ---> 2 ---> 3
    ArrayList<Matrix> outputMatrices = new ArrayList<>();   //3 ---> 2 ---> 1 ---> 0
    ArrayList<Matrix> errorMatrices = new ArrayList<>();    //3 ---> 2 ---> 1 ---> 0

    Matrix inputMatrix;
    int inputNodes;
    int outputNodes;
    double learningRate;
    private boolean sigmoidRequired = true;

    int[] hidden_nodes_array;
    int hidden_nodes_array_length;
//</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Set Neural Network Structure">
    //<editor-fold defaultstate="collapsed" desc="Initialise MLNN">
    public MultipleLayerNeuralNetwork(int input_nodes, int[] hidden_nodes_array, int output_nodes) {
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
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Set Learning Rate">
    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void sigmoidRequired(boolean val) {
        this.sigmoidRequired = val;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Create Weight and bias Matrices">
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
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="train(inputs, outputs)">
    public void train(double[] inputs, double[] outputs) {

        initializeVariables();
        this.inputMatrix = Matrix.toMatrix(inputs);

        calulateOutput(inputs);

        fixAllWeightsAndBiases(outputs);

        //timme for back propogation
    }

    //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> outputMatrices = calulateOutput(inputs)">
    public void calulateOutput(double[] inputs) {

        for (int i = 0; i <= this.hidden_nodes_array_length; i++) {
            if (i == 0) {
                this.outputMatrices.add(calculateOuputOfCurrentLayer(this.weightMatrices.get(i), this.biasMatrices.get(i), Matrix.toMatrix(inputs)));
            } else {
                this.outputMatrices.add(calculateOuputOfCurrentLayer(this.weightMatrices.get(i), this.biasMatrices.get(i), this.outputMatrices.get(i - 1)));

            }

        }

    }

    //<editor-fold defaultstate="collapsed" desc="calculateOuputOfCurrentLayer(weight, bias, input)">
    private Matrix calculateOuputOfCurrentLayer(Matrix wieghtMatrix, Matrix biasMatrix, Matrix inputMatrix) {

        Matrix result = Matrix.vectorMultiply(wieghtMatrix, inputMatrix);
        result.add(biasMatrix);
        if (!this.sigmoidRequired == false) {
            result.doSigmoid();
        }
        return result;

    }
    //</editor-fold>

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="fixAllWeightsAndBiases(outputs)">
    public void fixAllWeightsAndBiases(double[] desiredOutput) {

        //calculate Error Matrices
        //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> errorMatrices = calculateErrorMatrices(desiredOutput)">
        calculateErrorMatrices(desiredOutput);
//</editor-fold>

        //fix All Weight Matrices
        //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> weightMatrices = fixWieghtMatrices()">
        fixWieghtMatrices();
        //</editor-fold>

        //fix All Bias Matrices
        //<editor-fold defaultstate="collapsed" desc="ArrayList<Matrix> biasMatrices = fixBiasMatrices()">
        fixBiasMatrices();
        //</editor-fold>

    }

    //<editor-fold defaultstate="collapsed" desc="calculateErrorMatrices(desiredOutput)">
    private void calculateErrorMatrices(double[] desiredOutput) {
        int k = 0;

        this.errorMatrices.add(Matrix.toMatrix(desiredOutput).subtract(this.outputMatrices.get(this.hidden_nodes_array_length)));

        for (int i = this.hidden_nodes_array_length; i > 0; i--) {
//            if (i == this.hidden_nodes_array_length) {

//                this.errorMatrices.add(this.outputMatrices.get(this.hidden_nodes_array_length).subtract(Matrix.toMatrix(desiredOutput)));
//            } else {/
            this.errorMatrices.add(Matrix.vectorMultiply(Matrix.transpose(this.weightMatrices.get(i)), this.errorMatrices.get(k)));
            k++;

//            }
        }

        Collections.reverse(this.errorMatrices);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Fix Wieght Martices">
    private void fixWieghtMatrices() {
        //we have to start fixing from the back
//        int k = 0;
        Matrix deltaW = null;
        for (int i = hidden_nodes_array_length; i >= 0; i--) {

            if (i != 0) {
                deltaW = fixWeightMatrix(this.errorMatrices.get(i), this.outputMatrices.get(i), this.outputMatrices.get(i - 1)/**
                 * , this.weightMatrices.get(i)*
                 */
                );
            } else {
                deltaW = fixWeightMatrix(this.errorMatrices.get(i), this.outputMatrices.get(i), this.inputMatrix/**
                 * , this.weightMatrices.get(i)*
                 */
                );

            }
//            if (deltaW.rows == 1) {
//                double numDelta = deltaW.data[0][0];
//                this.weightMatrices.get(i).add(numDelta);

//            } else {
            //add weight to deltas
            this.weightMatrices.get(i).add(deltaW);
//            }
//            k++;
        }

    }

    private Matrix fixWeightMatrix(Matrix error, Matrix output, Matrix previousOutput /**
     * ,Matrix weight*
     */
    ) {
        return Matrix.vectorMultiply(
                Matrix.scalarMultiply(
                        Matrix.scalarMultiply(
                                Matrix.doDerivaitveSigmoid(output), /**
                                 * Matrix.toMatrix(error.data[i])*
                                 */
                                error
                        ), this.learningRate
                ),
                Matrix.transpose(previousOutput)
        ) //                            )
                ;
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="fix Bias Matrices">
    private void fixBiasMatrices() {

        for (int i = hidden_nodes_array_length; i >= 0; i--) {

            Matrix deltaBias = fixBiasMatrix(this.errorMatrices.get(i), this.outputMatrices.get(i));
            //add bias to deltas
            if (deltaBias.rows == 1) {
                double numDelta = deltaBias.data[0][0];
                this.biasMatrices.get(i).add(numDelta);

            } else {
                this.biasMatrices.get(i).add(deltaBias);
            }
        }

    }

    private Matrix fixBiasMatrix(Matrix error, Matrix output) {

        return Matrix.scalarMultiply(
                Matrix.scalarMultiply(
                        Matrix.doDerivaitveSigmoid(output), error
                ), this.learningRate
        );
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="feedForward(inputs)">
    public double[] feedForward(double[] inputs) {
        //COMPUTATION OF A GUESS

        this.outputMatrices = new ArrayList<>();   //3 ---> 2 ---> 1 ---> 0
        this.errorMatrices = new ArrayList<>();    //3 ---> 2 ---> 1 ---> 0
        this.inputMatrix = Matrix.toMatrix(inputs);

        calulateOutput(inputs);

        return this.outputMatrices.get(this.hidden_nodes_array_length).toArray();
    }
    //</editor-fold>

    private void initializeVariables() {
        this.outputMatrices = new ArrayList<>();   //3 ---> 2 ---> 1 ---> 0
        this.errorMatrices = new ArrayList<>();    //3 ---> 2 ---> 1 ---> 0    
    }

}
