/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.ArrayList;

/**
 *
 * @author abhinav
 */
public class Convolution {

    //16 convolution
    ArrayList<Matrix> result;

    public Convolution(ArrayList<Matrix> channels, Matrix filter) {

        ArrayList<Matrix> resultChannels = new ArrayList<>(channels.size());

        for (int i = 0; i < channels.size(); i++) {
            Matrix dataMatrix = channels.get(i);
            Matrix resultMatrix = new Matrix(dataMatrix.rows - filter.rows + 1, dataMatrix.cols - filter.cols + 1);
            Matrix[][] miniData2DArray = new Matrix[resultMatrix.rows][resultMatrix.cols];
            for (int j = 0; j < resultMatrix.rows; j++) {
                for (int k = 0; k < resultMatrix.cols; k++) {
                    Matrix miniData = new Matrix(filter.rows, filter.cols);
                    for (int l = 0; l < filter.rows; l++) {
                        for (int m = 0; m < filter.cols; m++) {
                            miniData.data[l][m] = dataMatrix.data[j + l][k + m];
                        }
                    }
                    miniData = Matrix.vectorMultiply(miniData, filter);
                    miniData2DArray[j][k] = miniData;
                }
            }
            Matrix newResultMatrix = Matrix.convert2DArrayToMatrix(miniData2DArray);
            resultChannels.set(i, newResultMatrix);
        }
        this.result = resultChannels;
    }
}
