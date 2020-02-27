/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

/**
 *
 * @author abhinav
 */
public class Matrix {

    int rows;
    int cols;
    double[][] data;

    public Matrix(int rows, int cols) {

        this.rows = rows;
        this.cols = cols;
        this.data = new double[this.rows][this.cols];
//        Matrix matrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = new Double(0);
            }
        }

    }

    public void randomize() {

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    public void add(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] += num;
            }
        }
    }

    public void add(Matrix m) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] += m.data[i][j];
            }
        }
    }

    public void subtract(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] -= num;
            }
        }
    }

    public Matrix subtract(Matrix m) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] - m.data[i][j];
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }

    public static Matrix scalarMultiply(Matrix m, double num) {
        Matrix resutMatrix = new Matrix(m.rows, m.cols);

        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                resutMatrix.data[i][j] = m.data[i][j] * num;

            }
        }
        return resutMatrix;

    }

    public static Matrix scalarMultiply(Matrix m, Matrix n) {
        Matrix resulMatrix = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                resulMatrix.data[i][j] = m.data[i][j] * n.data[i][j];
            }
        }
        return resulMatrix;

    }

    public static Matrix vectorMultiply(Matrix m, Matrix n) {
        Matrix result = new Matrix(m.rows, n.cols);

        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                for (int k = 0; k < m.cols; k++) {
                    result.data[i][j] += m.data[i][k] * n.data[k][j];

                }
            }
        }
        return result;
    }

    public static Matrix toMatrix(double[] inputs) {
        Matrix result = new Matrix(inputs.length, 1);

        for (int i = 0; i < inputs.length; i++) {
            result.data[i][0] = inputs[i];

        }
        return result;
    }

    public double[] toArray() {
        double[] result = new double[this.rows * this.cols];
        int count = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result[count] = this.data[i][j];
            }
            count++;
        }
        return result;

//        result = new Float[]
    }

    public void print() {
        for (int k = 0; k < this.cols; k++) {
            System.out.print("    " + k + "   ");

        }
        System.out.println("");
        for (int i = 0; i < this.rows; i++) {

            for (int k = 0; k < this.cols; k++) {
                System.out.print("--------");

            }
            System.out.println("");
            System.out.print("|  ");
            for (int j = 0; j < this.cols; j++) {
                System.out.print(this.data[i][j] + "  |  ");

            }
            System.out.println("");

        }
        for (int k = 0; k < this.cols; k++) {
            System.out.print("--------");

        }
        System.out.println("");
        System.out.println("");
    }

    public void doSigmoid() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = sigmoid(this.data[i][j]);
            }
        }
    }

    public static Matrix doDerivaitveSigmoid(Matrix m) {
        Matrix resMatrix = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                resMatrix.data[i][j] = derivativeSigmoid(m.data[i][j]);
            }
        }
        return resMatrix;
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double derivativeSigmoid(double x) {
        //The values coming here will be already passed out through a derivative function

        return x * (1 - x);
    }
}
