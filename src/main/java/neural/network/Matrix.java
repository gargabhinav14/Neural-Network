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

    public static Matrix convert2DArrayToMatrix(Matrix[][] m) {

        Matrix[][] arrayMat = m;
        int mrows = arrayMat.length;
        int mCols = arrayMat[0].length;

        int p = 0;
        int q = 0;

        int newRows = arrayMat[0][0].rows * mrows;
        int newCols = arrayMat[0][0].cols * mCols;

        int innerGridRows = arrayMat[0][0].rows;
        int innerGridCols = arrayMat[0][0].cols;

        Matrix resultMatrix = new Matrix(newRows, newCols);
        for (int i = 0; i < mrows; i++) {
            for (int j = 0; j < mCols; j++) {

                Matrix matrix = arrayMat[i][j];
                for (int k = 0; k < matrix.rows; k++) {
                    for (int l = 0; l < matrix.cols; l++) {
                        resultMatrix.data[k + p][l + q] = matrix.data[k][l];
                    }
                }
                q = q + innerGridCols;
            }
            p = p + innerGridRows;
            q = 0;
        }
        return resultMatrix;
    }

    public static Matrix getRandomMatrix(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        m.randomize();
        return m;
    }

    static Matrix getRandomPositiveMatrix(int filterRows, int filterCols) {
        Matrix m = new Matrix(filterRows, filterCols);
        for (int i = 0; i < filterRows; i++) {
            for (int j = 0; j < filterCols; j++) {
                m.data[i][j] = Math.random();
            }
        }
        return m;
    }

    public void fixNegative() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                if (this.data[i][j] <= 0) {
                    this.data[i][j] = 0;
                }
            }
        }

    }

    public void fixPositive() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                if (this.data[i][j] >= 255) {
                    this.data[i][j] = 255;
                }
            }
        }
    }

    public double sumDiagonals() {
        double sum = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                if (i == j) {
                    sum = sum + this.data[i][j];
                }
            }
        }
        return sum;

    }

    public static Matrix filterLeftEdge() {
        Matrix filterM = new Matrix(3, 3);
        for (int i = 0; i < filterM.rows; i++) {
            for (int j = 0; j < filterM.cols; j++) {
                if (j == 0) {
                    filterM.data[i][j] = -1;
                }
                if (j == 1) {
                    filterM.data[i][j] = 1;
                }
                if (j == 2) {
                    filterM.data[i][j] = 0;
                }
            }
        }
        return filterM;
    }
        public static Matrix filterRightEdge() {
        Matrix filterM = new Matrix(3, 3);
        for (int i = 0; i < filterM.rows; i++) {
            for (int j = 0; j < filterM.cols; j++) {
                if (j == 0) {
                    filterM.data[i][j] = 0;
                }
                if (j == 1) {
                    filterM.data[i][j] = 1;
                }
                if (j == 2) {
                    filterM.data[i][j] = -1;
                }
            }
        }
        return filterM;
    }

    public int getAverageValue() {
        int sum = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        sum = sum / (this.rows + this.cols);
        return sum;
    }

    public double getSum() {
        double sum = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        return sum;
    }

    int getMaximumValue() {
        double check = -999999999;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                if (this.data[i][j] > check) {
                    check = this.data[i][j];
                }
            }
        }
        return (int) check;
    }

}
