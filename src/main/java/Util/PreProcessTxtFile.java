/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author abhunavgarg
 */
public class PreProcessTxtFile {

    //<editor-fold defaultstate="collapsed" desc="DataDNA">
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GLOBALS">
    public int noOfWords = 0;
    public int noOfUniqueWords = 0;
    public String processedTextdata = null;
    public String uniqueWordsString = "";
//    public ArrayList<DataDNA> dataset;

    public ArrayList<String> inputWords = new ArrayList<>();
    public ArrayList<String> outputWords = new ArrayList<>();
    public int windowSize = 5;
    public String[] allWords;
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Constructor">
    public PreProcessTxtFile(String path) throws Exception {
        System.out.println("Processing Data ...");
        String data = readFileAsString(path);
        preProcess(data);
        getUniqueWords();
        generateDataSet();
        System.out.println("Processing Finished.");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="generateDataSet">
    private void generateDataSet() throws IOException {
        this.allWords = this.processedTextdata.split(" ");
//        FileOutputStream fout = new FileOutputStream(new File("dataSet.ser"));
//        ObjectOutputStream oos = new ObjectOutputStream(fout);
        try {
            for (int i = 1; i < this.allWords.length; i++) {
                if (i % 10000 == 0) {
                    System.out.println("DataSet Created for" + i);
                }
                int counter = this.windowSize / 2;

                String inputWord = this.allWords[i];
                for (int j = 0; j < this.windowSize; j++) {
//                DataDNA dna = new DataDNA();
//                dna.input = this.allWords[i];
                    this.inputWords.add(inputWord);
                    if (i - counter >= 0) {
                        if (i + counter <= this.allWords.length) {
//                        dna.output = this.allWords[i - counter];
                            String outputWord = this.allWords[i - counter];
                            this.outputWords.add(outputWord);
//                        oos.writeObject(dna);
                        }
                    }
                    counter--;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Generated Data : dataSet.ser");
//        oos.close();
//        fout.close();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="readFileAsString">
    public String readFileAsString(String fileName) throws Exception {
        String rawFileData = "";
        rawFileData = new String(Files.readAllBytes(Paths.get(fileName)));
        return rawFileData;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="preProcess">
    private void preProcess(String str) throws IOException {
        str = str.replaceAll("[^a-zA-Z0-9]+", " ");
        str = str.replaceAll(" +", " ");
        str = str.replaceAll("\n+", " ");
        String str1 = str.toLowerCase();
        this.processedTextdata = str1;
        this.noOfWords = str1.split(" ").length + 1;
        FileWriter myWriter = new FileWriter("FormattedFile.txt");
        myWriter.write(this.processedTextdata);
        myWriter.close();
        System.out.println("Processed Data : processedFile.txt");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="getUniqueWords">
    private void getUniqueWords() throws IOException {
        String[] words = this.processedTextdata.split(" ");
        String unique = "";
        Map<String, String> vocabulary = new HashMap<>();
        StringBuilder builder = new StringBuilder(unique);
        for (int i = 0; i < words.length; i++) {
            if (!vocabulary.containsKey(words[i])) {
                vocabulary.put(words[i], "new");
                builder.append(words[i]);
                builder.append(" ");
            }
        }
        this.uniqueWordsString = builder.toString();
        this.noOfUniqueWords = this.uniqueWordsString.split(" ").length;
        FileWriter myWriter = new FileWriter("uniqueWords.txt");
        System.out.println("Unique Data : uniqueWords.txt");
        myWriter.write(this.uniqueWordsString);
        myWriter.close();
    }
    //</editor-fold>

}
