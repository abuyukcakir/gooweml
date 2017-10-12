package MainPackage;

import moa.core.Utils;
import moa.evaluation.BasicMultiLabelPerformanceEvaluator;
import moa.AbstractMOAObject;
import moa.core.Example;
import moa.core.Measurement;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.evaluation.MultiTargetPerformanceEvaluator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by abuyukcakir on 22.06.2017.
 */
public class AdvancedMultiLabelEvaluator extends AbstractMOAObject implements MultiTargetPerformanceEvaluator {

    protected int L;
    private boolean savingIncremental = false;

    /** running sum of accuracy */
    private double sumExactMatch = 0.0;
    private double sumHamming = 0.0;
    private double[] sumTP;
    private double[] sumFP;
    private double[] sumFN;

    private int windowSize;

    private double sumExamplePrecision, sumExampleRecall, sumExampleAccuracy, exampleFScore;
    private double microPrecision, microRecall, microFScore;
    private double macroPrecision, macroRecall, macroFScore;

    private ArrayList<Double> incrementalExactMatch = new ArrayList<>();
    private ArrayList<Double> incrementalHamming = new ArrayList<>();
    private ArrayList<Double> incrementalExAcc = new ArrayList<>();
    private ArrayList<Double> incrementalExF1 = new ArrayList<>();


    private double[] windowExactMatch;
    private double[] windowHamming;
    private double[] windowExAcc;
    private double[] windowExF1;

    private double sum_windowEM, sum_windowHamming, sum_windowExAcc, sum_windowExF1;

    /** running number of examples */
    int sumExamples = 0;

    /** preset threshold */
    private double t = 0.5;

    public AdvancedMultiLabelEvaluator(int windowSize, boolean savingIncremental){
        this.windowSize = windowSize;
        this.savingIncremental = savingIncremental;

        //for sliding window of size: windowsize
        windowExactMatch = new double[windowSize];
        windowHamming = new double[windowSize];
        windowExAcc = new double[windowSize];
        windowExF1 = new double[windowSize];
    }

    @Override
    public void reset() {
        sumExactMatch = 0.0;
        sumHamming = 0.0;
        sumExamples = 0;
        sumExampleAccuracy = 0;
        sumExamplePrecision = 0;
        sumExampleRecall = 0;
        exampleFScore = 0;
    }

    @Override
    public void addResult(Example<Instance> example, Prediction y) {

        MultiLabelInstance x = (MultiLabelInstance) example.getData();

        if (L == 0) {
            L = x.numberOutputTargets();

            sumTP = new double[L];
            sumFP = new double[L];
            sumFN = new double[L];
        }

        if (y == null) {
            System.err.print("[WARNING] Prediction is null! (Ignoring this prediction)");
        }
        else if (y.numOutputAttributes() < x.numOutputAttributes()) {
            System.err.println("[WARNING] Only "+y.numOutputAttributes()+" labels found! (Expecting "+x.numOutputAttributes()+")\n (Ignoring this prediction)");
        }
        else {
            sumExamples++;
            int correct = 0;
            double cur_tp = 0;
            double cur_fp = 0;
            double cur_fn = 0;

            for (int j = 0; j < y.numOutputAttributes(); j++) {
                int yp = (y.getVote(j,1) > t) ? 1 : 0;  //prediction itself

                //True Positive if 1 and predicted 1:
                sumTP[j] += ((int)x.classValue(j) == 1 && yp == 1) ? 1 : 0;
                cur_tp += ((int)x.classValue(j) == 1 && yp == 1) ? 1 : 0;   //for example-based

                //False Negative if 1 and predicted 0:
                sumFN[j] += ((int)x.classValue(j) == 1 && yp == 0) ? 1 : 0;
                cur_fn += ((int)x.classValue(j) == 1 && yp == 0) ? 1 : 0;   //for example-based

                //False Positive if 0 and predicted 1:
                sumFP[j] += ((int)x.classValue(j) == 0 && yp == 1) ? 1 : 0;
                cur_fp += ((int)x.classValue(j) == 0 && yp == 1) ? 1 : 0;       //for example-based

                correct += ((int)x.classValue(j) == yp) ? 1 : 0;
            }

            //  get example based metrics..
            double delta_EM = (correct == L) ? 1 : 0;
            double delta_ham = correct / (double)L;

            sumHamming+= delta_ham; 			// Hamming Score
            sumExactMatch += delta_EM; 		// Exact Match

            double delta_exPre = 0.0, delta_exRec = 0.0, delta_exAcc = 0.0;
            if(cur_tp != 0){
                delta_exPre = cur_tp / (cur_tp + cur_fp);
                sumExamplePrecision += delta_exPre;

                delta_exRec = cur_tp / (cur_tp + cur_fn);
                sumExampleRecall += delta_exRec;

                delta_exAcc = cur_tp / (cur_tp + cur_fn + cur_fp);
                sumExampleAccuracy += delta_exAcc;
            }

            // update incremental metrics that kept record of...
            if(savingIncremental){

                // add eval. to the sliding window. if full, replace the oldest.
                // this means add to index of i % windowSize
                int currentIndex = sumExamples % windowSize;
                updateWindowAccuracy(currentIndex, delta_EM, delta_ham, delta_exAcc, delta_exPre, delta_exRec);

            }

            //end incrementals
        }

    }

    private double getMicroPrecision(double[] tp, double[] fp){
        double sum_tp = 0;
        double sum_tp_fp = 0.0;

        for (int i = 0; i < L; i++) {
            sum_tp += tp[i];
            sum_tp_fp += tp[i];
            sum_tp_fp += fp[i];
        }

        double micro_precision = sum_tp / sum_tp_fp;
        return micro_precision;
    }

    private double getMicroRecall(double[] tp, double[] fn){
        double sum_tp = 0;
        double sum_tp_fn = 0.0;
        double micro_recall = 0.0;

        for (int i = 0; i < L; i++) {
            sum_tp += tp[i];
            sum_tp_fn += tp[i];
            sum_tp_fn += fn[i];
        }

        micro_recall = sum_tp / sum_tp_fn;
        return micro_recall;
    }

    private double getMacroPrecision(double[] tp, double[] fp){
        double[] macro_precision = tp.clone();
        for (int i = 0; i < L; i++) {
            double denom = tp[i] + fp[i];

            if(denom == 0.0)
                macro_precision[i] = 0.0;
            else
                macro_precision[i] = tp[i] / (tp[i] + fp[i]);
        }

        //average it over all labels
        return Utils.sum(macro_precision) / L;

    }

    private double getMacroRecall(double[] tp, double[] fn){
        double[] macro_recall = tp.clone();
        for (int i = 0; i < L; i++) {
            double denom = tp[i] + fn[i];

            if(denom == 0.0)
                macro_recall[i] = 0.0;
            else
                macro_recall[i] = tp[i] / (tp[i] + fn[i]);
        }

        //average it over all labels
        return Utils.sum(macro_recall) / L;
    }

    private void updateWindowAccuracy(int currentIndex, double delta_EM, double delta_ham, double delta_exAcc,
                                      double delta_exPre, double delta_exRec){
        // remove the old value from the window sum, overwrite, then add the new value to sum.
        // do it for every metric.

        sum_windowEM -= windowExactMatch[currentIndex];
        windowExactMatch[currentIndex] = delta_EM;
        sum_windowEM += delta_EM;

        sum_windowHamming -= windowHamming[currentIndex];
        windowHamming[currentIndex] = delta_ham;
        sum_windowHamming += delta_ham;

        sum_windowExAcc -= windowExAcc[currentIndex];
        windowExAcc[currentIndex] = delta_exAcc;
        sum_windowExAcc += delta_exAcc;

        sum_windowExF1 -= windowExF1[currentIndex];
        double delta_fscore;
        if(delta_exPre + delta_exRec > 0)
            delta_fscore = (2 * delta_exPre * delta_exRec) / (delta_exPre + delta_exRec);
        else
            delta_fscore = 0.0;
        windowExF1[currentIndex] = delta_fscore;
        sum_windowExF1 += delta_fscore;

        int numInstancesInTheWindow = windowSize < sumExamples ? windowSize : sumExamples;

        incrementalExactMatch.add(sum_windowEM / numInstancesInTheWindow);
        incrementalHamming.add(sum_windowHamming / numInstancesInTheWindow);
        incrementalExAcc.add(sum_windowExAcc / numInstancesInTheWindow);
        incrementalExF1.add(sum_windowExF1 / numInstancesInTheWindow);

    }

    private void saveDetailedWindowBasedEvaluation(String outdir){
        File fileIncrementalRecords = new File(outdir + "records1" + ".csv");
        int increase=1;
        while(fileIncrementalRecords.exists()){
            increase++;
            fileIncrementalRecords = new File(outdir + "records" + increase + ".csv");
        }
        if(!fileIncrementalRecords.exists()){
            try{
                fileIncrementalRecords.createNewFile();

                System.out.println("we re here:");
                System.out.println(fileIncrementalRecords.getAbsolutePath());

                BufferedWriter writer =  new BufferedWriter(new FileWriter(fileIncrementalRecords));
                StringBuilder out = new StringBuilder();

                //fill in the real time recordings
                out.append("numInstances,exactMatch,hammingScore,exampleAccuracy,exampleF1\n");
                for (int i = 0; i < sumExamples; i++) {
                    out.append((i+1) + "," + incrementalExactMatch.get(i) +
                            "," + incrementalHamming.get(i)+
                            "," + incrementalExAcc.get(i) +
                            "," + incrementalExF1.get(i) + "\n");
                }
                writer.write(String.valueOf(out));
                writer.close();

            }catch (IOException e){
            }
        }
    }

    private void saveShortWindowBasedEvaluation(String outdir){
        File fileIncrementalRecords = new File(outdir + "records1_short" + ".csv");     //short version. smoothened
        int increase=1;
        while(fileIncrementalRecords.exists()){
            increase++;
            fileIncrementalRecords = new File(outdir + "records" + increase + "_short.csv");
        }
        if(!fileIncrementalRecords.exists()){
            try{
                fileIncrementalRecords.createNewFile();

//                System.out.println("we re here:");
                System.out.println(fileIncrementalRecords.getAbsolutePath());

                BufferedWriter writer =  new BufferedWriter(new FileWriter(fileIncrementalRecords));
                StringBuilder out = new StringBuilder();

                //fill in the real time recordings
                out.append("numInstances,exactMatch,hammingScore,exampleAccuracy,exampleF1\n");

                //calculate average for each window.
                System.out.println("Window size: " + windowSize);
                int numberOfWindows = sumExamples / windowSize;       // integer division. no problem.
                System.out.println("Number of windows: " + numberOfWindows);
                System.out.println("Incremental arraylist size: " + incrementalExactMatch.size());

                double[] shorter_em = new double[numberOfWindows];
                double[] shorter_ham = new double[numberOfWindows];
                double[] shorter_exacc = new double[numberOfWindows];
                double[] shorter_exf1 = new double[numberOfWindows];

                for (int i = 0; i < numberOfWindows; i++) {
                    int index = (i+1) * windowSize - 1;
                    shorter_em[i] = incrementalExactMatch.get(index);
                    shorter_ham[i] = incrementalHamming.get(index);
                    shorter_exacc[i] = incrementalExAcc.get(index);
                    shorter_exf1[i] = incrementalExF1.get(index);
                }

                //write records..
                for (int i = 0; i < numberOfWindows; i++) {
                    out.append((i+1) + "," + shorter_em[i] +
                            "," + shorter_ham[i] +
                            "," + shorter_exacc[i] +
                            "," + shorter_exf1[i] + "\n");
                }
                writer.write(String.valueOf(out));
                writer.close();

            }catch (IOException e){
            }
        }
    }

    @Override
    public Measurement[] getPerformanceMeasurements() {
        //get the overall metrics...
        System.out.println("total examples: " + sumExamples);

        double examplePrecision = sumExamplePrecision / sumExamples;
        double exampleRecall = sumExampleRecall / sumExamples;
        double exampleAccuracy = sumExampleAccuracy / sumExamples;
        double exampleFScore = 2.0 * examplePrecision * exampleRecall / (examplePrecision + exampleRecall);

        // micro averaged measures:
        microPrecision = getMicroPrecision(sumTP, sumFP);   //micro averaged precision
        microRecall = getMicroRecall(sumTP, sumFN);         //micro averaged recall

        microFScore = 0;                                    //micro averaged fscore
        if(microPrecision != 0.0 && microRecall != 0.0)
            microFScore = 2 * microPrecision * microRecall / (microPrecision + microRecall);

        //macro averaged measures:
        macroPrecision = getMacroPrecision(sumTP, sumFP);   //macro averaged precision
        macroRecall = getMacroRecall(sumTP, sumFN);         //macro averaged recall

        macroFScore = 0;                                    //macro averaged fscore
        if(macroPrecision != 0.0 && macroRecall != 0.0)
            macroFScore = 2 * macroPrecision * macroRecall / (macroPrecision + macroRecall);

        // gather measurements
        Measurement m[] = new Measurement[]{
                new Measurement("Exact Match", sumExactMatch/sumExamples),
                new Measurement("Hamming Score", sumHamming/sumExamples),
                new Measurement("Example-Based Accuracy", exampleAccuracy),
                new Measurement("Example-Based Precision", examplePrecision),
                new Measurement("Example-Based Recall", exampleRecall),
                new Measurement("Example-Based F-Score", exampleFScore),
                new Measurement("Micro-Averaged Precision", microPrecision),
                new Measurement("Micro-Averaged Recall", microRecall),
                new Measurement("Micro-Averaged F1-Score", microFScore),
                new Measurement("Macro-Averaged Precision", macroPrecision),
                new Measurement("Macro-Averaged Recall", macroRecall),
                new Measurement("Macro-Averaged F1-Score", macroFScore)
        };

        // save incremental findings....
        if(savingIncremental){
            String outdir = "output//final-results//";

            saveDetailedWindowBasedEvaluation(outdir);
            saveShortWindowBasedEvaluation(outdir);
        }

        return m;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        sb.append("Basic Multi-label Performance Evaluator");
    }

    @Override
    public void addResult(Example<Instance> example, double[] classVotes) {
        // TODO Auto-generated method stub

    }
}