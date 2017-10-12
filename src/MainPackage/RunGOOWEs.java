package MainPackage;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Baselines.*;

import GOOWE.GOOWEML;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.MultiLabelClassifier;
import moa.classifiers.MultiLabelLearner;
import moa.classifiers.meta.OzaBoost;
import moa.classifiers.meta.WEKAClassifier;
import moa.classifiers.multilabel.MultilabelHoeffdingTree;
import moa.classifiers.multilabel.meta.OzaBagAdwinML;
import moa.classifiers.multilabel.meta.OzaBagML;
//import moa.core.InstanceExample;
import moa.classifiers.multilabel.trees.ISOUPTree;
import moa.core.*;
import moa.evaluation.BasicMultiLabelPerformanceEvaluator;
import moa.evaluation.F1;
import moa.options.ClassOption;
import moa.options.WEKAClassOption;
import moa.streams.MultiTargetArffFileStream;
import moa.classifiers.multilabel.MEKAClassifier;
import weka.classifiers.meta.AdaBoostM1;
import meka.classifiers.multilabel.RAkEL;
import weka.classifiers.meta.AdaBoostM1;
import moa.classifiers.Classifier;

public class RunGOOWEs {

    public static int numberOfWindows = 20;

    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.out.println("Missing or extra arguments. The arguments should be of the following form:\n" +
                    "datasetLocation numberOfInstances numberOfLabels algorithmID.\n" +
                    "Algorithm IDs are as follows:\n" +
                    "1. GOOWE-BR\n" +
                    "2. GOOWE-CC\n" +
                    "3. GOOWE-PS\n" +
                    "4. GOOWE-iSOUP-MTR" +
                    "");
        } else {
            //get the dataset name (or location) as argument
            String dataset = args[0];
            int numInstances = Integer.parseInt(args[1]);
            int numLabels = Integer.parseInt(args[2]);
            int algorithmIndex = Integer.parseInt(args[3]);

            System.out.println("Execution starts for the dataset " + dataset + "\n" +
                    "with NumInstances = " + numInstances + " and numLabels = " + numLabels + "\n" +
                    "for the GOOWE number " + algorithmIndex);

            String outdir = "output//statistics//" + dataset + "-goowe-" + algorithmIndex + "-";
//        File directory = new File(dir);
            String outFileName = outdir + "-results" + ".txt";
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(outFileName)));
            StringBuilder out;

            // Initialize the learners.
            GOOWEML gooweml_mlht = new GOOWEML();
            gooweml_mlht.baseLearnerOption.setValueViaCLIString("multilabel.MultilabelHoeffdingTree -a multilabel.MajorityLabelset");

            GOOWEML gooweml_br = new GOOWEML();
            gooweml_br.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            GOOWEML gooweml_cc = new GOOWEML();
            gooweml_cc.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            GOOWEML gooweml_ps = new GOOWEML();
            gooweml_ps.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 10 -W weka.classifiers.bayes.NaiveBayesUpdateable)");

            GOOWEML gooweml_isoup = new GOOWEML();
            gooweml_isoup.isIsoup = true;

            List<MultiLabelLearner> classifiers = new ArrayList<>();

            classifiers.add(gooweml_mlht);

            classifiers.add(gooweml_br);
            classifiers.add(gooweml_cc);
            classifiers.add(gooweml_ps);
            classifiers.add(gooweml_isoup);

            MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataset, numLabels + "");
            stream.prepareForUse();

//            MultiLabelLearner learner = classifiers.get(algorithmIndex);
            //prepare the learner
//            stream.restart();

            GOOWEML learner = gooweml_cc;


            learner.setModelContext(stream.getHeader());
            learner.prepareForUse();

            int windowSize = numInstances / numberOfWindows;
//            int windowSize = 1000;      //for imdb

            //prepare the learner
            ((GOOWEML)learner).setWindowSize(windowSize);
            learner.resetLearning();

            //create the evaluator for this learner and dataset
            AdvancedMultiLabelEvaluator evaluator =
                    new AdvancedMultiLabelEvaluator(windowSize, true);

            int index = 0;  // index shows the index of the current instance in the dataset.

            long starttime, endtime;
            starttime = System.currentTimeMillis();

            while (stream.hasMoreInstances() && index < numInstances) {
                InstanceExample instanceEx = stream.nextInstance();
                Instance instance = instanceEx.getData();

                try {
                    if (index > windowSize) {   // need to form the first classifier

                        Prediction mlp = learner.getPredictionForInstance(instanceEx);

//                        System.out.println("Prediction: " + mlp.toString());

                        evaluator.addResult(instanceEx, mlp);       //test
                    }

                    learner.trainOnInstanceImpl((MultiLabelInstance) instance);              //then train
                    index++;

                } catch (Exception e) {
                    e.printStackTrace();
                    writer.write(e.getMessage());
                }
                if (index % windowSize == 0) {
                    System.out.println("Size of " + learner.getPurposeString() + " : " + SizeOf.fullSizeOf(learner));
                }

            }
            // end of the stream. show results

            endtime = System.currentTimeMillis();
            out = new StringBuilder();

            System.out.println(learner.getPurposeString());
            System.out.println("Performance Measurements:");
            Measurement[] measurements = evaluator.getPerformanceMeasurements();
            Measurement.getMeasurementsDescription(measurements, out, 0);
            System.out.println(out.toString() + "\n");

            writer.write(learner.getPurposeString() + "\n");
            writer.write(String.valueOf(out));
            writer.write("\n");

            String timeString = "Time: " + (endtime - starttime) + " ms  \n";
            System.out.println(timeString + "\n");
            writer.write(timeString + "\n");

            writer.write("Size of the model: " + SizeOf.fullSizeOf(learner) + " bytes\n");

            writer.close();
        }
    }

    private static class Dataset {
        /*
        *  A little helper class to contain dataset information.
        * */

        private String datasetName;
        private int numInstances;
        private int numLabels;

        public Dataset(String name, int instances, int labels) {
            datasetName = name;
            numInstances = instances;
            numLabels = labels;
        }

        public String getName() {
            return datasetName;
        }

        public int getNumInstances() {
            return numInstances;
        }

        public int getNumLabels() {
            return numLabels;
        }

    }
}
