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
import moa.classifiers.rules.multilabel.AMRulesMultiTargetRegressor;
import moa.classifiers.rules.multilabel.meta.MultiLabelRandomAMRules;
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

public class RunMultiLabel {

    public static int globNumClassifiers = 10;
    public static int globNumLabels;

    public static String globNameOfDS;
    public static int globNumFeatures;
    public static int windowSize = 50;
    public static int numberOfWindows = 20;


    public static void main(String[] args) throws IOException {
        String datasetDir = "dataset//";
        String outdir = "output//goowe-all";
//        File directory = new File(dir);

        String outFileName = outdir + "-results" + ".txt";
        BufferedWriter writer =  new BufferedWriter(new FileWriter(new File(outFileName)));
        StringBuilder out;

        //Datasets with their corresponding names, number of instances and number of labels.
        Dataset scene = new Dataset("Scene.arff", 2407, 6);
        Dataset yeast = new Dataset("Yeast.arff", 2417, 14);
        Dataset music = new Dataset("Music.arff", 592, 6);
        Dataset ohsumed = new Dataset("OHSUMED-F.arff", 13529, 23);
        Dataset slashdot = new Dataset("SLASHDOT-F.arff", 3782, 22);
        Dataset reuters = new Dataset("REUTERS-K500-EX2.arff", 6000, 103);
        Dataset imdb = new Dataset("IMDB-F.arff", 120918, 28);
        Dataset tmc2007 = new Dataset("A-TMC7-REDU-X2-500.arff", 28596, 22);
        Dataset enron = new Dataset("ENRON-F.arff", 1702, 53);

        List<Dataset> datasetList = new ArrayList<>();
        datasetList.add(scene);
//        datasetList.add(yeast);
//        datasetList.add(music);
//        datasetList.add(ohsumed);
//        datasetList.add(slashdot);
//        datasetList.add(reuters);
//        datasetList.add(imdb);
//        datasetList.add(tmc2007);
//        datasetList.add(enron);

        // Initialize the learners.
//
//        MEKAClassifier br = new MEKAClassifier();
//        br.baseLearnerOption = new WEKAClassOption("baseLearner", 'l',
//                "Classifier to train.", weka.classifiers.Classifier.class,
//                "meka.classifiers.multilabel.incremental.BRUpdateable");
//

        MEKAClassifier cc = new MEKAClassifier();
        cc.baseLearnerOption = new WEKAClassOption("baseLearner", 'l',
                "Classifier to train.", weka.classifiers.Classifier.class,
                "meka.classifiers.multilabel.incremental.CCUpdateable");

//
//        MyISOUPTree isoup = new MyISOUPTree();

        OzaBagML ebr = new OzaBagML();
        ebr.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");

        OzaBagML ecc = new OzaBagML();
        ecc.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");



        GOOWEML gooweml_br = new GOOWEML();
        gooweml_br.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");

        GOOWEML gooweml_cc = new GOOWEML();
        gooweml_cc.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");

        GOOWEML gooweml_ps = new GOOWEML();
        gooweml_ps.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 10 -W weka.classifiers.bayes.NaiveBayesUpdateable)");

        GOOWEML gooweml_isoup = new GOOWEML();
        gooweml_isoup.isIsoup = true;

        MultilabelHoeffdingTree mlht = new MultilabelHoeffdingTree();
        mlht.learnerOption.setValueViaCLIString("multilabel.MajorityLabelset");



        GOOWEML gooweml_mlht = new GOOWEML();
        gooweml_mlht.baseLearnerOption.setValueViaCLIString("multilabel.MultilabelHoeffdingTree  -a (multilabel.MajorityLabelset)");
        System.out.println("default : " + gooweml_mlht.baseLearnerOption.getDefaultCLIString());
        System.out.println("valueas cli : " + gooweml_mlht.baseLearnerOption.getValueAsCLIString());

        List<MultiLabelLearner> classifiers = new ArrayList<>();
//        classifiers.add(br);
//        classifiers.add(cc);
//        classifiers.add(isoup);

//        classifiers.add(ebr);
//        classifiers.add(ecc);
//        classifiers.add(ozaml_isoup);
//        classifiers.add(oza_mlht);


        classifiers.add(gooweml_br);
        classifiers.add(gooweml_cc);
        classifiers.add(gooweml_ps);
        classifiers.add(gooweml_isoup);
//        classifiers.add(gooweml_mlht);
//        classifiers.add(mlht);

        for (Dataset d: datasetList) {
            //Create the datastream...
            MultiTargetArffFileStream stream = new MultiTargetArffFileStream(datasetDir + d.getName(), d.getNumLabels() + "");
            stream.prepareForUse();

            //get some parameters...
            globNameOfDS = stream.getHeader().getRelationName();
            globNumFeatures = stream.getHeader().numInputAttributes();
            globNumLabels = stream.getHeader().numOutputAttributes();
            int numInstances = d.getNumInstances();

            //for debugging purposes
            System.out.println("Name of DS: " + globNameOfDS);
            System.out.println("Number of features:  " + globNumFeatures);
            System.out.println("Number of labels: " + globNumLabels);
            System.out.println("Number of instances: " + numInstances);

            writer.write("DATASET: " + globNameOfDS + "\n");
            writer.write("Number of Features: " + globNumFeatures + "\n");
            writer.write("Number of Labels: " + globNumLabels + "\n");
            writer.write("Number of instances: " + numInstances + "\n\n");


            //test each classifier..
            for (MultiLabelLearner learner: classifiers) {
                windowSize = d.getNumInstances() / numberOfWindows;

                //prepare the learner
                if(learner instanceof GOOWEML){
                    ((GOOWEML)learner).setWindowSize(windowSize);
                }

                stream.restart();
                learner.setModelContext(stream.getHeader());
                learner.prepareForUse();

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
                        if (index > windowSize) {

                            Prediction mlp = learner.getPredictionForInstance(instanceEx);
//                            System.out.println("Prediction for instance " + index);
//                            System.out.println(mlp.toString());

                                evaluator.addResult(instanceEx, mlp);       //test
                        }
                        learner.trainOnInstanceImpl((MultiLabelInstance) instance);              //then train
//                        learner.trainOnInstance((MultiLabelInstance) instance);              //

                        index++;

                        if( index % windowSize == 0){
                            System.out.println("Size of the learner : " + SizeOf.fullSizeOf(learner));
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                        writer.write(e.getMessage());
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

                String timeString = "Time: "+(endtime - starttime) + " ms  \n";
                System.out.println(timeString + "\n");
                writer.write(timeString + "\n");

            }
        }
        writer.close();
        System.out.println("END RUNMULTILABEL");
    }

    private static class Dataset{
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

    public String getName(){
        return datasetName;
    }

    public int getNumInstances(){
        return numInstances;
    }

    public int getNumLabels(){
        return numLabels;
    }

    }
}
