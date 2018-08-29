package MainPackage;

import Baselines.*;

import GOOWE.GOOWE;
import GOOWE.GOOWEML;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.MultiLabelLearner;
import moa.classifiers.multilabel.meta.OzaBagAdwinML;
import moa.classifiers.multilabel.meta.OzaBagML;
import moa.core.*;
import moa.streams.MultiTargetArffFileStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by abuyukcakir on 17.08.2017.
 */
public class RunClassifiers {
    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.out.println("Missing or extra arguments. The arguments should be of the following form:\n" +
                    "datasetLocation numberOfLabels windowSize algorithmID.\n" +
                    "Algorithm IDs are as follows:\n" +
                    "1. EBR\n" +
                    "2. ECC\n" +
                    "3. EPS\n" +
                    "4. iSOUP-MTR-Bagging\n" +
                    "5. EBR+ADWIN\n" +
                    "6. ECC+ADWIN\n" +
                    "7. EPS+ADWIN\n" +
                    "");
        } else {
            //get the dataset name (or location) as argument
            String dataset = args[0];
            int numLabels = Integer.parseInt(args[1]);
            int windowSize = Integer.parseInt(args[2]);
            int algorithmIndex = Integer.parseInt(args[3]);

            File datasetFile = new File(dataset);
            String datasetFileName = datasetFile.getName();

            System.out.println("Execution starts for the dataset " + dataset + "\n" +
                    "with numLabels = " + numLabels + "\n" +
                    "and windowSize = " + windowSize + "\n" +
                    "for the selected algorithm with ID = " + algorithmIndex);

            String outdir = "output//statistics//" + datasetFileName + "-alg-" + algorithmIndex + "-";

            String outFileName = outdir + "-results" + ".txt";
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(outFileName)));
            StringBuilder out;

            OzaBagML learner = null;
            OzaBagMLISOUP ozaml_isoup = null;
            OzaBagAdwinML adwinlearner = null;

            if(algorithmIndex == 1){
                learner = new OzaBagML();
                learner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");
            }
            else if(algorithmIndex == 2){
                learner = new OzaBagML();
                learner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");
            }
            else if (algorithmIndex == 3){
                learner = new OzaBagML();
                learner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 10 -W weka.classifiers.bayes.NaiveBayesUpdateable)");
            }
            else if (algorithmIndex == 4){
                ozaml_isoup = new OzaBagMLISOUP();
            }
            else if (algorithmIndex == 5){
                adwinlearner = new OzaBagAdwinML();
                adwinlearner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");
            }
            else if (algorithmIndex == 6){
                adwinlearner = new OzaBagAdwinML();
                adwinlearner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");
            }
            else if (algorithmIndex == 7){
                adwinlearner = new OzaBagAdwinML();
                adwinlearner.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 20 -W weka.classifiers.bayes.NaiveBayesUpdateable)");
            }
            else{
                System.out.println("No such algorithm with the given ID exists.");
            }

            MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataset, numLabels + "");
            stream.prepareForUse();

            MultiLabelLearner curLearner = null;
            if(algorithmIndex == 1 | algorithmIndex == 2 | algorithmIndex == 3 )
                curLearner = learner;
            else if (algorithmIndex == 5 | algorithmIndex == 6 | algorithmIndex == 7 )
                curLearner = adwinlearner;
            else if (algorithmIndex == 4)
                curLearner = ozaml_isoup;

            //prepare the learner

            curLearner.setModelContext(stream.getHeader());
            curLearner.prepareForUse();

            //create the evaluator for this learner and dataset
            AdvancedMultiLabelEvaluator evaluator =
                    new AdvancedMultiLabelEvaluator(windowSize, true);

            int index = 0;  // index shows the index of the current instance in the dataset.

            long starttime, endtime;
            starttime = System.currentTimeMillis();

            while (stream.hasMoreInstances()) {
                InstanceExample instanceEx = stream.nextInstance();
                Instance instance = instanceEx.getData();

                try {
                    Prediction mlp = curLearner.getPredictionForInstance(instanceEx);
//                        System.out.println("Prediction for instance " + index);
//                        System.out.println(mlp.toString());
                    evaluator.addResult(instanceEx, mlp);       //test

                    curLearner.trainOnInstanceImpl((MultiLabelInstance) instance);              //then train
//                        learner.trainOnInstance(instance);              //then train
                    index++;

                } catch (Exception e) {
                    e.printStackTrace();
                    writer.write(e.getMessage());
                }
                if (index % windowSize == 0) {
                    System.out.println("Size of " + curLearner.getPurposeString() + " : " + SizeOf.fullSizeOf(curLearner));
                }

            }
            // end of the stream. show results

            endtime = System.currentTimeMillis();
            out = new StringBuilder();

            System.out.println(curLearner.getPurposeString());
            System.out.println("Performance Measurements:");
            Measurement[] measurements = evaluator.getPerformanceMeasurements();
            Measurement.getMeasurementsDescription(measurements, out, 0);
            System.out.println(out.toString() + "\n");

            writer.write(curLearner.getPurposeString() + "\n");
            writer.write(String.valueOf(out));
            writer.write("\n");

            String timeString = "Time: " + (endtime - starttime) + " ms  \n";
            System.out.println(timeString + "\n");
            writer.write(timeString + "\n");


            writer.write("Size of the model: " + SizeOf.fullSizeOf(curLearner) + " bytes\n");

            writer.close();
        }
    }
}
