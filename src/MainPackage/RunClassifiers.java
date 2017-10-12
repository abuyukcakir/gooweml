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
    public static int numberOfWindows = 20;

    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.out.println("Missing or extra arguments. The arguments should be of the following form:\n" +
                    "datasetLocation numberOfInstances numberOfLabels algorithmID.\n" +
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
            int numInstances = Integer.parseInt(args[1]);
            int numLabels = Integer.parseInt(args[2]);
            int algorithmIndex = Integer.parseInt(args[3]);

//            System.out.println("Execution starts for the dataset " + dataset + "\n" +
//                    "with NumInstances = " + numInstances + " and numLabels = " + numLabels + "\n" +
//                    "for the algorithm number " + algorithmIndex);

            String outdir = "output//statistics//" + dataset + "-alg-" + algorithmIndex + "-";
//        File directory = new File(dir);

            String outFileName = outdir + "-results" + ".txt";
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(outFileName)));
            StringBuilder out;

//        GOOWEML gooweml_br = new GOOWEML();
//        gooweml_br.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");
//
//        GOOWEML gooweml_cc = new GOOWEML();
//        gooweml_cc.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");
//
//        GOOWEML gooweml_isoup = new GOOWEML();
//        gooweml_isoup.isIsoup = true;

//        GOOWEML gooweml_mlht = new GOOWEML();
//        gooweml_mlht.baseLearnerOption.setValueViaCLIString("multilabel.MultilabelHoeffdingTree -a multilabel.MajorityLabelset");

            OzaBagML ebr = new OzaBagML();
            ebr.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            OzaBagML ecc = new OzaBagML();
            ecc.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            OzaBagML eps = new OzaBagML();
            eps.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 10 -W weka.classifiers.bayes.NaiveBayesUpdateable)");

            OzaBagMLISOUP ozaml_isoup = new OzaBagMLISOUP();

            OzaBagML oza_mlht = new OzaBagML();
            oza_mlht.baseLearnerOption.setValueViaCLIString("multilabel.MultilabelHoeffdingTree -a multilabel.MajorityLabelset");

            OzaBagAdwinML ebr_adwin = new OzaBagAdwinML();
            ebr_adwin.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.BRUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            OzaBagAdwinML ecc_adwin = new OzaBagAdwinML();
            ecc_adwin.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.CCUpdateable -W weka.classifiers.trees.HoeffdingTree)");

            OzaBagAdwinML eps_adwin = new OzaBagAdwinML();
            eps_adwin.baseLearnerOption.setValueViaCLIString("multilabel.MEKAClassifier -l (meka.classifiers.multilabel.incremental.PSUpdateable -I 100 -S 20 -W weka.classifiers.bayes.NaiveBayesUpdateable)");

            List<MultiLabelLearner> classifiers = new ArrayList<>();
//        classifiers.add(br);
//        classifiers.add(cc);
//        classifiers.add(isoup);
//        classifiers.add(ps);
//        classifiers.add(mlht);

            classifiers.add(oza_mlht);

            classifiers.add(ebr);
            classifiers.add(ecc);
            classifiers.add(eps);

            classifiers.add(ozaml_isoup);

            classifiers.add(ebr_adwin);
            classifiers.add(ecc_adwin);
            classifiers.add(eps_adwin);


//        classifiers.add(gooweml_mlht);
//        classifiers.add(gooweml_br);
//        classifiers.add(gooweml_cc);
//        classifiers.add(gooweml_isoup);

            MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataset, numLabels + "");
            stream.prepareForUse();

            MultiLabelLearner learner = classifiers.get(algorithmIndex);
            //prepare the learner
//            stream.restart();
//            learner.resetLearning();

            learner.setModelContext(stream.getHeader());
            learner.prepareForUse();

            //create the evaluator for this learner and dataset
            AdvancedMultiLabelEvaluator evaluator =
                    new AdvancedMultiLabelEvaluator(numInstances / numberOfWindows, true);

            int index = 0;  // index shows the index of the current instance in the dataset.

            long starttime, endtime;
            starttime = System.currentTimeMillis();

            while (stream.hasMoreInstances() && index < numInstances) {
                InstanceExample instanceEx = stream.nextInstance();
                Instance instance = instanceEx.getData();

                try {
                    Prediction mlp = learner.getPredictionForInstance(instanceEx);
//                        System.out.println("Prediction for instance " + index);
//                        System.out.println(mlp.toString());
                    evaluator.addResult(instanceEx, mlp);       //test

                    learner.trainOnInstanceImpl((MultiLabelInstance) instance);              //then train
//                        learner.trainOnInstance(instance);              //then train
                    index++;

                } catch (Exception e) {
                    e.printStackTrace();
                    writer.write(e.getMessage());
                }
                if (index % numInstances / numberOfWindows == 0) {
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
}