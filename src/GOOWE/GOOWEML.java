package GOOWE;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.MultiLabelLearner;
import moa.classifiers.MultiTargetRegressor;
import moa.core.Utils;

/**
 * Created by abuyukcakir on 10.06.2017.
 */
public class GOOWEML extends GOOWE implements MultiLabelLearner, MultiTargetRegressor {

    public GOOWEML(){
        super();
//        this.outputFileName = "";
//        try {
//            output = new BufferedWriter(new FileWriter("goowe-out.txt", false));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

    }

    @Override
    public void setWindowSize(int windowSize) {
        super.setWindowSize(windowSize);
    }

    @Override
    public void trainOnInstanceImpl(MultiLabelInstance multiLabelInstance) {
        trainOnInstanceImpl((Instance) multiLabelInstance);
    }

    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance multiLabelInstance) {
//        System.out.println("inside this fnc");
        Prediction result = new MultiLabelPrediction(multiLabelInstance.numOutputAttributes());
        double[] votes = getVotesForInstance(multiLabelInstance);
//        votes = normalizeVotes(votes);
        int maxIndex = Utils.maxIndex(votes);

        for (int i = 0; i < votes.length; i++) {
            votes[i] = votes[i] / votes[maxIndex];
//            System.out.println(votes[i] + " ");
        }
//        System.out.println();

//        result.setVotes(votes);

        for(int i = 0; i < numberOfLabels; i++){  //for each label
            result.setVote(i, 0, 1 - votes[i]);
            result.setVote(i, 1, votes[i]);
        }

        System.out.println("Result votes:");
        System.out.println(result.toString());

        return result;
    }

    @Override
    public Prediction getPredictionForInstance(Instance multiLabelInstance) {
//        System.out.println("Making the prediction...");
        Prediction result = new MultiLabelPrediction(multiLabelInstance.numOutputAttributes());

        double[] votes = getVotesForInstance(multiLabelInstance);

        //OLD
        if(votes != null || votes.length > 0) {

            votes = normalizeVotes(votes);
//            System.out.println(votes.length);
        }
        else{
            System.out.println("votes null or length 0");
        }

//        System.out.println("GOOWEML --- Votes: ");


        double threshold = 1.0 / numberOfLabels;
//        System.out.println("Threshold: " + threshold);

//        System.out.println("Votes:");
        for(int i = 0; i < numberOfLabels; i++){  //for each label
            result.setVote(i, 1, votes[i] > threshold ? 1.0 : 0.0);
            result.setVote(i, 0, 1 - result.getVote(i, 1));
        }
//        System.out.println();

        // END OLD
        return result;
    }

}
