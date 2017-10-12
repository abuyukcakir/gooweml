package GOOWE;

import Baselines.MyISOUPTree;
import MainPackage.RunMultiLabel;
import com.github.javacliparser.IntOption;
import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.meta.WEKAClassifier;
import moa.classifiers.multilabel.MEKAClassifier;
import moa.classifiers.multilabel.trees.ISOUPTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.options.WEKAClassOption;
import moa.tasks.TaskMonitor;

/**
 *
 * @author Hamed R. Bonab
 *  Date 17 March 2017
 *
 *  @author Alican Büyükçakır
 *  7.10.17
 */

public class GOOWE extends AbstractClassifier{
    
    // options 
    public ClassOption baseLearnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class,
			"trees.HoeffdingTree -e 2000000 -g 100 -c 0.01");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public final boolean fuseOutput = true;
    public boolean isIsoup;

    public final int numOfHypo = ensembleSizeOption.getValue();
    public int numberOfLabels;
    
//    final int fixedWindowPeriod = 500; //this specifies if no change in this period of time happens we should train new hypo and compare it with existings
    public int fixedWindowPeriod;

    public Classifier[] hypo; // array of classifiers in ensemble

    public double[] glob_weight;  // weights of each classifier in an ensemble
    
    InstanceList window;
    int num_proccessed_instance;
    int curNumOfHypo; //number of hypothesis till now     
    int candidateIndex;

    public GOOWE(){
        fixedWindowPeriod = 50;     //default
        isIsoup = false;

    }

    public void setWindowSize(int windowSize){
        fixedWindowPeriod = windowSize;
    }
    
    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        super.prepareForUseImpl(monitor, repository);
    }
    
    
    @Override
    public void resetLearningImpl() {
        window = new InstanceList(fixedWindowPeriod);
        this.num_proccessed_instance = 0;
        this.curNumOfHypo = 0;
        this.hypo = new Classifier[numOfHypo+1];
        glob_weight = new double[numOfHypo];

//        System.out.println("Ensemble size: " + numOfHypo);

        for(int i=0; i< hypo.length ; i++){
            if(!isIsoup){
                Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);

                baseLearner.setModelContext(this.getModelContext());
//                baseLearner.resetLearning();
                hypo[i] = baseLearner.copy();

                hypo[i].prepareForUse();
            }
            else
            {
                ISOUPTree baseLearner = new ISOUPTree();
                baseLearner.setModelContext(this.getModelContext());
                baseLearner.resetLearning();

                hypo[i] = baseLearner;
                hypo[i].prepareForUse();
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        numberOfLabels = instnc.numOutputAttributes();

        double[][] votes = new double[curNumOfHypo][numberOfLabels];
        for(int i=0; i<curNumOfHypo ; i++){
            Prediction p = hypo[i].getPredictionForInstance(instnc);
//            System.out.println("Prediction " + i + " : " + p.toString());
            double[] vote = new double[p.numOutputAttributes()];
            for (int k = 0; k < p.numOutputAttributes(); k++) {
                vote[k] = p.getVote(k, 1);
            }
            vote = normalizeVotes(vote);

            for(int j=0; j<vote.length ; j++){
                votes[i][j] = vote[j];
            }            
        }
        if(num_proccessed_instance<fixedWindowPeriod)
            votes = null;
        
        window.add(instnc, votes);
        this.num_proccessed_instance++;
                
        if(this.num_proccessed_instance % fixedWindowPeriod == 0){ //chunk is full
//            System.out.println("Chunk will be processed now...");
            processChunk();
        }
        
    }

    public void trainOnInstance(Instance instnc){
        trainOnInstanceImpl(instnc);
    }


    public double[] normalizeVotes(double[] votes){
        double[] newVotes = new double[numberOfLabels];

        //check if all values are zero
        boolean allZero = true;
        for(int i=0; i<votes.length; i++){
            if(votes[i]>0)
                allZero=false;
        }

        if(allZero){ // all the votes are equal to zero

            double equalVote = 1.0/numberOfLabels;
            for(int i=0; i<numberOfLabels; i++){
                newVotes[i]=equalVote;
            }

        }else{ // votes are not equal to zero
            // pick one way to normalize
            newVotes = centerThanNormalize(votes);
//            newVotes = softmax(votes);
        }

        return newVotes;
        
    }
    
    //process a new given chunk of instances
    private void processChunk() {
        Classifier newClassifier = hypo[numOfHypo];
        if(!isIsoup){
            newClassifier = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        }
        else{
            newClassifier = new ISOUPTree();
        }

        newClassifier.setModelContext(this.getModelContext());
//        newClassifier.resetLearning();
        Classifier temp = newClassifier.copy();

//        newClassifier.prepareForUse();        // this fucks things up

        //train new classifier on this new chunk
        for(int i=fixedWindowPeriod; i>0; i--){
            temp.trainOnInstance(window.getIns(i-1));
        }

        // weight and train new and rest classifiers 
        if(curNumOfHypo==0) { //there is no one
//            System.out.println("no classifier yet.. training from chunk");
            candidateIndex = 0;
            hypo[candidateIndex] = temp;
            glob_weight[candidateIndex] = 1.0;
            curNumOfHypo++;
            
        } else if (curNumOfHypo < numOfHypo) { //still has space
            //when there is still space for new classifiers,
            // train the old ones with this instance window.
            // then, add the new one.

//            System.out.println("still has space.. training from chunk");

            candidateIndex = curNumOfHypo;

            hypo[candidateIndex] = temp;
            glob_weight[candidateIndex] = 1.0;

            double[] newWeights = window.getWeight();
            for(int i=0; i<newWeights.length; i++){
                glob_weight[i] = newWeights[i];
            }

            System.out.println("Weights:");
            for (int i = 0; i < glob_weight.length; i++) {
                System.out.print(glob_weight[i] + ", ");
            }
            System.out.println();


            curNumOfHypo++;
        } else { // is full
            
            glob_weight = window.getWeight();

            //find minimum weight
            candidateIndex = 0;

            System.out.println("Weights:");
            for (int i = 0; i < glob_weight.length; i++) {
                System.out.print(glob_weight[i] + ", ");
            }
            System.out.println();

            for(int i=1; i<glob_weight.length ; i++){
                if(glob_weight[i]<glob_weight[candidateIndex])
                    candidateIndex = i;
            }

            // print out the candidate index.
            System.out.println("Change model " + candidateIndex + " !");


            //substitute
            hypo[candidateIndex] = temp;
            glob_weight[candidateIndex] = 1.0;
        }

        //  train the rest of classifiers 
        for(int i=0;i<curNumOfHypo;i++){               // for each old classifier
            for(int j=0; j<fixedWindowPeriod; j++){     // for each instance in the window
                hypo[i].trainOnInstance(window.getIns(j));
            }
        }
//
//
//        System.out.println("Ensemble classifiers:");
//        for (int i = 0; i < numOfHypo; i++) {
//            System.out.print( hypo[i].hashCode() + ", ");
//        }
//        System.out.println();
        
        hypo[numOfHypo].resetLearning();
    }
    
    
    @Override
    public boolean correctlyClassifies(Instance inst) {
        int expectedClass = Utils.maxIndex(getVotesForInstance(inst));
        int realClass = (int) inst.classValue();
        
        return expectedClass == realClass;
    }
    
    
    @Override
    public double[] getVotesForInstance(Instance instnc) {
        DoubleVector combinedVote = new DoubleVector();        
        double[] hypo_weight = glob_weight;
        
        for (int i = 0; i < curNumOfHypo; i++) {
            Prediction p = hypo[i].getPredictionForInstance(instnc);
//
//            System.out.println("Prediction of classifier[" + i + "]");
//            System.out.println(p.toString());

            DoubleVector vote = new DoubleVector();

            for (int j = 0; j < p.numOutputAttributes(); j++) {
                vote.setValue(j, p.getVote(j, 1));
            }

            double[] v = vote.getArrayCopy();
            v = normalizeVotes(v);
            vote = new DoubleVector(v);
            vote.scaleValues(hypo_weight[i]);
            combinedVote.addValues(vote);

//            if (vote.sumOfValues() > 0.0) {
//                double[] v = vote.getArrayCopy();
//                v = normalizeVotes(v);
//                vote = new DoubleVector(v);
////                vote.normalize();
//                vote.scaleValues(hypo_weight[i]);
//                combinedVote.addValues(vote);
//            }
//            else{
//                vote.scaleValues(hypo_weight[i]);
//                combinedVote.addValues(vote);
//            }
        }

        return combinedVote.getArrayRef();
    }
    
    
    public void printArray(double[] arr){
        for(int i=0 ; i<arr.length; i++){
            System.out.println(arr[i] + " "); 
        }
        System.out.println("");
    }

    public double[] centerThanNormalize(double[] votes){
        double[] newVotes = new double[votes.length];

        int min = Utils.minIndex(votes);
        for (int i = 0; i < votes.length; i++) {
            newVotes[i] = votes[i] - votes[min];
        }
        //end edit

        double sum=0;
        for(int i=0; i<votes.length; i++){
            sum+=newVotes[i];
        }
        for(int i=0; i<votes.length; i++){
            newVotes[i]=(newVotes[i]/sum);
        }

        return newVotes;
    }

    public double[] softmax(double[] votes){
        double[] newVotes = new double[votes.length];

        for (int i = 0; i < votes.length; i++) {
            newVotes[i] = Math.pow(Math.E, votes[i]);
        }

        double sum=0;
        for(int i=0; i<votes.length; i++){
            sum+=newVotes[i];
        }
        for(int i=0; i<votes.length; i++){
            newVotes[i]=(newVotes[i]/sum);
        }

        return newVotes;
    }
    
    @Override
    public boolean isRandomizable() {
        return true;
    }   
    
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String getPurposeString() {
        String purposeString = "GOOWE with base classifier: " + (isIsoup ? "ISOUP Tree MTR":  baseLearnerOption.getValueAsCLIString());
        return purposeString;
    }
}
