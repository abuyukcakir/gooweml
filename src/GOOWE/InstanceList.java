
package GOOWE;

import Jama.Matrix;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.ArrayList;

/**
 *
 * @author Hamed R. Bonab
 * Date 17 March 2017
 */

public class InstanceList {

    int capacity; // fix window size
    int num_of_inst; // till now how many instances are there 
    int num_of_hypo;
    Instance[] list;
    VoteNode[] voteList;
    
    int not_unique;
    int unique;
    
    public InstanceList(int capacity){
        
        not_unique = 0;
        unique = 0;
        
        this.capacity = capacity;
        num_of_inst = 0;
        num_of_hypo = 0;
        
        list = new Instance[capacity];
        voteList =  new VoteNode[capacity];        
    }
    
    public void add(Instance ins, double[][] votes){
        
        for(int i=(list.length-1); i>0; i--){
            list[i] = list[i-1];
            voteList[i] = voteList[i-1]; 
        }
        list[0] = (Instance) ins.copy();

        // for the single label case, we just get ins.classValue().
//        voteList[0] = new VoteNode(votes, (int) ins.classValue());

        // for multi label, we get the list of the labels that are relevant.

//        if(votes != null){
//            System.out.println("before getting relevant indices, votes:");
//            for (int i = 0; i < votes.length; i++) {
//                for (int j = 0; j < votes[0].length; j++) {
//                    System.out.print(votes[i][j] + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }

        voteList[0] = new VoteNode(votes, getRelevantLabelIndices(ins));

        if(votes==null)
            num_of_hypo = 0;
        else
            num_of_hypo = votes.length;
        
        if(num_of_inst<capacity)
            num_of_inst++;
    }
    
    public Instance getIns(int index){
        return list[index];
    }
    
    public VoteNode getVote(int index){
        return voteList[index];
    }
    
    public boolean isReady(){
        if(num_of_inst>=capacity && num_of_hypo>=2)
            return true;
        else
            return false;
    }
 
    
    // calculate weights for current instances
    public double[] getWeight(){
        
        double[] weights;
        if(!this.isReady()){
            weights = new double[num_of_hypo];
            for(int i=0; i<num_of_hypo; i++){
                weights[i] = 1.0;
            }
            return weights;
        }
        
        double[][] A = new double[num_of_hypo][num_of_hypo];
        double[] D = new double[num_of_hypo];
        
        for(int i=0; i<capacity; i++){
            for(int q=0; q<num_of_hypo; q++){
                for(int j=0; j<num_of_hypo; j++){
                    A[q][j] += (voteList[i].getAt())[q][j];
                }
                D[q] += (voteList[i].getDt())[q];
            }
        }
        
        // get optimum weights by solving the linear equation
        weights = matrixSolver(A, D);

        //normalizing with "standart normalization"
        double min =weights[0];
        double max =weights[0];
        for(int i=1; i<weights.length;i++){
            if(weights[i]<min)
                min = weights[i];
            if(weights[i]>max)
                max = weights[i];
        }
        if(min != max){
            for(int i=0; i<weights.length;i++){
                weights[i] = ((weights[i]-min)/(max-min));
            }


        }

        return weights;        

    }
    
    
    private double[] matrixSolver(double[][] a, double[] d){
        //preparing matrix objects for JAMA package
        double[][] di = new double[d.length][1];
        for(int i=0; i<d.length; i++){
            di[i][0] = d[i];
        }
        Matrix A = new Matrix(a);
        Matrix D = new Matrix(di);
        double[][] res;
        
        //solve equation and change the result to sensible weight vector 
        double[] w = new double[d.length];
        try{            
            Matrix x = A.solve(D);
            res= x.getArray();
            for(int i=0;i<w.length; i++)
                w[i] = res[i][0];
            unique++;
        }catch(RuntimeException e){
            //System.out.println("Not a unique solution!!!! " + GOOWTester.index);
            for(int i=0;i<w.length; i++)
                w[i] = 1.0;
            not_unique++;
        }
        
        return w;
    }

    private ArrayList<Integer> getRelevantLabelIndices(Instance ins){
        // Relevant label indices will be sent to GOOWE to create ideal point.
        ArrayList<Integer> relevantLabelIndices = new ArrayList<>();

        int numLabels = ins.numOutputAttributes();

//        System.out.println("Ground truth:");

        for(int i = 0; i < numLabels; i++){
            if((int) ins.valueOutputAttribute(i) == 1 )    // track which labels are relevant
                relevantLabelIndices.add(i);
        }

//        System.out.println(relevantLabelIndices.toString());
        return relevantLabelIndices;
    }
    
    
    public void printStatus(){
        System.out.println("not unique : "+ not_unique + "  unique : "+ unique);
    }
    
    // votes for an Instance produced by different classifiers 
    private class VoteNode{
        private double[][] At; // matrix A for instance It
        private double[] Dt;   // matrix D for instance It
        private int num_of_cur_hypos;
        private int num_of_classes;
        
        //private double[][] votes;

        public VoteNode(double[][] votes, int classIndex) {
            //class index should start from zero
            //this.votes = votes;
            if(votes == null) {
                num_of_cur_hypos = 0;
                num_of_classes = 0;
            } else {
                num_of_cur_hypos = votes.length;
                num_of_classes = votes[0].length;
            }
            
            At = new double[num_of_cur_hypos][num_of_cur_hypos];
            Dt = new double[num_of_cur_hypos];
            
            for(int i=0; i<num_of_cur_hypos ; i++){
                for(int j=i; j<num_of_cur_hypos ; j++){
                    double ss = 0;                    
                    for(int k=0; k<num_of_classes ; k++){
                        ss += (votes[i][k]*votes[j][k]);
                    }
                    At[i][j] = ss;
                    At[j][i] = ss;
                }
                // for single-label, Dt[i] = votes[i][classIndex]. for multi-label,
                // each of the relevant class indices must change.
                Dt[i] = votes[i][classIndex];
            }
            System.out.println("A matrix:");
            for (int i = 0; i < At.length; i++) {
                for (int j = 0; j < At[0].length; j++) {
                    System.out.print( At[i][j] + " ");
                }
                System.out.println();
            }
            System.out.println();


            System.out.println("d vector:");
            for (int j = 0; j < Dt.length; j++) {
                System.out.print( Dt[j] + " ");
            }
            System.out.println();
        }

        public VoteNode(double[][] votes, ArrayList<Integer> classIndices) { //class index should start from zero
            //this.votes = votes;
            if(votes == null) {
                num_of_cur_hypos = 0;
                num_of_classes = 0;
            } else {
                num_of_cur_hypos = votes.length;
                num_of_classes = votes[0].length;
            }

            At = new double[num_of_cur_hypos][num_of_cur_hypos];
            Dt = new double[num_of_cur_hypos];

            for(int i=0; i<num_of_cur_hypos ; i++){
                for(int j=i; j<num_of_cur_hypos ; j++){
                    double ss = 0;
                    for(int k=0; k<num_of_classes ; k++){
                        ss += (votes[i][k]*votes[j][k]);
                    }
                    At[i][j] = ss;
                    At[j][i] = ss;
                }
                // for single-label, Dt[i] = votes[i][classIndex]. for multi-label,
                // each of the relevant class indices must change.
                for(int k = 0; k < classIndices.size(); k++)
                    Dt[i] += votes[i][classIndices.get(k)];

            }
//            System.out.println("A matrix:");
//            for (int i = 0; i < At.length; i++) {
//                for (int j = 0; j < At[0].length; j++) {
//                    System.out.print( At[i][j] + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//
//
//            System.out.println("d vector:");
//            for (int j = 0; j < Dt.length; j++) {
//                System.out.print( Dt[j] + " ");
//            }
//            System.out.println();
        }
        
        public double[][] getAt(){
            return At;
        }
        
        public double[] getDt(){
            return Dt;
        }

        public boolean ensembleIsReady(){
           if(num_of_cur_hypos<2)
               return false;
           else
               return true;
        }
        
    }// end of VoteNode
    
    
}
