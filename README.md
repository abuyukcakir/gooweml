# GOOWE-ML

GOOWE-ML is a stacked online ensemble for MLSC (multi-label stream classification) task. Its name is an abbreviation for Geometrically-Optimum Online Weighted Ensemble for Multi-Label Classification.

It is introduced in the ACM CIKM 2018 paper: 

"A Novel Online Stacked Ensemble for Multi-Label Stream Classification" by
Alican Büyükçakır, Hamed Bonab and Fazli Can.

### Dependencies

* MOA
* MEKA
* Jama - For matrix operations such as solving LSQ.
* sizeofag - for measuring memory consumption of each model

### Datasets

Can be downloaded from http://meka.sourceforge.net/#datasets. In case of this link crashing, I put the datasets that are used in our experiments into this repository as well.

### Running Models


Assuming you generated .jar files that run the main method for the files RunClassifiers.java and RunGOOWEs.java

Create the following directories in the same directory as your jar files:

- ./output/final-results
- ./output/statistics

The jar file will generate window-based evaluations of the models (in detailed and short formats) in the former; overall results wrt many metrics in the latter.

Run the experiments in the following format (arrange the virtual memory size according to your config.):

```shell
java -jar -Xmx32G -javaagent:sizeofag.jar Model.jar Dataset.arff NumLabels BatchSize AlgorithmNo
```


For instance, WLOG, for the dataset 20NG, run the experiments as:

* For GOOWE-ML-based ensembles:

```shell
java -jar -Xmx32G -javaagent:sizeofag.jar RunGOOWEs.jar 20NG-F.arff 20 1000 ${j}
```

where ${j} = 1 to 4 corresponds to [GOBR, GOCC, GOPS, GORT].

* For the baselines:

```shell
java -jar -Xmx32G -javaagent:sizeofag.jar RunClassifiers.jar 20NG-F.arff 20 1000 ${j}
```

where ${j} = 1 to 7 corresponds to [EBR, ECC, EPS, EBRT, EaBR, EaCC, EaPS].

#### To cite, use the following bibtex entry:
```
@inproceedings{buyukccakir2018novel,
  title={A novel online stacked ensemble for multi-label stream classification},
  author={B{\"u}y{\"u}k{\c{c}}ak{\i}r, Alican and Bonab, Hamed and Can, Fazli},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={1063--1072},
  year={2018},
  organization={ACM}
}

