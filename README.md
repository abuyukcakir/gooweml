# GOOWE-ML

GOOWE-ML: A Novel Online Ensemble for Multi-Label Stream Classification

### Dependencies

* MOA
* MEKA
* Jama - For matrix operations such as solving LSQ.
* sizeofag - for measuring memory consumption of each model

### Datasets

Can be downloaded from http://meka.sourceforge.net/#datasets.

### Running Models

Assuming you generated .jar files that run the main method for the files RunClassifiers.java and RunGOOWEs.java  

For the Scene dataset (Scene.arff, L=6, N=2407)

Running Baselines:

```shell
for j in `seq 1 7`;
do
	java -jar -Xmx16G -javaagent:sizeofag.jar RunClassifiers.jar Scene.arff 2407 6 ${j}
done
```

Running GOOWE-ML based models:

```shell
for j in `seq 1 4`;
do
        java -jar -Xmx16G -javaagent:sizeofag.jar RunGOOWEs.jar Scene.arff 2407 6 ${j}
done
```
