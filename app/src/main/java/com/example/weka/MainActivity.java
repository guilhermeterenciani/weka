package com.example.weka;

import androidx.appcompat.app.AppCompatActivity;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;

import weka.core.neighboursearch.BallTree;
import weka.core.EuclideanDistance;



import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AssetManager assetManager = getResources().getAssets();
        InputStream inputStream;

        System.out.println("Estou aqui");
        try {

            Instances mnistTrainWeka = new Instances(new BufferedReader
                    (new FileReader( new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),"train_60.arff"))));
            mnistTrainWeka.setClassIndex(mnistTrainWeka.numAttributes() - 1);
            Instances mnistTestWeka = new Instances(new BufferedReader
                    (new FileReader(new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),"mexidao_test.arff"))));
            mnistTestWeka.setClassIndex(mnistTestWeka.numAttributes() - 1);

            Log.i("MODEL_TIME","carreguei os arquivos");

            System.out.println("Vou carregar 1000 instâncias");
            Integer ninstance = mnistTrainWeka.numInstances();
            Integer quantinstance = 10;

//            for(int i = 0; i< ninstance; i+=quantinstance) {
//                Instances newData = new Instances(mnistTrainWeka, i+quantinstance);
//                for (int j = 0; j < ninstance && j < i + quantinstance; j++) {
//                    newData.add(mnistTestWeka.instance(j));
//                }
            System.out.println("Carreguei " + mnistTrainWeka.numInstances() + " instâncias");
            System.out.println("1-NN (Ball Tree)");
            IBk wekaNN = new IBk(1);
            BallTree ballTree = new BallTree();
            ballTree.setDistanceFunction(new EuclideanDistance());
            wekaNN.setNearestNeighbourSearchAlgorithm(ballTree);
            wekaNN.setDistanceWeighting(new SelectedTag(wekaNN.WEIGHT_INVERSE, wekaNN.TAGS_WEIGHTING));
            wekaNN.setCrossValidate(false);
            Toast.makeText(this, "Entrando no eval", Toast.LENGTH_SHORT).show();
            evalModel(wekaNN, mnistTrainWeka, mnistTestWeka);
            Toast.makeText(this, "Saindo do eval", Toast.LENGTH_SHORT).show();

//            }

        }catch(FileNotFoundException e){
            System.out.println("Deu certo carregar o arquivo não meu filho :("+ e);
        }
        catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }
    private static void evalModel(Classifier wekaModel, Instances train, Instances test) throws Exception {
        long start;
        long end;
        start = System.currentTimeMillis();
        wekaModel.buildClassifier(train);
        end = System.currentTimeMillis();
        System.out.println("\tTraining took: " + (end - start) / 1000.0);
        Log.i("MODEL_TIME","\tTraining took: " + (end - start) / 1000.0);
        Evaluation eval = new Evaluation(train);

        start = System.currentTimeMillis();
        wekaModel.classifyInstance(test.instance(0));
        wekaModel.classifyInstance(test.instance(1));
        wekaModel.classifyInstance(test.instance(2));
        wekaModel.classifyInstance(test.instance(3));
        wekaModel.classifyInstance(test.instance(4));
        wekaModel.classifyInstance(test.instance(5));
        wekaModel.classifyInstance(test.instance(6));
        wekaModel.classifyInstance(test.instance(7));
        wekaModel.classifyInstance(test.instance(8));
        wekaModel.classifyInstance(test.instance(9));
        wekaModel.classifyInstance(test.instance(10));
        wekaModel.classifyInstance(test.instance(11));
        //3 = 53
        //6 = 99
        //9 = 147
        //12= 194
        end = System.currentTimeMillis();
        System.out.println("Time 3 instances: "+ (end-start));


        start = System.currentTimeMillis();
        eval.evaluateModel(wekaModel, test);
        end = System.currentTimeMillis();
        System.out.println("Quantidade de corretos:"+eval.correct());

        Log.i("MODEL_TIME","Evaluation took " + (end - start) + " seconds with an error rate " + eval.errorRate());
        System.out.println("\tEvaluation took " + (end - start) + " seconds with an error rate " + eval.errorRate());
    }
}