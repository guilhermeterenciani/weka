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
import java.io.FileReader;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AssetManager assetManager = getResources().getAssets();
        InputStream inputStream;

        //String trainPath = R.id.
//        int trainPath = R.raw.train;
//        int testPath = R.raw.train;
        // tem que incluir os arquivos aqui
        System.out.println("Estou aqui");
        try {

            Instances mnistTrainWeka = new Instances(new BufferedReader(new FileReader( new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),"mexidao_train.arff"))));
            mnistTrainWeka.setClassIndex(mnistTrainWeka.numAttributes() - 1);
            Instances mnistTestWeka = new Instances(new BufferedReader(new FileReader(new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),"mexidao_test.arff"))));
            mnistTestWeka.setClassIndex(mnistTestWeka.numAttributes() - 1);
            Log.i("MODEL_TIME","carreguei os arquivos");


//            Primeira mente voce deve criar uma passa chamada "assets" dentro da pasta main ao lado de "res"...
//
//            depois, coloque os arquivos txt dentro desta pasta "assets" e use o codigo abaixo:
//
//
//            LinkedList<String> linhas=null;
//
//            try {
//                inputStream = assetManager.open("nome-do-arquivo.txt");
//                InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
//                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
//


            //BallTree ballTree = new BallTree();

            System.out.println("1-NN (Ball Tree)");
            IBk wekaNN = new IBk(1);
            BallTree ballTree = new BallTree();
            try {
                ballTree.setDistanceFunction(new EuclideanDistance());
            } catch (Exception e) {
                e.printStackTrace();
            }
            wekaNN.setNearestNeighbourSearchAlgorithm(ballTree);
            wekaNN.setDistanceWeighting(new SelectedTag(wekaNN.WEIGHT_INVERSE,wekaNN.TAGS_WEIGHTING));
            wekaNN.setCrossValidate(false);
            Toast.makeText(this, "Entrando no eval", Toast.LENGTH_SHORT).show();
            evalModel(wekaNN, mnistTrainWeka, mnistTestWeka);
            Toast.makeText(this, "Saindo do eval", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
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
        eval.evaluateModel(wekaModel, test);
        end = System.currentTimeMillis();
        Log.i("MODEL_TIME","Evaluation took " + (end - start) / 1000.0 + " seconds with an error rate " + eval.errorRate());
        System.out.println("\tEvaluation took " + (end - start) / 1000.0 + " seconds with an error rate " + eval.errorRate());
    }
}