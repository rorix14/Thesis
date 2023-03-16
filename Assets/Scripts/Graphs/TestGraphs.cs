using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;
using Random = Unity.Mathematics.Random;

namespace Graphs
{
    public class TestGraphs : MonoBehaviour
    {
        [SerializeField] private ComputeShader shader;

        void Start()
        {
            //var (x, y) = GenerateLinearSample();
            var (x, y) = GenerateSinSample();
            //var (x, y) = GenerateClassSample();

            var preds = new List<float>(y.Length);
            
            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 64, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(64, 64, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(64, 1, ActivationFunction.Linear, Instantiate(shader))
            };
            
            var model = new NetworkModel(layers, new MeanSquaredError(Instantiate(shader)));

            const int epochs = 1000;

            // model.Train(epochs, x, y, 999);
            //var tt = model.Predict(x);
            model.Dispose();

            var tt = RunCPUSingle(x, y, epochs);
             foreach (var pred in tt)
                 preds.Add(pred);
            
            // foreach (var pred in y)
            // {
            //     preds.Add(pred);
            //     if (pred >= 1.0f || pred <= -1.0f)
            //     {
            //         print(pred);
            //     }
            // }

            
            var graph = FindObjectOfType<WindowGraph>();
            if (graph)
                graph.SetGraph(new List<float>(), preds, GraphType.LineGraph, "Sin Wave", "Frequency", "Amplitude");
        }

        private IEnumerator RunEverySecond(WindowGraph graph)
        {
            //int i = 0;
            while (true)
            {
                yield return new WaitForSeconds(0.1f);
                //graph.UpdateValue(_graphVisual, 0, i++);
                //graph.AddValue(_graphVisual, Random.Range(0, 100));
            }
        }

        private Tuple<List<float>, List<float>> GenerateClassSample()
        {
            const int size = 15;
            var xValues = new List<float>(size);
            var yValues = new List<float>(size);

            for (int i = 0; i < size; i++)
            {
                xValues.Add(i * 5);
                yValues.Add(UnityEngine.Random.Range(0, 100));
            }

            return new Tuple<List<float>, List<float>>(xValues, yValues);
        }

        private Tuple<float[,], float[,]> GenerateSinSample()
        {
            var xValues = new List<float>();
            var yValues = new List<float>();
            float timeAdditive = 0;

            while (timeAdditive <= 1.0f)
            {
                xValues.Add(timeAdditive);
                yValues.Add(Mathf.Sin(Mathf.Deg2Rad * (58 * Mathf.PI * 2 * timeAdditive)));
                timeAdditive += Time.deltaTime / 6;
            }

            var x = new float[xValues.Count, 1];
            var y = new float[yValues.Count, 1];
            for (int i = 0; i < xValues.Count; i++)
            {
                x[i, 0] = xValues[i];
                y[i, 0] = yValues[i];
            }

            return new Tuple<float[,], float[,]>(x, y);
        }

        private Tuple<List<float>, List<float>> GenerateLinearSample()
        {
            var xValues = new List<float>();
            var yValues = new List<float>();
            float timeAdditive = 0;

            while (timeAdditive <= 10.0f)
            {
                xValues.Add(timeAdditive);
                yValues.Add(2 * timeAdditive + 5);
                timeAdditive += Time.deltaTime / 2;
            }

            return new Tuple<List<float>, List<float>>(xValues, yValues);
        }

        private float[,] RunCPUSingle(float[,] dataX, float[,] dataY, int epochs)
        {
            var layers = new BaseLayer[]
            {
                new DenseLayer(dataX.GetLength(1), 128),
                new ActivationTanh(),
                new DenseLayer(128, 128),
                new ActivationTanh(),
                new DenseLayer(128, 1),
                new ActivationLinear()
            };
            
            var mse = new LossFunctionMeanSquaredError();
            var Adam = new OptimizerAdam(0.005f, 1e-3f);

            var accuracyPrecision = NnMath.StandardDivination(dataY) / 250;

            for (int i = 0; i < epochs; i++)
            {
                layers[0].Forward(dataX);
                for (int j = 1; j < layers.Length; j++)
                {
                    layers[j].Forward(layers[j - 1].Output);
                }

                if (i % 999 == 0)
                {
                    var accuracy = 0.0f;
                    for (int j = 0; j < dataY.GetLength(0); j++)
                    {
                        for (int k = 0; k < dataY.GetLength(1); k++)
                        {
                            accuracy += Mathf.Abs(layers[layers.Length - 1].Output[j, k] - dataY[j, k]) <
                                        accuracyPrecision
                                ? 1
                                : 0;
                        }
                    }

                    print("(cpu) At " + i + ", loss: " + mse.Calculate(layers[layers.Length - 1].Output, dataY) +
                          ", accuracy: " + accuracy / dataY.GetLength(0));
                }
                
                mse.Backward(layers[layers.Length - 1].Output, dataY);
                layers[layers.Length - 1].Backward(mse.DInputs);
                for (int j = layers.Length - 2; j >= 0; j--)
                {
                    layers[j].Backward(layers[j + 1].DInputs);
                }

                Adam.PreUpdateParams();
                Adam.UpdateParams((DenseLayer)layers[4]);
                Adam.UpdateParams((DenseLayer)layers[2]);
                Adam.UpdateParams((DenseLayer)layers[0]);
                Adam.PostUpdateParams();
            }

            return layers[layers.Length - 1].Output;
        }
    }
}