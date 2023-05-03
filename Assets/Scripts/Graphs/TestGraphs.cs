using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;
using Random = UnityEngine.Random;

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

            //const int epochs = 1000;

            // model.Train(epochs, x, y, 999);
            //var tt = model.Predict(x);
            model.Dispose();

            // var tt = RunCPUSingle(x, y, epochs);
            //  foreach (var pred in tt)
            //      preds.Add(pred);

            // foreach (var pred in y)
            // {
            //     preds.Add(pred);
            //     if (pred >= 1.0f || pred <= -1.0f)
            //     {
            //         print(pred);
            //     }
            // }
            // distributional 

            var dist = new List<float> { 0f, 0f, 0f, 0f, 0f, 0f };
            var gamma = 0.99f;
            var reward = 3f;
            var support = new float[] { -10f, -5f, 0f, 5f, 10f };
            var delta = (10f + 10f) / 4;
            preds = new List<float> { 0.15f, 0.1f, 0.5f, 0.1f, 0.15f, 0f };
            for (int i = 0; i < 5; i++)
            {
                var v = reward + support[i] * gamma;
                var tz = Mathf.Clamp(v, -10f, 10f);
                var b = (tz + 10f) / delta;
                var l = (int)b;
                var u = Mathf.CeilToInt(b);
            
                if (l == u)
                {
                    dist[l] += preds[i];
                }
                else
                {
                    dist[l] += preds[i] * (u - b);
                    dist[u] += preds[i] * (b - l);
                }
            }
            // var b =  Mathf.RoundToInt((reward + 10f) / delta);
            // b = Mathf.Clamp(b, 0, 4);
            // var m = new float[6];
            // preds.CopyTo(m);
            // var j = 1;
            // for (int i = b; i > 0; i--)
            // {
            //     m[i] += Mathf.Pow(gamma, j) * m[i - 1];
            //     j++;
            // }
            // j = 1;
            // for (int i = b; i < 4; i++)
            // {
            //     m[i] += Mathf.Pow(gamma, j) * m[i + 1];
            //     j++;
            // }
            // var sum = 0f;
            // for (int i = 0; i < 5; i++)
            // {
            //     sum += m[i];
            // }
            // for (int i = 0; i < 5; i++)
            // {
            //     m[i] /= sum;
            // }
            
            for (int i = 0; i < 6; i++)
            {
                print("pred: " + preds[i] + ", dist: " + dist[i]);
                //print("pred: " + preds[i] + ", m: " + m[i]);
            }

            var graph = FindObjectOfType<WindowGraph>();
            if (graph)
                graph.SetGraph(new List<float>(), preds, GraphType.BarGraph, "Sin Wave", "Frequency", "Amplitude");
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