using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using NN;
using UnityEngine;
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

            const int epochs = 1000;

            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 64, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(64, 64, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(64, 1, ActivationFunction.Linear, Instantiate(shader))
            };

            var model = new NetworkModel(layers, new MeanSquaredError(Instantiate(shader)));

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            model.Train(epochs, x, y, 999);

            var preds = new List<float>(y.Length);
            foreach (var pred in model.Predict(x))
                preds.Add(pred);
            
            // foreach (var pred in y)
            //     preds.Add(pred);

            model.Dispose();
            
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

        // private Tuple<List<float>, List<float>> GenerateSinSample()
        // {
        //     var xValues = new List<float>();
        //     var yValues = new List<float>();
        //     float timeAdditive = 0;
        //
        //     while (timeAdditive <= 1.0f)
        //     {
        //         xValues.Add(timeAdditive);
        //         yValues.Add(Mathf.Sin(Mathf.Deg2Rad * (58 * Mathf.PI * 20 * timeAdditive)));
        //         timeAdditive += Time.deltaTime / 6;
        //     }
        //
        //     return new Tuple<List<float>, List<float>>(xValues, yValues);
        // }

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
    }
}