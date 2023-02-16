using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace Graphs
{
    public class TestGraphs : MonoBehaviour
    {
        void Start()
        {
            //var (x, y) = GenerateLinearSample();
            var (x, y) = GenerateSinSample();
            //var (x, y) = GenerateClassSample();

            foreach (var num in x)
            {
                //print(num);
            }
            var graph = FindObjectOfType<WindowGraph>();
            if (graph)
                graph.SetGraph(x, y, GraphType.LineGraph, "Sin Wave", "Frequency", "Amplitude");
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

        private Tuple<List<float>, List<float>> GenerateSinSample()
        {
            var xValues = new List<float>();
            var yValues = new List<float>();
            float timeAdditive = 0;

            while (timeAdditive <= 1.0f)
            {
                xValues.Add(timeAdditive);
                yValues.Add(Mathf.Sin(Mathf.Deg2Rad * (58 * Mathf.PI * 20 * timeAdditive)));
                timeAdditive += Time.deltaTime / 6;
            }

            return new Tuple<List<float>, List<float>>(xValues, yValues);
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
    }
}