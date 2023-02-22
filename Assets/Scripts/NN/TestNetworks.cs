using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

namespace NN
{
    public class TestNetworks : MonoBehaviour
    {
        [SerializeField] private ComputeShader shader;
        [SerializeField] private ComputeShader test_shader;

        private void Start()
        {
            var (x, y) = GenerateSinSample();
            //print(x.GetLength(0));

            const int epochs = 1;

            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 64, Instantiate(shader)),
                new NetworkLayer(64, 64, Instantiate(shader)),
                new NetworkLayer(64, 4, Instantiate(shader))
            };

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            
            for (int i = 0; i < epochs; i++)
            {
                layers[0].Forward(x);
                for (int j = 1; j < layers.Length; j++)
                {
                    layers[j].Forward(layers[j - 1].Output);
                }
            }
            
            for (int i = 0; i < epochs; i++)
            {
                layers[layers.Length - 1].Backward(layers[layers.Length - 1].Output);
                for (int j = layers.Length - 2; j >= 0; j--)
                {
                    layers[j].Backward(layers[j + 1].DInputs);
                }
            }

            stopwatch.Stop();

            float result = 0;
            // foreach (var value in layers[layers.Length - 1].Output)
            //     result += value;
            
            foreach (var value in layers[0].DInputs)
                result += value;

            //print("(GPU compute) Final value sum: " + result);
            //print("(GPU compute) Took: " + stopwatch.ElapsedMilliseconds + " ms");

            foreach (var layer in layers)
            {
                layer.Dispose();
            }

            //TestBuffer();
        }

        private void TestBuffer()
        {
            var testShader = Instantiate(test_shader);
            var buffer = new ComputeBuffer(4, sizeof(int));
            var readBuffer = new ComputeBuffer(4, sizeof(int));
            var tt = new int[4];
            var tt_1 = new int[4];
            
            int kernelIndexA = testShader.FindKernel("KernelA");
            testShader.SetBuffer(kernelIndexA, "buffer", buffer);
            int kernelIndexB = testShader.FindKernel("KernelB");
            testShader.SetBuffer(kernelIndexB, "buffer", buffer);
            testShader.SetBuffer(kernelIndexB, "read_buffer", readBuffer);
            
            buffer.SetData(tt);
            readBuffer.SetData(tt_1);
            
            testShader.Dispatch(kernelIndexA, 128, 1, 1);
            testShader.Dispatch(kernelIndexB, 128, 1, 1);

            readBuffer.GetData(tt_1);
            foreach (var t in tt_1)
            {
                print(t);
            }

            buffer.GetData(tt);
            print(tt[0]);
            
            buffer.Dispose();
            readBuffer.Dispose();
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
    }
}