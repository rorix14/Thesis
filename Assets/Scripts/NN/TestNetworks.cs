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

            const int epochs = 1000;
            
            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 64, ActivationFunction.ReLu,Instantiate(shader)),
                new NetworkLayer(64, 64, ActivationFunction.ReLu,Instantiate(shader)),
                new NetworkLayer(64, 1, ActivationFunction.Linear,Instantiate(shader))
            };

            var model = new NetworkModel(layers, new MeanSquaredError(Instantiate(shader)));

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            
           model.Train(epochs, x, y, 999);

            // for (int i = 0; i < epochs; i++)
            // {
            //     layers[0].Forward(x);
            //     for (int j = 1; j < layers.Length; j++)
            //     {
            //         layers[j].Forward(layers[j - 1].Output);
            //     }
            
                // print("(GPU) loss: " + layers[layers.Length - 1].ForwardLoss(y));
                // //layers[layers.Length - 1].ForwardLoss(y);
                // layers[layers.Length - 1].BackwardLoss();
                //
                // layers[layers.Length - 1].Backward(layers[layers.Length - 1].DInputsLoss);
                // for (int j = layers.Length - 2; j >= 0; j--)
                // {
                //     layers[j].Backward(layers[j + 1].DInputs);
                // }
            //}

            //print("(GPU) loss: " + layers[layers.Length - 1].ForwardLoss(y));

            // for (int i = 0; i < epochs; i++)
            // {
            //     layers[layers.Length - 1].BackwardLoss();
            //     layers[layers.Length - 1].Backward(layers[layers.Length - 1].DInputsLoss);
            //     for (int j = layers.Length - 2; j >= 0; j--)
            //     {
            //         layers[j].Backward(layers[j + 1].DInputs);
            //     }
            // }

            stopwatch.Stop();
            model.Dispose();
            
            //float result = 0;
            // foreach (var value in layers[layers.Length - 1].Output)
            //     result += value;

            // foreach (var value in layers[0].DInputs)
            //     result += value;

            //print("(GPU compute) Final value sum: " + result);
          // print("(GPU compute) Took: " + stopwatch.ElapsedMilliseconds + " ms");

            // foreach (var layer in layers)
            // {
            //     layer.Dispose();
            // }

            //TestBuffer();
        }

        private void TestBuffer()
        {
            var testShader = Instantiate(test_shader);
            var buffer = new ComputeBuffer(4, sizeof(float));
            var readBuffer = new ComputeBuffer(4, sizeof(int));
            var tt = new float[4];
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
                //print(t);
            }

            buffer.GetData(tt);
            //print("CPU: " + Mathf.Pow(-0.54442121213476f, 6));
            print(tt[0]);
            print(tt[1]);

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
                yValues.Add(Mathf.Sin(Mathf.Deg2Rad * (58 * Mathf.PI * 20 * timeAdditive)));
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