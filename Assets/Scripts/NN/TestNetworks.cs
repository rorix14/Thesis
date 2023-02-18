using System;
using System.Collections.Generic;
using UnityEngine;

namespace NN
{
    public class TestNetworks : MonoBehaviour
    {
        [SerializeField] private ComputeShader shader;
        [SerializeField] private ComputeShader test_shader;

        // public static ComputeBuffer Buffer;
        private void Start()
        {
            var (x, y) = GenerateSinSample();
            print(x.GetLength(0));

            const int epochs = 100;

            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 64, Instantiate(shader)),
                new NetworkLayer(64, 64, Instantiate(shader)),
                new NetworkLayer(64, 1, Instantiate(shader))
            };

            for (int i = 0; i < epochs; i++)
            {
                layers[0].Forward(x);
                for (int j = 1; j < layers.Length; j++)
                {
                    layers[j].Forward(layers[j - 1].Output);
                }
            }

            float result = 0;
            foreach (var value in layers[layers.Length - 1].Output)
            {
                result += value;
            }

            print("Final value sum: " + result);

            foreach (var layer in layers)
            {
                layer.Dispose();
            }

            TestBuffer();
        }

        private void TestBuffer()
        {
            var testShader = Instantiate(test_shader);
            var Buffer = new ComputeBuffer(4, sizeof(float));
            var readBuffer = new ComputeBuffer(4, sizeof(float));
            float[] tt = new float[4];
            Buffer.SetData(tt);

            int kernelIndexA = testShader.FindKernel("KernelA");
            testShader.SetBuffer(kernelIndexA, "buffer", Buffer);
            int kernelIndexB = testShader.FindKernel("KernelB");
            testShader.SetBuffer(kernelIndexB, "buffer", Buffer);
            testShader.SetBuffer(kernelIndexB, "read_buffer", readBuffer);
            
            testShader.Dispatch(kernelIndexA, 128, 1, 1);
            testShader.Dispatch(kernelIndexB, 128, 1, 1);

            readBuffer.GetData(tt);
            foreach (var t in tt)
            {
                print(t);
            }

            Buffer.Dispose();
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