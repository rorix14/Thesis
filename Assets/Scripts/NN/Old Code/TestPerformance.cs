using System;
using System.Collections.Generic;
using System.Diagnostics;
using NN.CPU_Multi;
using NN.CPU_Single;
using NN.GPU_Compute;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NN
{
    public class TestPerformance : MonoBehaviour
    {
        [SerializeField] private ComputeShader shader;

        private float[,] mat1 = { { -1, 1, 1 }, { 2, -2, 2 }, { 3, 3, -3 } };
        private float[,] mat2 = { { 3, 2, 4 }, { 5, 2, 1 } };
        private float[,] mat3 = { { -1, 1 }, { 2, -2 }, { -3, 3 } };
        private float[,] mat4 = { { 1 }, { 2 }, { 3 }, { 4 } };
        private float[,] mat5 = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };
        private float[,] mat6 = { { 1, 2, 3 } };
        private float[,] mat7 = { { -1, 4, 1, 0 }, { 2, -2, 2, -1 } };

        void Start()
        {
            var (x, y) = GenerateSinSample();

            var test = new float[x.GetLength(0), x.GetLength(1)];
            for (int i = 0; i < test.GetLength(0); i++)
            {
                for (int j = 0; j < test.GetLength(1); j++)
                {
                    test[i, j] = x[i, j];
                }
            }

            var layers = new BaseLayer[]
            {
                new DenseLayer(test.GetLength(1), 128),
                new ActivationTanh(),
                new DenseLayer(128, 128),
                new ActivationTanh(),
                //new DenseLayer(64, 64),
                //new ActivationReLu(),
                new DenseLayer(128, 1),
                new ActivationLinear()
            };

            const int epochs = 1000;
            //print("(CPU single) " + x.GetLength(0));
            // long singleCPU = 0;
            // long muliCPU = 0;
            // long GPU = 0;

            // for (int i = 0; i < 5000; i++)
            // {
            //     singleCPU += RunCPUSingle(x, epochs, layers);
            //     muliCPU += RunCPUMulti(x, epochs, layers);
            //     GPU += RunGPUCompute(x, epochs, layers);
            //
            //     if (i % 100 == 0)
            //         print("iter: " + i);
            // }

            var runtime = RunCPUSingleBackwards(x, y, epochs, layers);
            //print("(cpu single) took: " + runtime + " ms");

            //print("GPU single took: " + RunCPUSingle(test, epochs, layers) + " ms");

            //runtime = RunCPUMulti(x, epochs, layers);
            //print("CPU multi took: " + RunCPUMulti(test, epochs, layers) + " ms");

            //runtime = RunGPUCompute(x, epochs, layers);
            //print("GPU compute took: " + RunGPUCompute(test, epochs, layers) + " ms");
        }

        private long RunCPUSingleBackwards(float[,] dataX, float[,] dataY, int epochs, params BaseLayer[] layers)
        {
            var mse = new LossFunctionMeanSquaredError();
            var Adam = new OptimizerAdam(0.005f, 1e-3f);

            var accuracyPrecision = NnMath.StandardDivination(dataY) / 250;

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < epochs; i++)
            {
                layers[0].Forward(dataX);
                for (int j = 1; j < layers.Length; j++)
                {
                    layers[j].Forward(layers[j - 1].Output);
                }

                if (i % 99 == 0)
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

                mse.Calculate(layers[layers.Length - 1].Output, dataY);

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

            //print("(cpu) loss: " + mse.Calculate(layers[layers.Length - 1].Output, dataY));
            stopwatch.Stop();
            return stopwatch.ElapsedMilliseconds;
        }

        private long RunCPUSingle(float[,] data, int epochs, params BaseLayer[] layers)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < epochs; i++)
            {
                layers[0].Forward(data);
                for (int j = 1; j < layers.Length; j++)
                {
                    layers[j].Forward(layers[j - 1].Output);
                }
            }

            stopwatch.Stop();
            // for (int i = 0; i < 10; i++)
            // {
            //     print("CPU single output: " + layers[layers.Length - 1].Output[i,0]);
            // }
            float result = 0;
            foreach (var value in layers[layers.Length - 1].Output)
            {
                result += value;
            }

            //print("(CPU single) Final value sum: " + result);
            return stopwatch.ElapsedMilliseconds;
        }

        private long RunCPUMulti(float[,] data, int epochs, params BaseLayer[] layers)
        {
            var cpuMultiLayers = new List<CpuMultiNnLayer>(layers.Length / 2);
            foreach (var layer in layers)
            {
                if (layer.GetType() != typeof(DenseLayer)) continue;

                var denseLayer = (DenseLayer)layer;
                var cpuMultiLayer = new CpuMultiNnLayer(denseLayer.Weights.GetLength(0),
                    denseLayer.Weights.GetLength(1));
                cpuMultiLayers.Add(cpuMultiLayer);
            }

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < epochs; i++)
            {
                cpuMultiLayers[0].Forward(data);
                for (int j = 1; j < cpuMultiLayers.Count; j++)
                {
                    cpuMultiLayers[j].Forward(cpuMultiLayers[j - 1].Output);
                }
            }

            stopwatch.Stop();
            foreach (var gpuLayer in cpuMultiLayers)
            {
                gpuLayer.Dispose();
            }

            float result = 0;
            foreach (var value in cpuMultiLayers[cpuMultiLayers.Count - 1].Output)
            {
                result += value;
            }

            //print("Final value sum: " + result);
            return stopwatch.ElapsedMilliseconds;
        }

        private long RunGPUCompute(float[,] data, int epochs, params BaseLayer[] layers)
        {
            var gpuLayers = new List<GpuNnLayer>(layers.Length / 2);
            foreach (var layer in layers)
            {
                if (layer.GetType() != typeof(DenseLayer)) continue;

                var denseLayer = (DenseLayer)layer;
                var gpuLayer = new GpuNnLayer(Instantiate(shader), denseLayer.Weights.GetLength(0),
                    denseLayer.Weights.GetLength(1));
                gpuLayers.Add(gpuLayer);
            }

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < epochs; i++)
            {
                gpuLayers[0].Forward(data);
                for (int j = 1; j < gpuLayers.Count; j++)
                {
                    gpuLayers[j].Forward(gpuLayers[j - 1].Output);
                }
            }

            stopwatch.Stop();
            foreach (var gpuLayer in gpuLayers)
            {
                gpuLayer.Dispose();
            }

            // for (int i = 0; i < 10; i++)
            // {
            //     print("GPU compute output: " + gpuLayers[gpuLayers.Count - 1].Output[i,0]);
            // }
            //print("Weight: " + gpuLayers[0].Weights[0, 0] + ", bias: " + gpuLayers[0].Biases[0, 0]);
            float result = 0;
            foreach (var value in gpuLayers[gpuLayers.Count - 1].Output)
            {
                result += value;
            }

            //print("Final value sum: " + result);
            return stopwatch.ElapsedMilliseconds;
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

        private Tuple<float[,], float[,]> GenerateLinearSample()
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

// var comp1 = Instantiate(shader);
//
// var buffer1 = new ComputeBuffer(1, sizeof(float));
// var buffer2 = new ComputeBuffer(1, sizeof(float));
//             
// buffer2.SetData(new float[]{5});
// comp1.SetBuffer(comp1.FindKernel("test"), "output", buffer1);
// comp1.SetBuffer(comp1.FindKernel("test"), "input", buffer2);
// comp1.SetInt(Shader.PropertyToID("input_column_size"), 2);
// comp1.SetInt("input_row_size", 1);
//
// buffer2.SetData(new float[]{8});
// comp1.Dispatch(comp1.FindKernel("test"), 1, 1, 1);
// var result1 = new float[1];
// buffer1.GetData(result1);
// buffer1.Release();
//             
// print(result1[0]);