using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using NN.CPU_Single;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Random = UnityEngine.Random;

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

            var layers = new NetworkLayer[]
            {
                new NetworkLayer(x.GetLength(1), 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, 1, ActivationFunction.Linear, Instantiate(shader))
            };
            var model = new NetworkModel(layers, new MeanSquaredError(Instantiate(shader)));

            //const int epochs = 1000;

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            //model.Train(epochs, x, y, 99);

            stopwatch.Stop();
            model.Dispose();
            print("(GPU compute) Took: " + stopwatch.ElapsedMilliseconds + " ms");

            //TestBuffer();
            //LookUpArrayVsSwitch(1000);
            Test(100000);
        }

        private void Test(int iterations)
        {
            Random.InitState(42);
            var single = 51;
            var headNumber = 10;
            var x = 32;
            var y = 128;
            var k = single * headNumber;

            var input = new float[x, y];
            var weights = new float[y, k];
            var biases = new float[1, k];
            var outputs = new float[x, k];

            for (int i = 0; i < k; i++)
            {
                biases[0, i] = Random.value;
            }
            for (int i = 0; i < y; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    weights[i, j] = Random.value;
                }
            }
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    input[i, j] = Random.value;
                }
            }

            var testShader = Instantiate(test_shader);
            testShader.SetInt("input_column_size", x);
            testShader.SetInt("input_row_size", y);
            testShader.SetInt("weights_row_size", k);
            testShader.SetInt("head_number", headNumber);
            testShader.SetInt("distribution_lenght", single);
            var inputBuffer = new ComputeBuffer(input.Length, sizeof(float));
            var weightsBuffer = new ComputeBuffer(weights.Length, sizeof(float));
            var biasesBuffer = new ComputeBuffer(biases.Length, sizeof(float));
            var outputsBuffer = new ComputeBuffer(outputs.Length, sizeof(float));
            inputBuffer.SetData(input);
            weightsBuffer.SetData(weights);
            biasesBuffer.SetData(biases);
            outputsBuffer.SetData(outputs);
            int kernelIndex1 = testShader.FindKernel("forward_pass_linear");
            int kernelIndex2 = testShader.FindKernel("forward_pass_softmax");
            testShader.SetBuffer(kernelIndex1, "input", inputBuffer);
            testShader.SetBuffer(kernelIndex1, "weights", weightsBuffer);
            testShader.SetBuffer(kernelIndex1, "biases", biasesBuffer);
            testShader.SetBuffer(kernelIndex1, "output", outputsBuffer);  
            testShader.SetBuffer(kernelIndex2, "output", outputsBuffer);
            var groupX = Mathf.CeilToInt(x / 8f);
            var groupY1 = Mathf.CeilToInt(k / 8f);
            var groupY2 = Mathf.CeilToInt(headNumber / 8f);

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            testShader.Dispatch(kernelIndex1, groupX, groupY1, 1);
            testShader.Dispatch(kernelIndex2, groupX, groupY2, 1);
            outputsBuffer.GetData(outputs);
            stopwatch.Stop();
            print(stopwatch.ElapsedTicks);
            
            inputBuffer.Dispose();
            weightsBuffer.Dispose();
            biasesBuffer.Dispose();
            outputsBuffer.Dispose();

            var sum1 = 0f;
            var sum2 = 0f;
            for (int i = 0; i < single; i++)
            {
                sum1 += outputs[0, i];
                sum2 += outputs[0, k - single + i];
            }

            print("sum 1: " + sum1 + ", sum2: " + sum2);

            stopwatch.Restart();
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < headNumber; j++)
                {
                    float maxValue = float.MinValue;
                    sum1 = 0;
                    for (int l = 0; l < single; l++)
                    {
                        var result = 0f;
                        for (int m = 0; m < y; m++)
                        {
                            result += input[i, m] * weights[m, l];
                        }
                        result += biases[0, l];
                        outputs[i, j * single + l] = result;
                        if (maxValue > result) continue;
                        maxValue = result;
                    }
                    for (int l = 0; l < single; l++)
                    {
                        var indexY = j * single + l;
                        var res = Mathf.Exp( outputs[i, indexY] - maxValue);
                        outputs[i, indexY] = res;
                        sum1 += res;
                    }
                    for (int l = 0; l < single; l++)
                    {
                        outputs[i, j * single + l] /= sum1;
                    }
                }
            }
            stopwatch.Stop();
            print(stopwatch.ElapsedTicks);
            
            sum1 = 0f;
            sum2 = 0f;
            for (int i = 0; i < single; i++)
            {
                sum1 += outputs[0, i];
                sum2 += outputs[0, k - single + i];
            }

            print("sum 1: " + sum1 + ", sum2: " + sum2);
        }

        // Tests the max calculation of a matrix, compares times of: c# multithreading, single thread, jobs, compute shader
        // Parallel for is only better then single for loop at a size of > 20,000. The tests had two assignments being made inside the for loops
        // test ray casting multi and single
        private void CompareTimes(int iterations)
        {
            long parallel = 0;
            long single = 0;
            long job = 0;
            long compute = 0;

            int x = 64;
            int y = 10;

            var valueMatrix = new float[x, y];
            var valueMatrixJob = new NativeArray<float>(x * y, Allocator.Persistent);
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    var value = Random.value;
                    valueMatrix[i, j] = value;
                    valueMatrixJob[i * y + j] = value;
                }
            }

            var resultMatrix1 = new float[x];
            var resultMatrix2 = new float[x];
            var resultMatrix3 = new NativeArray<float>(x, Allocator.Persistent);
            var resultMatrix4 = new float[x];

            var testJob = new TestJobsFor()
            {
                ValueMatrix = valueMatrixJob,
                MaxValues = resultMatrix3,
                RowSize = y
            };

            var testShader = Instantiate(test_shader);
            var valueBuffer = new ComputeBuffer(x * y, sizeof(float));
            var maxBuffer = new ComputeBuffer(x, sizeof(float));

            int kernelIndexA = testShader.FindKernel("Max");
            testShader.SetBuffer(kernelIndexA, "buffer", valueBuffer);
            testShader.SetBuffer(kernelIndexA, "read_buffer", maxBuffer);
            testShader.SetInt("row_size", y);

            valueBuffer.SetData(valueMatrix);
            int groupX = Mathf.CeilToInt(x / (float)32);

            for (int k = 0; k < iterations; k++)
            {
                var stopwatch = new Stopwatch();
                stopwatch.Start();

                Parallel.For(0, x, i =>
                {
                    var max = float.MinValue;
                    for (int j = 0; j < y; j++)
                    {
                        var currentVal = valueMatrix[i, j];
                        if (max > currentVal) continue;

                        max = currentVal;
                    }

                    resultMatrix1[i] = max;
                });

                stopwatch.Stop();
                parallel += stopwatch.ElapsedTicks;

                stopwatch.Restart();

                for (int i = 0; i < x; i++)
                {
                    var max = float.MinValue;
                    for (int j = 0; j < y; j++)
                    {
                        var currentVal = valueMatrix[i, j];
                        if (max > currentVal) continue;

                        max = currentVal;
                    }

                    resultMatrix2[i] = max;
                }

                stopwatch.Stop();
                single += stopwatch.ElapsedTicks;

                stopwatch.Restart();

                var matrixDotProductJobHandle = testJob.Schedule(resultMatrix3.Length, 100);
                matrixDotProductJobHandle.Complete();

                stopwatch.Stop();
                job += stopwatch.ElapsedTicks;

                stopwatch.Restart();

                testShader.Dispatch(kernelIndexA, groupX, 1, 1);
                maxBuffer.GetData(resultMatrix4);

                stopwatch.Stop();
                compute += stopwatch.ElapsedTicks;
            }

            print("Parallel: " + parallel);
            print("Single: " + single);
            print("Job: " + job);
            print("Compute: " + compute);

            for (int i = 0; i < x; i++)
            {
                if (Math.Abs(resultMatrix1[i] - resultMatrix4[i]) <= 0) continue;

                print("Not the same: " + i + ", " + resultMatrix1[i] + ", " + resultMatrix4[i]);
                break;
            }

            valueMatrixJob.Dispose();
            resultMatrix3.Dispose();
            valueBuffer.Dispose();
            maxBuffer.Dispose();
        }


        [BurstCompile]
        private struct TestJobsFor : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> ValueMatrix;
            [WriteOnly] public NativeArray<float> MaxValues;

            public int RowSize;

            public void Execute(int index)
            {
                var result = float.MinValue;
                for (int i = 0; i < RowSize; i++)
                {
                    var currentVal = ValueMatrix[index * RowSize + i];
                    if (result > currentVal) continue;

                    result = currentVal;
                }

                MaxValues[index] = result;
            }
        }

        private void TestBuffer()
        {
            var testShader = Instantiate(test_shader);
            var buffer = new ComputeBuffer(4, sizeof(float));
            var readBuffer = new ComputeBuffer(4, sizeof(float));
            var tt = new float[4];
            var tt_1 = new float[4];

            testShader.SetInt("row_size", 4);

            int kernelIndexA = testShader.FindKernel("KernelA");
            testShader.SetBuffer(kernelIndexA, "buffer", buffer);
            testShader.SetBuffer(kernelIndexA, "read_buffer", readBuffer);
            int kernelIndexB = testShader.FindKernel("KernelB");
            testShader.SetBuffer(kernelIndexB, "buffer", buffer);
            testShader.SetBuffer(kernelIndexB, "read_buffer", readBuffer);

            buffer.SetData(tt);
            readBuffer.SetData(tt_1);

            testShader.Dispatch(kernelIndexA, 128, 100, 1);
            testShader.Dispatch(kernelIndexB, 128, 1, 1);

            readBuffer.GetData(tt_1);
            foreach (var t in tt_1)
            {
                print(t);
            }

            print("Exp: " + Mathf.Exp(-0.54689f));
            buffer.GetData(tt);
            //print("CPU: " + Mathf.Sqrt(0.0000000000445868f));
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