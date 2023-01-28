using System;
using System.Threading.Tasks;
using NN.CPU_Single;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace NN.CPU_Multi
{
    //TODO: inheriting from the dense layer class is not viable since we have to convert every matrix to arrays and vice versa
    public class CpuMultiNnLayer : DenseLayer, IDisposable
    {
        private NativeArray<float> _inputs;
        private NativeArray<float> _weights;
        private NativeArray<float> _biases;
        private NativeArray<float> _output;

        private MatrixDotProductJob _matrixDotProductJob;

        private bool _init;

        public CpuMultiNnLayer(int nInputs, int nNeurons, float weightRegularizerL2 = 0, float biasRegularizerL2 = 0) :
            base(nInputs, nNeurons, weightRegularizerL2, biasRegularizerL2)
        {
            _weights = new NativeArray<float>(Weights.Length, Allocator.Persistent);
            Parallel.For(0, Weights.Length, i =>
            {
                var y = i % Weights.GetLength(1);
                _weights[i] = Weights[(i + Weights.GetLength(1) - y) / Weights.GetLength(1) - 1, y];
            });

            _biases = new NativeArray<float>(Biases.Length, Allocator.Persistent);
            Parallel.For(0, Biases.Length, i => { _biases[i] = Biases[0, i]; });

            _matrixDotProductJob = new MatrixDotProductJob()
            {
                Weights = _weights,
                Biases = _biases,
                InputRowSize = Weights.GetLength(0),
                WeightsRowSize = Weights.GetLength(1)
            };
        }

        public override void Forward(float[,] input)
        {
            Inputs = input;

            if (!_init)
            {
                _init = true;
                _inputs = new NativeArray<float>(Inputs.Length, Allocator.Persistent);
                _output = new NativeArray<float>(Inputs.GetLength(0) * Weights.GetLength(1), Allocator.Persistent);
                Output = new float[Inputs.GetLength(0), Weights.GetLength(1)];

                _matrixDotProductJob.Inputs = _inputs;
                _matrixDotProductJob.Output = _output;
            }

            // for (int i = 0; i < _inputs.Length; i++)
            // {
            //     var y = i % Inputs.GetLength(1);
            //     _inputs[i] = Inputs[(i + Inputs.GetLength(1) - y) / Inputs.GetLength(1) - 1, y];
            // }

            float[] tt = _inputs.ToArray();
            
            var matrixDotProductJobHandle = _matrixDotProductJob.Schedule(_output.Length, 64);
            matrixDotProductJobHandle.Complete();
           
            // for (int i = 0; i < _output.Length; i++)
            // {
            //     var y = i % Output.GetLength(1);
            //     Output[(i + Output.GetLength(1) - y) / Output.GetLength(1) - 1, y] = _output[i];
            // }
        }

        [BurstCompile]
        private struct MatrixDotProductJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> Inputs;
            [ReadOnly] public NativeArray<float> Weights;
            [ReadOnly] public NativeArray<float> Biases;
            [WriteOnly] public NativeArray<float> Output;

            public int InputRowSize;
            public int WeightsRowSize;

            public void Execute(int index)
            {
                var y = index % WeightsRowSize;
                var x = (index + WeightsRowSize - y) / WeightsRowSize - 1;
                var result = 0f;
                for (int i = 0; i < InputRowSize; i++)
                {
                    result += Inputs[x * InputRowSize + i] * Weights[i * WeightsRowSize + y];
                }

                result += Biases[y];
                Output[index] = result < 0 ? 0 : result;
            }
        }

        public void Dispose()
        {
            _inputs.Dispose();
            _weights.Dispose();
            _biases.Dispose();
            _output.Dispose();
        }
    }
}