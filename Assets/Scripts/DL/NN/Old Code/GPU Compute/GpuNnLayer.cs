using System;
using NN.CPU_Single;
using UnityEngine;

namespace NN.GPU_Compute
{
    // we mainly inherit from this class so we have the weights and biases initialized,
    // if we want this initialization to be done on the GPU instead, we can just inherit from the BaseLayer class 
    public class GpuNnLayer : DenseLayer, IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _kernelHandle;
        private readonly uint _threadSizeX;
        private readonly uint _threadSizeY;

        private ComputeBuffer _inputBuffer;
        private ComputeBuffer _outputBuffer;
        private readonly ComputeBuffer _weightsBuffer;
        private readonly ComputeBuffer _biasesBuffer;

        private int _threadGroupX;
        private int _threadGroupY;

        private readonly int _tt;

        public GpuNnLayer(ComputeShader shader, int nInputs, int nNeurons, float weightRegularizerL2 = 0,
            float biasRegularizerL2 = 0) : base(nInputs, nNeurons, weightRegularizerL2, biasRegularizerL2)
        {
            _shader = shader;
            _kernelHandle = _shader.FindKernel("matrix_dot_product");
            _shader.GetKernelThreadGroupSizes(_kernelHandle, out _threadSizeX, out _threadSizeY, out _);

            _shader.SetInt("input_row_size", Weights.GetLength(0));
            _shader.SetInt("weights_row_size", Weights.GetLength(1));

            _weightsBuffer = new ComputeBuffer(Weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(Biases.Length, sizeof(float));

            _shader.SetBuffer(_kernelHandle, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandle, "biases", _biasesBuffer);

            _tt = Shader.PropertyToID("input_column_size");
        }

        public override void Forward(float[,] inputs)
        {
            Inputs = inputs;

            if (_inputBuffer == null)
            {
                Output = new float[Inputs.GetLength(0), Weights.GetLength(1)];

                _threadGroupX = Mathf.CeilToInt(Output.GetLength(0) / (float)_threadSizeX);
                _threadGroupY = Mathf.CeilToInt(Output.GetLength(1) / (float)_threadSizeY);

                _inputBuffer = new ComputeBuffer(Inputs.Length, sizeof(float));
                _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));

                _shader.SetBuffer(_kernelHandle, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandle, "output", _outputBuffer);
                _shader.SetInt(_tt, Inputs.GetLength(0));
                
                _weightsBuffer.SetData(Weights);
                _biasesBuffer.SetData(Biases);
            }

            // Weights and Biases only need to be sent if we are training, as they keep changing 
            _inputBuffer.SetData(Inputs);
            _shader.Dispatch(_kernelHandle, _threadGroupX, _threadGroupY, 1);
            _outputBuffer.GetData(Output);
        }

        public override void Backward(float[,] dValues)
        {
        }
        
        public void Dispose()
        {
            _inputBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _weightsBuffer?.Dispose();
            _biasesBuffer?.Dispose();
        }
    }
}