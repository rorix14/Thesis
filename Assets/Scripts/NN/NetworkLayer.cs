// TODO: see if the math NN library needs to be in this name space

using System.Linq;
using NN.CPU_Single;
using UnityEngine;

namespace NN
{
    public class NetworkLayer
    {
        // Neural network variables
        private float[,] _inputs;
        private readonly float[,] _weights;
        private readonly float[,] _biases;
        public float[,] Output;

        // Compute buffer variables
        private readonly ComputeShader _shader;

        private ComputeBuffer _inputBuffer;
        private ComputeBuffer _outputBuffer;
        private readonly ComputeBuffer _weightsBuffer;
        private readonly ComputeBuffer _biasesBuffer;

        private readonly int _kernelHandleForward;
        private readonly uint _threadSizeXForward;
        private readonly uint _threadSizeYForward;
        private int _threadGroupXForward;
        private int _threadGroupYForward;

        public NetworkLayer(int nInputs, int nNeurons, ComputeShader shader)
        {
            Random.InitState(42);
            _weights = new float[nInputs, nNeurons];
            _biases = new float[1, nNeurons];

            for (int i = 0; i < _weights.GetLength(1); i++)
            {
                _biases[0, i] = 0.01f * NnMath.RandomGaussian(-4.0f, 4.0f);
                for (int j = 0; j < _weights.GetLength(0); j++)
                {
                    _weights[j, i] = 0.01f * NnMath.RandomGaussian(-4.0f, 4.0f); //* Random.value;
                }
            }

            _shader = shader;
            _kernelHandleForward = _shader.FindKernel("forward_pass_ReLU");
            _shader.GetKernelThreadGroupSizes(_kernelHandleForward, out _threadSizeXForward, out _threadSizeYForward,
                out _);

            _shader.SetInt("input_row_size", _weights.GetLength(0));
            _shader.SetInt("weights_row_size", _weights.GetLength(1));

            _weightsBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleForward, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleForward, "biases", _biasesBuffer);
            _weightsBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
        }

        public void Forward(float[,] inputs)
        {
            if (_inputBuffer is null)
            {
                _shader.SetInt("input_column_size", inputs.GetLength(0));

                Output = new float[inputs.GetLength(0), _weights.GetLength(1)];

                _threadGroupXForward = Mathf.CeilToInt(Output.GetLength(0) / (float)_threadSizeXForward);
                _threadGroupYForward = Mathf.CeilToInt(Output.GetLength(1) / (float)_threadSizeYForward);

                _inputBuffer = new ComputeBuffer(inputs.Length, sizeof(float));
                _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));
                _shader.SetBuffer(_kernelHandleForward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleForward, "output", _outputBuffer);
            }

            _inputBuffer.SetData(inputs);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXForward, _threadGroupYForward, 1);
            _outputBuffer.GetData(Output);
        }

        public float[,] _dWeights;
        public float[,] _dBiases;
        public float[,] DInputs;

        private ComputeBuffer _dValuesBuffer;
        private ComputeBuffer _dInputsBuffer;
        private ComputeBuffer _dWeightsBuffer;
        private ComputeBuffer _dBiasesBuffer;

        private int _kernelHandleWeightsBiasesBackward;
        private int _kernelHandleInputsBackward;
        private uint _threadSizeXBackward;
        private uint _threadSizeYBackward;
        private int _threadGroupXWeightsBackward;
        private int _threadGroupYWeightsBackward;
        private int _threadGroupXInputsBackward;
        private int _threadGroupYInputsBackward;

        public void Backward(float[,] dValues)
        {
            if (_dWeights is null)
            {
                _dWeights = new float[_weights.GetLength(0), _weights.GetLength(1)];
                _dBiases = new float[_biases.GetLength(0), _biases.GetLength(1)];
                DInputs = new float[dValues.GetLength(0), _weights.GetLength(0)];

                _kernelHandleWeightsBiasesBackward = _shader.FindKernel("backwards_pass_ReLU_weights_biases");
                _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_ReLU_inputs");

                _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBiasesBackward, out _threadSizeXBackward,
                    out _threadSizeYBackward, out _);

                _dValuesBuffer = new ComputeBuffer(dValues.Length, sizeof(float));
                _dInputsBuffer = new ComputeBuffer(DInputs.Length, sizeof(float));
                _dWeightsBuffer = new ComputeBuffer(_dWeights.Length, sizeof(float));
                _dBiasesBuffer = new ComputeBuffer(_dBiases.Length, sizeof(float));

                _shader.SetBuffer(_kernelHandleInputsBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "d_inputs", _dInputsBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_weights", _dWeightsBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_biases", _dBiasesBuffer);

                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "weights", _weightsBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "output", _outputBuffer);

                _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBiasesBackward, out _threadSizeXBackward,
                    out _threadSizeYBackward, out _);
                _threadGroupXWeightsBackward = Mathf.CeilToInt(_dWeights.GetLength(0) / (float)_threadSizeXBackward);
                _threadGroupYWeightsBackward = Mathf.CeilToInt(_dWeights.GetLength(1) / (float)_threadSizeYBackward);

                _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out _);
                _threadGroupXInputsBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)x);
                _threadGroupYInputsBackward = Mathf.CeilToInt(DInputs.GetLength(1) / (float)y);
            }
            
            _dValuesBuffer.SetData(dValues);
            _shader.Dispatch(_kernelHandleWeightsBiasesBackward, _threadGroupXWeightsBackward,
                _threadGroupYWeightsBackward, 1);
            _dWeightsBuffer.GetData(_dWeights);
            _dBiasesBuffer.GetData(_dBiases);

            _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
                _threadGroupYInputsBackward, 1);
            _dInputsBuffer.GetData(DInputs);
        }

        public void Dispose()
        {
            _inputBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _weightsBuffer?.Dispose();
            _biasesBuffer?.Dispose();

            _dValuesBuffer?.Dispose();
            _dInputsBuffer?.Dispose();
            _dWeightsBuffer?.Dispose();
            _dBiasesBuffer?.Dispose();
        }
    }
}